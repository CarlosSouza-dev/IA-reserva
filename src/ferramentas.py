"""
FASE 3 - Ferramentas que o agente com LLM pode acionar.

O LLM NÃO calcula rota (modelos de linguagem erram contas e inventam
caminhos). Ele interpreta o pedido e chama estas funções, que devolvem
resultados verificáveis vindos da busca A* (Fase 1) sobre o custo aprendido
(Fase 2). Base: Laboratório 9 (ferramentas, `executar`) e Laboratório 8
(validar os dados antes de chamar o A*).

Contrato:
  * toda ferramenta devolve um dict serializável em JSON;
  * sucesso -> {"ok": True, ...}; falha -> {"ok": False, "erro": "...", ...},
    com mensagem clara para o LLM explicar ao usuário (nunca uma exceção);
  * nomes de cidade passam por `normalizar_cidade` (maiúsculas, acentos,
    sufixos "City/Town", apelidos e erros de digitação). Correções feitas
    voltam no campo "avisos" para o LLM não esconder isso do usuário.

`FERRAMENTAS` traz os esquemas no formato aceito por
`tokenizer.apply_chat_template(..., tools=FERRAMENTAS)`.
"""
import difflib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone

from mapas import MAPA_GRANDE, PONTOS_VERDES
from planejamento import (
    MAX_PARADAS_EXATO, RotaInalcancavel, avaliar_rota, montar_grafo_custo,
    planejar_entregas,
)
from trechos import TRECHOS, chave_trecho

# Horário de Brasília (sem horário de verão desde 2019). O Colab roda em UTC,
# então datetime.now() sem fuso daria "agora" 3 horas adiantado.
FUSO_BRASILIA = timezone(timedelta(hours=-3))

LOCAIS = sorted(MAPA_GRANDE)
CIDADES = [c for c in LOCAIS if c not in PONTOS_VERDES]

# Apelidos e grafias comuns -> nome oficial do mapa
APELIDOS = {
    "vermilion": "Vermillion City",
    "vermilion city": "Vermillion City",
    "indigo": "Indigo Plateau",
    "liga pokemon": "Indigo Plateau",
    "liga": "Indigo Plateau",
    "planalto indigo": "Indigo Plateau",
    "cinnabar": "Cinnabar Island",
    "ilha cinnabar": "Cinnabar Island",
    "monte prateado": "Mt. Silver",
    "mt silver": "Mt. Silver",
    "mount silver": "Mt. Silver",
    "mt moon": "Mt. Moon",
    "mount moon": "Mt. Moon",
    "mt mortar": "Mt. Mortar",
    "floresta de viridian": "Viridian Forest",
    "floresta ilex": "Ilex Forest",
    "lago da furia": "Lake of Rage",
}


# =====================================================================
# 1. NORMALIZAÇÃO E VALIDAÇÃO DOS ARGUMENTOS
# =====================================================================
class ArgumentoInvalido(ValueError):
    def __init__(self, mensagem, sugestoes=None):
        super().__init__(mensagem)
        self.sugestoes = sugestoes or []


def _chave(texto):
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", sem_acento.lower()).split()


def _forma_curta(nome):
    palavras = [p for p in _chave(nome) if p not in {"city", "town", "cidade", "de"}]
    return " ".join(palavras)


_INDICE = {}
for _local in LOCAIS:
    _INDICE[" ".join(_chave(_local))] = _local
    _INDICE.setdefault(_forma_curta(_local), _local)
for _apelido, _local in APELIDOS.items():
    _INDICE[" ".join(_chave(_apelido))] = _local


def normalizar_cidade(nome, avisos=None):
    """Nome livre -> nome oficial do mapa. Levanta ArgumentoInvalido com
    sugestões se não houver correspondência segura."""
    if not isinstance(nome, str) or not nome.strip():
        raise ArgumentoInvalido("Nome de cidade vazio ou inválido")
    if nome in MAPA_GRANDE:
        return nome
    chave = " ".join(_chave(nome))
    curta = _forma_curta(nome)
    for candidata in (chave, curta):
        if candidata in _INDICE:
            oficial = _INDICE[candidata]
            # "Pallet" -> "Pallet Town" é óbvio; só avisa quando foi um apelido
            if avisos is not None and _forma_curta(oficial) != curta:
                avisos.append(f"'{nome}' interpretado como '{oficial}'")
            return oficial
    proximas = difflib.get_close_matches(curta or chave, list(_INDICE), n=3, cutoff=0.75)
    oficiais = list(dict.fromkeys(_INDICE[p] for p in proximas))
    if len(oficiais) == 1:
        if avisos is not None:
            avisos.append(f"'{nome}' interpretado como '{oficiais[0]}' (correção de digitação)")
        return oficiais[0]
    sugestoes = oficiais or difflib.get_close_matches(nome, LOCAIS, n=3, cutoff=0.4)
    raise ArgumentoInvalido(f"'{nome}' não existe no mapa de Kanto-Johto", sugestoes)


def interpretar_hora(valor, agora=None):
    """'agora', 18, 18.5, '18h', '18h30', '18:30', '6pm' -> hora decimal [0, 24)."""
    if valor is None or (isinstance(valor, str) and _chave(valor) in (["agora"], ["now"], [])):
        agora = agora or datetime.now(FUSO_BRASILIA)
        return round(agora.hour + agora.minute / 60, 2)
    if isinstance(valor, bool):
        raise ArgumentoInvalido(f"Horário inválido: {valor!r}")
    if isinstance(valor, (int, float)):
        hora = float(valor)
    else:
        texto = str(valor).strip().lower()
        m = re.fullmatch(r"(\d{1,2})\s*(?:h|:|hs|horas?)?\s*(\d{2})?\s*(?:min)?\s*(am|pm)?", texto)
        if not m:
            raise ArgumentoInvalido(f"Horário inválido: {valor!r} (use 0-23, '18h30' ou 'agora')")
        hora = int(m.group(1)) + (int(m.group(2)) / 60 if m.group(2) else 0)
        if m.group(3) == "pm" and hora < 12:
            hora += 12
        if m.group(3) == "am" and hora >= 12:
            hora -= 12
    if not 0 <= hora < 24:
        raise ArgumentoInvalido(f"Horário fora do intervalo 0-24h: {valor!r}")
    return round(hora, 2)


def _lista(valor):
    if valor is None:
        return []
    if isinstance(valor, str):
        return [v for v in re.split(r"\s*[,;]\s*", valor) if v]
    return list(valor)


def _pares(valor, avisos):
    """[["A","B"], "C-D", {"de":"E","para":"F"}] -> [(A, B), (C, D), (E, F)],
    exigindo que cada par seja uma estrada existente."""
    pares = []
    for item in _lista(valor):
        if isinstance(item, dict):
            a, b = item.get("de") or item.get("origem"), item.get("para") or item.get("destino")
        elif isinstance(item, str):
            partes = re.split(r"\s*(?:->|–|—| - |-| e | a | até )\s*", item, maxsplit=1)
            a, b = (partes + [None])[:2]
        else:
            a, b = (list(item) + [None, None])[:2]
        a, b = normalizar_cidade(a, avisos), normalizar_cidade(b, avisos)
        if chave_trecho(a, b) not in TRECHOS:
            raise ArgumentoInvalido(f"Não existe estrada direta entre {a} e {b}",
                                    sorted(MAPA_GRANDE[a]))
        pares.append((a, b))
    return pares


def _erro(mensagem, sugestoes=None, avisos=None):
    saida = {"ok": False, "erro": mensagem}
    if sugestoes:
        saida["sugestoes"] = sugestoes
    if avisos:
        saida["avisos"] = list(dict.fromkeys(avisos))
    return saida


# =====================================================================
# 2. PREDITOR DA FASE 2 (carregado uma vez)
# =====================================================================
_PREDITOR = None


def preditor_padrao():
    global _PREDITOR
    if _PREDITOR is None:
        from modelo_custo import obter_preditor
        _PREDITOR = obter_preditor()
    return _PREDITOR


def definir_preditor(preditor):
    """Permite injetar outro preditor (testes, comparação com o simulador)."""
    global _PREDITOR
    _PREDITOR = preditor


def _fmt(minutos):
    return round(float(minutos), 1)


# =====================================================================
# 3. FERRAMENTAS
# =====================================================================
def buscar_rota(origem, destino=None, paradas=None, hora="agora", evitar=None,
                acidentes=None, bloqueios=None, retornar_origem=False, produto=None):
    """Melhor rota em tempo previsto, de `origem` até `destino` e/ou passando
    por todas as `paradas` (a ordem de visita é otimizada)."""
    avisos = []
    try:
        origem = normalizar_cidade(origem, avisos)
        alvos = [normalizar_cidade(p, avisos) for p in _lista(paradas)]
        if destino:
            alvos.append(normalizar_cidade(destino, avisos))
        if not alvos:
            return _erro("Informe ao menos um destino ou parada", avisos=avisos)
        hora_dec = interpretar_hora(hora)
        evitar = [normalizar_cidade(c, avisos) for c in _lista(evitar)]
        acidentes = _pares(acidentes, avisos)
        bloqueios = _pares(bloqueios, avisos)

        restricoes = dict(evitar_cidades=evitar, bloqueios=bloqueios,
                          obrigatorias=[origem, *alvos])
        g_tempo = montar_grafo_custo(hora_dec, "tempo", preditor_padrao(), acidentes, **restricoes)
        plano = planejar_entregas(origem, alvos, g_tempo, retornar_origem)

        # Comparação com a decisão ingênua da Fase 1 (menor distância)
        g_dist = montar_grafo_custo(hora_dec, "distancia", **restricoes)
        curto = planejar_entregas(origem, alvos, g_dist, retornar_origem)
        tempo_curto = avaliar_rota(curto["rota"], g_tempo)
    except ArgumentoInvalido as e:
        return _erro(str(e), e.sugestoes, avisos)
    except (RotaInalcancavel, ValueError) as e:
        return _erro(str(e), avisos=avisos)

    saida = {
        "ok": True,
        "origem": origem,
        "ordem_entregas": plano["ordem"],
        "rota": plano["rota"],
        "tempo_previsto_min": _fmt(plano["custo_total"]),
        "distancia_km": round(avaliar_rota(plano["rota"], g_dist), 1),
        "hora_saida": hora_dec,
        "pernas": [{"de": a, "para": b, "caminho": cam, "tempo_previsto_min": _fmt(c)}
                   for a, b, cam, c in plano["pernas"]],
        "rota_mais_curta": {
            "rota": curto["rota"],
            "distancia_km": round(curto["custo_total"], 1),
            "tempo_previsto_min": _fmt(tempo_curto),
        },
        "economia_vs_mais_curta_min": _fmt(max(0.0, tempo_curto - plano["custo_total"])),
        "metodo_ordem": plano["metodo"],
        "nos_expandidos": plano["nos_expandidos"],
        "restricoes": {"evitar": evitar, "acidentes": [list(p) for p in acidentes],
                       "bloqueios": [list(p) for p in bloqueios]},
    }
    if produto:
        saida["produto"] = str(produto)
    if avisos:
        saida["avisos"] = list(dict.fromkeys(avisos))
    return saida


def estimar_tempo_trecho(origem, destino, hora="agora", acidente=False):
    """Tempo previsto (min) de UMA estrada direta entre duas cidades vizinhas."""
    avisos = []
    try:
        a, b = normalizar_cidade(origem, avisos), normalizar_cidade(destino, avisos)
        if chave_trecho(a, b) not in TRECHOS:
            raise ArgumentoInvalido(f"{a} e {b} não são vizinhas (não há estrada direta)",
                                    sorted(MAPA_GRANDE[a]))
        hora_dec = interpretar_hora(hora)
    except ArgumentoInvalido as e:
        return _erro(str(e), e.sugestoes, avisos)
    trecho = TRECHOS[chave_trecho(a, b)]
    saida = {
        "ok": True, "de": a, "para": b, "hora": hora_dec, "acidente": bool(acidente),
        "tempo_previsto_min": _fmt(preditor_padrao()(trecho, hora_dec, bool(acidente))),
        "distancia_km": trecho.distancia, "tipo_via": trecho.tipo_via,
        "lombadas": trecho.lombadas, "semaforos": trecho.semaforos,
    }
    if avisos:
        saida["avisos"] = avisos
    return saida


def listar_cidades():
    """Locais válidos do mapa (cidades e pontos naturais)."""
    return {"ok": True, "cidades": CIDADES, "pontos_naturais": sorted(PONTOS_VERDES)}


def listar_vizinhas(cidade):
    """Locais ligados por estrada direta a `cidade`, com a distância em km."""
    avisos = []
    try:
        cidade = normalizar_cidade(cidade, avisos)
    except ArgumentoInvalido as e:
        return _erro(str(e), e.sugestoes)
    saida = {"ok": True, "cidade": cidade,
             "vizinhas": [{"local": v, "distancia_km": d}
                          for v, d in sorted(MAPA_GRANDE[cidade].items())]}
    if avisos:
        saida["avisos"] = avisos
    return saida


# =====================================================================
# 4. ESQUEMAS PARA O LLM E DESPACHO
# =====================================================================
_CIDADE = {"type": "string", "description": "Nome de um local do mapa de Kanto-Johto"}
_TRECHOS = {
    "type": "array",
    "description": "Estradas diretas, cada uma como [cidade_a, cidade_b]",
    "items": {"type": "array", "items": {"type": "string"}},
}
_HORA = {"type": "string",
         "description": "Hora de saída: 'agora', '8', '18h30' ou '18:30'. Use 'agora' se o usuário não disser."}

FERRAMENTAS = [
    {"type": "function", "function": {
        "name": "buscar_rota",
        "description": ("Calcula a rota de entrega mais rápida (tempo previsto pelo modelo de "
                        "tráfego) da origem até o destino e/ou passando por várias paradas, "
                        "otimizando a ordem de visita. Use SEMPRE que o usuário pedir uma rota."),
        "parameters": {"type": "object", "properties": {
            "origem": {**_CIDADE, "description": "Cidade de partida"},
            "destino": {**_CIDADE, "description": "Destino final (opcional se houver paradas)"},
            "paradas": {"type": "array", "items": {"type": "string"},
                        "description": "Endereços (cidades) de entrega, em qualquer ordem"},
            "hora": _HORA,
            "evitar": {"type": "array", "items": {"type": "string"},
                       "description": "Cidades ou regiões por onde NÃO passar"},
            "acidentes": {**_TRECHOS, "description": "Estradas com acidente (ficam mais lentas)"},
            "bloqueios": {**_TRECHOS, "description": "Estradas interditadas (não podem ser usadas)"},
            "retornar_origem": {"type": "boolean", "description": "Voltar à origem no final"},
            "produto": {"type": "string", "description": "Produto a ser entregue, se informado"},
        }, "required": ["origem"]},
    }},
    {"type": "function", "function": {
        "name": "estimar_tempo_trecho",
        "description": "Tempo previsto (min) de uma única estrada direta entre duas cidades vizinhas.",
        "parameters": {"type": "object", "properties": {
            "origem": _CIDADE, "destino": _CIDADE, "hora": _HORA,
            "acidente": {"type": "boolean", "description": "Há acidente nessa estrada?"},
        }, "required": ["origem", "destino"]},
    }},
    {"type": "function", "function": {
        "name": "listar_cidades",
        "description": "Lista todos os locais válidos do mapa.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "listar_vizinhas",
        "description": "Lista os locais ligados por estrada direta a uma cidade.",
        "parameters": {"type": "object", "properties": {"cidade": _CIDADE},
                       "required": ["cidade"]},
    }},
]

_FUNCOES = {
    "buscar_rota": buscar_rota,
    "estimar_tempo_trecho": estimar_tempo_trecho,
    "listar_cidades": listar_cidades,
    "listar_vizinhas": listar_vizinhas,
}


def executar_ferramenta(nome, argumentos=None):
    """Executa a ferramenta pedida pelo LLM. Nunca levanta exceção: ferramenta
    inexistente ou argumento inesperado viram {"ok": False, "erro": ...}."""
    funcao = _FUNCOES.get(nome)
    if funcao is None:
        return _erro(f"A ferramenta '{nome}' não existe", sorted(_FUNCOES))
    if isinstance(argumentos, str):
        try:
            argumentos = json.loads(argumentos or "{}")
        except json.JSONDecodeError:
            return _erro("Argumentos não estão em JSON válido")
    if not isinstance(argumentos or {}, dict):
        return _erro("Argumentos devem ser um objeto JSON")
    try:
        return funcao(**(argumentos or {}))
    except TypeError as e:
        return _erro(f"Argumentos inválidos para '{nome}': {e}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    exemplo = executar_ferramenta("buscar_rota", {
        "origem": "pallet", "paradas": ["Celadon", "lavender", "Saffron City"],
        "hora": "18h", "acidentes": [["Saffron City", "Vermillion City"]], "produto": "Poké Balls",
    })
    print(json.dumps(exemplo, ensure_ascii=False, indent=2))
    print(f"(ordem exata até {MAX_PARADAS_EXATO} paradas)")
