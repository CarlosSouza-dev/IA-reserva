"""
FASE 3 - Núcleo do agente com LLM (Laboratórios 8, 9 e 10).

O modelo de linguagem ORQUESTRA; ele não calcula. O ciclo é o do Laboratório 9:

    1. pensar   o LLM lê o pedido e decide se chama uma ferramenta
                (saída estruturada: <tool_call>{"name": ..., "arguments": ...}</tool_call>)
    2. agir     executamos a ferramenta pedida (ferramentas.py: A* + modelo de custo)
    3. observar o resultado (JSON) volta para a conversa como mensagem "tool"
    4. responder o LLM explica a rota em português, usando só os números recebidos

Modelo padrão: Qwen/Qwen2.5-1.5B-Instruct (o mesmo dos laboratórios), rodando
localmente via `transformers`. Não há chave de API no código: se precisar de
token do Hugging Face, ele vem da variável de ambiente HF_TOKEN ou dos Secrets
do Colab.

Uso:
    python src/agente_llm.py "Saio de Pallet às 18h e entrego em Celadon e Lavender"
    python src/agente_llm.py              (modo conversa; "sair" encerra)

No notebook:
    from agente_llm import criar_agente
    agente = criar_agente()
    agente.responder("...")["resposta"]
"""
import json
import os
import re
import time
from pathlib import Path

from ferramentas import FERRAMENTAS, executar_ferramenta

RAIZ = Path(__file__).resolve().parent.parent
PASTA_PROMPTS = RAIZ / "prompts"
MODELO_PADRAO = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_PADRAO = "v3"
MAX_PASSOS = 3          # pensar/agir/observar no máximo 3 vezes por pedido
MAX_TOKENS_SAIDA = 320  # resposta final curta (o prompt pede até 5 frases)

NOMES_FERRAMENTAS = {f["function"]["name"] for f in FERRAMENTAS}


# =====================================================================
# 1. PROMPTS VERSIONADOS
# =====================================================================
def carregar_prompt(versao=PROMPT_PADRAO):
    """Lê prompts/<versao>.md, sem os comentários HTML de documentação."""
    caminho = PASTA_PROMPTS / f"{versao}.md"
    if not caminho.exists():
        disponiveis = sorted(p.stem for p in PASTA_PROMPTS.glob("*.md"))
        raise FileNotFoundError(f"Prompt '{versao}' não existe. Disponíveis: {disponiveis}")
    texto = caminho.read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", texto, flags=re.S).strip()


# =====================================================================
# 2. O MODELO DE LINGUAGEM
# =====================================================================
def _token_hf():
    """Token do Hugging Face, se houver: variável de ambiente ou Colab Secrets."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:  # pragma: no cover - só existe dentro do Colab
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


class LLMQwen:
    """Envolve um modelo de chat do Hugging Face. Chamar o objeto com
    (mensagens, ferramentas) devolve o texto gerado; cada chamada fica em
    `self.uso` (tokens de entrada/saída e segundos) para a análise de custo."""

    def __init__(self, nome=MODELO_PADRAO, dispositivo=None, max_tokens=MAX_TOKENS_SAIDA):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.nome = nome
        self.max_tokens = max_tokens
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        tipo = torch.float16 if self.dispositivo == "cuda" else torch.float32
        token = _token_hf()
        self.tokenizador = AutoTokenizer.from_pretrained(nome, token=token)
        try:
            modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=tipo, token=token)
        except TypeError:  # transformers antigos usam torch_dtype
            modelo = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=tipo, token=token)
        self.modelo = modelo.to(self.dispositivo).eval()
        self.uso = []

    def __call__(self, mensagens, ferramentas=None):
        entrada = self.tokenizador.apply_chat_template(
            mensagens, tools=ferramentas, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.dispositivo)
        n_entrada = entrada["input_ids"].shape[-1]
        inicio = time.perf_counter()
        with self.torch.no_grad():
            saida = self.modelo.generate(**entrada, max_new_tokens=self.max_tokens,
                                         do_sample=False, temperature=None, top_p=None,
                                         top_k=None)
        gerado = saida[0][n_entrada:]
        self.uso.append({"tokens_entrada": int(n_entrada), "tokens_saida": int(len(gerado)),
                         "segundos": round(time.perf_counter() - inicio, 3)})
        return self.tokenizador.decode(gerado, skip_special_tokens=True).strip()


# =====================================================================
# 3. LER A SAÍDA ESTRUTURADA DO LLM
# =====================================================================
def extrair_json(texto):
    """Primeiro objeto JSON do texto (do 1º '{' ao último '}'), ou None."""
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim <= inicio:
        return None
    try:
        return json.loads(texto[inicio:fim + 1])
    except json.JSONDecodeError:
        return None


_BLOCO_TOOL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def ler_chamadas(texto):
    """Pedidos de ferramenta na resposta do LLM -> [{"name", "arguments"}].

    Formato esperado (template do Qwen): blocos <tool_call>{...}</tool_call>.
    Tolerâncias: bloco sem fechamento (resposta cortada) e JSON "solto" com
    "name" de uma ferramenta conhecida (modelos pequenos às vezes esquecem a tag).
    Lista vazia = o LLM não pediu ferramenta (é a resposta final).
    """
    blocos = _BLOCO_TOOL.findall(texto)
    candidatos = [extrair_json(b) for b in blocos] if blocos else [extrair_json(texto)]
    chamadas = []
    for obj in candidatos:
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        if not blocos and obj["name"] not in NOMES_FERRAMENTAS:
            continue  # JSON solto qualquer não é pedido de ferramenta
        argumentos = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(argumentos, str):
            argumentos = extrair_json(argumentos) or {}
        chamadas.append({"name": obj["name"], "arguments": argumentos or {}})
    return chamadas


def remover_chamadas(texto):
    return _BLOCO_TOOL.sub("", texto).strip()


# =====================================================================
# 4. O QUE VOLTA PARA O LLM E A EXPLICAÇÃO DE RESERVA
# =====================================================================
def resumir_para_llm(resultado):
    """Versão enxuta do JSON da ferramenta: menos tokens e menos chance de o
    modelo pequeno se perder (tira os caminhos de cada perna e contadores)."""
    if not resultado.get("ok") or "rota" not in resultado:
        return resultado
    resumo = {k: resultado[k] for k in (
        "ok", "origem", "ordem_entregas", "rota", "tempo_previsto_min", "distancia_km",
        "hora_saida", "economia_vs_mais_curta_min", "produto", "avisos") if k in resultado}
    # A rota mais curta vai COMPLETA: sem ela, o Qwen inventava uma no teste.
    resumo["rota_mais_curta"] = {k: resultado["rota_mais_curta"][k]
                                 for k in ("rota", "distancia_km", "tempo_previsto_min")}
    if len(resultado["pernas"]) > 1:  # com 1 perna, repetiria o tempo total
        resumo["pernas"] = [{"de": p["de"], "para": p["para"],
                             "tempo_previsto_min": p["tempo_previsto_min"]}
                            for p in resultado["pernas"]]
    restricoes = {k: v for k, v in resultado.get("restricoes", {}).items() if v}
    if restricoes:
        resumo["restricoes"] = restricoes
    return resumo


def _hora_texto(hora):
    h, m = int(hora), round((hora - int(hora)) * 60)
    return f"{h:02d}h{m:02d}"


def explicar_resultado(resultado):
    """Explicação determinística (sem LLM) de um resultado de ferramenta.
    Usada como reserva quando o modelo não produz resposta final."""
    if not resultado.get("ok"):
        texto = f"Não consegui calcular: {resultado.get('erro')}."
        if resultado.get("sugestoes"):
            texto += " Você quis dizer: " + ", ".join(resultado["sugestoes"]) + "?"
        return texto
    if "rota" not in resultado:
        return json.dumps(resultado, ensure_ascii=False)
    partes = []
    if resultado.get("avisos"):
        partes.append("Obs.: " + "; ".join(resultado["avisos"]) + ".")
    if len(resultado["ordem_entregas"]) > 1:
        # vírgulas, não setas: setas ficam reservadas para estradas reais (fidelidade)
        partes.append("Ordem das entregas: " + ", depois ".join(resultado["ordem_entregas"]) + ".")
    partes.append(f"Rota saindo às {_hora_texto(resultado['hora_saida'])}: "
                  + " → ".join(resultado["rota"]) + ".")
    partes.append(f"Tempo previsto: {resultado['tempo_previsto_min']:.0f} min "
                  f"({resultado['distancia_km']:.0f} km).")
    economia = resultado.get("economia_vs_mais_curta_min", 0)
    if economia > 0:
        partes.append(f"Economiza {economia:.0f} min em relação à rota mais curta "
                      f"({resultado['rota_mais_curta']['distancia_km']:.0f} km, "
                      f"{resultado['rota_mais_curta']['tempo_previsto_min']:.0f} min).")
    else:
        partes.append("É também a rota mais curta em distância.")
    return " ".join(partes)


# =====================================================================
# 5. CHECAGEM DE FIDELIDADE (barreira contra alucinação)
#
# No teste com o Qwen2.5-1.5B, a escolha da ferramenta foi correta, mas a
# resposta final às vezes inventava trechos, tempos por perna e uma "rota mais
# curta" inexistente. Esta checagem compara a resposta com o que as
# ferramentas devolveram:
#   * todo número citado precisa vir do resultado ou do pedido (com tolerância
#     de arredondamento e conversão de minutos para "Xh Ymin");
#   * todo local do mapa citado precisa aparecer no resultado ou no pedido;
#   * toda sequência "A → B" precisa ser uma estrada que existe no mapa.
# Se falhar, o agente descarta o texto do LLM e usa explicar_resultado().
# =====================================================================
from ferramentas import LOCAIS, _chave  # noqa: E402
from mapas import MAPA_GRANDE  # noqa: E402

_NUMERO = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
_SETA = re.compile(r"\s*(?:→|->|=>|⇒)\s*")
_FORMAS = sorted(
    {(" ".join(_chave(local)), local) for local in LOCAIS}
    | {(" ".join(p for p in _chave(local) if p not in {"city", "town"}), local) for local in LOCAIS},
    key=lambda par: -len(par[0]))


def _locais_no_texto(texto):
    """Locais do mapa citados no texto, na ordem em que aparecem."""
    normal = " " + " ".join(_chave(texto)) + " "
    achados = []
    for forma, local in _FORMAS:
        for m in re.finditer(rf"(?<= ){re.escape(forma)}(?= )", normal):
            achados.append((m.start(), local))
            normal = normal[:m.start()] + "#" * len(forma) + normal[m.end():]
    return [local for _, local in sorted(achados)]


def _numeros_de(valor, saida):
    if isinstance(valor, bool):
        return
    if isinstance(valor, (int, float)):
        saida.add(float(valor))
        if valor >= 60:
            horas, minutos = divmod(round(valor), 60)
            saida.update({float(horas), float(minutos), round(valor / 60, 1)})
    elif isinstance(valor, dict):
        for v in valor.values():
            _numeros_de(v, saida)
    elif isinstance(valor, (list, tuple)):
        for v in valor:
            _numeros_de(v, saida)
    elif isinstance(valor, str):
        for n in _NUMERO.findall(valor):
            _numeros_de(float(n.replace(",", ".")), saida)


def verificar_fidelidade(resposta, resultados, pedido=""):
    """Lista de problemas da resposta em relação aos resultados (vazia = fiel)."""
    permitidos = set()
    _numeros_de(list(resultados), permitidos)
    _numeros_de(pedido, permitidos)
    locais_permitidos = set(_locais_no_texto(json.dumps(list(resultados), ensure_ascii=False)))
    locais_permitidos |= set(_locais_no_texto(pedido))
    pequenos = len(max((r.get("rota", []) for r in resultados), key=len, default=[])) + 1

    problemas = []
    for bruto in _NUMERO.findall(resposta):
        n = float(bruto.replace(",", "."))
        if n <= pequenos and n == int(n):
            continue  # numeração de lista, "2 paradas"...
        if not any(abs(n - p) <= (1.0 if p >= 10 else 0.05) for p in permitidos):
            problemas.append(f"número {bruto} não veio das ferramentas")
    for local in _locais_no_texto(resposta):
        if local not in locais_permitidos:
            problemas.append(f"local '{local}' não está no resultado")
    if not resultados and _SETA.search(resposta) and _locais_no_texto(resposta):
        problemas.append("rota citada sem consultar a ferramenta")
    for linha in resposta.splitlines():
        partes = _SETA.split(linha)
        for esquerda, direita in zip(partes, partes[1:]):
            a, b = _locais_no_texto(esquerda), _locais_no_texto(direita)
            if a and b and b[0] not in MAPA_GRANDE.get(a[-1], {}):
                problemas.append(f"trecho inexistente: {a[-1]} → {b[0]}")
    return list(dict.fromkeys(problemas))


RESPOSTA_SEM_DADOS = ("Não consegui calcular a rota com segurança. Informe a origem, "
                      "os destinos e o horário de saída.")


# =====================================================================
# 6. O AGENTE
# =====================================================================
class Agente:
    def __init__(self, llm, versao_prompt=PROMPT_PADRAO, max_passos=MAX_PASSOS,
                 ferramentas=FERRAMENTAS):
        self.llm = llm
        self.versao_prompt = versao_prompt
        self.prompt_sistema = carregar_prompt(versao_prompt)
        self.max_passos = max_passos
        self.ferramentas = ferramentas

    def responder(self, pedido, mostrar=False):
        """Executa o ciclo pensar -> agir -> observar -> responder.

        Retorna dict com: resposta, chamadas [{nome, argumentos, resultado}],
        passos, usou_reserva (True se a resposta final veio de
        `explicar_resultado` e não do LLM), segundos, tokens e versão do prompt.
        """
        mensagens = [{"role": "system", "content": self.prompt_sistema},
                     {"role": "user", "content": pedido}]
        uso_antes = len(getattr(self.llm, "uso", []))
        inicio = time.perf_counter()
        chamadas, resposta, passos = [], "", 0

        for passos in range(1, self.max_passos + 1):
            texto = self.llm(mensagens, self.ferramentas)                 # 1. pensar
            pedidos = ler_chamadas(texto)
            if not pedidos:
                resposta = remover_chamadas(texto)                        # 4. responder
                break
            mensagens.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": p["name"], "arguments": p["arguments"]}}
                for p in pedidos]})
            for p in pedidos:
                resultado = executar_ferramenta(p["name"], p["arguments"])  # 2. agir
                chamadas.append({"nome": p["name"], "argumentos": p["arguments"],
                                 "resultado": resultado})
                if mostrar:
                    print(f"   [ferramenta] {p['name']} {json.dumps(p['arguments'], ensure_ascii=False)}")
                    print(f"   [resultado ] ok={resultado.get('ok')} "
                          f"{resultado.get('erro') or resultado.get('tempo_previsto_min', '')}")
                mensagens.append({"role": "tool", "name": p["name"],              # 3. observar
                                  "content": json.dumps(resumir_para_llm(resultado),
                                                        ensure_ascii=False)})

        # Barreira contra alucinação: a resposta do LLM só passa se for fiel
        resposta_llm = resposta
        resultados = [c["resultado"] for c in chamadas]
        problemas = verificar_fidelidade(resposta, resultados, pedido) if resposta else []
        usou_reserva = False
        if not resposta or problemas:
            usou_reserva = True
            resposta = (explicar_resultado(resultados[-1]) if resultados
                        else RESPOSTA_SEM_DADOS)

        uso = getattr(self.llm, "uso", [])[uso_antes:]
        return {
            "pedido": pedido,
            "resposta": resposta,
            "resposta_llm": resposta_llm,
            "problemas_fidelidade": problemas,
            "chamadas": chamadas,
            "passos": passos,
            "usou_reserva": usou_reserva,
            "segundos": round(time.perf_counter() - inicio, 3),
            "chamadas_llm": len(uso),
            "tokens_entrada": sum(u["tokens_entrada"] for u in uso),
            "tokens_saida": sum(u["tokens_saida"] for u in uso),
            "versao_prompt": self.versao_prompt,
            "modelo": getattr(self.llm, "nome", type(self.llm).__name__),
        }


def criar_agente(versao_prompt=PROMPT_PADRAO, modelo=MODELO_PADRAO, dispositivo=None):
    return Agente(LLMQwen(modelo, dispositivo), versao_prompt)


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Agente de rotas com LLM (Fase 3)")
    parser.add_argument("pedido", nargs="*", help="pedido em linguagem natural")
    parser.add_argument("--prompt", default=PROMPT_PADRAO, help="versão do prompt (v1, v2)")
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    args = parser.parse_args()

    print(f"Carregando {args.modelo}...")
    agente = criar_agente(args.prompt, args.modelo)
    pedidos = [" ".join(args.pedido)] if args.pedido else None
    while True:
        pedido = pedidos.pop(0) if pedidos else (None if pedidos == [] else input("\nVocê: "))
        if pedido is None or pedido.strip().lower() in {"sair", "exit", "quit"}:
            break
        r = agente.responder(pedido, mostrar=True)
        print(f"\nAgente: {r['resposta']}")
        print(f"   ({r['segundos']} s, {r['chamadas_llm']} chamadas ao LLM, "
              f"{r['tokens_entrada']}+{r['tokens_saida']} tokens"
              + (", resposta de reserva" if r["usou_reserva"] else "") + ")")
