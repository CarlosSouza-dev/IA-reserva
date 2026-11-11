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
PROMPT_PADRAO = "v2"
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
    resumo["rota_mais_curta"] = {k: resultado["rota_mais_curta"][k]
                                 for k in ("distancia_km", "tempo_previsto_min")}
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
        partes.append("Ordem das entregas: " + " → ".join(resultado["ordem_entregas"]) + ".")
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
# 5. O AGENTE
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

        usou_reserva = False
        if not resposta and chamadas:
            resposta, usou_reserva = explicar_resultado(chamadas[-1]["resultado"]), True
        if not resposta:
            resposta = "Não consegui entender o pedido. Informe a origem, os destinos e o horário."
            usou_reserva = True

        uso = getattr(self.llm, "uso", [])[uso_antes:]
        return {
            "pedido": pedido,
            "resposta": resposta,
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
