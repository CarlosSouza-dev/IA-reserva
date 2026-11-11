"""
FASE 3 - Avaliação do agente com a bateria de pedidos (issue #9, Laboratório 10).

Para cada pedido de testes/pedidos_teste.json, roda o agente e mede, com
critérios definidos ANTES de rodar:

    ferramenta_ok  a 1ª ferramenta chamada é a esperada (ou nenhuma);
    argumentos_ok  argumentos equivalentes aos esperados (pelo significado:
                   "Pallet" == "Pallet Town", "18h" == 18, paradas como conjunto);
    rota_ok        a rota devolvida é a mesma calculada com os argumentos esperados;
    resposta_fiel  a resposta final cita o tempo correto e passa na checagem de
                   fidelidade (ou, sem rota válida, não inventa rota);
    acerto         todos os critérios aplicáveis.

Também registra o que interessa à seção de IA responsável (#10): tokens,
latência, se a resposta do LLM foi descartada pela checagem e por quê.

Uso (a partir da raiz do repositório):
    python src/avaliacao_agente.py                     # prompt v3, Qwen2.5-1.5B
    python src/avaliacao_agente.py --prompt v2 --limite 5
"""
import json
import math
import re
from pathlib import Path

import pandas as pd

import ferramentas as F
from agente_llm import verificar_fidelidade
from trechos import chave_trecho

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CASOS = RAIZ / "testes" / "pedidos_teste.json"
SAIDA = RAIZ / "outputs" / "fase3"
CLASSES = ["buscar_rota", "estimar_tempo_trecho", "listar_vizinhas", "listar_cidades", "nenhuma"]


def carregar_casos(caminho=CAMINHO_CASOS):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))["casos"]


# =====================================================================
# 1. COMPARAÇÃO DE ARGUMENTOS PELO SIGNIFICADO
# =====================================================================
def _cidade(nome):
    """Nome oficial, ou o texto em minúsculas se não existir no mapa
    (casos 'fora do mapa' esperam justamente um nome inválido)."""
    try:
        return F.normalizar_cidade(nome)
    except F.ArgumentoInvalido:
        return str(nome).strip().lower()


def _hora(valor):
    if valor is None or (isinstance(valor, str) and valor.strip().lower() in {"agora", "now", ""}):
        return "agora"
    try:
        return F.interpretar_hora(valor)
    except F.ArgumentoInvalido:
        return f"inválida: {valor}"


def _pares(valor):
    try:
        return {chave_trecho(a, b) for a, b in F._pares(valor, [])}
    except (F.ArgumentoInvalido, TypeError, ValueError):
        return {"inválido"}


def _conjunto(valor):
    return {_cidade(c) for c in F._lista(valor)}


def normalizar_argumentos(ferramenta, args):
    """Forma canônica dos argumentos, para comparar esperado x obtido."""
    args = args or {}
    if ferramenta == "buscar_rota":
        destinos = _conjunto(args.get("paradas"))
        if args.get("destino"):
            destinos.add(_cidade(args["destino"]))
        return {
            "origem": _cidade(args.get("origem")) if args.get("origem") else None,
            "destinos": destinos,
            "hora": _hora(args.get("hora")),
            "evitar": _conjunto(args.get("evitar")),
            "acidentes": _pares(args.get("acidentes")),
            "bloqueios": _pares(args.get("bloqueios")),
            "retornar_origem": bool(args.get("retornar_origem", False)),
        }
    if ferramenta == "estimar_tempo_trecho":
        return {"trecho": chave_trecho(_cidade(args.get("origem")), _cidade(args.get("destino"))),
                "hora": _hora(args.get("hora")), "acidente": bool(args.get("acidente", False))}
    if ferramenta == "listar_vizinhas":
        return {"cidade": _cidade(args.get("cidade"))}
    return {}


def comparar_argumentos(ferramenta, esperados, obtidos):
    """(ok, diferenças). O produto só é exigido quando o caso o espera."""
    e, o = normalizar_argumentos(ferramenta, esperados), normalizar_argumentos(ferramenta, obtidos)
    diferencas = [f"{k}: esperado {e[k]!r}, obtido {o.get(k)!r}" for k in e if e[k] != o.get(k)]
    if (esperados or {}).get("produto"):
        produto = str((obtidos or {}).get("produto", "")).lower()
        if esperados["produto"].lower() not in produto:
            diferencas.append(f"produto: esperado {esperados['produto']!r}, obtido {produto!r}")
    return not diferencas, diferencas


# =====================================================================
# 2. AVALIAÇÃO DE UM CASO
# =====================================================================
def _tempo_citado(resposta, minutos):
    """A resposta cita o tempo previsto (arredondado ou em horas e minutos)?"""
    numeros = {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", resposta)}
    h, m = divmod(round(minutos), 60)
    return any(abs(n - minutos) <= 1 for n in numeros) or {float(h), float(m)} <= numeros


def avaliar_caso(caso, resultado_agente):
    chamadas = resultado_agente["chamadas"]
    primeira = chamadas[0] if chamadas else None
    escolhida = primeira["nome"] if primeira else "nenhuma"
    esperada = caso["ferramenta"] or "nenhuma"

    ferramenta_ok = escolhida == esperada
    if caso["ferramenta"] is None:
        argumentos_ok, diferencas = ferramenta_ok, []
    elif ferramenta_ok:
        argumentos_ok, diferencas = comparar_argumentos(esperada, caso["argumentos"], primeira["argumentos"])
    else:
        argumentos_ok, diferencas = False, []

    resposta = resultado_agente["resposta"]
    rota_ok, fiel = None, None
    esperado = F.executar_ferramenta(esperada, caso["argumentos"]) if caso["ferramenta"] else None
    rota_valida = bool(esperado and esperado.get("ok") and "rota" in esperado)
    if rota_valida:
        obtido = primeira["resultado"] if primeira else {}
        rota_ok = obtido.get("rota") == esperado["rota"]
        fiel = (_tempo_citado(resposta, esperado["tempo_previsto_min"])
                and not verificar_fidelidade(resposta, [c["resultado"] for c in chamadas],
                                             caso["pedido"]))
    elif esperado and esperado.get("ok"):          # outras ferramentas com sucesso
        fiel = not verificar_fidelidade(resposta, [c["resultado"] for c in chamadas], caso["pedido"])
    else:                                          # sem ferramenta ou fora do mapa
        fiel = not verificar_fidelidade(resposta, [c["resultado"] for c in chamadas], caso["pedido"])

    acerto = ferramenta_ok and argumentos_ok and (rota_ok is not False) and bool(fiel)
    return {
        "id": caso["id"], "categoria": caso["categoria"], "pedido": caso["pedido"],
        "ferramenta_esperada": esperada, "ferramenta_escolhida": escolhida,
        "ferramenta_ok": ferramenta_ok, "argumentos_ok": argumentos_ok,
        "rota_ok": rota_ok, "resposta_fiel": fiel, "acerto": acerto,
        "diferencas_argumentos": "; ".join(diferencas),
        "argumentos_obtidos": json.dumps(primeira["argumentos"], ensure_ascii=False) if primeira else "",
        "resultado_ok": primeira["resultado"].get("ok") if primeira else None,
        "llm_descartado": bool(resultado_agente.get("problemas_fidelidade")),
        "problemas_fidelidade": "; ".join(resultado_agente.get("problemas_fidelidade", [])),
        "usou_reserva": resultado_agente["usou_reserva"],
        "resposta_llm": resultado_agente.get("resposta_llm", ""),
        "resposta": resposta,
        "passos": resultado_agente["passos"],
        "chamadas_llm": resultado_agente["chamadas_llm"],
        "tokens_entrada": resultado_agente["tokens_entrada"],
        "tokens_saida": resultado_agente["tokens_saida"],
        "segundos": resultado_agente["segundos"],
    }


# =====================================================================
# 3. MÉTRICAS AGREGADAS
# =====================================================================
def precisao_revocacao(esperadas, escolhidas, classe):
    vp = sum(e == classe and c == classe for e, c in zip(esperadas, escolhidas))
    fp = sum(e != classe and c == classe for e, c in zip(esperadas, escolhidas))
    fn = sum(e == classe and c != classe for e, c in zip(esperadas, escolhidas))
    precisao = vp / (vp + fp) if vp + fp else math.nan
    revocacao = vp / (vp + fn) if vp + fn else math.nan
    return {"classe": classe, "VP": vp, "FP": fp, "FN": fn,
            "precisão": round(precisao, 3), "revocação": round(revocacao, 3)}


def resumir(df):
    por_categoria = (df.groupby("categoria", sort=False)
                       [["ferramenta_ok", "argumentos_ok", "acerto"]].mean().mul(100).round(0))
    por_categoria.insert(0, "casos", df.groupby("categoria", sort=False).size())
    matriz = pd.crosstab(df["ferramenta_esperada"], df["ferramenta_escolhida"],
                         rownames=["esperada"], colnames=["escolhida"])
    classes = [c for c in CLASSES if c in set(df["ferramenta_esperada"]) | set(df["ferramenta_escolhida"])]
    pr = pd.DataFrame([precisao_revocacao(df["ferramenta_esperada"], df["ferramenta_escolhida"], c)
                       for c in classes])
    geral = {
        "casos": len(df),
        "acerto (%)": round(100 * df["acerto"].mean(), 1),
        "ferramenta certa (%)": round(100 * df["ferramenta_ok"].mean(), 1),
        "argumentos certos (%)": round(100 * df["argumentos_ok"].mean(), 1),
        "rota certa": f"{int(df['rota_ok'].fillna(False).sum())}/{int(df['rota_ok'].notna().sum())}",
        "resposta fiel": f"{int(df['resposta_fiel'].fillna(False).sum())}/{int(df['resposta_fiel'].notna().sum())}",
        "resposta do LLM descartada": int(df["llm_descartado"].sum()),
        "segundos por pedido (média)": round(df["segundos"].mean(), 1),
        "tokens por pedido (média)": round((df["tokens_entrada"] + df["tokens_saida"]).mean()),
    }
    return geral, por_categoria, matriz, pr


def _md_tabela(df, indice=True):
    df = df.reset_index() if indice else df
    linhas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    linhas += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(linhas)


def escrever_relatorio(df, rotulo, caminho):
    geral, por_categoria, matriz, pr = resumir(df)
    erros = df[~df["acerto"]]
    linhas = [f"# Bateria de pedidos do agente — {rotulo}", "",
              *[f"- **{k}:** {v}" for k, v in geral.items()], "",
              "## Acerto por categoria (%)", "", _md_tabela(por_categoria), "",
              "## Matriz de confusão da escolha de ferramenta", "", _md_tabela(matriz), "",
              "## Precisão e revocação por ferramenta", "", _md_tabela(pr, indice=False), "",
              f"## Casos com erro ({len(erros)})", ""]
    for _, e in erros.iterrows():
        motivo = []
        if not e["ferramenta_ok"]:
            motivo.append(f"ferramenta {e['ferramenta_escolhida']} (esperada {e['ferramenta_esperada']})")
        if e["ferramenta_ok"] and not e["argumentos_ok"]:
            motivo.append(f"argumentos: {e['diferencas_argumentos']}")
        if e["rota_ok"] is False:
            motivo.append("rota diferente da esperada")
        if not e["resposta_fiel"]:
            motivo.append("resposta não fiel")
        linhas += [f"- **#{e['id']} ({e['categoria']})** {e['pedido']}",
                   f"  - motivo: {'; '.join(motivo)}",
                   f"  - chamada: `{e['argumentos_obtidos'] or '(nenhuma)'}`",
                   f"  - resposta: {e['resposta'][:220].replace(chr(10), ' ')}"]
    descartes = df[df["llm_descartado"]]
    if len(descartes):
        linhas += ["", f"## Respostas do LLM barradas pela checagem de fidelidade ({len(descartes)})", ""]
        for _, d in descartes.iterrows():
            linhas += [f"- **#{d['id']}** {d['problemas_fidelidade']}",
                       f"  - texto do LLM: {d['resposta_llm'][:220].replace(chr(10), ' ')}"]
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return geral


# =====================================================================
# 4. EXECUÇÃO
# =====================================================================
def _tipos(df):
    """CSV relido volta com booleanos como texto e vazios como NaN."""
    def booleano(v):
        return v in (True, "True")

    def opcional(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else booleano(v)

    for coluna in ("rota_ok", "resposta_fiel", "resultado_ok"):
        df[coluna] = df[coluna].map(opcional).astype(object)
    for coluna in ("ferramenta_ok", "argumentos_ok", "acerto", "llm_descartado", "usou_reserva"):
        df[coluna] = df[coluna].map(booleano)
    for coluna in ("diferencas_argumentos", "argumentos_obtidos", "problemas_fidelidade",
                   "resposta_llm", "resposta"):
        df[coluna] = df[coluna].fillna("").astype(str)
    return df


def executar_bateria(agente, casos=None, mostrar=True, parcial=None):
    """Roda os casos. Se `parcial` (caminho de CSV) for dado, grava o progresso
    a cada caso e pula os que já estiverem nele: no CPU a bateria leva ~20 min
    e uma queda no meio não perde o que já rodou."""
    casos = casos or carregar_casos()
    linhas = []
    if parcial and Path(parcial).exists():
        linhas = pd.read_csv(parcial).to_dict(orient="records")
    feitos = {int(linha["id"]) for linha in linhas}
    for caso in casos:
        if caso["id"] in feitos:
            continue
        r = agente.responder(caso["pedido"])
        linha = avaliar_caso(caso, r)
        linhas.append(linha)
        if parcial:
            Path(parcial).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(linhas).to_csv(parcial, index=False)
        if mostrar:
            print(f"{'certo ' if linha['acerto'] else 'ERRADO'} | #{caso['id']:>2} {caso['categoria']:<18} "
                  f"| {linha['segundos']:>5.1f}s | {caso['pedido']}", flush=True)
    return _tipos(pd.DataFrame(linhas).sort_values("id").reset_index(drop=True))


def salvar(df, rotulo, pasta=SAIDA):
    pasta.mkdir(parents=True, exist_ok=True)
    nome = "bateria_" + rotulo.lower().replace(" ", "_").replace("/", "_")
    df.to_csv(pasta / f"{nome}.csv", index=False)
    geral = escrever_relatorio(df, rotulo, pasta / f"{nome}.md")
    return pasta / f"{nome}.csv", geral


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    from agente_llm import MODELO_PADRAO, PROMPT_PADRAO, criar_agente

    parser = argparse.ArgumentParser(description="Avalia o agente na bateria de pedidos")
    parser.add_argument("--prompt", default=PROMPT_PADRAO)
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument("--limite", type=int, default=None, help="rodar só os N primeiros casos")
    args = parser.parse_args()

    casos = carregar_casos()[:args.limite]
    agente = criar_agente(args.prompt, args.modelo)
    rotulo = f"{args.modelo.split('/')[-1]} {args.prompt}"
    parcial = SAIDA / ("parcial_" + rotulo.lower().replace(" ", "_") + ".csv")
    df = executar_bateria(agente, casos, parcial=parcial)
    caminho, geral = salvar(df, rotulo)
    parcial.unlink(missing_ok=True)
    print()
    for k, v in geral.items():
        print(f"{k:>28}: {v}")
    print(f"\nResultados em '{caminho}' e '{caminho.with_suffix('.md')}'")
