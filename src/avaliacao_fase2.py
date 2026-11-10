"""
FASE 2 - Treino, comparação e avaliação do modelo de custo.

Rode a partir da raiz do repositório:
    python src/avaliacao_fase2.py      (ou: python src/modelo_custo.py)

Gera em outputs/:
    csv/dataset_trechos_fase2.csv         dados usados no treino (com faltantes)
    csv/fase2_comparacao_modelos.csv      MAE/RMSE/R² de cada modelo, 2 cenários
    csv/fase2_erro_por_grupo.csv          MAE por tipo de via, acidente, horário...
    csv/fase2_erro_por_trecho.csv         MAE de cada estrada (estradas novas)
    graficos/fase2_*.png                  comparação, previsto x real, resíduos
    modelos/modelo_custo.joblib (+ .json) modelo final usado pela busca
    fase2_resumo.md                       resumo em texto para o relatório
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from modelo_custo import (  # noqa: E402
    ALVO, CENARIOS_VALIDACAO, PreditorTempo, carregar_dataset, comparar_modelos,
    escolher_melhor, metricas, previsoes_fora_da_amostra, resumo_faltantes,
    salvar_modelo, treinar_final,
)

SAIDA = Path(__file__).resolve().parent.parent / "outputs"
CORES = {"estradas conhecidas": "#1D76DB", "estradas novas": "#C8102E"}


# =====================================================================
# SEGMENTOS PARA A ANÁLISE DE ERROS
# =====================================================================
def faixa_horaria(hora):
    if hora < 6:
        return "madrugada (0-6h)"
    if 6.5 <= hora < 10:
        return "pico manhã (6h30-10h)"
    if 16 <= hora < 20.5:
        return "pico tarde (16h-20h30)"
    return "fora de pico"


def adicionar_segmentos(df):
    df = df.copy()
    df["faixa_horaria"] = df["hora"].map(faixa_horaria)
    df["tem_acidente"] = df["acidente"].map({0: "sem acidente", 1: "com acidente"})
    mediana = df["distancia"].median()
    df["comprimento"] = (df["distancia"] > mediana).map(
        {False: f"curto (<= {mediana:g} km)", True: f"longo (> {mediana:g} km)"})
    df["tipo_via_obs"] = df["tipo_via"].fillna("(faltante)")
    return df


SEGMENTOS = {
    "tipo_via_obs": "Tipo de via",
    "tem_acidente": "Acidente",
    "faixa_horaria": "Faixa horária",
    "comprimento": "Comprimento do trecho",
}


def erro_por_grupo(previsoes_por_cenario):
    """MAE e viés (erro médio) por segmento, lado a lado nos dois cenários."""
    linhas = []
    for cenario, df in previsoes_por_cenario.items():
        df = adicionar_segmentos(df)
        for coluna, rotulo in SEGMENTOS.items():
            for valor, g in df.groupby(coluna):
                linhas.append({
                    "Cenário": cenario, "Segmento": rotulo, "Valor": valor,
                    "Amostras": len(g),
                    "Tempo médio real (min)": round(g[ALVO].mean(), 1),
                    "MAE (min)": round(g["erro"].abs().mean(), 2),
                    "Erro relativo (%)": round(100 * (g["erro"].abs() / g[ALVO]).mean(), 1),
                    "Viés (min)": round(g["erro"].mean(), 2),
                })
    return pd.DataFrame(linhas)


def erro_por_trecho(previsoes):
    df = adicionar_segmentos(previsoes)
    return (df.groupby("trecho")
              .agg(distancia=("distancia", "first"),
                   tipo_via=("tipo_via", lambda s: s.mode().iat[0] if s.notna().any() else None),
                   tempo_medio=(ALVO, "mean"),
                   mae=("erro", lambda e: e.abs().mean()),
                   vies=("erro", "mean"))
              .round(2).sort_values("mae", ascending=False).reset_index())


# =====================================================================
# GRÁFICOS
# =====================================================================
def grafico_comparacao(tabela, pasta):
    fig, ax = plt.subplots(figsize=(9, 5))
    modelos = tabela["Modelo"].tolist()
    largura = 0.38
    for i, cenario in enumerate(CENARIOS_VALIDACAO):
        pos = [k + (i - 0.5) * largura for k in range(len(modelos))]
        barras = ax.bar(pos, tabela[f"MAE {cenario} (min)"], largura,
                        yerr=tabela[f"MAE {cenario} desvio"], capsize=3,
                        color=CORES[cenario], label=cenario.capitalize())
        ax.bar_label(barras, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(range(len(modelos)), modelos, rotation=15)
    ax.set_ylabel("MAE (min) — menor é melhor")
    ax.set_title("Fase 2: erro dos modelos na validação cruzada (5 dobras)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "fase2_comparacao_modelos.png", dpi=200)
    plt.close(fig)


def grafico_previsto_real(previsoes_por_cenario, nome_modelo, pasta):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (cenario, df) in zip(eixos, previsoes_por_cenario.items()):
        cores = df["acidente"].map({0: CORES[cenario], 1: "#F59E0B"})
        ax.scatter(df[ALVO], df["previsto"], s=6, alpha=0.4, c=cores)
        lim = max(df[ALVO].max(), df["previsto"].max())
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        m = metricas(df[ALVO], df["previsto"])
        ax.set_title(f"{cenario.capitalize()}\nMAE {m['MAE']:.1f} min · R² {m['R2']:.3f}")
        ax.set_xlabel("Tempo real observado (min)")
    eixos[0].set_ylabel("Tempo previsto (min)")
    fig.suptitle(f"{nome_modelo}: previsto x real (laranja = trecho com acidente)")
    fig.tight_layout()
    fig.savefig(pasta / "fase2_previsto_vs_real.png", dpi=200)
    plt.close(fig)


def grafico_residuos(previsoes_por_cenario, nome_modelo, pasta):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (cenario, df) in zip(eixos, previsoes_por_cenario.items()):
        ax.scatter(df["previsto"], df["erro"], s=6, alpha=0.4, c=CORES[cenario])
        ax.axhline(0, color="k", lw=1)
        ax.set_title(cenario.capitalize())
        ax.set_xlabel("Tempo previsto (min)")
    eixos[0].set_ylabel("Resíduo = previsto − real (min)")
    fig.suptitle(f"{nome_modelo}: resíduos")
    fig.tight_layout()
    fig.savefig(pasta / "fase2_residuos.png", dpi=200)
    plt.close(fig)


def grafico_erro_por_grupo(tabela_grupos, pasta):
    segmentos = list(SEGMENTOS.values())
    fig, eixos = plt.subplots(1, len(segmentos), figsize=(16, 4.5), sharey=True)
    for ax, seg in zip(eixos, segmentos):
        t = tabela_grupos[tabela_grupos["Segmento"] == seg]
        pivot = t.pivot(index="Valor", columns="Cenário", values="MAE (min)")
        pivot = pivot[[c for c in CENARIOS_VALIDACAO if c in pivot]]
        pivot.plot.bar(ax=ax, color=[CORES[c] for c in pivot.columns], legend=False, rot=25)
        ax.set_title(seg)
        ax.set_xlabel("")
    eixos[0].set_ylabel("MAE (min)")
    eixos[-1].legend(title="Cenário")
    fig.suptitle("Onde o modelo erra mais")
    fig.tight_layout()
    fig.savefig(pasta / "fase2_erro_por_grupo.png", dpi=200)
    plt.close(fig)


# =====================================================================
# RESUMO EM TEXTO
# =====================================================================
def _linha_tabela(valores):
    return "| " + " | ".join(str(v) for v in valores) + " |"


def escrever_resumo(df, faltantes, tabela, melhor, desempenho, grupos, trechos, caminho):
    colunas = ["Modelo", "MAE estradas conhecidas (min)", "R² estradas conhecidas",
               "MAE estradas novas (min)", "R² estradas novas"]
    linhas = [
        "# Fase 2 — Resumo do modelo de custo (gerado automaticamente)",
        "",
        f"- Observações: **{len(df)}** ({df['trecho'].nunique()} trechos × "
        f"{len(df) // df['trecho'].nunique()} travessias), tempo médio "
        f"{df[ALVO].mean():.1f} min (mín. {df[ALVO].min():.1f}, máx. {df[ALVO].max():.1f}).",
        "- Faltantes injetados e tratados (mediana para numéricas, moda para `tipo_via`): "
        + ", ".join(f"`{c}` {p}%" for c, p in faltantes.items()) + ".",
        "- Codificação: one-hot em `tipo_via`; hora como seno/cosseno; padronização z-score "
        "nas numéricas. Tudo dentro do `Pipeline`, ajustado só nas dobras de treino.",
        "",
        "## Comparação (validação cruzada, 5 dobras)",
        "",
        _linha_tabela(colunas),
        _linha_tabela(["---"] * len(colunas)),
        *[_linha_tabela([r[c] for c in colunas]) for _, r in tabela.iterrows()],
        "",
        f"**Modelo escolhido:** {melhor} (menor MAE em estradas novas, o cenário mais exigente).",
        "",
        "## Desempenho do modelo escolhido (previsões fora da amostra)",
        "",
    ]
    for cenario, m in desempenho.items():
        linhas.append(f"- {cenario.capitalize()}: MAE {m['MAE']:.2f} min, "
                      f"RMSE {m['RMSE']:.2f} min, R² {m['R2']:.3f}")
    linhas += ["", "## Onde o modelo erra mais", ""]
    for cenario in CENARIOS_VALIDACAO:
        g = grupos[grupos["Cenário"] == cenario]
        piores = g.sort_values("MAE (min)", ascending=False).head(4)
        linhas.append(f"- {cenario.capitalize()}: " + "; ".join(
            f"{r['Segmento'].lower()} = {r['Valor']} (MAE {r['MAE (min)']} min, "
            f"viés {r['Viés (min)']:+} min)" for _, r in piores.iterrows()))
    linhas += ["", "Trechos mais difíceis quando a estrada nunca foi vista (estradas novas):", ""]
    for _, r in trechos.head(5).iterrows():
        linhas.append(f"- {r['trecho']} ({r['distancia']:g} km, {r['tipo_via']}): "
                      f"MAE {r['mae']} min, viés {r['vies']:+} min, tempo médio {r['tempo_medio']} min")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


# =====================================================================
# EXECUÇÃO COMPLETA
# =====================================================================
def executar_treino_e_avaliacao(saida=SAIDA, verbose=True):
    pasta_csv, pasta_graf = saida / "csv", saida / "graficos"
    pasta_csv.mkdir(parents=True, exist_ok=True)
    pasta_graf.mkdir(parents=True, exist_ok=True)

    df = carregar_dataset()
    df.to_csv(pasta_csv / "dataset_trechos_fase2.csv", index=False)
    faltantes = resumo_faltantes(df)

    tabela = comparar_modelos(df)
    tabela.to_csv(pasta_csv / "fase2_comparacao_modelos.csv", index=False)
    melhor = escolher_melhor(tabela)

    previsoes = {c: previsoes_fora_da_amostra(df, melhor, c) for c in CENARIOS_VALIDACAO}
    desempenho = {c: metricas(p[ALVO], p["previsto"]) for c, p in previsoes.items()}
    grupos = erro_por_grupo(previsoes)
    grupos.to_csv(pasta_csv / "fase2_erro_por_grupo.csv", index=False)
    trechos = erro_por_trecho(previsoes["estradas novas"])
    trechos.to_csv(pasta_csv / "fase2_erro_por_trecho.csv", index=False)

    grafico_comparacao(tabela, pasta_graf)
    grafico_previsto_real(previsoes, melhor, pasta_graf)
    grafico_residuos(previsoes, melhor, pasta_graf)
    grafico_erro_por_grupo(grupos, pasta_graf)

    pipeline = treinar_final(df, melhor)
    caminho_modelo = salvar_modelo(pipeline, melhor, tabela)
    escrever_resumo(df, faltantes, tabela, melhor, desempenho, grupos, trechos,
                    saida / "fase2_resumo.md")

    if verbose:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("\n" + "=" * 70 + "\nFASE 2: COMPARAÇÃO DOS MODELOS DE CUSTO\n" + "=" * 70)
            print(tabela.drop(columns=[c for c in tabela if "desvio" in c]).to_string(index=False))
        print(f"\nModelo escolhido: {melhor}")
        for c, m in desempenho.items():
            print(f"  {c:>20}: MAE {m['MAE']:.2f} min | RMSE {m['RMSE']:.2f} | R² {m['R2']:.3f}")
        print(f"Modelo salvo em '{caminho_modelo}'. Resumo em '{saida / 'fase2_resumo.md'}'.")

    return PreditorTempo(pipeline, melhor), tabela


if __name__ == "__main__":
    executar_treino_e_avaliacao()
