"""
FASE 2 - Modelo de custo APRENDIDO: prevê o tempo (min) de cada trecho.

Na Fase 1 o custo de uma aresta era a distância. Em `trechos.py` existe um
simulador que sabe o tempo "verdadeiro" de cada trecho, mas o agente NÃO pode
consultá-lo: no mundo real ninguém tem essa fórmula. O agente só vê DADOS
(observações de travessias com ruído) e precisa aprender com eles.

Fluxo deste módulo (base: Laboratório 4, preparação sem vazamento):

    1. carregar_dataset()        observações sintéticas + faltantes injetados
    2. montar_preprocessador()   imputação, one-hot, hora cíclica, padronização
    3. comparar_modelos()        baseline, Ridge, k-NN, árvore e floresta com
                                 validação cruzada (estradas conhecidas x novas)
    4. treinar_final()/salvar()  melhor modelo ajustado em todos os dados
    5. PreditorTempo             a interface que a busca A* usa como custo

Todo o pré-processamento fica DENTRO do Pipeline do scikit-learn: em cada
dobra da validação cruzada a mediana, a moda e a média/desvio da padronização
são calculadas só com as linhas de treino daquela dobra (sem vazamento).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from trechos import SEED, gerar_dataset_sintetico

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "outputs" / "modelos"
CAMINHO_MODELO = PASTA_MODELOS / "modelo_custo.joblib"
CAMINHO_METADADOS = PASTA_MODELOS / "modelo_custo.json"

# =====================================================================
# 1. CONFIGURAÇÃO (documentada para o relatório)
# =====================================================================
COLUNAS_NUMERICAS = ["distancia", "lombadas", "semaforos", "acidente"]
COLUNA_HORA = ["hora"]
COLUNAS_CATEGORICAS = ["tipo_via"]
COLUNAS_ENTRADA = COLUNAS_NUMERICAS + COLUNA_HORA + COLUNAS_CATEGORICAS
ALVO = "tempo_min"
GRUPO = "trecho"  # identificador da estrada, usado para agrupar a validação

# Fração de valores apagados de propósito: dados reais têm buracos (sensor que
# falhou, cadastro incompleto) e o PDF pede tratamento de faltantes documentado.
COLUNAS_COM_FALTANTES = ("lombadas", "semaforos", "tipo_via")
FRACAO_FALTANTES = 0.02

N_DOBRAS = 5               # validação cruzada GroupKFold
TEMPO_MINIMO_MIN = 0.1     # previsão nunca <= 0 (a heurística do A* exige custo > 0)


# =====================================================================
# 2. DADOS
# =====================================================================
def carregar_dataset(n_por_trecho=60, fracao_faltantes=FRACAO_FALTANTES, seed=SEED):
    """Gera as observações (trecho, hora, acidente) -> tempo e injeta faltantes.

    Cada linha é uma travessia observada. O tempo vem do simulador com ruído
    multiplicativo (ver `trechos.gerar_dataset_sintetico`). Depois, uma fração
    `fracao_faltantes` de cada coluna em COLUNAS_COM_FALTANTES vira NaN, de
    forma reprodutível (semente fixa).
    """
    df = pd.DataFrame(gerar_dataset_sintetico(n_por_trecho=n_por_trecho, seed=seed))
    df[GRUPO] = df["origem"] + " | " + df["destino"]
    df["lombadas"] = df["lombadas"].astype(float)
    df["semaforos"] = df["semaforos"].astype(float)

    rng = np.random.default_rng(seed)
    for coluna in COLUNAS_COM_FALTANTES:
        apagar = rng.random(len(df)) < fracao_faltantes
        df[coluna] = df[coluna].astype(object if coluna == "tipo_via" else float)
        df.loc[apagar, coluna] = np.nan
    return df


def resumo_faltantes(df):
    """Percentual de faltantes por coluna (apenas as que têm algum)."""
    pct = df.isna().mean().mul(100).round(2)
    return pct[pct > 0]


# =====================================================================
# 3. PRÉ-PROCESSAMENTO
# =====================================================================
def codificar_hora(X):
    """Hora (0-24) -> (seno, cosseno). Assim 23h e 0h ficam vizinhas, o que um
    número de 0 a 24 não representa (para ele, 23 e 0 estão no extremo oposto)."""
    angulo = 2 * np.pi * np.asarray(X, dtype=float) / 24.0
    return np.hstack([np.sin(angulo), np.cos(angulo)])


def _nomes_hora(_transformador, _entradas):
    return np.array(["hora_sin", "hora_cos"], dtype=object)


def montar_preprocessador():
    """ColumnTransformer com as três decisões de pré-processamento:

    * numéricas: faltante -> mediana do treino; depois padronização (z-score),
      indispensável para o k-NN, que mede distância entre exemplos;
    * hora: codificação cíclica seno/cosseno (já fica em [-1, 1]);
    * tipo_via: faltante -> moda do treino; depois one-hot (sem ordem falsa
      entre "asfalto" e "terra").
    """
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("padronizar", StandardScaler()),
    ])
    hora = FunctionTransformer(codificar_hora, feature_names_out=_nomes_hora)
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numericas", numericas, COLUNAS_NUMERICAS),
        ("hora", hora, COLUNA_HORA),
        ("categoricas", categoricas, COLUNAS_CATEGORICAS),
    ])


# =====================================================================
# 4. MODELOS CANDIDATOS (todos vistos na disciplina)
# =====================================================================
def modelos_candidatos(seed=SEED):
    """Dicionário nome -> estimador. O baseline prevê sempre a média do treino:
    qualquer modelo útil precisa ganhar dele com folga."""
    return {
        "Baseline (média)": DummyRegressor(strategy="mean"),
        "Regressão Ridge": Ridge(alpha=1.0),
        "k-NN (k=10)": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "Árvore de Decisão": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5,
                                                    random_state=seed),
        "Floresta Aleatória": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                    n_jobs=-1, random_state=seed),
    }


def montar_pipeline(estimador):
    return Pipeline([("preprocessamento", montar_preprocessador()), ("modelo", estimador)])


def _x_y_grupos(df):
    return df[COLUNAS_ENTRADA], df[ALVO].to_numpy(), df[GRUPO].to_numpy()


# =====================================================================
# 5. COMPARAÇÃO COM VALIDAÇÃO CRUZADA EM DOIS CENÁRIOS
#
# O mesmo trecho aparece 60 vezes no dataset (horas e acidentes diferentes).
# Isso pede duas perguntas distintas, e cada uma tem sua validação:
#
#   * "estradas conhecidas" (KFold embaralhado): o modelo prevê uma estrada já
#     observada em OUTRAS horas. É o uso real do agente, que só roda sobre as
#     45 estradas do próprio mapa.
#   * "estradas novas" (GroupKFold por trecho): cada estrada fica inteira numa
#     dobra só, então o modelo prevê uma estrada que NUNCA viu (ex.: uma via
#     nova). Com KFold comum essa pergunta teria resposta otimista, porque o
#     modelo "decora" a estrada.
#
# O modelo é ESCOLHIDO pelo cenário mais exigente (estradas novas), e os dois
# números são reportados.
# =====================================================================
CENARIOS_VALIDACAO = ("estradas conhecidas", "estradas novas")


def _validador(cenario, n_dobras, grupos, seed=SEED):
    """(divisor de dobras, grupos a repassar) para o cenário de validação."""
    if cenario == "estradas conhecidas":
        return KFold(n_splits=n_dobras, shuffle=True, random_state=seed), None
    if cenario == "estradas novas":
        return GroupKFold(n_splits=n_dobras), grupos
    raise ValueError(f"cenário de validação desconhecido: {cenario}")


def comparar_modelos(df, modelos=None, n_dobras=N_DOBRAS):
    """Tabela com MAE, RMSE e R² (média e desvio nas dobras) de cada modelo,
    nos dois cenários de validação."""
    modelos = modelos or modelos_candidatos()
    X, y, grupos = _x_y_grupos(df)
    pontuacoes = {"mae": "neg_mean_absolute_error",
                  "rmse": "neg_root_mean_squared_error",
                  "r2": "r2"}
    linhas = []
    for nome, estimador in modelos.items():
        linha = {"Modelo": nome}
        for cenario in CENARIOS_VALIDACAO:
            cv, g = _validador(cenario, n_dobras, grupos)
            r = cross_validate(montar_pipeline(estimador), X, y, groups=g, cv=cv,
                               scoring=pontuacoes)
            mae, rmse, r2 = -r["test_mae"], -r["test_rmse"], r["test_r2"]
            linha.update({
                f"MAE {cenario} (min)": round(mae.mean(), 3),
                f"MAE {cenario} desvio": round(mae.std(), 3),
                f"RMSE {cenario} (min)": round(rmse.mean(), 3),
                f"R² {cenario}": round(r2.mean(), 4),
            })
        linha["Tempo de treino (s)"] = round(r["fit_time"].mean(), 3)
        linhas.append(linha)
    return (pd.DataFrame(linhas)
            .sort_values("MAE estradas novas (min)").reset_index(drop=True))


def escolher_melhor(tabela):
    """Menor MAE em estradas novas (cenário exigente), ignorando o baseline."""
    candidatos = tabela[~tabela["Modelo"].str.startswith("Baseline")]
    return candidatos.sort_values("MAE estradas novas (min)").iloc[0]["Modelo"]


def previsoes_fora_da_amostra(df, nome_modelo, cenario="estradas novas", n_dobras=N_DOBRAS):
    """Previsão de cada linha feita por um modelo que NÃO a viu no treino,
    segundo as dobras do `cenario`. É a base da análise de erros."""
    X, y, grupos = _x_y_grupos(df)
    estimador = modelos_candidatos()[nome_modelo]
    cv, g = _validador(cenario, n_dobras, grupos)
    previsto = cross_val_predict(montar_pipeline(estimador), X, y, groups=g, cv=cv)
    saida = df.copy()
    saida["previsto"] = np.maximum(previsto, TEMPO_MINIMO_MIN)
    saida["erro"] = saida["previsto"] - saida[ALVO]
    return saida


def metricas(y_real, y_previsto):
    return {
        "MAE": float(mean_absolute_error(y_real, y_previsto)),
        "RMSE": float(math.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": float(r2_score(y_real, y_previsto)),
    }


# =====================================================================
# 6. MODELO FINAL, PERSISTÊNCIA E PREDITOR USADO PELA BUSCA
# =====================================================================
def treinar_final(df, nome_modelo):
    """Ajusta o pipeline escolhido em TODOS os dados (para uso em produção)."""
    X, y, _ = _x_y_grupos(df)
    pipeline = montar_pipeline(clone(modelos_candidatos()[nome_modelo]))
    return pipeline.fit(X, y)


def salvar_modelo(pipeline, nome_modelo, tabela=None, caminho=CAMINHO_MODELO):
    import joblib

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho)
    meta = {"modelo": nome_modelo, "colunas_entrada": COLUNAS_ENTRADA, "alvo": ALVO}
    if tabela is not None:
        meta["validacao_cruzada"] = tabela.to_dict(orient="records")
    caminho.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return caminho


class PreditorTempo:
    """Envolve o pipeline treinado com a interface que `montar_grafo_tempo` usa.

    * `prever_lote(trechos, hora, acidentes)` -> array de minutos (uma chamada
      ao modelo por grafo: 45 trechos de uma vez, rápido);
    * `preditor(trecho, hora, acidente)` -> minutos de um trecho (compatível
      com o parâmetro `prever` antigo).
    As previsões são limitadas a TEMPO_MINIMO_MIN para o custo nunca ser <= 0.
    """

    def __init__(self, pipeline, nome="modelo"):
        self.pipeline = pipeline
        self.nome = nome

    @classmethod
    def carregar(cls, caminho=CAMINHO_MODELO):
        import joblib

        caminho = Path(caminho)
        nome = "modelo"
        meta = caminho.with_suffix(".json")
        if meta.exists():
            nome = json.loads(meta.read_text(encoding="utf-8")).get("modelo", nome)
        return cls(joblib.load(caminho), nome)

    def prever_lote(self, trechos, hora, acidentes):
        X = pd.DataFrame({
            "distancia": [t.distancia for t in trechos],
            "lombadas": [float(t.lombadas) for t in trechos],
            "semaforos": [float(t.semaforos) for t in trechos],
            "acidente": [int(bool(a)) for a in acidentes],
            "hora": [float(hora)] * len(trechos),
            "tipo_via": [t.tipo_via for t in trechos],
        })[COLUNAS_ENTRADA]
        return np.maximum(self.pipeline.predict(X), TEMPO_MINIMO_MIN)

    def __call__(self, trecho, hora, acidente=False):
        return float(self.prever_lote([trecho], hora, [acidente])[0])


def selecionar_e_treinar(df=None):
    """Compara os modelos, escolhe o melhor, treina em tudo e salva.
    Retorna (PreditorTempo, tabela_de_comparacao)."""
    df = carregar_dataset() if df is None else df
    tabela = comparar_modelos(df)
    melhor = escolher_melhor(tabela)
    pipeline = treinar_final(df, melhor)
    salvar_modelo(pipeline, melhor, tabela)
    return PreditorTempo(pipeline, melhor), tabela


def obter_preditor(caminho=CAMINHO_MODELO):
    """Carrega o modelo salvo; se ainda não existir, treina e salva."""
    if Path(caminho).exists():
        return PreditorTempo.carregar(caminho)
    preditor, _ = selecionar_e_treinar()
    return preditor


if __name__ == "__main__":
    # Treino + avaliação completos (tabelas e gráficos): ver avaliacao_fase2.py
    from avaliacao_fase2 import executar_treino_e_avaliacao

    executar_treino_e_avaliacao()
