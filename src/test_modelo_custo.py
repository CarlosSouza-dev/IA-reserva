"""Testes da Fase 2: pré-processamento, modelo de custo e integração com o A*.

Rode, a partir da raiz do repositório, com:
    python -m unittest discover -s src -v
"""
import itertools
import math
import unittest

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from buscas import a_estrela, busca_custo_uniforme, calibrar_escala_heuristica
from mapas import MAPA_GRANDE, MAPA_PEQUENO
from modelo_custo import (
    ALVO, COLUNAS_ENTRADA, TEMPO_MINIMO_MIN, PreditorTempo, carregar_dataset,
    codificar_hora, comparar_modelos, escolher_melhor, montar_pipeline,
)
from trechos import TRECHOS, montar_grafo_tempo

HORAS = [3, 8, 12, 18, 22]


class TestPreprocessamento(unittest.TestCase):
    def test_hora_ciclica_aproxima_23h_e_0h(self):
        perto = codificar_hora([[23.9], [0.1]])
        longe = codificar_hora([[0.0], [12.0]])
        self.assertLess(np.linalg.norm(perto[0] - perto[1]), 0.1)
        self.assertAlmostEqual(np.linalg.norm(longe[0] - longe[1]), 2.0)

    def test_faltantes_injetados_e_reprodutiveis(self):
        a = carregar_dataset(n_por_trecho=20)
        b = carregar_dataset(n_por_trecho=20)
        self.assertTrue(a.equals(b))
        for coluna in ("lombadas", "semaforos", "tipo_via"):
            self.assertGreater(a[coluna].isna().sum(), 0, coluna)
            self.assertLess(a[coluna].isna().mean(), 0.06, coluna)
        self.assertEqual(a[ALVO].isna().sum(), 0)

    def test_pipeline_trata_faltantes(self):
        df = carregar_dataset(n_por_trecho=10)
        pipe = montar_pipeline(Ridge()).fit(df[COLUNAS_ENTRADA], df[ALVO])
        X = pipe.named_steps["preprocessamento"].transform(df[COLUNAS_ENTRADA])
        self.assertFalse(np.isnan(np.asarray(X, dtype=float)).any())

    def test_comparacao_e_escolha_ignoram_baseline(self):
        df = carregar_dataset(n_por_trecho=8)
        tabela = comparar_modelos(df, {"Baseline (média)": DummyRegressor(),
                                       "Regressão Ridge": Ridge()})
        self.assertEqual(len(tabela), 2)
        self.assertEqual(escolher_melhor(tabela), "Regressão Ridge")
        ridge = tabela.set_index("Modelo").loc["Regressão Ridge"]
        base = tabela.set_index("Modelo").loc["Baseline (média)"]
        self.assertLess(ridge["MAE estradas conhecidas (min)"], base["MAE estradas conhecidas (min)"])
        self.assertLess(ridge["MAE estradas novas (min)"], base["MAE estradas novas (min)"])


class TestIntegracaoComBusca(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = carregar_dataset(n_por_trecho=20)
        modelo = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, random_state=0)
        cls.preditor = PreditorTempo(montar_pipeline(modelo).fit(df[COLUNAS_ENTRADA], df[ALVO]),
                                     "floresta (teste)")

    def test_previsao_positiva(self):
        tempos = self.preditor.prever_lote(list(TRECHOS.values()), 18, [True] * len(TRECHOS))
        self.assertTrue((tempos >= TEMPO_MINIMO_MIN).all())

    def test_lote_igual_a_chamada_individual(self):
        trechos = list(TRECHOS.values())[:5]
        lote = self.preditor.prever_lote(trechos, 8, [False, True, False, True, False])
        um_a_um = [self.preditor(t, 8, a) for t, a in zip(trechos, [False, True, False, True, False])]
        np.testing.assert_allclose(lote, um_a_um)

    def test_grafo_do_modelo_simetrico_e_igual_ao_modo_por_aresta(self):
        g_lote = montar_grafo_tempo(MAPA_GRANDE, 18, prever=self.preditor)
        g_aresta = montar_grafo_tempo(MAPA_GRANDE, 18, prever=lambda t, h, a: self.preditor(t, h, a))
        self.assertEqual(g_lote, g_aresta)
        for a, viz in g_lote.items():
            for b, t in viz.items():
                self.assertGreater(t, 0)
                self.assertEqual(g_lote[b][a], t)

    def test_astar_igual_ucs_em_todos_os_pares_com_modelo(self):
        for hora in HORAS:
            g = montar_grafo_tempo(MAPA_GRANDE, hora, acidentes=[("Mahogany Town", "Ice Path")],
                                   prever=self.preditor)
            escala = calibrar_escala_heuristica(g)
            for o, d in itertools.permutations(g, 2):
                with self.subTest(hora=hora, origem=o, destino=d):
                    self.assertTrue(math.isclose(a_estrela(o, d, g, escala)[1],
                                                 busca_custo_uniforme(o, d, g)[1], abs_tol=1e-6))

    def test_modelo_aprendeu_efeito_do_acidente_e_do_pico(self):
        trechos = list(TRECHOS.values())
        livre = self.preditor.prever_lote(trechos, 12, [False] * len(trechos)).mean()
        acidente = self.preditor.prever_lote(trechos, 12, [True] * len(trechos)).mean()
        madrugada = self.preditor.prever_lote(trechos, 3, [False] * len(trechos)).mean()
        pico = self.preditor.prever_lote(trechos, 18, [False] * len(trechos)).mean()
        self.assertGreater(acidente, livre)
        self.assertGreater(pico, madrugada)

    def test_mapa_pequeno_funciona_com_modelo(self):
        g = montar_grafo_tempo(MAPA_PEQUENO, 8, prever=self.preditor)
        caminho, custo, _, _ = a_estrela("Cianwood City", "Blackthorn City", g)
        self.assertEqual((caminho[0], caminho[-1]), ("Cianwood City", "Blackthorn City"))
        self.assertGreater(custo, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
