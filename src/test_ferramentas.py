"""Testes da Fase 3 (parte determinística): planejamento com várias paradas e
ferramentas do agente. O LLM não entra aqui; estes testes garantem que o que
ele recebe das ferramentas é correto.

Para serem rápidos e reprodutíveis, as ferramentas usam o SIMULADOR como
preditor (definir_preditor). O modelo aprendido é testado em test_modelo_custo.

    python -m unittest discover -s src -v
"""
import inspect
import json
import unittest
from datetime import datetime

from buscas import a_estrela, custo_do_caminho
import ferramentas as F
from mapas import MAPA_GRANDE
from planejamento import (
    RotaInalcancavel, _ordem_heuristica, aplicar_restricoes, matriz_de_rotas,
    montar_grafo_custo, planejar_entregas,
)
from trechos import chave_trecho, tempo_trecho


def rota_valida(rota, grafo):
    return all(b in grafo[a] for a, b in zip(rota, rota[1:]))


class TestPlanejamento(unittest.TestCase):
    def setUp(self):
        self.g = montar_grafo_custo(18, "tempo")

    def test_um_destino_igual_ao_astar(self):
        plano = planejar_entregas("Pewter City", ["Fuchsia City"], self.g)
        caminho, custo, _, _ = a_estrela("Pewter City", "Fuchsia City", self.g)
        self.assertEqual(plano["rota"], caminho)
        self.assertAlmostEqual(plano["custo_total"], custo)

    def test_varias_paradas_rota_completa_e_consistente(self):
        paradas = ["Lavender Town", "Celadon City", "Cerulean City", "Fuchsia City"]
        plano = planejar_entregas("Pallet Town", paradas, self.g)
        self.assertEqual(plano["rota"][0], "Pallet Town")
        self.assertEqual(sorted(plano["ordem"]), sorted(paradas))
        self.assertTrue(set(paradas) <= set(plano["rota"]))
        self.assertTrue(rota_valida(plano["rota"], self.g))
        self.assertAlmostEqual(plano["custo_total"], custo_do_caminho(plano["rota"], self.g))
        self.assertEqual(plano["metodo"], "exato")

    def test_heuristica_nunca_melhor_que_o_exato(self):
        paradas = ["Lavender Town", "Celadon City", "Cerulean City", "Fuchsia City",
                   "Azalea Town", "Olivine City"]
        exato = planejar_entregas("Pallet Town", paradas, self.g)
        rotas, _ = matriz_de_rotas(["Pallet Town", *paradas], self.g)
        ordem = _ordem_heuristica("Pallet Town", paradas, rotas, False)
        custo_h = sum(rotas[(a, b)][1] for a, b in zip(["Pallet Town", *ordem], ordem))
        self.assertGreaterEqual(custo_h, exato["custo_total"] - 1e-9)
        self.assertEqual(sorted(ordem), sorted(paradas))

    def test_muitas_paradas_usa_heuristica(self):
        paradas = [c for c in F.CIDADES if c != "Cianwood City"][:10]
        plano = planejar_entregas("Cianwood City", paradas, self.g)
        self.assertEqual(plano["metodo"], "heurístico")
        self.assertTrue(set(paradas) <= set(plano["rota"]))
        self.assertTrue(rota_valida(plano["rota"], self.g))

    def test_retornar_a_origem(self):
        plano = planejar_entregas("Pallet Town", ["Viridian City", "Pewter City"], self.g,
                                  retornar_origem=True)
        self.assertEqual(plano["rota"][-1], "Pallet Town")

    def test_evitar_cidade_e_bloqueio(self):
        livre = planejar_entregas("Pewter City", ["Fuchsia City"], self.g)["rota"]
        evitar = next(c for c in livre[1:-1] if c not in ("Pewter City", "Fuchsia City"))
        g = montar_grafo_custo(18, "tempo", evitar_cidades=[evitar],
                               obrigatorias=["Pewter City", "Fuchsia City"])
        self.assertNotIn(evitar, planejar_entregas("Pewter City", ["Fuchsia City"], g)["rota"])

        a, b = livre[0], livre[1]
        g = montar_grafo_custo(18, "tempo", bloqueios=[(b, a)])
        rota = planejar_entregas("Pewter City", ["Fuchsia City"], g)["rota"]
        usados = {chave_trecho(x, y) for x, y in zip(rota, rota[1:])}
        self.assertNotIn(chave_trecho(a, b), usados)

    def test_inalcancavel(self):
        vizinhos = [("Cinnabar Island", v) for v in MAPA_GRANDE["Cinnabar Island"]]
        g = montar_grafo_custo(12, "tempo", bloqueios=vizinhos)
        with self.assertRaises(RotaInalcancavel):
            planejar_entregas("Pallet Town", ["Cinnabar Island"], g)

    def test_nao_pode_evitar_ponto_do_pedido(self):
        with self.assertRaises(ValueError):
            aplicar_restricoes(MAPA_GRANDE, ["Celadon City"], obrigatorias=["Celadon City"])


class TestNormalizacao(unittest.TestCase):
    def test_nomes(self):
        casos = {
            "Pallet Town": "Pallet Town", "pallet": "Pallet Town",
            "GOLDENROD city": "Goldenrod City", "Goldenrd": "Goldenrod City",
            "vermilion": "Vermillion City", "Ilha Cinnabar": "Cinnabar Island",
            "mt. moon": "Mt. Moon", "cerulian": "Cerulean City",
        }
        for entrada, esperado in casos.items():
            with self.subTest(entrada=entrada):
                self.assertEqual(F.normalizar_cidade(entrada), esperado)

    def test_avisa_correcao_mas_nao_forma_curta(self):
        avisos = []
        F.normalizar_cidade("pallet", avisos)
        self.assertEqual(avisos, [])
        F.normalizar_cidade("Goldenrd", avisos)
        self.assertEqual(len(avisos), 1)

    def test_cidade_inexistente(self):
        for nome in ("Nárnia", "", None, "São Paulo"):
            with self.subTest(nome=nome), self.assertRaises(F.ArgumentoInvalido):
                F.normalizar_cidade(nome)

    def test_horas(self):
        agora = datetime(2026, 11, 11, 17, 30)
        casos = {"agora": 17.5, None: 17.5, 8: 8.0, "18h": 18.0, "18h30": 18.5,
                 "18:30": 18.5, "6pm": 18.0, "12am": 0.0}
        for entrada, esperado in casos.items():
            with self.subTest(entrada=entrada):
                self.assertAlmostEqual(F.interpretar_hora(entrada, agora), esperado)
        for invalida in ("25", "abc", -1, 24, True):
            with self.subTest(invalida=invalida), self.assertRaises(F.ArgumentoInvalido):
                F.interpretar_hora(invalida)


class TestFerramentas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)  # simulador: rápido e determinístico

    @classmethod
    def tearDownClass(cls):
        F.definir_preditor(cls._anterior)

    def test_buscar_rota_sucesso(self):
        r = F.executar_ferramenta("buscar_rota", {
            "origem": "pewter", "destino": "Fuchsia", "hora": "18h", "produto": "Poção"})
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["rota"][0], r["rota"][-1]), ("Pewter City", "Fuchsia City"))
        self.assertLessEqual(r["tempo_previsto_min"], r["rota_mais_curta"]["tempo_previsto_min"])
        self.assertGreaterEqual(r["economia_vs_mais_curta_min"], 0)
        self.assertEqual(r["produto"], "Poção")
        json.dumps(r)  # serializável para voltar ao LLM

    def test_paradas_como_texto_e_destino(self):
        r = F.buscar_rota("Pallet", paradas="Celadon, Saffron", destino="Lavender", hora=9)
        self.assertTrue(r["ok"], r)
        self.assertEqual(sorted(r["ordem_entregas"]),
                         ["Celadon City", "Lavender Town", "Saffron City"])

    def test_bloqueio_muda_a_rota(self):
        livre = F.buscar_rota("Pewter City", "Fuchsia City", hora=18)
        a, b = livre["rota"][1], livre["rota"][2]
        r = F.buscar_rota("Pewter City", "Fuchsia City", hora=18, bloqueios=[[a, b]])
        self.assertTrue(r["ok"], r)
        self.assertNotEqual(r["rota"], livre["rota"])
        self.assertGreaterEqual(r["tempo_previsto_min"], livre["tempo_previsto_min"])

    def test_acidente_aumenta_tempo_do_trecho(self):
        sem = F.estimar_tempo_trecho("Ice Path", "Mahogany Town", 12, False)
        com = F.estimar_tempo_trecho("Ice Path", "Mahogany Town", 12, True)
        self.assertGreater(com["tempo_previsto_min"], sem["tempo_previsto_min"])

    def test_erros_claros(self):
        casos = [
            ("buscar_rota", {"origem": "Nárnia", "destino": "Pallet Town"}),
            ("buscar_rota", {"origem": "Pallet Town"}),
            ("buscar_rota", {"origem": "Pallet Town", "destino": "Pallet Town"}),
            ("buscar_rota", {"origem": "Pallet Town", "destino": "Celadon",
                             "evitar": ["Celadon City"]}),
            ("buscar_rota", {"origem": "Pallet Town", "destino": "Celadon",
                             "acidentes": [["Pallet Town", "Celadon City"]]}),
            ("buscar_rota", {"origem": "Pallet Town", "destino": "Celadon", "hora": "99"}),
            ("buscar_rota", {"origim": "Pallet Town"}),
            ("estimar_tempo_trecho", {"origem": "Pallet Town", "destino": "Celadon City"}),
            ("voar", {}),
            ("buscar_rota", "{json quebrado"),
            ("buscar_rota", ["lista"]),
        ]
        for nome, args in casos:
            with self.subTest(nome=nome, args=args):
                r = F.executar_ferramenta(nome, args)
                self.assertFalse(r["ok"])
                self.assertTrue(r["erro"])
                json.dumps(r)

    def test_argumentos_em_json_texto(self):
        r = F.executar_ferramenta("listar_vizinhas", '{"cidade": "pallet"}')
        self.assertTrue(r["ok"])
        self.assertEqual({v["local"] for v in r["vizinhas"]}, set(MAPA_GRANDE["Pallet Town"]))

    def test_listar_cidades(self):
        r = F.listar_cidades()
        self.assertEqual(len(r["cidades"]) + len(r["pontos_naturais"]), len(MAPA_GRANDE))

    def test_esquemas_batem_com_as_funcoes(self):
        nomes = [f["function"]["name"] for f in F.FERRAMENTAS]
        self.assertEqual(sorted(nomes), sorted(F._FUNCOES))
        for esquema in F.FERRAMENTAS:
            fn = F._FUNCOES[esquema["function"]["name"]]
            params = inspect.signature(fn).parameters
            props = esquema["function"]["parameters"]["properties"]
            self.assertEqual(set(props), set(params), esquema["function"]["name"])
            for obrigatorio in esquema["function"]["parameters"].get("required", []):
                self.assertIs(params[obrigatorio].default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
