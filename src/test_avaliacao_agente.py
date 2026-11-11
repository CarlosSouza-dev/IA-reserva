"""Testes da avaliação da bateria de pedidos (issue #9), com LLMs simulados:
um "oráculo" que sempre faz a chamada esperada e responde com fidelidade (tem
de tirar 100%), e um LLM ruim que sempre pede a mesma rota."""
import json
import unittest

import ferramentas as F
from agente_llm import Agente, explicar_resultado
from avaliacao_agente import (
    carregar_casos, comparar_argumentos, executar_bateria, precisao_revocacao, resumir,
)
from mapas import MAPA_GRANDE
from trechos import TRECHOS, chave_trecho, tempo_trecho

CASOS = carregar_casos()


def _tool_call(nome, argumentos):
    return "<tool_call>" + json.dumps({"name": nome, "arguments": argumentos}) + "</tool_call>"


class LLMOraculo:
    nome = "oráculo"

    def __init__(self):
        self.uso = []
        self.por_pedido = {c["pedido"]: c for c in CASOS}

    def __call__(self, mensagens, ferramentas=None):
        self.uso.append({"tokens_entrada": 10, "tokens_saida": 5, "segundos": 0.0})
        ultima = mensagens[-1]
        if ultima["role"] == "user":
            caso = self.por_pedido[ultima["content"]]
            if caso["ferramenta"] is None:
                return "Desculpe, eu só planejo rotas de entrega em Kanto e Johto."
            return _tool_call(caso["ferramenta"], caso["argumentos"])
        return explicar_resultado(json.loads(ultima["content"]))


class LLMTeimoso(LLMOraculo):
    nome = "teimoso"

    def __call__(self, mensagens, ferramentas=None):
        self.uso.append({"tokens_entrada": 10, "tokens_saida": 5, "segundos": 0.0})
        if mensagens[-1]["role"] == "user":
            return _tool_call("buscar_rota", {"origem": "Pallet", "destino": "Viridian"})
        return "Vá de Pallet Town -> Viridian City em 37 minutos."


class TestBateria(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)

    @classmethod
    def tearDownClass(cls):
        F.definir_preditor(cls._anterior)

    def test_casos_bem_formados(self):
        self.assertGreaterEqual(len(CASOS), 15)
        self.assertEqual(len({c["id"] for c in CASOS}), len(CASOS))
        categorias = {c["categoria"] for c in CASOS}
        for exigida in ("difícil", "fora do mapa", "sem ferramenta", "várias paradas", "restrição"):
            self.assertIn(exigida, categorias)
        nomes = {f["function"]["name"] for f in F.FERRAMENTAS}
        for c in CASOS:
            if c["ferramenta"]:
                self.assertIn(c["ferramenta"], nomes)
            for chave in ("acidentes", "bloqueios"):
                for a, b in (c["argumentos"] or {}).get(chave, []):
                    self.assertIn(chave_trecho(a, b), TRECHOS, f"caso {c['id']}: {a}-{b}")
            if c["categoria"] not in ("fora do mapa",) and c["argumentos"]:
                for chave in ("origem", "destino", "cidade"):
                    if chave in c["argumentos"]:
                        self.assertIn(c["argumentos"][chave], MAPA_GRANDE, f"caso {c['id']}")

    def test_oraculo_acerta_tudo(self):
        df = executar_bateria(Agente(LLMOraculo()), mostrar=False)
        self.assertTrue(df["acerto"].all(), df.loc[~df["acerto"], ["id", "diferencas_argumentos",
                                                                      "rota_ok", "resposta_fiel"]])
        geral, por_categoria, matriz, pr = resumir(df)
        self.assertEqual(geral["acerto (%)"], 100.0)
        self.assertTrue((pr["precisão"] == 1).all())

    def test_llm_teimoso_e_reprovado(self):
        df = executar_bateria(Agente(LLMTeimoso()), mostrar=False)
        self.assertLess(df["acerto"].mean(), 0.2)
        self.assertTrue(df["llm_descartado"].any())  # "37 minutos" é barrado
        _, _, matriz, pr = resumir(df)
        nenhuma = pr.set_index("classe").loc["nenhuma"]
        self.assertEqual(nenhuma["VP"], 0)

    def test_comparacao_pelo_significado(self):
        esperado = {"origem": "Pallet Town", "paradas": ["Celadon City", "Lavender Town"], "hora": 18}
        ok, _ = comparar_argumentos("buscar_rota", esperado,
                                    {"origem": "pallet", "paradas": ["Lavender", "Celadon"], "hora": "18h"})
        self.assertTrue(ok)
        ok, dif = comparar_argumentos("buscar_rota", esperado,
                                      {"origem": "Pallet", "paradas": ["Celadon"], "hora": "agora"})
        self.assertFalse(ok)
        self.assertEqual(len(dif), 2)
        ok, dif = comparar_argumentos("buscar_rota", {"origem": "Pallet Town", "destino": "Viridian City"},
                                      {"origem": "Pallet", "destino": "Viridian", "evitar": ["Pewter"]})
        self.assertFalse(ok)  # restrição inventada conta como erro

    def test_precisao_revocacao(self):
        esperadas = ["a", "a", "b", "nenhuma"]
        escolhidas = ["a", "b", "b", "a"]
        r = precisao_revocacao(esperadas, escolhidas, "a")
        self.assertEqual((r["VP"], r["FP"], r["FN"]), (1, 1, 1))
        self.assertEqual((r["precisão"], r["revocação"]), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
