"""Testes do núcleo do agente (Fase 3) com um LLM SIMULADO.

O LLM real (Qwen) é lento e não roda no CI; aqui um "LLM roteirizado" devolve
respostas fixas para testar o que é nosso: leitura da saída estruturada, o
ciclo pensar/agir/observar, as mensagens devolvidas ao modelo e a resposta de
reserva. O Qwen é exercitado na bateria de pedidos (issue #9) e no Colab.
"""
import json
import unittest

import ferramentas as F
from agente_llm import (
    Agente, carregar_prompt, explicar_resultado, extrair_json, ler_chamadas,
    remover_chamadas, resumir_para_llm, verificar_fidelidade,
)
from trechos import tempo_trecho


class LLMRoteirizado:
    """Devolve as respostas da lista, em ordem, e guarda o que recebeu."""

    nome = "roteirizado"

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.recebidas = []
        self.uso = []

    def __call__(self, mensagens, ferramentas=None):
        self.recebidas.append([dict(m) for m in mensagens])
        self.uso.append({"tokens_entrada": 100, "tokens_saida": 20, "segundos": 0.01})
        return self.respostas.pop(0) if self.respostas else ""


def tool_call(nome, **argumentos):
    return f'<tool_call>\n{json.dumps({"name": nome, "arguments": argumentos})}\n</tool_call>'


class TestSaidaEstruturada(unittest.TestCase):
    def test_extrair_json(self):
        self.assertEqual(extrair_json('bla {"a": 1} bla'), {"a": 1})
        self.assertIsNone(extrair_json("sem json"))
        self.assertIsNone(extrair_json("{quebrado"))

    def test_ler_chamada_padrao_qwen(self):
        texto = tool_call("buscar_rota", origem="Pallet", destino="Celadon")
        self.assertEqual(ler_chamadas(texto), [
            {"name": "buscar_rota", "arguments": {"origem": "Pallet", "destino": "Celadon"}}])

    def test_tolerancias(self):
        cortada = '<tool_call>{"name": "listar_cidades", "arguments": {}}'
        self.assertEqual(ler_chamadas(cortada)[0]["name"], "listar_cidades")
        solta = 'Vou chamar: {"name": "buscar_rota", "arguments": "{\\"origem\\": \\"Pallet\\"}"}'
        self.assertEqual(ler_chamadas(solta)[0]["arguments"], {"origem": "Pallet"})
        self.assertEqual(ler_chamadas('Resposta com JSON {"name": "Ash"}'), [])
        self.assertEqual(ler_chamadas("A rota é Pallet → Viridian."), [])

    def test_duas_chamadas(self):
        texto = tool_call("listar_cidades") + "\n" + tool_call("listar_vizinhas", cidade="Pallet")
        self.assertEqual([c["name"] for c in ler_chamadas(texto)],
                         ["listar_cidades", "listar_vizinhas"])
        self.assertEqual(remover_chamadas(texto), "")


class TestPrompts(unittest.TestCase):
    def test_versoes_existem_sem_comentarios(self):
        for versao in ("v1", "v2", "v3", "v4"):
            texto = carregar_prompt(versao)
            self.assertIn("Kanto", texto)
            self.assertNotIn("<!--", texto)
        self.assertIn("buscar_rota", carregar_prompt("v2"))
        with self.assertRaises(FileNotFoundError):
            carregar_prompt("v99")


class TestCicloDoAgente(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)

    @classmethod
    def tearDownClass(cls):
        F.definir_preditor(cls._anterior)

    def test_pedido_com_ferramenta_e_resposta_final(self):
        llm = LLMRoteirizado([
            tool_call("buscar_rota", origem="Pewter", destino="Fuchsia", hora="18h"),
            "Saia de Pewter City e siga até Fuchsia City.",
        ])
        r = Agente(llm).responder("Rota de Pewter até Fuchsia às 18h")
        self.assertEqual(r["passos"], 2)
        self.assertFalse(r["usou_reserva"])
        self.assertEqual(len(r["chamadas"]), 1)
        self.assertTrue(r["chamadas"][0]["resultado"]["ok"])
        self.assertTrue(r["resposta"].startswith("Saia de Pewter"))
        self.assertEqual((r["chamadas_llm"], r["tokens_entrada"], r["tokens_saida"]), (2, 200, 40))

        # 2ª chamada ao LLM recebeu o pedido de ferramenta e o resultado (observar)
        segunda = llm.recebidas[1]
        self.assertEqual([m["role"] for m in segunda], ["system", "user", "assistant", "tool"])
        self.assertEqual(segunda[2]["tool_calls"][0]["function"]["name"], "buscar_rota")
        resultado = json.loads(segunda[3]["content"])
        self.assertEqual(resultado["rota"][-1], "Fuchsia City")
        self.assertNotIn("nos_expandidos", resultado)  # resumo enxuto

    def test_fora_de_escopo_sem_ferramenta(self):
        llm = LLMRoteirizado(["Desculpe, eu só planejo rotas de entrega em Kanto e Johto."])
        r = Agente(llm).responder("Me passa uma receita de bolo?")
        self.assertEqual(r["chamadas"], [])
        self.assertEqual(r["passos"], 1)
        self.assertIn("só planejo rotas", r["resposta"])

    def test_erro_da_ferramenta_volta_para_o_llm(self):
        llm = LLMRoteirizado([
            tool_call("buscar_rota", origem="Nárnia", destino="Pallet"),
            "Não encontrei Nárnia no mapa.",
        ])
        r = Agente(llm).responder("Rota de Nárnia até Pallet")
        self.assertFalse(r["chamadas"][0]["resultado"]["ok"])
        self.assertEqual(json.loads(llm.recebidas[1][3]["content"])["ok"], False)

    def test_reserva_quando_llm_nao_responde(self):
        llm = LLMRoteirizado([tool_call("buscar_rota", origem="Pallet", paradas=["Celadon", "Lavender"]), ""])
        r = Agente(llm).responder("Pallet para Celadon e Lavender")
        self.assertTrue(r["usou_reserva"])
        self.assertIn("Tempo previsto", r["resposta"])
        self.assertIn("Ordem das entregas", r["resposta"])

    def test_limite_de_passos(self):
        repetida = tool_call("listar_cidades")
        llm = LLMRoteirizado([repetida] * 10)
        r = Agente(llm, max_passos=3).responder("liste")
        self.assertEqual(r["passos"], 3)
        self.assertEqual(len(r["chamadas"]), 3)
        self.assertTrue(r["usou_reserva"])

    def test_prompt_v1_tambem_funciona(self):
        llm = LLMRoteirizado(["Olá!"])
        r = Agente(llm, versao_prompt="v1").responder("oi")
        self.assertEqual(r["versao_prompt"], "v1")
        self.assertEqual(llm.recebidas[0][0]["content"], carregar_prompt("v1"))


class TestFidelidade(unittest.TestCase):
    """Casos reais do teste com o Qwen2.5-1.5B (prompt v2), reproduzidos aqui."""

    @classmethod
    def setUpClass(cls):
        anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)
        try:
            cls.multi = F.buscar_rota("Pallet", paradas=["Celadon", "Lavender"], hora="18h")
            cls.simples = F.buscar_rota("Pewter", "Fuchsia", hora="8")
        finally:
            F.definir_preditor(anterior)

    def test_explicacao_de_reserva_e_fiel(self):
        for r in (self.multi, self.simples):
            self.assertEqual(verificar_fidelidade(explicar_resultado(r), [r]), [])

    def test_rota_mais_curta_inventada(self):
        resposta = ("Em comparação, a rota mais curta seria:\n"
                    "1. **Pallet Town** -> **Celadon City**\n"
                    "2. **Celadon City** -> **Lavender Town**")
        problemas = verificar_fidelidade(resposta, [self.multi])
        self.assertIn("trecho inexistente: Pallet Town → Celadon City", problemas)

    def test_tempo_por_perna_inventado(self):
        resposta = "- **Pewter City** -> **Mt. Moon**: 1 hora e 15 minutos, totalizando 9999 minutos"
        self.assertTrue(any("9999" in p for p in verificar_fidelidade(resposta, [self.simples])))

    def test_local_que_nao_esta_no_resultado(self):
        problemas = verificar_fidelidade("Passe por Goldenrod City.", [self.simples])
        self.assertIn("local 'Goldenrod City' não está no resultado", problemas)

    def test_conversao_horas_e_arredondamento_aceitos(self):
        t = self.simples["tempo_previsto_min"]
        h, m = divmod(round(t), 60)
        resposta = f"Tempo previsto: {t:.0f} min ({h} horas e {m} minutos)."
        self.assertEqual(verificar_fidelidade(resposta, [self.simples]), [])

    def test_rota_sem_ferramenta(self):
        self.assertEqual(verificar_fidelidade("Desculpe, eu só planejo rotas de entrega em Kanto e Johto.", []), [])
        problemas = verificar_fidelidade("Vá de Pallet Town -> Viridian City.", [], "rota de pallet")
        self.assertIn("rota citada sem consultar a ferramenta", problemas)

    def test_agente_troca_resposta_alucinada_pela_reserva(self):
        anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)
        try:
            alucinada = "A rota é Pewter City -> Fuchsia City em 12 minutos."
            llm = LLMRoteirizado([tool_call("buscar_rota", origem="Pewter", destino="Fuchsia", hora="8"),
                                  alucinada])
            r = Agente(llm).responder("Pewter até Fuchsia às 8h")
        finally:
            F.definir_preditor(anterior)
        self.assertTrue(r["usou_reserva"])
        self.assertEqual(r["resposta_llm"], alucinada)
        self.assertTrue(r["problemas_fidelidade"])
        self.assertIn("Tempo previsto", r["resposta"])


class TestExplicacao(unittest.TestCase):
    def test_resumo_e_explicacao(self):
        anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)
        try:
            r = F.buscar_rota("Pewter", "Fuchsia", hora=18)
        finally:
            F.definir_preditor(anterior)
        resumo = resumir_para_llm(r)
        self.assertNotIn("pernas", resumo)  # 1 perna só repetiria o total
        self.assertEqual(resumo["rota_mais_curta"]["rota"], r["rota_mais_curta"]["rota"])
        texto = explicar_resultado(r)
        self.assertIn(f"{r['tempo_previsto_min']:.0f} min", texto)
        self.assertIn("18h00", texto)
        anterior = F._PREDITOR
        F.definir_preditor(tempo_trecho)
        try:
            trecho = F.estimar_tempo_trecho("Ice Path", "Mahogany", 18, True)
        finally:
            F.definir_preditor(anterior)
        texto_trecho = explicar_resultado(trecho)
        self.assertNotIn("{", texto_trecho)  # nada de JSON cru para o usuário
        self.assertIn(f"{trecho['tempo_previsto_min']:.0f} min", texto_trecho)
        self.assertEqual(verificar_fidelidade(texto_trecho, [trecho]), [])
        vizinhas = explicar_resultado(F.listar_vizinhas("Saffron"))
        self.assertIn("Celadon City", vizinhas)
        erro = explicar_resultado({"ok": False, "erro": "X não existe", "sugestoes": ["Y"]})
        self.assertIn("X não existe", erro)
        self.assertIn("Y", erro)


if __name__ == "__main__":
    unittest.main(verbosity=2)
