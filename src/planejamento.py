"""
Planejamento de ENTREGAS: várias paradas, restrições e acidentes.

O PDF descreve um veículo que "precisa visitar um conjunto de endereços"
("preciso entregar nestes oito endereços... evitando a região do acidente").
O A* da Fase 1 resolve origem -> destino; aqui ele vira o bloco de construção:

    1. aplicar_restricoes()  remove do mapa as cidades a evitar e os trechos
                             bloqueados (acidente com interdição total);
    2. grafo de custo         tempo previsto pelo modelo da Fase 2 (ou distância);
    3. matriz de A*           melhor rota entre cada par de pontos do pedido;
    4. ordem de visita        permutações exatas até MAX_PARADAS_EXATO paradas;
                              acima disso, vizinho mais próximo + melhoria 2-opt.

A busca continua a mesma; o planejador só decide em que ORDEM chamar o A*.
"""
import itertools
import math

from buscas import a_estrela, calibrar_escala_heuristica, custo_do_caminho
from mapas import MAPA_GRANDE
from trechos import chave_trecho, montar_grafo_tempo

MAX_PARADAS_EXATO = 8  # 8! = 40.320 ordens: ainda instantâneo em Python


class RotaInalcancavel(Exception):
    """Algum ponto do pedido não pode ser alcançado com as restrições dadas."""


# =====================================================================
# 1. RESTRIÇÕES
# =====================================================================
def aplicar_restricoes(mapa, evitar_cidades=(), bloqueios=(), obrigatorias=()):
    """Cópia do mapa sem as cidades de `evitar_cidades` e sem os trechos de
    `bloqueios` (pares (a, b), em qualquer ordem).

    `obrigatorias` (origem e paradas) nunca podem ser evitadas: se o usuário
    pedir para evitar o próprio destino, é erro de pedido, não de rota.
    """
    evitar = set(evitar_cidades)
    conflito = evitar & set(obrigatorias)
    if conflito:
        raise ValueError(f"Não dá para evitar um ponto do próprio pedido: {sorted(conflito)}")
    bloqueados = {chave_trecho(a, b) for a, b in bloqueios}
    return {
        a: {b: c for b, c in viz.items()
            if b not in evitar and chave_trecho(a, b) not in bloqueados}
        for a, viz in mapa.items() if a not in evitar
    }


def montar_grafo_custo(hora, criterio="tempo", preditor=None, acidentes=(),
                       evitar_cidades=(), bloqueios=(), obrigatorias=(), mapa=MAPA_GRANDE):
    """Grafo {cidade: {vizinho: custo}} já com as restrições aplicadas.

    criterio = "tempo"     -> minutos previstos (modelo da Fase 2, ou simulador
                              se `preditor` for None), com atraso de acidente;
    criterio = "distancia" -> km (critério da Fase 1).
    """
    restrito = aplicar_restricoes(mapa, evitar_cidades, bloqueios, obrigatorias)
    if criterio == "distancia":
        return restrito
    if criterio != "tempo":
        raise ValueError(f"critério desconhecido: {criterio}")
    acidentes_validos = [(a, b) for a, b in acidentes if b in restrito.get(a, {})]
    return montar_grafo_tempo(restrito, hora, acidentes_validos, prever=preditor)


# =====================================================================
# 2. MATRIZ DE ROTAS ENTRE OS PONTOS DO PEDIDO
# =====================================================================
def matriz_de_rotas(pontos, grafo):
    """{(a, b): (caminho, custo)} com o A* entre todos os pares de `pontos`.
    Também devolve o total de nós expandidos (esforço da busca)."""
    escala = calibrar_escala_heuristica(grafo) if any(grafo.values()) else 0.0
    rotas, expandidos = {}, 0
    for a, b in itertools.permutations(dict.fromkeys(pontos), 2):
        caminho, custo, nos, _ = a_estrela(a, b, grafo, escala)
        expandidos += nos
        if caminho is None:
            raise RotaInalcancavel(f"Não existe rota de {a} até {b} com as restrições informadas")
        rotas[(a, b)] = (caminho, custo)
    return rotas, expandidos


# =====================================================================
# 3. ORDEM DE VISITA
# =====================================================================
def _custo_ordem(ordem, origem, rotas, retornar):
    pontos = [origem, *ordem] + ([origem] if retornar else [])
    return sum(rotas[(a, b)][1] for a, b in zip(pontos, pontos[1:]))


def _ordem_exata(origem, paradas, rotas, retornar):
    return min(itertools.permutations(paradas),
               key=lambda ordem: _custo_ordem(ordem, origem, rotas, retornar))


def _ordem_heuristica(origem, paradas, rotas, retornar):
    """Vizinho mais próximo seguido de 2-opt (inverte trechos da ordem enquanto
    melhorar). Não garante o ótimo, mas é rápido para muitas paradas."""
    restantes, atual, ordem = set(paradas), origem, []
    while restantes:
        proximo = min(restantes, key=lambda p: (rotas[(atual, p)][1], p))
        ordem.append(proximo)
        restantes.remove(proximo)
        atual = proximo
    melhorou = True
    while melhorou:
        melhorou = False
        for i, j in itertools.combinations(range(len(ordem)), 2):
            nova = ordem[:i] + ordem[i:j + 1][::-1] + ordem[j + 1:]
            if _custo_ordem(nova, origem, rotas, retornar) < _custo_ordem(ordem, origem, rotas, retornar) - 1e-9:
                ordem, melhorou = nova, True
    return tuple(ordem)


# =====================================================================
# 4. PLANO COMPLETO
# =====================================================================
def planejar_entregas(origem, paradas, grafo, retornar_origem=False):
    """Melhor ordem de visita e rota completa sobre `grafo`.

    Retorna dict com: ordem (paradas na ordem de visita), rota (lista de
    cidades, origem incluída), custo_total, pernas [(de, para, caminho, custo)],
    metodo ("exato" ou "heurístico") e nos_expandidos.
    """
    paradas = [p for p in dict.fromkeys(paradas) if p != origem]
    if not paradas:
        raise ValueError("Informe ao menos um destino diferente da origem")
    for p in [origem, *paradas]:
        if p not in grafo:
            raise RotaInalcancavel(f"'{p}' não está disponível no mapa com as restrições informadas")

    rotas, expandidos = matriz_de_rotas([origem, *paradas], grafo)
    if len(paradas) <= MAX_PARADAS_EXATO:
        ordem, metodo = _ordem_exata(origem, paradas, rotas, retornar_origem), "exato"
    else:
        ordem, metodo = _ordem_heuristica(origem, paradas, rotas, retornar_origem), "heurístico"

    pontos = [origem, *ordem] + ([origem] if retornar_origem else [])
    pernas, rota = [], [origem]
    for a, b in zip(pontos, pontos[1:]):
        caminho, custo = rotas[(a, b)]
        pernas.append((a, b, caminho, custo))
        rota += caminho[1:]
    return {
        "ordem": list(ordem),
        "rota": rota,
        "custo_total": sum(p[3] for p in pernas),
        "pernas": pernas,
        "metodo": metodo,
        "nos_expandidos": expandidos,
    }


def avaliar_rota(rota, grafo):
    """Custo de uma rota já pronta em outro grafo (ex.: rota curta avaliada em
    tempo). Devolve math.inf se a rota usar um trecho que não existe nele."""
    try:
        return custo_do_caminho(rota, grafo)
    except KeyError:
        return math.inf
