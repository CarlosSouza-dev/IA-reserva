# Projeto Acadêmico de IA — Agente de Logística no Mundo Pokémon

IESB · Ciência da Computação · Inteligência Artificial (2026/2) · Prof. Rodrigo Gonçalves

Um agente que evolui em três fases: **busca** (A*) → **custo aprendido** (tempo de viagem) → **agente com LLM**.
Cenário: entregas entre cidades de Kanto e Johto, modeladas como um grafo.

| Fase | Custo da aresta | Status |
|------|-----------------|--------|
| 1 — Busca | distância | feito (UCS x A*, 3+ mapas) |
| 2 — Aprendizado | tempo **previsto** por Floresta Aleatória a partir de lombadas, semáforos, acidente, horário e tipo de via | feito (modelo treinado, avaliado e integrado ao A*) |
| 3 — Agente com LLM | tempo previsto | ferramentas e rotas com várias paradas prontas; falta o núcleo do LLM |

## Como rodar
Use **Python 3.11 ou 3.12** (versões alfa como 3.15 ainda não têm pandas/scikit-learn).
O VS Code já aponta para `.venv` (ver `.vscode/settings.json`); crie-o uma vez:
```bash
py -3.11 -m venv .venv
.venv\Scriptsctivate          # Windows (no Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python src/main.py                        # Fase 1 + treino/avaliação do modelo + experimento da Fase 2 -> outputs/
python src/avaliacao_fase2.py             # só treina, compara e avalia os modelos (outputs/fase2_resumo.md)
python src/mapas.py                       # valida os três mapas
python src/trechos.py                     # gera outputs/csv/dataset_trechos.csv (treino da Fase 2)
python src/ferramentas.py                 # exemplo de chamada da ferramenta buscar_rota (JSON)
python -m unittest discover -s src -v     # testes
```

## Estrutura (`src/`)
- `mapas.py` — coordenadas, grafos (pequeno/médio/grande) e validação (`validar_mapa`).
- `buscas.py` — custo uniforme e A* sobre um núcleo comum; heurística de distância direta (HDD) com escala calibrada.
- `trechos.py` — características de cada trecho, horário → trânsito, `tempo_trecho`, `montar_grafo_tempo(..., prever=modelo)`, dataset sintético.
- `modelo_custo.py` — Fase 2: dataset com faltantes, pré-processamento (Pipeline), modelos candidatos, validação cruzada (estradas conhecidas x novas) e `PreditorTempo`, usado como custo do A*.
- `avaliacao_fase2.py` — comparação dos modelos, análise de erros por segmento, gráficos e resumo `outputs/fase2_resumo.md`.
- `planejamento.py` — várias paradas (ordem exata até 8, vizinho mais próximo + 2-opt acima), cidades a evitar, acidentes e bloqueios.
- `ferramentas.py` — Fase 3: ferramentas do agente (`buscar_rota`, `estimar_tempo_trecho`, `listar_cidades`, `listar_vizinhas`), validação de nomes/horários, esquemas `FERRAMENTAS` e `executar_ferramenta`.
- `desenhar_mapas.py`, `tabela.py`, `graficos.py` — saídas visuais e CSV.
- `test_buscas.py` — otimalidade do A* em todos os pares, consistência da HDD, custos.
- `test_ferramentas.py` — planejamento, normalização de nomes/horários, erros das ferramentas e esquemas.
- `test_modelo_custo.py` — pré-processamento, previsão positiva, lote x individual, A* == UCS no grafo previsto pelo modelo.

## Decisões de modelagem
- **Estado** = cidade; **ação** = ir a uma cidade vizinha; **custo** = distância (Fase 1) ou tempo em minutos (Fase 2).
- **Heurística (HDD)**: `h(n) = escala × distância em linha reta até o destino`, com
  `escala = min(custo / distância reta)` sobre todas as arestas. Pela desigualdade triangular isso garante
  `h(a) ≤ c(a,b) + h(b)` (consistente, logo admissível), para distância ou para tempo.
- **Tempo do trecho** (min): `deslocamento × (1 + sens_via × (fator_horário − 1)) + lombadas×0,25 + semáforos×0,7×fator_horário + acidente×20`.
- O horário é uma "foto" (hora de saída para todo o trajeto); ver issue sobre custo dependente do horário de chegada.

Tarefas e prazos: [ISSUES.md](ISSUES.md).
