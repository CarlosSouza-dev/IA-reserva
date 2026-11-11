# Bateria de pedidos do agente — Qwen2.5-1.5B-Instruct v3

- **casos:** 20
- **acerto (%):** 85.0
- **ferramenta certa (%):** 100.0
- **argumentos certos (%):** 85.0
- **rota certa:** 10/12
- **resposta fiel:** 18/20
- **resposta do LLM descartada:** 2
- **segundos por pedido (média):** 33.2
- **tokens por pedido (média):** 3281

## Acerto por categoria (%)

| categoria | casos | ferramenta_ok | argumentos_ok | acerto |
|---|---|---|---|---|
| fácil | 2 | 100.0 | 100.0 | 100.0 |
| horário | 2 | 100.0 | 100.0 | 100.0 |
| várias paradas | 2 | 100.0 | 50.0 | 50.0 |
| restrição | 3 | 100.0 | 67.0 | 67.0 |
| difícil | 3 | 100.0 | 100.0 | 100.0 |
| outras ferramentas | 2 | 100.0 | 50.0 | 50.0 |
| fora do mapa | 2 | 100.0 | 100.0 | 100.0 |
| sem ferramenta | 3 | 100.0 | 100.0 | 100.0 |
| ambíguo | 1 | 100.0 | 100.0 | 100.0 |

## Matriz de confusão da escolha de ferramenta

| esperada | buscar_rota | estimar_tempo_trecho | listar_vizinhas | nenhuma |
|---|---|---|---|---|
| buscar_rota | 14 | 0 | 0 | 0 |
| estimar_tempo_trecho | 0 | 1 | 0 | 0 |
| listar_vizinhas | 0 | 0 | 1 | 0 |
| nenhuma | 0 | 0 | 0 | 4 |

## Precisão e revocação por ferramenta

| classe | VP | FP | FN | precisão | revocação |
|---|---|---|---|---|---|
| buscar_rota | 14 | 0 | 0 | 1.0 | 1.0 |
| estimar_tempo_trecho | 1 | 0 | 0 | 1.0 | 1.0 |
| listar_vizinhas | 1 | 0 | 0 | 1.0 | 1.0 |
| nenhuma | 4 | 0 | 0 | 1.0 | 1.0 |

## Casos com erro (3)

- **#6 (várias paradas)** Saindo de Goldenrod às 7h30, entrego em Azalea, Olivine e Ecruteak e depois volto para Goldenrod.
  - motivo: argumentos: retornar_origem: esperado True, obtido False; rota diferente da esperada; resposta não fiel
  - chamada: `{"origem": "Goldenrod", "paradas": ["Azalea", "Olivine", "Ecruteak"], "hora": "7h30"}`
  - resposta: Rota: Goldenrod City → Ilex Forest → Azalea Town → Ilex Forest → Goldenrod City → National Park → Ecruteak City → Olivine City. Ordem das entregas: Azalea Town, Ecruteak City, Olivine City. Tempo previsto: 590.8 minutos 
- **#9 (restrição)** A estrada entre Celadon e Fuchsia está interditada. Como vou de Celadon até Fuchsia às 10h?
  - motivo: argumentos: bloqueios: esperado {('Celadon City', 'Fuchsia City')}, obtido set(); rota diferente da esperada; resposta não fiel
  - chamada: `{"origem": "Celadon", "destino": "Fuchsia", "hora": "10"}`
  - resposta: Rota: Celadon City → Fuchsia City. Ordem das entregas: Fuchsia City. Tempo previsto: 182.5 min (110 km). Comparação: a rota mais curta tem 110 km e levaria 182.5 min; economia de 0.0 min. Observações: A rota escolhida fo
- **#13 (outras ferramentas)** Quanto tempo leva a estrada entre Ice Path e Mahogany às 18h?
  - motivo: argumentos: hora: esperado 18.0, obtido 'agora'
  - chamada: `{"origem": "Ice Path", "destino": "Mahogany"}`
  - resposta: {"ok": true, "de": "Ice Path", "para": "Mahogany Town", "hora": 0.38, "acidente": false, "tempo_previsto_min": 36.8, "distancia_km": 22, "tipo_via": "terra", "lombadas": 1, "semaforos": 1}

## Respostas do LLM barradas pela checagem de fidelidade (2)

- **#7** número 0.9 não veio das ferramentas
  - texto do LLM: Rota: Pewter City → Mt. Moon → Cerulean City → Saffron City → Celadon City → Fuchsia City. Ordem das entregas: Fuchsia City. Tempo previsto: 635 minutos (10 horas e 35 minutos). Comparação: a rota mais curta tem 312 km e
- **#13** número 5 não veio das ferramentas
  - texto do LLM: A estrada entre Ice Path e Mahogany Town leva aproximadamente 37 minutos (5 horas e 37 minutos). A distância é de 22 km. Esta via está rodovia e possui 1 lombada e 1 semáforo.
