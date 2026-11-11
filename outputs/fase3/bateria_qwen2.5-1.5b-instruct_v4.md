# Bateria de pedidos do agente — Qwen2.5-1.5B-Instruct v4

- **casos:** 20
- **acerto (%):** 50.0
- **ferramenta certa (%):** 70.0
- **argumentos certos (%):** 55.0
- **rota certa:** 6/12
- **resposta fiel:** 13/20
- **resposta do LLM descartada:** 0
- **segundos por pedido (média):** 26.9
- **tokens por pedido (média):** 3332

## Acerto por categoria (%)

| categoria | casos | ferramenta_ok | argumentos_ok | acerto |
|---|---|---|---|---|
| fácil | 2 | 50.0 | 50.0 | 0.0 |
| horário | 2 | 50.0 | 50.0 | 50.0 |
| várias paradas | 2 | 0.0 | 0.0 | 0.0 |
| restrição | 3 | 100.0 | 33.0 | 33.0 |
| difícil | 3 | 67.0 | 67.0 | 67.0 |
| outras ferramentas | 2 | 100.0 | 100.0 | 100.0 |
| fora do mapa | 2 | 50.0 | 0.0 | 0.0 |
| sem ferramenta | 3 | 100.0 | 100.0 | 100.0 |
| ambíguo | 1 | 100.0 | 100.0 | 100.0 |

## Matriz de confusão da escolha de ferramenta

| esperada | buscar_rota | estimar_tempo_trecho | listar_vizinhas | nenhuma |
|---|---|---|---|---|
| buscar_rota | 8 | 0 | 0 | 6 |
| estimar_tempo_trecho | 0 | 1 | 0 | 0 |
| listar_vizinhas | 0 | 0 | 1 | 0 |
| nenhuma | 0 | 0 | 0 | 4 |

## Precisão e revocação por ferramenta

| classe | VP | FP | FN | precisão | revocação |
|---|---|---|---|---|---|
| buscar_rota | 8 | 0 | 6 | 1.0 | 0.571 |
| estimar_tempo_trecho | 1 | 0 | 0 | 1.0 | 1.0 |
| listar_vizinhas | 1 | 0 | 0 | 1.0 | 1.0 |
| nenhuma | 4 | 6 | 0 | 0.4 | 1.0 |

## Casos com erro (10)

- **#1 (fácil)** Qual a rota de Pewter até Fuchsia às 8h?
  - motivo: resposta não fiel
  - chamada: `{"origem": "Pewter", "destino": "Fuchsia", "hora": "8"}`
  - resposta: A rota mais curta é via Mt. Moon, Cerclane City, Saffron City e Celadon City, totalizando 312 km e levando aproximadamente 570 minutos. Esta rota economiza 9.1 minutos em relação à rota original.   Rota: Pewter City → Mt
- **#2 (fácil)** Preciso ir de Goldenrod City para Ecruteak City às 14h.
  - motivo: ferramenta nenhuma (esperada buscar_rota); rota diferente da esperada; resposta não fiel
  - chamada: `(nenhuma)`
  - resposta: De qual cidade o veículo sai e em que horário?
- **#3 (horário)** Saindo agora de Cerulean, como chego em Vermillion?
  - motivo: ferramenta nenhuma (esperada buscar_rota); rota diferente da esperada; resposta não fiel
  - chamada: `(nenhuma)`
  - resposta: De qual cidade o veículo sai e em que horário?
- **#5 (várias paradas)** Saio de Pallet às 18h e preciso entregar em Celadon e Lavender.
  - motivo: ferramenta nenhuma (esperada buscar_rota); rota diferente da esperada; resposta não fiel
  - chamada: `(nenhuma)`
  - resposta: De qual cidade o veículo sai e em que horário?
- **#6 (várias paradas)** Saindo de Goldenrod às 7h30, entrego em Azalea, Olivine e Ecruteak e depois volto para Goldenrod.
  - motivo: ferramenta nenhuma (esperada buscar_rota); rota diferente da esperada; resposta não fiel
  - chamada: `(nenhuma)`
  - resposta: De qual cidade o veículo sai e em que horário?
- **#8 (restrição)** Leve 3 Poções de Goldenrod até Ecruteak agora, evitando Violet.
  - motivo: argumentos: produto: esperado 'Poções', obtido ''
  - chamada: `{"origem": "Goldenrod", "destino": "Ecruteak", "hora": "agora", "evitar": ["Violet"]}`
  - resposta: Rota: Goldenrod City → National Park → Ecruteak City. Ordem das entregas: Ecruteak City. Tempo previsto: 104.4 min (65 km).  A rota mais curta tem 65 km e levaria 104.4 min; economia de 0.0 min.  Observações: Evite Viole
- **#9 (restrição)** A estrada entre Celadon e Fuchsia está interditada. Como vou de Celadon até Fuchsia às 10h?
  - motivo: argumentos: bloqueios: esperado {('Celadon City', 'Fuchsia City')}, obtido set(); rota diferente da esperada; resposta não fiel
  - chamada: `{"origem": "Celadon", "destino": "Fuchsia", "hora": "10"}`
  - resposta: Rota: Celadon City → Fuchsia City. Ordem das entregas: Fuchsia City. Tempo previsto: 182.5 minutos (110 km).  Comparação: a rota mais curta tem 110 km e levaria 182.5 minutos; economia de 0.0 minutos.  Observações: Esta 
- **#11 (difícil)** Quero mandar um pacote pra Cerulian saindo de Saffon ao meio-dia.
  - motivo: ferramenta nenhuma (esperada buscar_rota); rota diferente da esperada; resposta não fiel
  - chamada: `(nenhuma)`
  - resposta: De qual cidade o veículo sai e em que horário?
- **#15 (fora do mapa)** Me leva de Pallet para São Paulo às 10h.
  - motivo: ferramenta nenhuma (esperada buscar_rota)
  - chamada: `(nenhuma)`
  - resposta: De qual cidade o veículo sai e em que horário?
- **#16 (fora do mapa)** Rota de New Bark até Hogwarts agora.
  - motivo: argumentos: destinos: esperado {'hogwarts'}, obtido set()
  - chamada: `{"origem": "New Bark", "hora": "agora"}`
  - resposta: Não consegui calcular: Informe ao menos um destino ou parada. Por exemplo, você poderia dizer "Nova Horta e Hogwarts".
