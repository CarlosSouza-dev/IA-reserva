# Fase 2 — Resumo do modelo de custo (gerado automaticamente)

- Observações: **2700** (45 trechos × 60 travessias), tempo médio 103.4 min (mín. 9.5, máx. 436.1).
- Faltantes injetados e tratados (mediana para numéricas, moda para `tipo_via`): `lombadas` 1.78%, `semaforos` 1.33%, `tipo_via` 1.7%.
- Codificação: one-hot em `tipo_via`; hora como seno/cosseno; padronização z-score nas numéricas. Tudo dentro do `Pipeline`, ajustado só nas dobras de treino.

## Comparação (validação cruzada, 5 dobras)

| Modelo | MAE estradas conhecidas (min) | R² estradas conhecidas | MAE estradas novas (min) | R² estradas novas |
| --- | --- | --- | --- | --- |
| Floresta Aleatória | 8.445 | 0.971 | 20.586 | 0.6917 |
| Regressão Ridge | 20.197 | 0.8722 | 20.997 | 0.7711 |
| Árvore de Decisão | 9.929 | 0.9595 | 24.205 | 0.6355 |
| k-NN (k=10) | 10.16 | 0.958 | 30.976 | 0.5619 |
| Baseline (média) | 61.313 | -0.001 | 61.518 | -0.0621 |

**Modelo escolhido:** Floresta Aleatória (menor MAE em estradas novas, o cenário mais exigente).

## Desempenho do modelo escolhido (previsões fora da amostra)

- Estradas conhecidas: MAE 8.45 min, RMSE 13.51 min, R² 0.971
- Estradas novas: MAE 20.59 min, RMSE 33.42 min, R² 0.823

## Onde o modelo erra mais

- Estradas conhecidas: tipo de via = (faltante) (MAE 21.43 min, viés +9.82 min); comprimento do trecho = longo (> 50 km) (MAE 13.38 min, viés -0.44 min); acidente = com acidente (MAE 12.52 min, viés -6.95 min); faixa horária = pico tarde (16h-20h30) (MAE 11.36 min, viés -1.45 min)
- Estradas novas: comprimento do trecho = longo (> 50 km) (MAE 35.31 min, viés +5.05 min); faixa horária = pico tarde (16h-20h30) (MAE 25.14 min, viés +0.39 min); tipo de via = (faltante) (MAE 24.11 min, viés +7.5 min); tipo de via = paralelepipedo (MAE 23.07 min, viés -5.42 min)

Trechos mais difíceis quando a estrada nunca foi vista (estradas novas):

- Tohjo Falls | Viridian City (170 km, asfalto): MAE 122.42 min, viés +122.42 min, tempo médio 160.4 min
- Pewter City | Viridian Forest (105 km, terra): MAE 80.49 min, viés -80.49 min, tempo médio 192.12 min
- Cherrygrove City | Violet City (110 km, asfalto): MAE 64.83 min, viés +64.83 min, tempo médio 105.37 min
- Fuchsia City | Lavender Town (170 km, paralelepipedo): MAE 62.36 min, viés -61.28 min, tempo médio 291.2 min
- Lavender Town | Vermillion City (135 km, paralelepipedo): MAE 39.23 min, viés +21.8 min, tempo médio 232.41 min
