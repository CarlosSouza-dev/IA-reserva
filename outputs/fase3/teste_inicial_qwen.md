# Teste inicial do agente com o Qwen2.5-1.5B-Instruct (CPU, fp32, 10/11/2026)

Mesmos 4 pedidos com os prompts v3 (padrão) e v2, já com a checagem de fidelidade ativa.
Rodado localmente em CPU (Intel, sem CUDA), por isso ~40-75 s por pedido; no Colab com GPU T4 o
esperado é bem menor. Primeira rodada (v2, sem checagem): as 3 respostas com rota inventaram
dados (rota mais curta inexistente "Pallet → Celadon → Lavender", tempos de "1 hora e 15 minutos"
por perna, "a rota inclui Violet" quando era para evitar), o que motivou o prompt v3 e
`verificar_fidelidade`. Ver `prompts/v3.md`.

Observações desta rodada:
- Escolha de ferramenta e argumentos: 8/8 corretos (inclusive acidente como par de cidades,
  `evitar` e `produto`); o pedido fora de escopo foi recusado sem chamar ferramenta nas duas versões.
- v3: respostas fiéis ao JSON em 3/3 pedidos de rota; ~150-210 tokens de saída.
- v2: no 1º pedido o modelo repetiu "182 minutos" para cada perna (só a perna Celadon → Lavender
  tem 182,9 min). A checagem NÃO pegou: o número existe no resultado, mas foi associado à perna
  errada. Limitação conhecida (issue #10).

```text
modelo carregado em 16.9s (cpu)
================================================================================
[v3] PEDIDO: Saio de Pallet às 18h e preciso entregar em Celadon e Lavender.
  FERRAMENTA: buscar_rota {"origem": "Pallet", "paradas": ["Celadon", "Lavender"], "hora": "18h"}
  RESULTADO : True ['Pallet Town', 'Cinnabar Island', 'Seafoam Islands', 'Fuchsia City', 'Celadon City', 'Saffron City', 'Lavender Town'] 684.3
  RESPOSTA  : Rota: Pallet Town → Cinnabar Island → Seafoam Islands → Fuchsia City → Celadon City → Saffron City → Lavender Town.
Ordem das entregas: Celadon City, Lavender Town.
Tempo previsto: 684.3 min (344 km).
Comparação: a rota mais curta tem 344 km e levaria 684.3 min; economia de 0.0 min.
Observações: A rota escolhida foi a menos longa possível, mas consideramos todas as opções disponíveis.
  (49.754s, passos=2, tokens=3677+175, reserva=False)
================================================================================
[v3] PEDIDO: Qual a rota mais rápida de Pewter até Fuchsia às 8h? Teve um acidente entre Saffron e Vermillion.
  FERRAMENTA: buscar_rota {"origem": "Pewter", "destino": "Fuchsia", "hora": "8", "acidentes": [["Saffron", "Vermillion"]], "retornar_origem": false}
  RESULTADO : True ['Pewter City', 'Mt. Moon', 'Cerulean City', 'Saffron City', 'Celadon City', 'Fuchsia City'] 570.8
  RESPOSTA  : A rota mais rápida de Pewter até Fuchsia às 8h, considerando um acidente entre Saffron e Vermillion, é:

Rota: Pewter City → Mt. Moon → Cerulean City → Saffron City → Celadon City → Fuchsia City.

Ordem das entregas: Fuchsia City.

Tempo previsto: 570.8 minutos (ou 9 horas e 30 minutos).

Comparação: A rota mais curta tem 312 km e levaria 570.8 minutos; economia de 0.0 minutos.

Observações: Não há restrições adicionais neste caso.
  (47.531s, passos=2, tokens=3646+207, reserva=False)
================================================================================
[v3] PEDIDO: Leve 3 Poções de Goldenrod até Ecruteak agora, evitando Violet.
  FERRAMENTA: buscar_rota {"origem": "Goldenrod", "destino": "Ecruteak", "hora": "agora", "evitar": ["Violet"], "produto": "Poções"}
  RESULTADO : True ['Goldenrod City', 'National Park', 'Ecruteak City'] 97.2
  RESPOSTA  : Rota: Goldenrod City → National Park → Ecruteak City.
Ordem das entregas: Ecruteak City.
Tempo previsto: 97.2 min (65 km).
Comparação: a rota mais curta tem 65 km e levaria 97.2 min; economia de 0.0 min.
Observações: Não há restrições adicionais.
  (37.574s, passos=2, tokens=3584+145, reserva=False)
================================================================================
[v3] PEDIDO: Me passa uma receita de bolo de cenoura?
  RESPOSTA  : Desculpe, eu só planejo rotas de entrega em Kanto e Johto. Diga a origem, os destinos e o horário que eu calculo a melhor rota.
  (13.508s, passos=1, tokens=1664+43, reserva=False)
================================================================================
[v2] PEDIDO: Saio de Pallet às 18h e preciso entregar em Celadon e Lavender.
  FERRAMENTA: buscar_rota {"origem": "Pallet", "paradas": ["Celadon", "Lavender"], "hora": "18h"}
  RESULTADO : True ['Pallet Town', 'Cinnabar Island', 'Seafoam Islands', 'Fuchsia City', 'Celadon City', 'Saffron City', 'Lavender Town'] 684.3
  RESPOSTA  : Aqui está a rota ideal para sua entrega:

1. **Pallet Town** -> **Cinnabar Island**
   - Tempo estimado: 501 minutos

2. **Cinnabar Island** -> **Seafoam Islands**
   - Tempo estimado: 182 minutos

3. **Seafoam Islands** -> **Fuchsia City**
   - Tempo estimado: 182 minutos

4. **Fuchsia City** -> **Celadon City**
   - Tempo estimado: 182 minutos

5. **Celadon City** -> **Saffron City**
   - Tempo estimado: 182 minutos

6. **Saffron City** -> **Lavender Town**
   - Tempo estimado: 182 minutos

Total de tempo estimado: **684 minutos**

Esta rota economiza **0 minutos** em comparação com a rota mais curta possível.

Onde você gostaria de retornar? Se sim, escolha uma opção abaixo:
- **Retornar à origem**: Pallet Town
- **Não retornar**: Continue para Lavender Town

Selecione uma opção:
  (70.198s, passos=2, tokens=3703+302, reserva=False)
================================================================================
[v2] PEDIDO: Qual a rota mais rápida de Pewter até Fuchsia às 8h? Teve um acidente entre Saffron e Vermillion.
  FERRAMENTA: buscar_rota {"origem": "Pewter", "destino": "Fuchsia", "hora": "8", "acidentes": [["Saffron", "Vermillion"]], "retornar_origem": false}
  RESULTADO : True ['Pewter City', 'Mt. Moon', 'Cerulean City', 'Saffron City', 'Celadon City', 'Fuchsia City'] 570.8
  RESPOSTA  : A rota mais rápida de Pewter até Fuchsia às 8h, considerando o acidente entre Saffron e Vermillion, é:

1. **Pewter City** -> **Mt. Moon**
2. **Mt. Moon** -> **Cerulean City**
3. **Cerulean City** -> **Saffron City**
4. **Saffron City** -> **Celadon City**
5. **Celadon City** -> **Fuchsia City**

O tempo estimado para essa rota é de aproximadamente **570 minutos**, ou **9 horas e 30 minutos**.

Esta rota economiza **0 minutos** em comparação com a rota mais curta possível, que seria:

- **Pewter City** -> **Mt. Moon**
- **Mt. Moon** -> **Cerulean City**
- **Cerulean City** -> **Saffron City**
- **Saffron City** -> **Celadon City**
- **Celadon City** -> **Fuchsia City**

A diferença de tempo entre essas duas roteiras é zero, pois ambas têm o mesmo tempo previsto de **570 minutos**.

Por favor, note que esta rota inclui paradas nos seguintes lugares: Mt. Moon, Cerulean City, Saffron City, e Celadon City.
  (74.224s, passos=2, tokens=3672+356, reserva=False)
================================================================================
[v2] PEDIDO: Leve 3 Poções de Goldenrod até Ecruteak agora, evitando Violet.
  FERRAMENTA: buscar_rota {"origem": "Goldenrod", "destino": "Ecruteak", "hora": "agora", "evitar": ["Violet"], "produto": "3 Poções"}
  RESULTADO : True ['Goldenrod City', 'National Park', 'Ecruteak City'] 96.8
  RESPOSTA  : A rota ideal para levar 3 Poções de Goldenrod até Ecruteak, evitando Violet, é:

1. Goldenrod City -> National Park -> Ecruteak City
O tempo estimado para essa rota é de aproximadamente **96 minutos**.

Esta rota economiza **0 minutos** em comparação com a rota mais curta possível.

Regras aplicáveis:
- Evitar: Violet City

Desta forma, você chegará na Ecruteak City **aproximadamente 96 minutos após sair de Goldenrod City**, seguindo a rota indicada.
  (60.418s, passos=2, tokens=3612+191, reserva=False)
================================================================================
[v2] PEDIDO: Me passa uma receita de bolo de cenoura?
  RESPOSTA  : Desculpe, eu só planejo rotas de entrega em Kanto e Johto. Diga a origem, os destinos e o horário que eu calculo a melhor rota.
  (13.734s, passos=1, tokens=1677+43, reserva=False)

```
