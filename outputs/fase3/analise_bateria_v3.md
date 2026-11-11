# Análise da bateria de pedidos — Qwen2.5-1.5B-Instruct, prompt v3

Fonte: `bateria_qwen2.5-1.5b-instruct_v3.csv` e `.md` (20 pedidos de `testes/pedidos_teste.json`,
critérios definidos antes de rodar). Execução local em CPU (PyTorch sem CUDA), 11/11/2026.

## Resultado

| Métrica | Valor |
|---|---|
| Acerto completo | **17/20 (85%)** |
| Ferramenta certa | 20/20 (100%); precisão e revocação 1,0 em todas as classes |
| Argumentos certos | 17/20 (85%) |
| Rota igual à esperada | 10/12 pedidos de rota válidos |
| Resposta final fiel | 18/20 |
| Respostas do LLM barradas pela checagem de fidelidade | 2 (ambas alucinações reais) |
| Latência média (CPU) | 33 s por pedido (10 s sem ferramenta, ~45 s com rota) |
| Tokens médios por pedido | ~3.300 (≈ 90% são a descrição das ferramentas e o prompt de sistema) |

O agente **sempre escolheu a ferramenta certa**: pediu rota quando era rota, consultou o trecho
ou as vizinhas quando era isso, e não chamou nada em saudações, perguntas fora do domínio e no
pedido sem origem (caso 20). No caso 20 ele não chutou uma origem, mas também não perguntou de
onde o veículo sai: respondeu "Você quer que eu faça a entrega diretamente para Celadon?". A
métrica conta como acerto, mas a resposta poderia ser mais útil. Também acertou os
casos difíceis: erros de digitação ("Cerulian", "Saffon"), grafia alternativa ("vermilion"),
horário por extenso ("6 da tarde", "meio-dia") e pedido em inglês. Nos dois pedidos fora do
mapa (São Paulo, Hogwarts) a ferramenta recusou o destino e o agente explicou o erro, sem
inventar rota.

## Onde errou (3 casos) e por quê

Os três erros são do mesmo tipo: **o modelo deixou de extrair uma informação do pedido**. Ele
nunca preencheu um argumento com valor errado.

1. **#6, várias paradas com retorno.** No pedido "...e depois volto para Goldenrod", ele não
   preencheu `retornar_origem`. A rota calculada é válida, mas termina em Olivine. O v3 só cita
   `retornar_origem` na lista de argumentos e nenhum exemplo usa esse campo.
2. **#9, estrada interditada.** "A estrada entre Celadon e Fuchsia está interditada" não virou
   `bloqueios`. O agente devolveu justamente a estrada interditada, que é **o erro mais grave da
   bateria**. Os exemplos do prompt só mostram acidente (atraso), e o modelo não associou
   "interditada" a bloqueio.
3. **#13, tempo de um trecho às 18h.** Em `estimar_tempo_trecho` ele omitiu `hora`, e a
   ferramenta usou "agora". O prompt descreve a regra de horário só para `buscar_rota`. Além
   disso, a resposta do LLM inventou "5 horas e 37 minutos" e foi barrada. A explicação de
   reserva mostrava o JSON cru ao usuário; isso foi corrigido em `explicar_resultado`.

Hipótese comum: com 1,5 bilhão de parâmetros, o modelo segue bem o padrão dos exemplos
(few-shot), mas generaliza pouco para argumentos que não aparecem em nenhum exemplo. Uma
versão v4 do prompt com regras e exemplos para "interditada/fechada → bloqueios",
"volto/retorno → retornar_origem" e "hora também em estimar_tempo_trecho" ataca exatamente os
três erros. Ela precisa ser medida na mesma bateria antes de se dizer que funciona.

## A checagem de fidelidade em ação

| Caso | O que o LLM escreveu | O que a ferramenta devolveu | Resultado |
|---|---|---|---|
| #7 | "economia de 0.9 minutos" | economia 0,0 min | resposta descartada, usada a reserva |
| #13 | "37 minutos (5 horas e 37 minutos)" | 36,8 min | resposta descartada, usada a reserva |

Limitação conhecida: a checagem compara números, locais e trechos, mas não o **contexto** de
cada número. No teste inicial com o v2 (`teste_inicial_qwen.md`), o modelo repetiu um tempo
verdadeiro de uma perna ("182 min") em pernas onde ele não valia, e isso passou.

## Custo e latência

- Não há custo de API: o modelo roda localmente, no Colab ou no computador.
- O custo está em tempo e memória. Em CPU, fp32 usa ~6 GB de RAM e leva 30 a 60 s por pedido.
  Os pedidos de rota fazem duas chamadas ao LLM (escolher a ferramenta e explicar o resultado).
- Cerca de 3.000 dos ~3.300 tokens de entrada são fixos (esquemas das ferramentas e prompt).
  Encurtar as descrições das ferramentas reduziria o custo de todo pedido.
