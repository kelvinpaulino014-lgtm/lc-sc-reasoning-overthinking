# Roteiro de Apresentação Oral (5 Minutos — Exatos 4 Slides)
**Disciplina:** PPGI0034 — Redes Neurais e Aprendizado Profundo (UnB — 2026/2)  
**Docente:** Prof. Dr. Díbio Leandro Borges  
**Autor:** Kelvin Rafael Paulino Buss (`kelvin.buss@aluno.unb.br`)  
**Título:** Controle Dinâmico de "Overthinking" em Modelos de Linguagem via Parada Precoce Calibrada por Entropia em Cadeias de Pensamento (*Chain-of-Thought*)

---

## Distribuição de Tempo (Total: 5m00s)

| Slide | Título do Slide | Tempo Alvo | Foco Central da Fala |
| :---: | :--- | :---: | :--- |
| **Slide 1** | **Contexto, Problema ("Overthinking") e os 4 Artigos do ICLR 2026** | `0m00s – 1m15s` (75s) | Conectar com a Aula 11 (*Self-Consistency*, Slide 17), apresentar a curva em U invertido de (WU et al., 2026), o limite do CaTS de (HUANG et al., 2026) e a sobrecorreção de (HAO et al., 2026). |
| **Slide 2** | **A Inovação Proposta: Decodificador Online LC-SC ($N \times L$)** | `1m15s – 2m35s` (80s) | Explicar **QUAL** é a inovação (controle bidimensional $N \times L$ em tempo real) e **POR QUE** funciona (poda intratrilha $L$-*Stopping* por pico de entropia de Shannon $H_t > \tau_H$ + votação ponderada por comprimento ótimo $L^*$ + $N$-*Stopping*). |
| **Slide 3** | **Resultados Experimentais: Curva em U Invertido e Fronteira de Pareto** | `2m35s – 4m00s` (85s) | Mostrar **COMO se observa a inovação**: Figura 1 (Curva em U invertido no GSM8K e MATH-500) e Figura 3 (Deslocamento superior da Fronteira de Pareto com redução de 69,0% dos tokens no GSM8K e 65,1% no MATH-500, com ganho de +10,0 p.p. e +22,0 p.p. sobre o SC). |
| **Slide 4** | **Calibração (ECE), Estudo de Ablação e Conclusões** | `4m00s – 5m00s` (60s) | Mostrar a queda do ECE (Figura 4a), a ablação dos 3 componentes (Figura 4b) e responder diretamente à Pergunta de Pesquisa. |

---

## Texto Falado Sugerido (Slide a Slide)

### Slide 1 (0m00s – 1m15s): Contexto, Pergunta de Pesquisa e Base ICLR 2026
> "Bom dia/boa tarde, Professor Díbio e colegas. Meu Projeto 2 investiga o controle dinâmico do fenômeno de *overthinking* em modelos de linguagem compactos (1.5B) durante o raciocínio *Chain-of-Thought* (visto na Aula 11).
> A literatura recente do **ICLR 2026** revelou um paradoxo importante: **Wu et al. (2026)** provaram que a acurácia em função do tamanho $L$ da cadeia segue uma **curva em U invertido** — pensar demais acumula erros. **Hao et al. (2026)** demonstraram que passos extras de revisão em modelos menores causam *sobrecorreção* (*over-correction*), trocando uma resposta já correta por uma alucinação, enquanto **Feng et al. (2026)** mostraram a diluição de atenção em contextos longos.
> Por outro lado, **Huang et al. (2026)** propuseram o **CaTS** para parar precocemente o número de amostras $N$ do *Self-Consistency*, mas não controlam o comprimento $L$ de cada cadeia e sofrem descalibração severa em trilhas longas. Nossa pergunta de pesquisa é: podemos controlar **simultaneamente e online** a matriz $N \times L$ monitorando a entropia de Shannon para cortar o *overthinking* antes que ele corrompa o voto?"

### Slide 2 (1m15s – 2m35s): A Inovação — Length-Calibrated Self-Consistency (LC-SC)
> "Para responder a isso, propomos o **LC-SC** (*Length-Calibrated Self-Consistency*).
> Enquanto o *Length-Filtered Vote* de Wu et al. (2026) é apenas *post-hoc* — obrigando a GPU a gerar todas as cadeias até o fim para só depois filtrá-las — e o CaTS de Huang et al. (2026) atua apenas no eixo $N$, nossa inovação opera em três frentes simultâneas:
> 1. **$L$-Stopping Online (Poda Intratrilha):** rastreamos a entropia de Shannon $H_t$ passo a passo. Quando a cadeia já obteve uma resposta candidata após $L_{\min}$ e dispara um pico de entropia $\bar{H}_k > \tau_H$ (sinalizando hesitação circular), truncamos a trilha imediatamente, salvando a resposta pré-sobrecorreção.
> 2. **Ponderação Calibrada por Comprimento:** cada voto recebe o peso $w_i = \hat{c}_i \cdot \exp(-\lambda \max(0, L_i - L^*)/\sigma_L)$, penalizando trilhas que excedem a zona ótima $L^*$ calibrada no conjunto de validação.
> 3. **$N$-Stopping Calibrado:** encerramos a geração de novas amostras assim que o consenso ponderado atinge o limiar $\gamma$."

### Slide 3 (2m35s – 4m00s): Resultados — Curva em U Invertido e Fronteira de Pareto
> "Validamos o LC-SC nos benchmarks matemáticos **GSM8K** e **MATH-500** com o modelo `Qwen2.5-1.5B-Instruct` em GPU RTX 3050, separando estritamente calibração e teste.
> À esquerda, comprovamos empiricamente a **curva em U invertido** de Wu et al. (2026): a acurácia atinge o pico na zona ótima $L^*$ ($134$ tokens no GSM8K e $211$ no MATH-500) e cai drasticamente para cadeias longas em *overthinking* (de $77{,}9\%$ para $24{,}6\%$ no GSM8K). A curva tracejada vermelha mostra como o nosso **$L$-Stopping** resgata **794 sobrecorreções** no total ($395$ no GSM8K e $399$ no MATH-500).
> À direita, na **Fronteira de Pareto**, o LC-SC domina todos os baselines: comparado ao *Self-Consistency* padrão ($N=16$), o LC-SC reduz o consumo médio de tokens em **69,0%** no GSM8K e **65,1%** no MATH-500, enquanto eleva a acurácia final para **85,00%** no GSM8K (+10,0 p.p. sobre SC e +5,0 p.p. sobre CaTS) e **85,50%** no MATH-500 (+22,0 p.p. sobre SC e +14,0 p.p. sobre CaTS)."

### Slide 4 (4m00s – 5m00s): Calibração (ECE), Ablação e Conclusão
> "Por fim, no Slide 4 analisamos o porquê desse ganho.
> No diagrama de confiabilidade à esquerda, vemos que a confiança bruta e mesmo o CaTS sofrem com trilhas longas superconfiantes, enquanto o LC-SC reduz o Erro de Calibração Esperado (**ECE**) de **20,33% para 6,60%** no GSM8K e de **19,83% para 8,19%** no MATH-500 (com ROC-AUC de **91,80%**).
> No estudo de ablação à direita, remover o $L$-Stopping ou a penalização de comprimento $\lambda$ degrada tanto a acurácia quanto o custo computacional, provando que os três mecanismos propostos são complementares.
> Concluímos positivamente nossa pergunta de pesquisa: o controle conjunto $N \times L$ guiado por entropia mitiga o *overthinking* em SLMs com zero necessidade de retreinamento. Obrigado!"
