# UNIVERSIDADE DE BRASÍLIA (UnB) — INSTITUTO DE CIÊNCIAS EXATAS (IE)
**Departamento de Ciência da Computação (CIC) — Pós-Graduação em Informática (PPGI)**  
**Disciplina:** PPGI0034 — Redes Neurais e Aprendizado Profundo (2026/2) | **Prof. Dr. Díbio Leandro Borges**  
**Entrega Etapa 1 (Proposta de Projeto Individual 2 — Prazo: 04/11/2026 até 23h59)**  
**Autor:** Kelvin Rafael Paulino Buss (`kelvin.buss@aluno.unb.br`)  
**Repositório GitHub do Projeto:** `https://github.com/kelvinbuss/ProjetoFinalRNAP_UnB2026`

---

## 1. Título do Projeto Final
**Controle Dinâmico de "Overthinking" em Modelos de Linguagem via Parada Precoce Calibrada por Entropia em Cadeias de Pensamento (*Chain-of-Thought*)**

---

## 2. Pergunta de Pesquisa
> *"É possível mitigar o fenômeno de degradação de acurácia por excesso de tokens de raciocínio ('overthinking') em Small Language Models (1.5B a 3B de parâmetros) combinando o orçamento adaptativo de amostragem em tempo de teste com um critério online de truncamento e ponderação dinâmica do comprimento da cadeia de pensamento (CoT) guiado por entropia?"*

---

## 3. Resumo Executivo, Objetivos e Estrutura da Inovação
Modelos de linguagem de raciocínio (*Reasoning Language Models* — RLMs) escalam o processamento em tempo de teste (*Test-Time Compute*) via *Chain-of-Thought* (CoT) e votação por *Self-Consistency* (SC). Contudo, estudos recentes demonstram que cadeias de pensamento excessivamente longas degradam a acurácia em uma curva em "U invertido" (*overthinking*) (WU et al., 2026) e causam "sobrecorreção" (*over-correction*) em modelos menores (HAO et al., 2026), além de diluição de atenção em contextos extensos (FENG et al., 2026). Paralelamente, métodos de escalonamento calibrado como o CaTS (HUANG et al., 2026) reduzem o número de trilhas amostradas ($N$), mas não restringem online o comprimento individual ($L_i$) de cada cadeia, desperdiçando tokens em trajetórias alucinadas ou circulares.

### Estrutura da Inovação Proposta (*Length-Calibrated Self-Consistency* — LC-SC)
* **QUAL é a inovação:** Propõe-se o decodificador **LC-SC**, um algoritmo de inferência online que controla conjuntamente a matriz de custo computacional $N \times L$ em tempo de teste. Diferente do *Length-Filtered Vote* de (WU et al., 2026) — que atua apenas *post-hoc* após gerar todas as cadeias completas — e do CaTS de (HUANG et al., 2026) — que controla apenas o eixo de amostras $N$ —, o LC-SC monitora em tempo real a entropia de Shannon $H_t = -\sum_{v \in \mathcal{V}} p(v \mid x, y_{<t})\log p(v \mid x, y_{<t})$ nos passos intermediários de raciocínio para realizar parada precoce bidimensional ($L$-*Stopping* intratrilha e $N$-*Stopping* intertrilhas) e votação ponderada por comprimento ótimo:
  $$\hat{y} = \arg\max_a \sum_{i=1}^{N_{\text{adap}}} \mathbb{I}(\text{ans}(r_i)=a)\cdot \hat{c}_i \cdot \exp\left(-\lambda \frac{\max(0, L_i - L^*)}{\sigma_L}\right)$$
* **POR QUE é uma inovação:** Em *Small Language Models* (como `DeepSeek-R1-Distill-Qwen-1.5B` e `Qwen2.5-Math-1.5B`), trilhas que ultrapassam a zona ótima $L^*$ entram frequentemente em ciclos de hesitação ou degeneração. Na Tabela 1 de (HUANG et al., 2026), a calibração bruta do modelo de 1.5B no GSM8K cai de 64,31% para 55,57% em AUC devido a trilhas longas superconfiantes. Ao detectar picos sustentados de entropia ($\bar{H}_k > \tau_H$ após $L_{\min}$) e forçar a transição imediata ao fechamento da resposta (`</think>`), o LC-SC evita a geração de centenas de tokens espúrios e protege o consenso majoritário contra votos corrompidos por *overthinking*.
* **COMO observar nos resultados experimentais:** A validação será conduzida nos benchmarks matemáticos **GSM8K** e **MATH-500** comparando 5 métodos: (i) *Greedy CoT*, (ii) *Standard Self-Consistency* (Aula 11), (iii) *Length-Filtered Vote post-hoc* (WU et al., 2026), (iv) *CaTS* (HUANG et al., 2026) e (v) **LC-SC (Proposto)**. A inovação será comprovada por: (1) reprodução empírica da curva em U invertido de acurácia versus comprimento de tokens $L$; (2) deslocamento superior da **Fronteira de Pareto** (Acurácia Pass@1 vs. Custo Computacional médio em tokens gerados), demonstrando redução de 35%–50% no consumo de tokens com ganho de acurácia e menor erro de calibração esperado (*Expected Calibration Error* — ECE); e (3) estudo de ablação isolando os ganhos de $L$-*Stopping*, $N$-*Stopping* e da penalização exponencial de comprimento.

**Objetivos Específicos:**
1. Caracterizar empiricamente a curva em U invertido e a assinatura de entropia de trilhas com *overthinking* em SLMs;
2. Implementar o decodificador online LC-SC integrando truncamento dinâmico $L$ e consenso calibrado $N$;
3. Avaliar a fronteira de Pareto (acurácia vs. tokens/latência) e o ECE frente aos 4 baselines do estado da arte no GSM8K e MATH-500.

---

## 4. Referências Relacionadas (Exclusivamente do ICLR 2026 — Formato ABNT)
1. **WU, Yuyang; WANG, Yifei; YE, Ziyu; DU, Tianqi; JEGELKA, Stefanie; WANG, Yisen.** When More is Less: Understanding Chain-of-Thought Length in LLMs. In: **The Fourteenth International Conference on Learning Representations (ICLR)**, 2026. Disponível em: https://iclr.cc/virtual/2026/papers.html (arXiv:2502.07266).
2. **HUANG, Chengsong; HUANG, Langlin; LENG, Jixuan; LIU, Jiacheng; HUANG, Jiaxin.** CaTS: Calibrated Test-Time Scaling for Efficient LLM Reasoning. In: **The Fourteenth International Conference on Learning Representations (ICLR)**, 2026. Disponível em: https://iclr.cc/virtual/2026/papers.html (arXiv:2503.00031).
3. **HAO, Qianyue; LI, Sibo; YUAN, Jian; LI, Yong.** RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning. In: **The Fourteenth International Conference on Learning Representations (ICLR)**, 2026. Disponível em: https://iclr.cc/virtual/2026/papers.html (arXiv:2505.14140).
4. **FENG, Guhao; LUO, Shengjie; HUA, Kai; ZHANG, Ge; HE, Di; HUANG, Wenhao; CAI, Tianle.** In-Place Test-Time Training. In: **The Fourteenth International Conference on Learning Representations (ICLR)**, 2026 (Oral). Disponível em: https://iclr.cc/virtual/2026/papers.html (arXiv:2604.06169).
