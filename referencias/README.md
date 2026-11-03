# Acervo de Referências Oficiais do ICLR 2026 — Projeto Individual 2

Este diretório armazena **localmente** todos os 4 artigos científicos do **ICLR 2026** (*The Fourteenth International Conference on Learning Representations*) e seus respectivos **repositórios oficiais de código-fonte do GitHub**, que fundamentam o método proposto **LC-SC (*Length-Calibrated Self-Consistency*)**.

---

## 1. Artigos em PDF (`artigos_iclr2026_pdfs/`)

| Arquivo Local | Citação ABNT (ICLR 2026) | Papel no Nosso Projeto (LC-SC) |
| :--- | :--- | :--- |
| [`01_Wu_et_al_2026_When_More_is_Less_CoT_Length.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/01_Wu_et_al_2026_When_More_is_Less_CoT_Length.pdf) | **WU, Yuyang; WANG, Yifei; YE, Ziyu; DU, Tianqi; JEGELKA, Stefanie; WANG, Yisen.** *When More is Less: Understanding Chain-of-Thought Length in LLMs*. In: **ICLR**, 2026. | Demonstra teoricamente (Teorema 4.3) e empiricamente a **curva em U invertido** da acurácia em função do comprimento $L$ do CoT (*overthinking*). Propõe o *Length-Filtered Vote* (Algoritmo 1, p. 27), que filtra comprimentos apenas *post-hoc* (após gastar todos os tokens). Nosso LC-SC transforma essa intuição em **parada precoce online ($L$-Stopping)** durante a decodificação. |
| [`02_Huang_et_al_2026_CaTS_Calibrated_Test_Time_Scaling.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/02_Huang_et_al_2026_CaTS_Calibrated_Test_Time_Scaling.pdf) | **HUANG, Chengsong; HUANG, Langlin; LENG, Jixuan; LIU, Jiacheng; HUANG, Jiaxin.** *CaTS: Calibrated Test-Time Scaling for Efficient LLM Reasoning*. In: **ICLR**, 2026. | Propõe escalonamento em tempo de teste calibrado (CaTS) para interromper o número de amostras $N$ do *Self-Consistency* no GSM8K e MATH usando `DeepSeek-R1-Distill-Qwen-1.5B`. Contudo, não controla o comprimento $L_i$ de cada trilha e sofre queda de AUC (de 64,31% para 55,57% na Tabela 1) em trilhas longas. Nosso LC-SC resolve essa falha combinando o consenso em $N$ com o truncamento e ponderação em $L$. |
| [`03_Hao_et_al_2026_RL_of_Thoughts_Navigating_LLM_Reasoning.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/03_Hao_et_al_2026_RL_of_Thoughts_Navigating_LLM_Reasoning.pdf) | **HAO, Qianyue; LI, Sibo; YUAN, Jian; LI, Yong.** *RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning*. In: **ICLR**, 2026. | Modela os passos lógicos de inferência como um MDP e comprova no Apêndice B.2 (Figura 7, p. 19) o fenômeno de **sobrecorreção (*over-correction*)**: passos extras de refinamento corrompem respostas que já estavam corretas em modelos menores. Justifica o gatilho de truncamento por entropia do nosso LC-SC. |
| [`04_Feng_et_al_2026_In_Place_Test_Time_Training.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/04_Feng_et_al_2026_In_Place_Test_Time_Training.pdf) | **FENG, Guhao; LUO, Shengjie; HUA, Kai; ZHANG, Ge; HE, Di; HUANG, Wenhao; CAI, Tianle.** *In-Place Test-Time Training*. In: **ICLR**, 2026 (Oral). | Demonstra como o acúmulo excessivo de tokens gerados em tempo de teste degrada o mecanismo de atenção (diluição de contexto) e propõe adaptação compacta de estados. Fundamenta teoricamente por que manter cadeias CoT enxutas em torno de $L^*$ preserva a qualidade da representação. |

---

## 2. Repositórios Oficiais do GitHub (`repos_iclr2026/`)

Todos os 4 repositórios oficiais foram clonados/extraídos localmente para inspeção de prompts, hiperparâmetros e scripts de avaliação:

1. **`repos_iclr2026/01_PKU-ML_CoT-Length/`**
   - **Origem:** `https://github.com/PKU-ML/CoT-Length` (Wu et al., ICLR 2026)
   - **Conteúdo útil:** Scripts de medição de comprimento de CoT, agrupamento por faixas de tokens e implementação de referência do *Length-Filtered Vote*.
2. **`repos_iclr2026/02_Chengsong-Huang_Self-Calibration_CaTS/`**
   - **Origem:** `https://github.com/Chengsong-Huang/Self-Calibration` (Huang et al., ICLR 2026)
   - **Conteúdo útil:** Prompts de extração de confiança, cálculo de ECE/AUC, *Entropy-based Dynamic Temperature* (EDT) e scripts de *Early Stopping* no GSM8K.
3. **`repos_iclr2026/03_tsinghua-fib-lab_RL-LLM-Reasoning/`**
   - **Origem:** `https://github.com/tsinghua-fib-lab/RL-LLM-Reasoning` (Hao et al., ICLR 2026)
   - **Conteúdo útil:** Estruturação de blocos de pensamento passo a passo e avaliação em benchmarks matemáticos.
4. **`repos_iclr2026/04_ByteDance-Seed_In-Place-TTT/`**
   - **Origem:** `https://github.com/ByteDance-Seed/In-Place-TTT` (Feng et al., ICLR 2026 Oral)
   - **Conteúdo útil:** Arquitetura e scripts de avaliação de eficiência em contexto para modelos da família Qwen.
