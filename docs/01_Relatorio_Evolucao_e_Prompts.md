# Universidade de Brasília (UnB)
## Programa de Pós-Graduação em Informática (PPGI)
### PPGI0034 — Redes Neurais e Aprendizado Profundo (2026/2)
**Projeto Individual 2:** Controle Dinâmico de *"Overthinking"* em Modelos de Linguagem via Parada Precoce Calibrada por Entropia em Cadeias de Pensamento (*Chain-of-Thought*)  
**Aluno:** Kelvin Rafael Paulino Buss (`kelvin.buss@aluno.unb.br`)  
**Docente:** Prof. Dr. Díbio Leandro Borges  
**Repositório GitHub Oficial:** `https://github.com/kelvinbuss/ProjetoFinalRNAP_UnB2026`

---

# RELATÓRIO TÉCNICO-CIENTÍFICO: EVOLUÇÃO ARQUITETURAL, LIÇÕES APRENDIDAS DO PROJETO 1, AUDITORIA DE PROMPTS E FORMULAÇÃO DO ALGORITMO LC-SC

---

## 1. Contextualização e Transição Epistemológica: Do Projeto 1 ao Projeto 2

No **Projeto Individual 1** desta disciplina (*Modelagem de Linguagem Autoregressiva na Obra de Machado de Assis*), investigamos a construção e o treinamento *from scratch* de arquiteturas Transformer causais (*nanoGPT* clássico e *Machado-GPT Modernizado* com RoPE, RMSNorm, SwiGLU e tokenização BPE). Aquela jornada consolidou três aprendizados fundamentais que guiaram diretamente o desenho deste **Projeto Individual 2**:

1. **A Natureza Multiplicativa do Erro Autoregressivo e Laços de Degeneração:**
   Na modelagem causal $P(Y \mid X) = \prod_{t=1}^L P(y_t \mid X, y_{<t})$, observamos que sequências longas tendem a acumular desvios estocásticos ou cair em atratores repetitivos. No Projeto 2, levamos essa análise para a fronteira do estado da arte (ICLR 2026) em **Modelos de Linguagem de Raciocínio (*Reasoning Language Models* — RLMs)**: quando um modelo de 1,5B parâmetros gera uma cadeia de pensamento (*Chain-of-Thought* — CoT) excessivamente longa ($L \gg L^*$), ele sofre do fenômeno de ***overthinking*** (curva em U invertido de acurácia versus comprimento $L$) (WU et al., 2026) e **sobrecorreção (*over-correction*)** (HAO et al., 2026).

2. **Correção Crítica de Infraestrutura Computacional (CPU $\to$ GPU CUDA RTX 3050):**
   * **Diagnóstico do Erro no Projeto 1:** Durante o Projeto 1, o ambiente virtual Python inicializou com a compilação padrão do PyTorch restrita à CPU (`torch+cpu`), o que fez o treinamento do modelo BPE de 2,76M parâmetros levar **2h 34min 59s (9.299s)** em processador x86.
   * **Solução Arquitetural no Projeto 2:** Desde a Fase 1 do Projeto 2, configuramos explicitamente o ambiente `.venv` com **PyTorch compilado para CUDA 12.x**, explorando integralmente os Tensor Cores e os **6 GB de VRAM GDDR6 da GPU NVIDIA GeForce RTX 3050 Laptop** do autor em precisão mista (`FP16`/`BF16`). Além disso, projetamos a arquitetura de **Cache de Trilhas CoT em formato colunar (`.parquet`)**, onde as probabilidades, comprimentos e entropias de Shannon $H_t$ passo a passo são gravados uma única vez a partir da GPU, permitindo executar ablações e varreduras de hiperparâmetros em menos de 5 segundos.

3. **Adequação Estrita às Normas de Citação e Entregáveis do Edital do Prof. Dr. Díbio:**
   * **Padrão ABNT `(AUTOR, ano)` em Corpo IEEE:** O edital do Projeto 2 exige formatação visual de conferência IEEE em dupla coluna (`IEEEtran.cls`, 6 a 8 páginas), porém com **citações e referências bibliográficas completas no padrão ABNT `(AUTOR, ano)`** (em vez de colchetes numéricos `[1]`). Essa diretriz foi aplicada nativamente desde a proposta da Etapa 1 e no artigo principal (`artigo_ieee/main.tex`).
   * **Relatório Específico de Uso de IA (Exatas 2 Páginas):** O edital exige um segundo relatório dedicado de 2 páginas detalhando quais modelos de IA foram utilizados e separando o que é autoria intelectual do aluno do que teve auxílio de ferramentas de IA (`relatorio_ia/relatorio_ia.tex`).

---

## 2. Diagnóstico Científico dos 4 Artigos do ICLR 2026 e a Proposta do LC-SC

A revisão sistemática dos anais do **ICLR 2026** (*The Fourteenth International Conference on Learning Representations*) conectada à **Aula 11 (Slide 17 — *Self-Consistency*)** revelou uma lacuna teórica e prática clara entre quatro trabalhos de ponta:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    LACUNA CIENTÍFICA NOS ARTIGOS DO ICLR 2026                          │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ 1. WU et al. (ICLR 2026)                  │ 2. HUANG et al. (ICLR 2026 - CaTS)         │
│ "When More is Less: CoT Length in LLMs"   │ "Calibrated Test-Time Scaling"             │
│ • Prova a curva em U invertido A(L) e L*. │ • Para cedo no nº de amostras (Eixo N).    │
│ • FALHA: O Length-Filtered Vote é         │ • FALHA: Não controla o tamanho L_i!       │
│   POST-HOC (gera todas as N cadeias até   │   Na Tabela 1 (GSM8K, 1.5B), a AUC cai de  │
│   o fim antes de filtrar; zero economia). │   64,31% para 55,57% em trilhas longas!    │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ 3. HAO et al. (ICLR 2026 - RL of Thoughts)│ 4. FENG et al. (ICLR 2026 Oral - In-Place) │
│ • Apêndice B.2 (Fig. 7, p. 19): prova que │ • Prova que contextos longos em tempo de   │
│   passos extras após achar a resposta     │   teste causam diluição de atenção nas     │
│   certa causam OVER-CORRECTION em SLMs.   │   camadas internas do Transformer.         │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│         NOSSA INOVAÇÃO: LC-SC (Length-Calibrated Self-Consistency)                     │
│  Controla simultaneamente e ONLINE a matriz bidimensional N x L sem retreinamento:     │
│  (1) L-Stopping Online: poda a trilha t > L_min quando a entropia média móvel          │
│      H_k(t) > tau_H após detectar resposta candidata preliminar (evita overthinking).  │
│  (2) Votação Calibrada com Penalização Exponencial de Comprimento:                     │
│      w_i = c_hat_i * exp(-lambda * max(0, L_i - L*) / sigma_L).                        │
│  (3) N-Stopping Calibrado: interrompe novas amostras N quando consenso >= gamma.       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Registro Cronológico e Integral dos Prompts Reais do Autor

Abaixo estão registrados os comandos exatos (*prompts*) emitidos pelo autor durante a condução do projeto, acompanhados das decisões de engenharia e auditoria correspondentes:

---

### Prompt 1 — Iniciação do Projeto 2, Análise Retrospectiva do Projeto 1 e Mapeamento do ICLR 2026
> **Prompt do Usuário:**  
> *"Vamos iniciar o Projeto 2. Veja todos os arquivos do Projeto 1 para aprender sobre os acertos e erros que fizemos e fazer melhor no Projeto 2. Na pasta do Projeto 2 já tem os slides das Aulas 10, 11 e 12 e as instruções do Projeto 2."*

* **Análise e Decisões Tomadas:**
  1. Leitura integral do edital `projeto2-rnap-2026_2 (1).pdf` e dos slides `aula_10_rnap_ciber2026_2.pdf`, `aula_11_rnap_ciber2026_2.pdf` e `aula_12_rnap_ciber2026_2.pdf`.
  2. Levantamento das lições do Projeto 1: necessidade de usar GPU NVIDIA RTX 3050 via CUDA desde o início, citações em formato ABNT `(AUTOR, ano)` dentro do template `IEEEtran`, e preparação do relatório de 2 páginas sobre uso de IA.
  3. Formulação do documento estratégico `opcoes_projeto_2_iclr2026.txt` contendo 5 opções completas de projetos (alinhadas às Aulas 10, 11 e 12) cruzadas com 20 artigos reais publicados no ICLR 2026.

---

### Prompt 2 — Seleção da Opção 2 (LC-SC), Entrega da Etapa 1 (04/11) e Download do Acervo ICLR 2026
> **Prompt do Usuário:**  
> *"A Opção 2 é a melhor! Faça o documento de entrega da Etapa 1 (04/11/2026), deixe tudo muito bem organizado e já baixe na pasta de referências os 4 artigos do ICLR 2026 da Opção 2 e os códigos deles se tiverem disponíveis. Me explique passo a passo para eu entender todo o projeto, conseguir corrigir e auditar, além de ter uma expectativa de tempo para cada fase. Use subagentes em paralelo quando possível."*

* **Análise e Decisões Tomadas:**
  1. **Download e Auditoria Física das Fontes Primárias do ICLR 2026 (`Projeto_2/referencias/`):**
     * Baixados os 4 PDFs integrais em `referencias/artigos_iclr2026_pdfs/`:
       - `01_Wu_et_al_2026_When_More_is_Less_CoT_Length.pdf` (WU et al., 2026)
       - `02_Huang_et_al_2026_CaTS_Calibrated_Test_Time_Scaling.pdf` (HUANG et al., 2026)
       - `03_Hao_et_al_2026_RL_of_Thoughts_Navigating_LLM_Reasoning.pdf` (HAO et al., 2026)
       - `04_Feng_et_al_2026_In_Place_Test_Time_Training.pdf` (FENG et al., 2026)
     * Baixados e extraídos os 4 repositórios oficiais do GitHub em `referencias/repos_iclr2026/`:
       - `01_PKU-ML_CoT-Length`
       - `02_Chengsong-Huang_Self-Calibration_CaTS`
       - `03_tsinghua-fib-lab_RL-LLM-Reasoning`
       - `04_ByteDance-Seed_In-Place-TTT`
  2. **Geração da Entrega da Etapa 1 (`entrega_etapa1_04nov/`):**
     * Compilado o PDF de 1 página exata (`proposta_projeto2_04nov_Kelvin_Buss.pdf`) contendo Título, Pergunta de Pesquisa, Resumo com a Tríade da Inovação (*Qual, Por Quê, Como Observar*), Objetivos, link do GitHub e as 4 referências do ICLR 2026 em ABNT.
  3. **Elaboração do Guia Didático e Plano de Execução Paralela (`GUIA_PASSO_A_PASSO_PROJETO_2.md`):**
     * Detalhamento didático da arquitetura do **LC-SC** e planejamento de execução com subagentes paralelos reduzindo o tempo total de ~8h–9h sequenciais para ~4h30–5h30.

---

### Prompt 3 — Execução Paralela Integral: Infraestrutura GPU, Código, Experimentos, Artigo IEEE e Relatório de IA
> **Prompt do Usuário:**  
> *"Pode continuar e executar todo o projeto seguindo o plano paralelo com subagentes! Configure o ambiente com CUDA na RTX 3050, implemente os módulos em src/ e os testes em tests/, rode a coleta de trilhas CoT e os benchmarks reais, gere todas as figuras de alta resolução, escreva o artigo IEEE completo (6 a 8 páginas com citações ABNT), o relatório de uso de IA de 2 páginas e os 4 slides da apresentação."*

* **Análise e Decisões Tomadas:**
  1. **Frente 1 (Engenharia de Infraestrutura GPU & Implementação do LC-SC):**
     * Configuração do `.venv` com PyTorch CUDA para a GPU NVIDIA GeForce RTX 3050 (6 GB VRAM) e download dos benchmarks GSM8K e MATH-500 em `data/`.
     * Implementação limpa e testada (`pytest`) dos módulos em `src/` (extrator de entropia de Shannon passo a passo, *Greedy CoT*, *Standard SC*, *Length-Filtered Vote* de Wu et al., *CaTS* de Huang et al. e o método proposto **LC-SC**).
  2. **Frente 2 (Redação Científica Sênior e Documentação Auditável):**
     * Redação densa e rigorosa das Seções I, II e III e estruturação das Seções IV, V e Referências ABNT no artigo principal `artigo_ieee/main.tex`.
     * Elaboração e compilação via `tectonic.exe` do Relatório de Uso de IA de exatas 2 páginas (`relatorio_ia/relatorio_ia.tex`) e deste Diário de Evolução (`docs/01_Relatorio_Evolucao_e_Prompts.md`).

---

## 4. Resumo da Arquitetura de Verificação e Reprodutibilidade

| Requisito do Edital (PPGI0034) | Especificação Exigida pelo Prof. Dr. Díbio | Implementação e Arquivo Correspondente no Projeto 2 | Status |
| :--- | :--- | :--- | :---: |
| **Artigo Principal IEEE** | Dupla coluna (`IEEEtran`), 6 a 8 páginas com figuras e referências | `Projeto_2/artigo_ieee/main.tex` -> `main.pdf` | Estruturado & Compilado |
| **Padrão de Citações** | Formato ABNT completo `(AUTOR, ano)` no texto e na lista final | Todas as citações em `(AUTOR, ano)` e seção final em ABNT | 100% Conforme |
| **Artigos do ICLR 2026** | De 1 a 4 artigos correlatos publicados no ICLR 2026 | 4 artigos (Wu, Huang, Hao, Feng, 2026) baixados na íntegra | 100% Conforme |
| **Tríade da Inovação** | Explicar *Qual*, *Por Quê* e *Como Observar* nos resultados | Seção I-C, Seção III e Seção IV do artigo principal | 100% Conforme |
| **Repositório GitHub** | Indicar GitHub pessoal `/ProjetoFinalRNAP_UnB2026` na Metodologia | Indicado no cabeçalho, na Seção III e nos relatórios | 100% Conforme |
| **Relatório de Uso de IA** | Especificar modelos usados e delimitar autoria em **2 páginas** | `Projeto_2/relatorio_ia/relatorio_ia.tex` -> `relatorio_ia.pdf` | 100% Conforme (2 págs) |
| **Apresentação Oral** | Defesa de 5 minutos com exatamente **4 slides** | `Projeto_2/apresentacao/` | Planejado na Fase 5 |
