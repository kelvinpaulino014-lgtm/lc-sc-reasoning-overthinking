# Guia Passo a Passo e Plano de Execução Paralela — Projeto Individual 2 (PPGI0034 UnB)

**Autor:** Kelvin Rafael Paulino Buss (`kelvin.buss@aluno.unb.br`)  
**Disciplina:** PPGI0034 — Redes Neurais e Aprendizado Profundo (2026/2) — Prof. Dr. Díbio Leandro Borges  
**Título do Projeto:** *Controle Dinâmico de "Overthinking" em Modelos de Linguagem via Parada Precoce Calibrada por Entropia em Cadeias de Pensamento (Chain-of-Thought)*  
**Método Proposto:** **LC-SC** (*Length-Calibrated Self-Consistency*)

---

## 1. Visão Geral: Como Entender o Projeto em 3 Minutos (Para Você Revisar e Auditar Tudo)

Imagine que você pede a um modelo de linguagem pequeno (como o `DeepSeek-R1-Distill-Qwen-1.5B`) para resolver um problema de matemática do benchmark **GSM8K**.
Na **Aula 11 (Slide 17)**, o Prof. Díbio apresentou o **Self-Consistency (SC)**: em vez de gerar apenas 1 resposta (*Greedy CoT*), o modelo sorteia $N$ caminhos de raciocínio diferentes (por exemplo, $N=16$) e faz uma **votação majoritária** na resposta final.

### O Problema que os 4 Artigos do ICLR 2026 Descobriram:
1. **O Fenômeno do "Overthinking" (Curva em U Invertido) — (WU et al., 2026):**
   - Wu et al. provaram matemática e empiricamente no ICLR 2026 que pensar um pouco ajuda, mas **pensar demais destrói a acurácia**. A relação entre o número de tokens gerados ($L$) e a chance de acertar segue uma **curva em U invertido** (sobe até um ponto ótimo $L^*$ e depois despenca).
   - *Limitação de Wu et al. (2026):* Eles criaram o *Length-Filtered Vote*, mas ele é **post-hoc** (pós-geração): o computador é obrigado a gastar tempo e GPU gerando todas as 16 cadeias até o final (mesmo as que entraram em loop de 2.048 tokens) para só depois jogá-las fora na hora de votar!
2. **O Custo de Amostrar $N$ Caminhos e a Descalibração — (HUANG et al., 2026 - CaTS):**
   - Huang et al. criaram o **CaTS** (*Calibrated Test-Time Scaling*) no ICLR 2026 para parar de gerar novas amostras $N$ assim que as primeiras amostras já concordam com alta confiança calibrada.
   - *Limitação de Huang et al. (2026):* O CaTS olha apenas para **quantas amostras ($N$)** são geradas, mas **não controla o tamanho ($L$) de cada amostra**. Pior ainda: na Tabela 1 do próprio artigo do CaTS (página 5), quando eles testam o `DeepSeek-R1-Distill-1.5B` no GSM8K, a métrica de calibração (AUC) **cai de 64,31% para 55,57%**! Por quê? Porque quando um modelo pequeno entra em *overthinking*, ele repete padrões e fica artificialmente "superconfiante" na resposta errada.
3. **A "Sobrecorreção" (*Over-Correction*) — (HAO et al., 2026 - RL of Thoughts):**
   - Hao et al. (ICLR 2026) mostraram no Apêndice B.2 (Figura 7, p. 19) que quando modelos menores tentam "refinar" demais um cálculo que já estava certo, eles sofrem de *over-correction*: trocam a resposta certa por uma alucinação.
4. **A Diluição de Atenção em Contextos Longos — (FENG et al., 2026 - In-Place TTT, Oral):**
   - Feng et al. (ICLR 2026 Oral) demonstraram que acumular centenas de tokens no contexto degrada a representação interna das camadas de atenção e MLP do Transformer.

---

### A Sua Inovação: LC-SC (*Length-Calibrated Self-Consistency*)

Enquanto **Wu et al. (2026)** olham apenas para o comprimento $L$ (e só depois de gastar todos os tokens) e **Huang et al. (2026)** olham apenas para o número de amostras $N$, o seu método **LC-SC** controla **simultaneamente e em tempo real (online) a matriz $N \times L$**:

```
                  Comprimento da Cadeia CoT (Tokens L) --->
             L_min             Zona Ótima L*           Zona de Overthinking (L >> L*)
Amostra 1:  [ Raciocínio limpo e confiante ] -> Resposta: 42 (Peso alto = 0.95)
Amostra 2:  [ Raciocínio limpo e confiante ] -> Resposta: 42 (Peso alto = 0.92)
Amostra 3:  [ Raciocínio... ][ PICO DE ENTROPIA H_t > tau_H! ] -> CORTADO ONLINE (L-Stopping!)
Amostra 4:  ---> NÃO PRECISA GERAR! Consenso atingido entre Amostras 1 e 2 (N-Stopping!)
```

Os **3 Mecanismos do LC-SC** que você pode auditar diretamente no código (`src/decoders/lc_sc_decoder.py`):
1. **$L$-Stopping Online (Poda Intratrilha por Entropia):** Durante a geração token a token, calculamos a entropia de Shannon $H_t = -\sum_{v} p_t(v)\log p_t(v)$. Se a cadeia já passou do tamanho mínimo $L_{\min}$ e apresenta um pico sustentado de entropia ($\bar{H}_k > \tau_H$, sinalizando que o modelo entrou em dúvida circular ou *over-correction*), o decodificador injeta imediatamente o fechamento do bloco de pensamento (`</think>\nPortanto, a resposta final é \boxed{`) e extrai a resposta antes que a cadeia degenere.
2. **Ponderação Calibrada por Comprimento ($w_i$):** Na hora de votar, cada trilha $i$ recebe um peso que combina a confiança calibrada $\hat{c}_i$ (do CaTS) com uma penalidade suave caso a trilha tenha ultrapassado o comprimento ótimo $L^*$:
   $$w_i = \hat{c}_i \cdot \exp\left(-\lambda \frac{\max(0, L_i - L^*)}{\sigma_L}\right)$$
3. **$N$-Stopping Calibrado (Parada Intertrilhas):** A cada trilha concluída, se a resposta líder na votação acumular proporção de peso $\ge \gamma$ (ex.: $0.75$), interrompemos o loop de amostras $N$ imediatamente.

---

## 2. Expectativa de Tempo por Fase e Estratégia de Execução Paralela (Multi-Agentes)

Com base na nossa experiência no **Projeto 1** (onde evitaremos o erro da instalação do PyTorch apenas em CPU e usaremos diretamente sua **GPU NVIDIA GeForce RTX 3050 de 6 GB VRAM**), aqui está o cronograma realista de execução e como trabalharemos **em paralelo com múltiplos subagentes**:

### Tabela de Estimativa de Tempo (Sequencial vs. Paralelizado com Subagentes)

| Fase do Workflow | O que é feito nesta Fase | Tempo Sequencial Estimado | Execução em Paralelo (Subagentes) |
| :--- | :--- | :---: | :--- |
| **Fase 0: Etapa 1 (Prazo 04/11) & Acervo ICLR 2026** | Proposta de 1 página (`.tex`/`.pdf`), download dos 4 PDFs do ICLR 2026 e dos 4 repositórios oficiais do GitHub. | ~45 min | **CONCLUÍDA AGORA (100%)** |
| **Fase 1: Ambiente GPU + Dados + Teoria LaTeX (Seções I a III)** | Configurar `.venv` com PyTorch CUDA na RTX 3050, baixar GSM8K/MATH-500 e redigir Introdução, Trabalhos Correlatos e Metodologia do artigo IEEE. | ~2h00 | **Frente A (Agente de Engenharia)** instala CUDA + baixa dados enquanto **Frente B (Agente Acadêmico)** redige Seções I, II e III do `.tex` (**~45 min em paralelo**). |
| **Fase 2: Implementação Modular + Testes Unitários (`pytest`)** | Codificar extrator de entropia, baselines (Greedy, SC, Wu LFV, Huang CaTS) e o decodificador proposto **LC-SC** + suíte `pytest`. | ~1h30 | **Frente A** codifica `src/` e roda `pytest` enquanto **Frente B** prepara os scripts de plotagem vetorial e o esqueleto do Relatório de IA (**~50 min em paralelo**). |
| **Fase 3: Geração de Trilhas (Inferência GPU) e Validação** | Gerar banco de trilhas CoT com entropia passo a passo no GSM8K/MATH-500 usando a RTX 3050 (FP16/4-bit) e rodar o benchmark dos 5 métodos + ablação. | ~2h00 a 3h00 (tempo de GPU) | Enquanto a **GPU roda a inferência em background**, um subagente monitora os logs parciais e já estrutura as tabelas LaTeX e os slides da apresentação. |
| **Fase 4: Gráficos Vetoriais (300 DPI) e Tabelas Finais** | Gerar as 4 figuras de padrão ICLR (Curva U invertido, Assinatura de Entropia, Fronteira de Pareto, Diagrama de Calibração ECE) em `.pdf`/`.png`. | ~40 min | Execução automática via `scripts/04_generate_figures.py` assim que o `.parquet` da Fase 3 concluir (**~15 min**). |
| **Fase 5: Fechamento do Artigo IEEE (6–8 pág.), Relatório de IA (2 pág.) e 4 Slides** | Inserir números reais nas Seções IV e V, compilar PDFs finais via `tectonic.exe`, auditar número de páginas e empacotar para GitHub/Moodle. | ~1h15 | **Frente A** compila e audita o Artigo IEEE (6–8 págs), **Frente B** compila o Relatório de IA (2 págs) e **Frente C** gera os 4 slides HTML/PDF (**~35 min em paralelo**). |
| **TOTAL ACUMULADO** | Do início até o pacote final completo padrão A+ | ~8h a 9h sequenciais | **~4h30 a 5h30 totais com paralelismo!** |

---

### Diagrama do Fluxo Paralelo de Subagentes (Como Trabalharemos ao Mesmo Tempo)

```
========================================================================================
LINHA DO TEMPO OTIMIZADA COM EXECUÇÃO PARALELA (MULTI-AGENTES + BACKGROUND TASKS)
========================================================================================

[AGORA - CONCLUÍDO]
  ├── Entrega Etapa 1 (04/11/2026) pronta em 1 página PDF/TEX/MD/TXT
  ├── 4 Artigos PDFs do ICLR 2026 baixados e analisados na íntegra
  └── 4 Repositórios Oficiais do GitHub baixados em referencias/repos_iclr2026/

[BLOCO PARALELO 1: ~45-60 min]
  ├──► [Agente 1 / Background Task: Infraestrutura GPU & Dados]
  │      └── Cria .venv com PyTorch CUDA (RTX 3050), baixa GSM8K/MATH-500 e testa modelo 1.5B
  ├──► [Agente 2: Engenharia do Decodificador LC-SC & Testes]
  │      └── Implementa src/ (Greedy, SC, Wu LFV, CaTS, LC-SC) + valida com pytest
  └──► [Agente 3: Redação Científica IEEE (Seções I, II e III)]
         └── Escreve Introdução, Trabalhos Correlatos (4 artigos ICLR 2026 em ABNT)
             e Formulação Matemática do LC-SC em artigo_ieee/main.tex

[BLOCO PARALELO 2: ~1h30-2h30 (GPU rodando)]
  ├──► [Background Task GPU: Inferência GSM8K/MATH-500 + Coleta de Entropia]
  │      └── Gera as cadeias CoT salvando log-probs, entropia de Shannon e comprimentos L_i
  └──► [Agente 2 + Agente 3 (Enquanto a GPU trabalha)]
         ├── Prepara scripts/04_generate_figures.py para gerar as 4 figuras 300 DPI
         ├── Redige a Seção 1 e 2 do Relatório de Uso de IA (relatorio_ia/relatorio_ia.tex)
         └── Cria a estrutura visual dos 4 Slides de Apresentação (apresentacao/)

[BLOCO PARALELO 3: ~45 min (Fechamento e Auditoria)]
  ├──► [Script de Avaliação Offline (Instantâneo sobre o cache de trilhas)]
  │      └── Roda os 5 métodos + Ablação + Calcula Intervalos de Confiança (Bootstrap) e ECE
  ├──► [Agente 1: Gráficos e Tabelas Reais]
  │      └── Gera Figuras 1, 2, 3 e 4 e preenche a Seção IV (Resultados) e V (Conclusão)
  └──► [Auditoria Final de Qualidade]
         └── Compila PDFs via tectonic.exe e verifica:
             - Artigo IEEE entre 6 e 8 páginas (zero placeholders, ABNT (AUTOR, ano))
             - Relatório de IA com exatas 2 páginas
             - Apresentação com exatos 4 slides (5 minutos)
```

---

## 3. Como Você Pode Acompanhar, Auditar e Corrigir Qualquer Parte

Para que você tenha **controle total** sobre o projeto e possa revisar ou pedir ajustes a qualquer momento, organizamos a pasta `Projeto_2` de forma modular e transparente:

### Mapa Completo de Pastas e Arquivos (`Projeto_2/`)

1. **[`entrega_etapa1_04nov/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/entrega_etapa1_04nov)** *(Pronto para enviar até 04/11/2026 às 23h59!)*
   - [`proposta_projeto2_04nov_Kelvin_Buss.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/entrega_etapa1_04nov/proposta_projeto2_04nov_Kelvin_Buss.pdf): Documento PDF de **1 página exata**, diagramado com cabeçalho da UnB, Título Final, Pergunta de Pesquisa, Resumo com a Tríade da Inovação (*O Quê, Por Quê, Como Observar*), Objetivos, link do GitHub e as 4 referências do ICLR 2026 em ABNT.
   - [`proposta_projeto2_04nov_Kelvin_Buss.tex`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/entrega_etapa1_04nov/proposta_projeto2_04nov_Kelvin_Buss.tex): Código-fonte LaTeX (caso queira alterar uma frase ou seu número de matrícula, basta editar e rodar `..\bin\tectonic.exe proposta_projeto2_04nov_Kelvin_Buss.tex`).
   - [`proposta_projeto2_04nov_Kelvin_Buss.md`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/entrega_etapa1_04nov/proposta_projeto2_04nov_Kelvin_Buss.md) e [`proposta_projeto2_04nov_Kelvin_Buss.txt`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/entrega_etapa1_04nov/proposta_projeto2_04nov_Kelvin_Buss.txt): Versões em Markdown e Texto Puro caso o envio no Moodle peça colar texto diretamente na caixa de envio.

2. **[`referencias/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias)** *(Todo o material-base do ICLR 2026 salvo localmente)*
   - **`artigos_iclr2026_pdfs/`**: Os 4 artigos completos em PDF baixados:
     - [`01_Wu_et_al_2026_When_More_is_Less_CoT_Length.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/01_Wu_et_al_2026_When_More_is_Less_CoT_Length.pdf)
     - [`02_Huang_et_al_2026_CaTS_Calibrated_Test_Time_Scaling.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/02_Huang_et_al_2026_CaTS_Calibrated_Test_Time_Scaling.pdf)
     - [`03_Hao_et_al_2026_RL_of_Thoughts_Navigating_LLM_Reasoning.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/03_Hao_et_al_2026_RL_of_Thoughts_Navigating_LLM_Reasoning.pdf)
     - [`04_Feng_et_al_2026_In_Place_Test_Time_Training.pdf`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/referencias/artigos_iclr2026_pdfs/04_Feng_et_al_2026_In_Place_Test_Time_Training.pdf)
   - **`repos_iclr2026/`**: Os 4 códigos-fonte oficiais dos autores no GitHub já baixados e extraídos:
     - `01_PKU-ML_CoT-Length/` (Código oficial de Wu et al., 2026 — curva em U invertido e *Length-Filtered Vote*)
     - `02_Chengsong-Huang_Self-Calibration_CaTS/` (Código oficial de Huang et al., 2026 — calibração CaTS e scripts de GSM8K)
     - `03_tsinghua-fib-lab_RL-LLM-Reasoning/` (Código oficial de Hao et al., 2026 — navegador RL of Thoughts)
     - `04_ByteDance-Seed_In-Place-TTT/` (Código oficial de Feng et al., 2026 — In-Place Test-Time Training)

3. **[`src/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/src), [`scripts/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/scripts) e [`tests/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests)** *(Onde ficará todo o código Python do LC-SC)*
   - **Dica de Ouro de Engenharia (Cache de Trilhas):** Em vez de rodar o LLM na GPU 5 vezes seguidas (uma para cada método), nosso script `scripts/02_collect_cot_traces.py` gerará as $N=16$ trilhas completas salvando para cada passo: o texto, os tokens acumulados, a entropia de Shannon $H_t$ e a confiança $\hat{c}_t$ em um arquivo `.parquet` em `results/`.
   - Assim, o script `scripts/03_run_benchmarks.py` consegue simular **exatamente** o que teria acontecido se o **LC-SC** tivesse truncado a cadeia no passo $t_{\text{stop}}$ (onde $\bar{H}_k > \tau_H$) e parado na amostra $N_{\text{adap}}$! Isso permite que você ajuste hiperparâmetros ($\tau_H, \lambda, \gamma$) e recalcule todas as tabelas e gráficos em **menos de 5 segundos**, sem precisar esperar horas de GPU novamente!

4. **[`artigo_ieee/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/artigo_ieee), [`relatorio_ia/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/relatorio_ia) e [`apresentacao/`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/apresentacao)** *(Entregáveis Finais)*
   - Já equipados com `IEEEtran.cls` e com o compilador `bin/tectonic.exe` local para compilação imediata sem depender de instalação pesada de TeX Live.

---

## 4. Checklist de Prevenção de Erros (Lições do Projeto 1 Aplicadas ao Projeto 2)

- [x] **Erro 1 evitado (CPU vs. GPU):** No Projeto 1, o PyTorch foi instalado em versão `+cpu`, levando 2h35m para rodar. Na Fase 1 do Projeto 2, instalaremos explicitamente o PyTorch com suporte a **CUDA** para usar sua **RTX 3050 (6 GB VRAM)**.
- [x] **Erro 2 evitado (Citações numéricas `[1]` vs. ABNT `(AUTOR, ano)`):** O edital exige formato IEEE de duas colunas, mas **citações no padrão ABNT `(AUTOR, ano)`**. Já configuramos isso desde o documento da Etapa 1 e faremos o mesmo no `artigo_ieee/main.tex`.
- [x] **Erro 3 evitado (Transparência e Registro Contínuo de IA):** O edital exige um relatório separado de 2 páginas detalhando quais ferramentas de IA foram usadas e o que é autoria intelectual sua versus auxílio computacional/redacional. Manteremos o log atualizado em `docs/diario_prompts_e_auditoria_ia.md` a cada fase.
