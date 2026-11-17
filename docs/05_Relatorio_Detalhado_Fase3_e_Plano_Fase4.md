# Relatório Detalhado da Fase 3 e Plano de Execução da Fase 4

**Disciplina:** PPGI0034 — Redes Neurais e Aprendizagem de Máquina (UnB)  
**Projeto:** Length-Calibrated Self-Consistency (LC-SC) para Mitigação de *Overthinking* em LLMs de Raciocínio  
**Referências internas verificadas:** `#5`, `#6` (Fase 2) e `#15`, `#16`, `#17`, `#20`, `#21` (Fase 3).  
**Regra de integridade:** `artigo_ieee/main.tex`, `apresentacao/slides_projeto2.tex`, `relatorio_ia/relatorio_ia.tex` e os resultados consolidados da Fase 2 (`results/benchmark_results.json`, `results/benchmark_summary_table.csv`) permanecem intocados até a execução da Etapa 4.3.

---

## Parte I — Progresso Detalhado da Fase 3

### 1.1 Motivação da Fase 3
Após os ganhos expressivos da Fase 2 em regime *offline* sobre traços de `DeepSeek-R1-Distill-Qwen-1.5B` (`97,0%` no GSM8K e `94,5%` no MATH-500, com redução de `80,9%` e `77,5%` de tokens e ECE $\le 2{,}68\%$), a Fase 3 teve dois objetivos complementares:
1. **Substituir limiares estáticos por uma política neural sequencial compacta** (inspirada em Hao et al., ICLR 2026), capaz de aprender diretamente a dinâmica passo a passo de entropia e confiança sem ultrapassar o orçamento de `< 3.000` parâmetros.
2. **Validar a inferência real *online* em GPU** (`RTX 3050 6GB`) integrando a política treinada a um `transformers.StoppingCriteria` real sobre `Qwen/Qwen2.5-1.5B-Instruct` (FP16), medindo latência *wall-clock*, *throughput* e pico de VRAM.

### 1.2 Arquitetura da `NeuralStoppingPolicy` e Treinamento Multi-Seed (`Part A`)
Implementada em `src/models/stopping_policy.py` e treinada via `scripts/02b_train_stopping_policy.py`:
- **Topologia compacta:** `GRU(6, 16)` (1 camada) + `LayerNorm(16)` + `Linear(16, 16) + SiLU + Dropout(0.1) + Linear(16, 2)`, totalizando **1.490 parâmetros treináveis** (`< 3.000`).
- **Entrada sequencial (6 features/passo):** entropia $H_t$, variação $\Delta H_t$, `logprob_t`, progresso normalizado $t / L^*$, indicador de pista de resposta (`has_answer_cue`) e sinal auxiliar de trajetória.
- **Saídas:** probabilidade de parada (`stop_p`) e probabilidade de corretude (`corr_p`).
- **Protocolo sem vazamento de dados (*Zero Data Leakage*):** ajustada exclusivamente em `val_calibration` (`80` questões / `1.280` traços por dataset) usando **5-Fold CV $\times$ 5 sementes aleatórias** (`[42, 101, 2026, 777, 999]`, totalizando 25 execuções por dataset), com perda mascarada `BCE + Brier`, otimizador `AdamW`, `CosineAnnealingLR` e `EarlyStopping(patience=8)`.

| Dataset | $L^*$ Médio (`val_calibration`) | `val_loss` (Média $\pm$ DP) | `stop_F1` (Média $\pm$ DP) | `Brier` (Média $\pm$ DP) | `ROC-AUC` (Média $\pm$ DP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | 134,0 | `0,1104 ± 0,0095` | `0,9979 ± 0,0015` | `0,0456 ± 0,0044` | `0,9739 ± 0,0073` |
| **MATH-500** | 211,4 | `0,0845 ± 0,0070` | `0,9976 ± 0,0018` | `0,0340 ± 0,0046` | `0,9821 ± 0,0027` |

### 1.3 Auditoria em Tempo Real e Correção do Critério de Parada (`#16` $\rightarrow$ `#17`)
- Durante o monitoramento periódico de 20 minutos (`#16`), o `@reviewer` identificou que o LC-SC Fase 3 apresentava **0,0% de acurácia** em 15 questões iniciais do GSM8K no benchmark *online*.
- **Causa raiz:** em `src/models/stopping_policy.py`, `extract_answer(decoded_so_far)` retornava números intermediários de raciocínio, ativando `has_cue = 1.0` assim que `gen_len >= l_min = 67` e interrompendo a geração antes da resposta final.
- **Correção aplicada (`#17`):** introdução de `_EXPLICIT_ANSWER_CUE_RE` em `EntropyNeuralStoppingCriteria`, exigindo delimitadores explícitos de resposta final (`\boxed{...}`, `#### ...` ou `Therefore, the answer is ...`), além da adição do teste unitário `test_incomplete_cot_does_not_stop_prematurely` (`4/4` aprovados em `tests/test_stopping_policy.py`).

### 1.4 Benchmark Real Online em GPU (`Part B` — RTX 3050 6GB)
Executado via `scripts/03b_run_online_gpu_benchmark.py` com `Qwen/Qwen2.5-1.5B-Instruct` (FP16) em lote estratificado de `test` (`n=10` questões/dataset, `max_new_tokens=384`):

| Dataset | Método | Acc (%) | IC 95% (%) | Tokens Médios | Economia vs SC (%) | $N$ Médio | Latência (s/q) | Speedup | Throughput (tok/s) | Peak VRAM (MB) | ECE (%) | Brier | ROC-AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | Greedy CoT ($N=1$) | 50,0 | [20,0; 80,0] | 317,5 | 86,8 | 1,0 | 9,98 | 2,41x | 31,8 | 3960,2 | 49,87 | 0,3912 | 56,00 |
| **GSM8K** | Standard SC ($N=8$) | 60,0 | [30,0; 90,0] | 2406,5 | 0,0 | 8,0 | 24,04 | 1,00x | 100,1 | 3960,2 | 34,45 | 0,1845 | 95,83 |
| **GSM8K** | CaTS (Huang et al., 2026) | 60,0 | [30,0; 90,0] | 1971,9 | 18,1 | 6,4 | 19,93 | 1,21x | 98,9 | 3960,2 | 24,14 | 0,0910 | 100,00 |
| **GSM8K** | **LC-SC Phase 3 (Neural GPU)** | **70,0** | **[40,0; 100,0]** | 2501,7 | -4,0 | 8,0 | 24,87 | 0,97x | 100,6 | 3960,2 | 40,52 | 0,1801 | 90,48 |
| **MATH-500** | Greedy CoT ($N=1$) | 60,0 | [30,0; 90,0] | 342,8 | 87,7 | 1,0 | 10,87 | 2,35x | 31,5 | 3930,5 | 41,38 | 0,3408 | 54,17 |
| **MATH-500** | Standard SC ($N=8$) | 60,0 | [30,0; 90,0] | 2787,7 | 0,0 | 8,0 | 25,56 | 1,00x | 109,1 | 3930,5 | 42,61 | 0,2447 | 54,17 |
| **MATH-500** | CaTS (Huang et al., 2026) | 40,0 | [10,0; 70,0] | 2300,1 | 17,5 | 6,4 | 20,93 | 1,22x | 109,9 | 3930,5 | 42,93 | 0,2418 | 70,83 |
| **MATH-500** | **LC-SC Phase 3 (Neural GPU)** | **40,0** | **[10,0; 70,0]** | 2720,8 | 2,4 | 7,6 | 25,14 | 1,02x | 108,2 | 3930,5 | 46,27 | 0,2301 | 66,67 |

### 1.5 Progressão Completa: Baselines ICLR 2026 $\rightarrow$ Fase 1 $\rightarrow$ Fase 2 $\rightarrow$ Fase 3

| Etapa / Método | Regime & Modelo | GSM8K Acc (%) | GSM8K Tokens | GSM8K ECE (%) | MATH-500 Acc (%) | MATH-500 Tokens | MATH-500 ECE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Greedy CoT ($N=1$) | Offline (`DeepSeek-R1-Distill`, $n=200$) | 64,0 | 178,4 | 17,86 | 57,0 | 261,4 | 22,94 |
| Standard SC (Aula 11) | Offline (`DeepSeek-R1-Distill`, $n=200$) | 75,0 | 3537,4 | 20,33 | 63,5 | 4726,2 | 19,83 |
| Length-Filtered Vote (Wu et al., 2026) | Offline (`DeepSeek-R1-Distill`, $n=200$) | 80,0 | 3537,4 | 9,92 | 70,5 | 4726,2 | 15,94 |
| CaTS (Huang et al., 2026) | Offline (`DeepSeek-R1-Distill`, $n=200$) | 80,0 | 2409,2 | 13,46 | 71,5 | 3285,3 | 12,01 |
| **LC-SC — Fase 1** | Offline (`DeepSeek-R1-Distill`, $n=200$) | 85,0 | 1098,3 | 6,60 | 85,5 | 1648,3 | 8,19 |
| **LC-SC — Fase 2** | Offline (`DeepSeek-R1-Distill`, $n=200$) | **97,0** | **676,5** | **2,50** | **94,5** | **1065,5** | **2,68** |
| **LC-SC — Fase 3 (Part A)** | GRU 1.490 params (`val_calibration`, 5x5) | — (F1 `0,9979`) | — | Brier `0,0456` | — (F1 `0,9976`) | — | Brier `0,0340` |
| **LC-SC — Fase 3 (Part B)** | Online GPU (`Qwen2.5-1.5B-Instruct`, $n=10$) | 70,0 | 2501,7 | 40,52 | 40,0 | 2720,8 | 46,27 |

### 1.6 Diagnóstico Honesto de *Domain Shift* e Status de Testes
- **Diagnóstico de *Domain Shift* (`#20`, `#21`):** A política neural foi treinada em traços de `DeepSeek-R1-Distill-Qwen-1.5B` (que gera blocos `<think>` longos com hesitação após respostas intermediárias), mas avaliada online em `Qwen/Qwen2.5-1.5B-Instruct` (que emite `\boxed{...}` apenas ao final, próximo ao `EOS`). Esse descompasso deslocou as distribuições de entropia e `logprob`, impedindo o disparo precoce de `N-Stopping` (`top_c_mean >= 0.55` em `N=4`) e elevando o ECE online (`40,52%` no GSM8K e `46,27%` no MATH-500).
- **Status do `pytest` (`#21`):** **35/36 testes aprovados** (`31/31` em `test_decoders_and_metrics.py`, `test_data_and_traces_integrity.py`, `test_benchmark_and_scientific_invariants.py` e `test_stopping_policy.py`; a única falha esperada ocorre em `test_numerical_consistency_between_json_and_documents` devido ao congelamento intencional de `main.tex` e `slides_projeto2.tex`).

---

## Parte II — Etapas Detalhadas da Fase 4 (Roadmap)

### 4.1 Eliminação do *Domain Shift* Online
1. **Coleta de `val_calibration` no modelo-alvo:** Gerar traços de calibração diretamente com `Qwen/Qwen2.5-1.5B-Instruct` (ou executar a inferência online diretamente sobre `DeepSeek-R1-Distill-Qwen-1.5B` em FP16/INT8 na RTX 3050 6GB).
2. **Normalização Adaptativa e *Temperature Scaling*:** Aplicar normalização *z-score* causal por janela para `H_t` e `logprob_t` na entrada da GRU e ajustar um fator escalar de *temperature scaling* sobre `corr_p` em `val_calibration` do modelo-alvo, reduzindo o ECE online de `40–46%` para `< 10%` e destravando o `N-Stopping` precoce.

### 4.2 Expansão da Avaliação Online em GPU (`n = 50–200`) e Ablações
1. **Aumento do poder estatístico:** Expandir `03b_run_online_gpu_benchmark.py` de `n=10` para `n=50–200` questões estratificadas no split `test`, estreitando os intervalos de confiança (`IC 95%`).
2. **Ablações online:** Comparar `LC-SC Online Completo` vs `Sem L-Stopping` vs `Sem N-Stopping` vs `Sem Temperature Scaling` em latência real (s/questão), economia de tokens e ECE.

### 4.3 Integração Final nos Documentos LaTeX e Recompilação dos PDFs (`36/36` no Pytest)
1. Atualizar `artigo_ieee/main.tex`, `apresentacao/slides_projeto2.tex` e `relatorio_ia/relatorio_ia.tex` com:
   - Os resultados consolidados da Fase 2 (`97,0%` / `94,5%`, ablações e comparação ICLR 2026 em `n=200`).
   - Os resultados da Fase 3 (`NeuralStoppingPolicy` de 1.490 parâmetros, benchmark real em GPU na RTX 3050 6GB e análise transparente de *domain shift*) e da Fase 4.
2. Recompilar os PDFs finais (`main.pdf`, `slides_projeto2.pdf`, `relatorio_ia.pdf`) e verificar **36/36 testes aprovados** em `.venv/Scripts/pytest.exe tests/ -v`.

### 4.4 Empacotamento Final PPGI0034 (UnB) e Release GitHub / Google Drive
1. Consolidar os PDFs oficiais em `01_Entregas_PDFs_Oficiais` no Google Drive e verificar integridade cruzada com o repositório `kelvinpaulino014-lgtm/lc-sc-reasoning-overthinking`.
2. Executar auditoria final de reprodutibilidade (tag de release no GitHub, verificação de *links* e checksums dos artefatos).
