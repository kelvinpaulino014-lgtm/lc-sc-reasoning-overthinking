# Fase 3 — Treinamento Neural Profundo e Benchmark Real Online em GPU

**Projeto:** Length-Calibrated Self-Consistency (LC-SC) para LLMs de Raciocínio  
**Referências internas:** Execução e métricas do `@trainer` (`#15`, `#17`, `#20`) e Auditoria Independente do `@reviewer` (`#16`, `#21`).  
**Regra de integridade:** Arquivos finais de entrega (`artigo_ieee/main.tex`, `apresentacao/slides_projeto2.tex`, `relatorio_ia/relatorio_ia.tex`) e os resultados da Fase 2 (`results/benchmark_results.json`, `results/benchmark_summary_table.csv`) permaneceram intocados conforme especificado.

---

## 1. Arquitetura da Política Neural de Parada (`NeuralStoppingPolicy`) e Treinamento Multi-Seed

Implementada em `src/models/stopping_policy.py` e treinada por `scripts/02b_train_stopping_policy.py` (inspirada em Hao et al., ICLR 2026):

- **Arquitetura (< 3.000 parâmetros):** `GRU(6, 16)` (1 camada) + `LayerNorm(16)` + `Linear(16, 16) + SiLU + Dropout(0.1) + Linear(16, 2)`, totalizando **1.490 parâmetros treináveis**.
- **Features passo a passo:** `H_t`, `delta_H_t`, `logprob_t`, `t / L*`, `has_answer_cue`, entre outras features sequenciais da trajetória de raciocínio.
- **Protocolo de treinamento (Zero Data Leakage):** Treinada exclusivamente sobre `val_calibration` (`n_val_questions=80`, `1.280` traços por dataset) em validação cruzada **5-Fold CV $\times$ 5 sementes aleatórias** (`[42, 101, 2026, 777, 999]`), com perda mascarada `BCE + Brier`, `AdamW`, `CosineAnnealingLR` e `EarlyStopping(patience=8)`.
- **Artefatos gerados:** Checkpoints em `results/phase3_deep_training/checkpoints/` (`gsm8k_best_policy.pt`, `math500_best_policy.pt`), `training_curves.csv`, `training_metrics.json`, `training_progress.log` e `fig_phase3_learning_curves.{png,pdf}`.

### Métricas de Validação (5-Fold CV $\times$ 5 Seeds em `val_calibration`)

| Dataset | $L^*$ Médio | `val_loss` (Média $\pm$ DP) | `stop_F1` (Média $\pm$ DP) | `Brier` (Média $\pm$ DP) | `ROC-AUC` (Média $\pm$ DP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | 134,0 | `0,1104 ± 0,0095` | `0,9979 ± 0,0015` | `0,0456 ± 0,0044` | `0,9739 ± 0,0073` |
| **MATH-500** | 211,4 | `0,0845 ± 0,0070` | `0,9976 ± 0,0018` | `0,0340 ± 0,0046` | `0,9821 ± 0,0027` |

---

## 2. Benchmark Real Online em GPU (`Qwen/Qwen2.5-1.5B-Instruct` na RTX 3050 6GB)

Implementado em `scripts/03b_run_online_gpu_benchmark.py` utilizando o modelo `Qwen/Qwen2.5-1.5B-Instruct` (FP16) com `EntropyNeuralStoppingCriteria` integrado diretamente ao `transformers.StoppingCriteria` em lote estratificado de `test` (`n=10` questões/dataset, `max_new_tokens=384`).

### Diagnóstico e Correção Durante a Execução (`#16` $\rightarrow$ `#17`)
- Na primeira execução online, a auditoria (`#16`) detectou `0,0%` de acurácia do LC-SC Fase 3 em 15 questões porque `extract_answer` atribuía `has_cue = 1.0` a números intermediários do raciocínio.
- O critério foi corrigido (`#17`) com `_EXPLICIT_ANSWER_CUE_RE` para exigir delimitadores explícitos de resposta final (`\boxed{...}`, `#### ...` ou `Therefore, the answer is ...`), validado pelo novo teste unitário `test_incomplete_cot_does_not_stop_prematurely` (`4/4` em `tests/test_stopping_policy.py`).

### Resultados Reais em GPU (RTX 3050 6GB, `n=10`/dataset)

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

### Achado Científico Honesto (*Domain Shift* Cross-Model)
- No GSM8K online, o **LC-SC Phase 3** superou Standard SC e CaTS em **+10,0 p.p.** de acurácia (`70,0%` vs `60,0%`), mantendo throughput de `100,6 tok/s` e Peak VRAM abaixo de `4,0 GB` (`3960,2 MB`).
- Contudo, no `Qwen2.5-1.5B-Instruct`, `\boxed{...}` é emitido próximo ao token `EOS` (diferente dos traços `<think>` do `DeepSeek-R1-Distill` usados em `val_calibration`), e houve *domain shift* nas distribuições de entropia e `logprob` recebidas pela GRU, impedindo que o `N-Stopping` (`top_c_mean >= 0.55`) disparasse em `N=4` e degradando a calibração (ECE `40,52%` / `46,27%`).

---

## 3. Tabela Completa de Progressão: ICLR 2026 Baselines $\rightarrow$ Fase 1 $\rightarrow$ Fase 2 $\rightarrow$ Fase 3

| Etapa / Regime | Modelo & Split | GSM8K Acc (%) | GSM8K Tokens | GSM8K ECE (%) | MATH-500 Acc (%) | MATH-500 Tokens | MATH-500 ECE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy CoT ($N=1$)** | DeepSeek-R1-Distill (`test`, $n=200$) | 64,0 | 178,4 | 17,86 | 57,0 | 261,4 | 22,94 |
| **Standard SC (Aula 11)** | DeepSeek-R1-Distill (`test`, $n=200$) | 75,0 | 3537,4 | 20,33 | 63,5 | 4726,2 | 19,83 |
| **Length-Filtered Vote (Wu et al., 2026)** | DeepSeek-R1-Distill (`test`, $n=200$) | 80,0 | 3537,4 | 9,92 | 70,5 | 4726,2 | 15,94 |
| **CaTS (Huang et al., 2026)** | DeepSeek-R1-Distill (`test`, $n=200$) | 80,0 | 2409,2 | 13,46 | 71,5 | 3285,3 | 12,01 |
| **LC-SC — Fase 1** | DeepSeek-R1-Distill (`test`, $n=200$) | 85,0 | 1098,3 | 6,60 | 85,5 | 1648,3 | 8,19 |
| **LC-SC — Fase 2** | DeepSeek-R1-Distill (`test`, $n=200$) | **97,0** | **676,5** | **2,50** | **94,5** | **1065,5** | **2,68** |
| **LC-SC — Fase 3 (Part A: Policy CV)** | GRU 1.490 params (`val_calibration`, 5x5) | — (F1 `0,9979`) | — | Brier `0,0456` | — (F1 `0,9976`) | — | Brier `0,0340` |
| **LC-SC — Fase 3 (Part B: Online GPU)** | `Qwen2.5-1.5B-Instruct` (`test`, $n=10$) | 70,0 | 2501,7 | 40,52 | 40,0 | 2720,8 | 46,27 |

**Status da Suíte Pytest (`#21`):** **35/36 aprovados** (`31/31` nos testes de código, integridade, invariantes científicos e política neural; única falha esperada em `test_numerical_consistency_between_json_and_documents` devido ao congelamento intencional de `main.tex` e `slides_projeto2.tex`).

---

## 4. Opções Concretas para Integração Final ou Refinamento (Fase 4)

1. **Coletar traços `val_calibration` do próprio `Qwen2.5-1.5B-Instruct` (ou usar `DeepSeek-R1-Distill-Qwen-1.5B` em GPU) para eliminar o *domain shift*:**
   - Re-treinar `NeuralStoppingPolicy` diretamente sobre traços do modelo-alvo online e recalibrar o limiar de `N-Stopping` (`top_c_mean`), expandindo o teste online de `n=10` para `n=50`–`200` questões.
2. **Integrar os resultados consolidados (Fases 1, 2 e 3) no Artigo IEEE, Slides e Relatório LaTeX:**
   - Atualizar `artigo_ieee/main.tex`, `apresentacao/slides_projeto2.tex` e `relatorio_ia/relatorio_ia.tex` com os números finais da Fase 2 (benchmark completo `n=200`) e da Fase 3 (treinamento neural + GPU online + análise de *domain shift*), restaurando `36/36` no `pytest` e recompilando os PDFs finais.
3. **Ablação de Temperature Scaling / Normalização Online de Entropia na GRU:**
   - Aplicar normalização *z-score* adaptativa por janela ou *temperature scaling* pós-treinamento em `corr_p` para corrigir o ECE online (`40,52%` $\rightarrow$ `<10%`) sem exigir nova coleta completa de traços.
