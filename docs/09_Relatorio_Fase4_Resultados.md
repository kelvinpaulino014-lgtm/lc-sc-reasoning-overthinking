# Fase 4 — Treinamento e Avaliação em Trilhas Reais (`Qwen/Qwen2.5-1.5B-Instruct`)

**Repositório:** `Projeto_2_Fase4` (branch `codex/fase4-trilhas-reais`)  
**Referências internas:** Execução do `@trainer` (`#46`, `#51`, `#52`, `#60`) e Auditoria Independente do `@reviewer` (`#61`).  
**Regra de integridade:** `artigo_ieee/main.tex` e `apresentacao/slides_projeto2.tex` permaneceram intocados conforme especificado.

---

## 1. Protocolo Experimental e Coleta de Trilhas Reais (`#51`, `#61`)

Seguindo `docs/08_Protocolo_Fase4.md`, a Fase 4 eliminou o *domain shift* da Fase 3 coletando trilhas reais diretamente do modelo-alvo `Qwen/Qwen2.5-1.5B-Instruct` (FP16) na GPU **RTX 3050 6GB** via `scripts/05_collect_real_traces.py` (com auditoria periódica limpa em `results/phase4/audits.jsonl`, `errors: []`):

- **Dataset coletado (`results/phase4/traces.jsonl`):** **416/416 questões** divididas estritamente em:
  - `train`: **160** questões (`80` GSM8K + `80` MATH-500)
  - `calibration`: **64** questões (`32` GSM8K + `32` MATH-500)
  - `holdout`: **192** questões (`96` GSM8K + `96` MATH-500)
- **Causalidade estrita:** `src/models/real_trace_policy.py` constrói janelas causais de **24 tokens** sem vazamento de futuro ou de `gold_answer`.
- **Treinamento (`scripts/06_train_real_policy.py`):** ajuste de `StandardScaler + LogisticRegression` e `NeuralStoppingPolicy` (GRU, **1.490 parâmetros**, 5 sementes) estritamente sobre `train`; seleção e congelamento dos limiares com verificação SHA256 em `calibration` (`deployment_seed=101`, `best_cal_loss=0.0137`, artefatos em `results/phase4/policy/`).

---

## 2. Replay Pareado em `holdout` Completo (`n = 96`/dataset, `192` questões)

Valores exatos de `results/phase4/replay_summary.csv` (IC 95% via bootstrap pareado):

| Dataset | Estratégia | Acc (%) | IC 95% (%) | Tokens Médios | Economia vs SC-4 (%) | Traços Médios | ECE (%) | ROC-AUC (%) | $\Delta$ Acc vs SC-4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K ($n=96$)** | `sc4` | 73,96 | [64,58; 82,29] | 1172,4 | 0,0 | 4,00 | 29,25 | 81,44 | 0,00 p.p. |
| **GSM8K ($n=96$)** | `consensus2` | 73,96 | [64,58; 82,29] | 927,5 | 20,9 | 3,06 | 27,26 | 79,04 | 0,00 p.p. |
| **GSM8K ($n=96$)** | `first_box` | 73,96 | [64,58; 82,29] | 926,0 | 21,0 | 3,06 | 27,26 | 79,04 | 0,00 p.p. |
| **GSM8K ($n=96$)** | `fixed_cap` | 15,62 | [9,38; 22,94] | 735,9 | 37,2 | 3,88 | 9,96 | 97,98 | $-$58,33 p.p. |
| **GSM8K ($n=96$)** | `logistic` | 73,96 | [64,58; 82,29] | 926,0 | 21,0 | 3,06 | 27,26 | 79,04 | 0,00 p.p. |
| **GSM8K ($n=96$)** | **`gru`** | **73,96** | **[64,58; 82,29]** | **926,1** | **21,0** | **3,06** | **27,26** | **79,04** | **0,00 p.p.** |
| **MATH-500 ($n=96$)** | `sc4` | 41,67 | [32,29; 51,04] | 1416,1 | 0,0 | 4,00 | 17,29 | 98,28 | 0,00 p.p. |
| **MATH-500 ($n=96$)** | `consensus2` | 41,67 | [32,29; 51,04] | 1290,9 | 8,8 | 3,58 | 14,53 | 98,30 | 0,00 p.p. |
| **MATH-500 ($n=96$)** | `first_box` | 41,67 | [32,29; 51,04] | 1290,5 | 8,9 | 3,58 | 14,53 | 98,30 | 0,00 p.p. |
| **MATH-500 ($n=96$)** | `fixed_cap` | 2,08 | [0,00; 5,21] | 763,3 | 46,1 | 3,98 | 1,77 | 100,00 | $-$39,58 p.p. |
| **MATH-500 ($n=96$)** | `logistic` | 41,67 | [32,29; 51,04] | 1290,5 | 8,9 | 3,58 | 14,53 | 98,30 | 0,00 p.p. |
| **MATH-500 ($n=96$)** | **`gru`** | **41,67** | **[32,29; 51,04]** | **1290,5** | **8,9** | **3,58** | **14,53** | **98,30** | **0,00 p.p.** |
| **Overall ($n=192$)** | `sc4` | 57,81 | [50,52; 64,60] | 1294,3 | 0,0 | 4,00 | 23,27 | 92,65 | 0,00 p.p. |
| **Overall ($n=192$)** | `consensus2` | 57,81 | [50,52; 64,60] | 1109,2 | 14,3 | 3,32 | 20,90 | 92,08 | 0,00 p.p. |
| **Overall ($n=192$)** | **`gru`** | **57,81** | **[50,52; 64,60]** | **1108,3** | **14,4** | **3,32** | **20,90** | **92,08** | **0,00 p.p.** |

---

## 3. Benchmark Real Online em GPU (`RTX 3050 6GB`, `n = 48`/dataset, `96` questões)

Executado via `scripts/06b_eval_phase4_online.py` com `Qwen/Qwen2.5-1.5B-Instruct` (FP16, Peak VRAM constante de **`4087,5 MB`** — `results/phase4/online_gpu_summary.csv`):

| Dataset | Método | Acc (%) | IC 95% (%) | Tokens Médios | Economia vs `sc4_2x2` | Traços | Latência (s/q) | Razão Latência | ECE (%) | Sem `\boxed{}` / Cap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K ($n=48$)** | `sc4_2x2` | 64,58 | [52,03; 77,08] | 1172,2 | 0,0% | 4,00 | 20,92 | 1,000x | 28,42 | 3 / 23 |
| **GSM8K ($n=48$)** | `sc4_4x1` | 68,75 | [54,17; 81,25] | 1134,6 | 3,2% | 4,00 | 11,46 | 0,548x | 31,98 | 4 / 16 |
| **GSM8K ($n=48$)** | `consensus2` | 64,58 | [52,03; 77,08] | 993,3 | 15,3% | 3,29 | 17,71 | 0,847x | 24,38 | 3 / 21 |
| **GSM8K ($n=48$)** | `logistic` | 64,58 | [52,03; 77,08] | 992,2 | 15,4% | 3,29 | 17,90 | 0,856x | 24,38 | 3 / 21 |
| **GSM8K ($n=48$)** | **`gru`** | **64,58** | **[52,03; 77,08]** | **992,4** | **15,3%** | **3,29** | **17,94** | **0,858x** | **24,38** | **3 / 21** |
| **MATH-500 ($n=48$)** | `sc4_2x2` | 45,83 | [31,25; 60,42] | 1415,7 | 0,0% | 4,00 | 24,28 | 1,000x | 21,92 | 22 / 42 |
| **MATH-500 ($n=48$)** | `sc4_4x1` | 37,50 | [25,00; 52,08] | 1394,9 | 1,5% | 4,00 | 13,28 | 0,547x | 19,31 | 21 / 41 |
| **MATH-500 ($n=48$)** | `consensus2` | 45,83 | [31,25; 60,42] | 1283,5 | 9,3% | 3,58 | 21,96 | 0,904x | 18,38 | 22 / 37 |
| **MATH-500 ($n=48$)** | `logistic` | 45,83 | [31,25; 60,42] | 1282,9 | 9,4% | 3,58 | 22,22 | 0,915x | 18,38 | 22 / 37 |
| **MATH-500 ($n=48$)** | **`gru`** | **45,83** | **[31,25; 60,42]** | **1282,9** | **9,4%** | **3,58** | **22,28** | **0,917x** | **18,38** | **22 / 37** |
| **Overall ($n=96$)** | `sc4_2x2` | 55,21 | [44,79; 64,58] | 1294,0 | 0,0% | 4,00 | 22,60 | 1,000x | 25,17 | 25 / 65 |
| **Overall ($n=96$)** | `sc4_4x1` | 53,12 | [42,71; 62,50] | 1264,8 | 2,3% | 4,00 | 12,37 | 0,547x | 25,65 | 25 / 57 |
| **Overall ($n=96$)** | `consensus2` | 55,21 | [44,79; 64,58] | 1138,4 | 12,0% | 3,44 | 19,84 | 0,878x | 21,37 | 25 / 58 |
| **Overall ($n=96$)** | `logistic` | 55,21 | [44,79; 64,58] | 1137,6 | 12,1% | 3,44 | 20,06 | 0,888x | 21,37 | 25 / 58 |
| **Overall ($n=96$)** | **`gru`** | **55,21** | **[44,79; 64,58]** | **1137,6** | **12,1%** | **3,44** | **20,11** | **0,890x** | **21,37** | **25 / 58** |

---

## 4. Achados Científicos e Verificação

1. **Preservação integral da acurácia + redução de latência e ECE:** No conjunto completo online (`n=96`), tanto `consensus2` (`55,21%`, `19,84s`, `0,878x`, `ECE=21,37%`) quanto `gru` (`55,21%`, `20,11s`, `0,890x`, `ECE=21,37%`) preservam **100%** da acurácia de `sc4_2x2` (`55,21%`, `22,60s`, `ECE=25,17%`), reduzindo tokens em `12,0–12,1%`, latência em `11–12%` e ECE em `3,80 p.p.`.
2. **Ganho intra-traço vs.\ paralelismo em lote:** Em `Qwen/Qwen2.5-1.5B-Instruct`, o delimitador `\boxed{...}` aparece apenas ao final do raciocínio (ao contrário de modelos destilados com blocos `<think>` longos), fazendo com que a poda intra-traço da `GRU` gere economia marginal sobre `Consensus-2` (`1137,6` vs `1138,4` tokens), enquanto o paralelismo em lote-4 (`sc4_4x1`) reduz a latência *wall-clock* para `0,547x` (`12,37s`).
3. **Verificação por testes (`#61`):** **35/35 PASSED** (`test_real_trace_policy.py` [4], `test_stopping_policy.py` [4], `test_decoders_and_metrics.py` [17], `test_data_and_traces_integrity.py` [4], `test_benchmark_and_scientific_invariants.py` [6]).
4. **Figuras geradas:** `results/phase4/figures/fig1_phase4_learning_curves.{png,pdf}`, `fig2_phase4_replay_pareto.{png,pdf}`, `fig3_phase4_online_gpu_pareto.{png,pdf}`.
