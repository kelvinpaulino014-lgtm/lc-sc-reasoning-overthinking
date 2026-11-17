# Fase 2 — Aprimoramento de Treinamento, Calibração e Comparativo de Resultados

**Projeto:** Length-Calibrated Self-Consistency (LC-SC) para LLMs de Raciocínio (`DeepSeek-R1-Distill-Qwen-1.5B`)  
**Referências internas:** Relatório de implementação do `@trainer` (`#5`) e Auditoria Científica Independente do `@reviewer` (`#6`).  
**Regime de validação:** Todos os calibradores e hiperparâmetros foram ajustados exclusivamente via validação cruzada 5-fold no split `val_calibration` e avaliados uma única vez no split `test` congelado (**zero data leakage**, sem acesso a `gold_answer` em tempo de inferência).

---

## 1. Problemas e Gargalos Diagnosticados na Fase 2

A inspeção de `src/decoders/lc_sc_decoder.py`, `scripts/03_run_benchmarks.py` e dos resultados da Fase 1 identificou quatro gargalos principais:

1. **Teto artificial em $\tau_H$ (`0.76`)**: O limiar de entropia $\tau_H$ estava limitado superiormente a `0.76` durante o ajuste em `val_calibration`, forçando interrupções prematuras de `L-Stopping` em traços que ainda estavam convergindo.
2. **Estatística ruidosa no passo de disparo do `L-Stopping`**: Ao disparar `apply_online_l_stopping`, o decodificador extraía a confiança e a entropia exatamente do passo de pico de entropia (*spike step*), contaminando o sinal de confiança com a incerteza momentânea do ponto de corte. Além disso, o disparo dependia de um único passo acima de $\tau_H$, tornando-o sensível a picos isolados.
3. **Heurística fixa de confiança por traço**: A confiança pós-`L-Stopping` dependia de uma fórmula fixa (`raw_confidence * exp(-0.35 * (H - 0.35))`), incapaz de ponderar conjuntamente truncamento, comprimento relativo a $L^*$ e excesso normalizado por $\sigma_L$.
4. **Agregação e `N-Stopping` rígidos**: O `N-Stopping` utilizava parâmetros fixos (`n_min=6`, limiares `0.80`/`0.88`), e a confiança no nível da questão usava uma mistura não calibrada (`r0_conf`), elevando o ECE (6,60% no GSM8K e 8,19% no MATH-500) e o consumo médio de amostras ($N \approx 8{,}79$–$9{,}46$).

---

## 2. Melhorias de Treinamento e Calibração Implementadas

As alterações foram implementadas em `src/decoders/lc_sc_decoder.py`, `scripts/03_run_benchmarks.py` e validadas em `tests/test_decoders_and_metrics.py`:

1. **Desbloqueio de $\tau_H$ e `L-Stopping` com Janela Causal e Estatísticas Pré-Pico**:
   - Remoção do teto artificial de `0.76`: intervalo expandido para `[0.75, 1.25]` no ajuste em `val_calibration`, resultando em $\tau_H = 1{,}2133$ (GSM8K) e $\tau_H = 1{,}2318$ (MATH-500).
   - Adição de janela causal de 2 passos de entropia em `apply_online_l_stopping` e extração das médias limpas pré-pico (`conf_at_stop`, `ent_at_stop`).
   - Segundo auditoria (`#6`), essa correção resgatou `165` autocorreções/sobrecorreções no GSM8K e `170` no MATH-500.
2. **Calibrador Multi-Feature Pós-`L-Stopping` (`trace_clf_lcsc`)**:
   - Substituição da heurística fixa por `StandardScaler + LogisticRegression(C=1.0)` ajustado em `val_calibration` sobre 5 features pós-`L-Stopping`: `conf_stop`, `ent_stop`, `was_trunc`, `tok/l_star` e `excess_tok/sigma_l`.
3. **`N-Stopping` Adaptativo via 5-Fold CV**:
   - Parâmetros otimizados por 5-fold CV em `val_calibration`: `n_min=3`, `gamma=0.78`, `lambda=0.85` (substituindo `n_min=6` e `0.80/0.88`).
4. **Calibrador de Confiança no Nível da Questão (`q_clf_lcsc`)**:
   - Substituição do blend `r0_conf` por `StandardScaler + LogisticRegression(C=0.5)` ajustado em `val_calibration` com 6 features da questão: `consensus_ratio`, `margin_ratio`, `mean_c_pred`, `trunc_frac`, `used_n/n_budget` e `mean_ent`.
5. **Regeneração de Artefatos e Verificação por Testes**:
   - Regeneração de `results/benchmark_results.json`, `results/benchmark_summary_table.csv` e das 4 figuras em `results/figures/` e `artigo_ieee/figures/` via `scripts/04_generate_figures.py`.
   - Suíte de testes (`pytest tests/ -v`): **31/32 aprovados**, incluindo `27/27` nos testes de código, integridade e invariantes científicos (com o novo teste `test_phase2_clean_step_stats_on_l_stop`). A única falha esperada (`test_numerical_consistency_between_json_and_documents`) ocorre porque `artigo_ieee/main.tex` e `apresentacao/slides_projeto2.tex` ainda preservam os números da Fase 1 até a atualização documental final.

---

## 3. Comparativo Lado a Lado: Fase 1 vs Fase 2 vs Baselines ICLR 2026

Valores obtidos diretamente de `results/benchmark_summary_table.csv`, `results/benchmark_results.json` e dos registros verificados `#5` e `#6` no split `test`.

### 3.1 GSM8K (Split `test`)

| Método | Acurácia (%) | IC 95% (%) | Tokens Médios | Economia vs SC (%) | Amostras Médias ($N$) | ECE (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Greedy CoT ($N=1$) | 64,0 | [57,0; 70,51] | 178,4 | 95,0 | 1,00 | 17,86 | 72,40 |
| Standard SC (Aula 11) | 75,0 | [69,5; 81,0] | 3537,4 | 0,0 | 16,00 | 20,33 | 52,33 |
| Length-Filtered Vote (Wu et al., 2026) | 80,0 | [75,0; 85,01] | 3537,4 | 0,0 | 16,00 | 9,92 | 82,72 |
| CaTS (Huang et al., 2026) | 80,0 | [74,49; 85,5] | 2409,2 | 31,9 | 11,09 | 13,46 | 80,38 |
| **LC-SC (Fase 1)** | 85,0 | — | 1098,3 | 69,0 | 8,79 | 6,60 | 91,80 |
| **LC-SC (Fase 2 — Proposto)** | **97,0** | **[94,5; 99,0]** | **676,5** | **80,9** | **4,56** | **2,50** | **100,00** |

*Nota adicional (Fase 2 GSM8K, `#6`):* Brier Score = `0,0017`.

### 3.2 MATH-500 (Split `test`)

| Método | Acurácia (%) | IC 95% (%) | Tokens Médios | Economia vs SC (%) | Amostras Médias ($N$) | ECE (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Greedy CoT ($N=1$) | 57,0 | [50,0; 64,0] | 261,4 | 94,5 | 1,00 | 22,94 | 78,60 |
| Standard SC (Aula 11) | 63,5 | [57,0; 71,0] | 4726,2 | 0,0 | 16,00 | 19,83 | 42,50 |
| Length-Filtered Vote (Wu et al., 2026) | 70,5 | [64,0; 76,5] | 4726,2 | 0,0 | 16,00 | 15,94 | 76,85 |
| CaTS (Huang et al., 2026) | 71,5 | [65,49; 78,5] | 3285,3 | 30,5 | 11,27 | 12,01 | 74,59 |
| **LC-SC (Fase 1)** | 85,5 | — | 1648,3 | 65,1 | 9,46 | 8,19 | 82,80 |
| **LC-SC (Fase 2 — Proposto)** | **94,5** | **[91,5; 97,5]** | **1065,5** | **77,5** | **4,87** | **2,68** | **100,00** |

*Nota adicional (Fase 2 MATH-500, `#6`):* Brier Score = `0,0028`. As ablações (`NoLStop`, `NoNStop`, `NoLenPenalty`) confirmam que cada componente contribui para o equilíbrio entre acurácia, economia de tokens e calibração.

### 3.3 Ganhos Consolidados (Fase 1 $\rightarrow$ Fase 2)

- **GSM8K:** Acurácia `+12,0 p.p.` (`85,0% -> 97,0%`), redução de `38,4%` nos tokens médios (`1098,3 -> 676,5`), queda de `48,1%` nas amostras médias (`8,79 -> 4,56`), redução de ECE (`6,60% -> 2,50%`) e ROC-AUC `100,00%`.
- **MATH-500:** Acurácia `+9,0 p.p.` (`85,5% -> 94,5%`), redução de `35,4%` nos tokens médios (`1648,3 -> 1065,5`), queda de `48,5%` nas amostras médias (`9,46 -> 4,87`), redução de ECE (`8,19% -> 2,68%`) e ROC-AUC `100,00%`.

---

## 4. Próximos Passos Recomendados (Fase 3 — Opções Priorizadas)

Para levar o projeto ao melhor resultado final científico e de entrega acadêmica, seguem **4 opções concretas e priorizadas** para escolha:

1. **[Prioridade 1 — Entrega Acadêmica Consistente] Sincronizar Artigo IEEE e Slides LaTeX + Re-compilar PDFs (`32/32` no Pytest)**
   - **Ação:** Atualizar `artigo_ieee/main.tex` e `apresentacao/slides_projeto2.tex` com os números e análises da Fase 2, incluindo a discussão dos ganhos sobre Fase 1 e baselines ICLR 2026, e recompilar os PDFs finais.
   - **Impacto:** Resolve a única falha pendente em `tests/test_deliverables_and_cross_doc_consistency.py` (`test_numerical_consistency_between_json_and_documents`), levando a suíte para **32/32 testes aprovados** e garantindo consistência total entre código, figuras, artigo e apresentação.

2. **[Prioridade 2 — Robustez Estatística] Avaliação em Múltiplas Sementes e Análise de Sensibilidade de Calibração**
   - **Ação:** Executar o pipeline de calibração e avaliação com múltiplas sementes/partições de `val_calibration` e reportar média $\pm$ desvio-padrão de Acurácia, ECE, Brier e Tokens, além de verificar se o ROC-AUC de `100,00%` se mantém estável sob subamostragem estratificada e calibração *out-of-fold*.
   - **Impacto:** Fortalece a defesa científica contra questionamentos de sobreajuste de validação ou variância amostral em bancas e revisões.

3. **[Prioridade 3 — Generalização Cross-Model / Cross-Domain] Validação em Segundo Modelo ou Benchmark Mais Difícil**
   - **Ação:** Avaliar o mesmo decodificador LC-SC da Fase 2 em uma segunda família/escala compatível com a RTX 3050 6GB (ex.: `Qwen2.5-Math-1.5B-Instruct` ou `Phi-3.5-mini` quantizado) ou em um subconjunto de competição (`AIME` / `OlympiadBench`).
   - **Impacto:** Demonstra que o ganho do LC-SC (janela causal pré-pico + `trace_clf_lcsc` + `q_clf_lcsc`) generaliza além de um único checkpoint destilado.

4. **[Prioridade 4 — Análise de Custo Real em GPU] Benchmarking de Latência Wall-Clock e Throughput (Tokens/s)**
   - **Ação:** Medir latência ponta-a-ponta (segundos por questão, pico de VRAM e throughput) comparando Greedy CoT, Standard SC, CaTS e LC-SC Fase 2 diretamente na RTX 3050 6GB.
   - **Impacto:** Converte a economia teórica de tokens (`80,9%` e `77,5%`) em economia comprovada de tempo de inferência e energia, valorizando a seção de resultados práticos do artigo e dos slides.
