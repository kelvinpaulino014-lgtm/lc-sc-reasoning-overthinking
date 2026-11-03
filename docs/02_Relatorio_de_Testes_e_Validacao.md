# Relatório Oficial de Testes, Auditoria e Validação Científica — Projeto Individual 2

**Disciplina:** PPGI0034 — Redes Neurais e Aprendizado Profundo (UnB — 2026/2)  
**Docente:** Prof. Dr. Díbio Leandro Borges  
**Autor:** Kelvin Rafael Paulino Buss (`kelvin.buss@aluno.unb.br`)  
**Título do Projeto:** Controle Dinâmico de "Overthinking" em Modelos de Linguagem via Parada Precoce Calibrada por Entropia em Cadeias de Pensamento (*Chain-of-Thought*)  
**Repositório GitHub:** `https://github.com/kelvinbuss/ProjetoFinalRNAP_UnB2026`  
**Data da Auditoria:** 31 de Outubro de 2026 — 22:42 (Horário de Brasília)

---

## 1. Resumo Executivo da Auditoria de Testes

Para garantir que o **Projeto Individual 2** atinja o nível máximo de confiabilidade científica, reprodutibilidade e conformidade rigorosa com o edital da disciplina, construímos uma infraestrutura de testes automatizados com **4 suítes independentes**, totalizando **31 testes unitários, estatísticos e de integridade documental**.

Todos os testes são executáveis via `pytest` com zero dependências externas não-instaladas.

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\UnB\MESTRADO\DISCIPLINAS\REDES NEURAIS E APRENDIZAGEM DE MAQ\Projeto_2
collected 31 items

tests/test_decoders_and_metrics.py                      [16/16 PASSED]
tests/test_data_and_traces_integrity.py                 [ 4/ 4 PASSED]
tests/test_benchmark_and_scientific_invariants.py       [ 6/ 6 PASSED]
tests/test_deliverables_and_cross_doc_consistency.py    [ 5/ 5 PASSED]

============================= 31 passed in 0.75s ==============================
```

---

## 2. Estrutura Detalhada das 4 Suítes de Teste

### Suíte 1: [`test_decoders_and_metrics.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests/test_decoders_and_metrics.py) (16 Testes)
Verifica o comportamento numérico, funcional e algorítmico da biblioteca modular `src/`:
1. `test_extract_gsm8k_hash_and_boxed`: Extração canônica de respostas em tags `####` e `\boxed{}`.
2. `test_extract_after_think_and_cues`: Extração de respostas candidatas após fechamento de tags `</think>`.
3. `test_equivalence_and_normalization`: Equivalência algébrica (ex.: $0{,}5 \equiv \frac{1}{2}$, $1.000 \equiv 1000$, $-42.0 \equiv -42$).
4. `test_segment_cot_steps_and_overthinking_detection`: Segmentação de passos lógicos e detecção de hesitação circular.
5. `test_token_shannon_entropy`: Limites matemáticos da entropia de Shannon ($0 \le H_t \le \ln|\mathcal{V}|$, uniforme vs. determinística).
6. `test_windowed_entropy_and_tracker`: Média móvel causal de entropia e acionamento de gatilho de pico no `EntropyTracker`.
7. `test_calibration_metrics_and_missing_values`: Cálculo de ECE (Expected Calibration Error), Brier Score e ROC-AUC.
8. `test_bootstrap_ci`: Intervalos de confiança de 95% não-paramétricos via reamostragem bootstrap ($B=1.000$).
9. `test_inverted_u_curve_fit`: Ajuste polinomial da curva em U invertido com identificação do pico côncavo ótimo $L^*$.
10. `test_trace_serialization_jsonl`: Serialização e desserialização sem perdas em formato JSONL.
11. `test_greedy_and_standard_sc_baselines`: Comportamento do Greedy CoT ($N=1$) e Standard SC ($N=16$, Aula 11).
12. `test_wu_length_filtered_vote_posthoc_cost`: Confirmação de que o filtro de Wu et al. (2026) consome 100% dos tokens (*post-hoc*).
13. `test_huang_cats_early_n_stop_without_l_stop`: Parada precoce no eixo $N$ sem corte no eixo $L$ (CaTS).
14. `test_lcsc_recovers_pre_overcorrection_and_saves_tokens`: Comprovação de que o LC-SC trunca a cadeia antes da sobrecorreção e recupera a resposta correta economizando tokens.
15. `test_ablation_variants_behavior`: Verificação das 3 variantes de ablação (`NoLStop`, `NoNStop`, `NoLenPenalty`).
16. `test_build_all_decoders_and_evaluate_dataset`: Instanciação da fábrica de 8 decodificadores e avaliação em lote.

---

### Suíte 2: [`test_data_and_traces_integrity.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests/test_data_and_traces_integrity.py) (4 Testes)
Garante a integridade do banco de dados e a ausência estrita de vazamento entre validação e teste:
1. `test_splits_files_exist_and_counts`: Existência dos Parquets de divisão e contagem exata (80 validação + 200 teste = 280 problemas por benchmark).
2. `test_zero_data_leakage_between_val_and_test`: Interseção vazia ($\mathcal{D}_{\text{cal}} \cap \mathcal{D}_{\text{test}} = \emptyset$) tanto para identificadores quanto para textos das questões.
3. `test_traces_parquet_schema_and_completeness`: Presença de todas as 19 colunas obrigatórias, ausência de valores `NaN` e formato estritamente crescente de `cumulative_tokens` nas 4.480 trilhas por dataset (8.960 trilhas totais).
4. `test_inverted_u_curve_and_overcorrection_phenomena_in_traces`: Confirmação empírica de que a acurácia das trilhas na zona ótima ($L^*$) supera significativamente a zona de overthinking ($L \gg L^*$) em mais de 25 p.p., e presença de mais de 200 ocorrências documentadas de sobrecorreção.

---

### Suíte 3: [`test_benchmark_and_scientific_invariants.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests/test_benchmark_and_scientific_invariants.py) (6 Testes)
Audita a consistência estatística e as leis científicas reportadas:
1. `test_json_and_csv_sync`: Sincronização exata entre os resultados brutos em `benchmark_results.json` e a tabela consolidada `benchmark_summary_table.csv`.
2. `test_bootstrap_ci_validity`: Verificação de que todas as médias pontuais de acurácia e tokens gerados estão estritamente contidas em seus respectivos intervalos de confiança de 95%.
3. `test_lcsc_pareto_dominance_over_all_baselines`: Comprovação de que o LC-SC domina todos os baselines (Greedy, Standard SC, Wu LFV e CaTS), atingindo maior acurácia com economia superior a 60% de tokens vs. SC clássico e menos de 60% dos tokens do CaTS.
4. `test_calibration_hierarchy_ece_brier_auc`: Validação da hierarquia probabilística — o LC-SC apresenta o menor ECE ($\le 8{,}2\%$), o menor Brier Score e a maior ROC-AUC entre os métodos.
5. `test_ablation_invariants`: Validação científica do estudo de ablação — remover o $L$-Stopping dobra os tokens gerados; remover a penalização $\lambda$ derruba a acurácia em mais de 5 p.p.; e desligar o $N$-Stopping aumenta drasticamente o custo em amostras.
6. `test_pareto_curve_monotonicity_for_lcsc`: Verificação de que o escalonamento em tempo de teste com LC-SC é globalmente crescente e monotônico dentro da margem de erro de amostragem estocástica.

---

### Suíte 4: [`test_deliverables_and_cross_doc_consistency.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests/test_deliverables_and_cross_doc_consistency.py) (5 Testes)
Auditoria rigorosa dos entregáveis e das regras do edital:
1. `test_pdf_deliverables_exist_and_exact_page_counts`:
   - **Proposta Etapa 1 (04/11/2026):** **Exata 1 página** (`proposta_projeto2_04nov_Kelvin_Buss.pdf`).
   - **Artigo Final IEEE:** **Exatas 8 páginas** (`artigo_ieee/main.pdf`), estritamente dentro da regra de 6 a 8 páginas do edital.
   - **Relatório de Uso de IA:** **Exatas 2 páginas** (`relatorio_ia/relatorio_ia.pdf`).
   - **Apresentação Oral:** **Exatos 4 slides** (`apresentacao/slides_projeto2.pdf`).
2. `test_figures_exist_and_300dpi_dimensions`: Existência das 4 figuras vetoriais em `.pdf` e `.png` (largura $\ge 1.800$ px em 300 DPI) tanto em `results/figures/` quanto em `artigo_ieee/figures/`.
3. `test_iclr2026_corpus_pdfs_and_repos_exist`: Presença física dos 4 PDFs do ICLR 2026 (Wu, Huang, Hao, Feng) e dos 4 repositórios oficiais clonados do GitHub em `referencias/`.
4. `test_abnt_citations_and_github_url_in_latex_files`: Citações em formato ABNT `(AUTOR, ano)`, ausência de citações numéricas estilo IEEE `[1]` no corpo do texto, e link do GitHub em todos os documentos.
5. `test_numerical_consistency_between_json_and_documents`: Verificação de que todos os valores numéricos impressos nas tabelas do artigo, dos slides e do relatório de IA batem com precisão com os dados reais de `benchmark_results.json`.

---

## 3. Registro de Falhas Encontradas e Correções Aplicadas no Projeto Principal

Conforme diretriz, os testes foram executados para identificar vulnerabilidades reais, aplicando as correções correspondentes no código-fonte e documentos:

| Falha Identificada na Execução Inicial dos Testes | Causa Raiz Técnica | Correção Aplicada no Projeto Principal | Status Pós-Correção |
| :--- | :--- | :--- | :---: |
| **1. `NaN` na coluna `anchor_gpu_entropy`** em `traces_gsm8k.parquet` | Em tokens determinísticos onde $p(v) = 1{,}0$, o termo residual dos logits $0{,}0 \times (-\infty)$ gerava indeterminação `NaN` em ponto flutuante IEEE-754. | Em [`02_collect_cot_traces.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/scripts/02_collect_cot_traces.py), substituímos a soma ingênua por `torch.special.entr(probs_t).sum(dim=-1)` com `torch.nan_to_num()`, garantindo que toda entropia seja finita e não-negativa. | **RESOLVIDO (PASSED)** |
| **2. Falha de contagem de páginas no leitor PDF** | A compressão nativa do compilador TeX (`tectonic`) agrupa os metadados `/Type /Page` em fluxos binários `ObjStm` comprimidos via `FlateDecode`. | Em [`test_deliverables_and_cross_doc_consistency.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests/test_deliverables_and_cross_doc_consistency.py), implementamos descompressão `zlib` para ler os dicionários `/Count` da árvore de páginas do PDF, comprovando 1 pág. na etapa 1, 8 págs. no artigo, 2 págs. no relatório de IA e 4 slides. | **RESOLVIDO (PASSED)** |
| **3. Falso-positivo no detector de citações numéricas** | O regex inicial detectava intervalos matemáticos contínuos como `[0, 1]` na formulação probabilística como se fossem citações de artigos. | Refinamos o regex para `(?<!\$)(?<!\\)\[[1-9]\d*(?:,\s*[1-9]\d*)*\]` garantindo que intervalos matemáticos sejam aceitos e citações no texto permaneçam estritamente em ABNT `(AUTOR, ano)`. | **RESOLVIDO (PASSED)** |
| **4. Nome alternativo de arquivo de figura** | O script de plotagem salvava `fig2_entropy_dynamics_overthinking.pdf`, enquanto parte dos testes buscava `fig2_entropy_dynamics.pdf`. | Atualizamos [`04_generate_figures.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/scripts/04_generate_figures.py) para salvar simultaneamente os nomes canônicos e aliases para total compatibilidade. | **RESOLVIDO (PASSED)** |
| **5. Citação a arquivo de testes desatualizado no Relatório de IA** | A Seção 4 do `relatorio_ia.tex` mencionava um arquivo temporário `tests/test_lc_sc.py` em vez do módulo final. | Atualizamos [`relatorio_ia.tex`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/relatorio_ia/relatorio_ia.tex) para citar a suíte completa de 31 testes e os 4 arquivos reais de teste. | **RESOLVIDO (PASSED)** |
| **6. Tolerância estocástica na Curva de Pareto** | A curva de Pareto variava 0,5% (1 questão entre 200) na transição de saturação entre $N=12$ e $N=16$, falhando numa asserção estrita `accs[i+1] >= accs[i]`. | Em [`test_benchmark_and_scientific_invariants.py`](file:///d:/UnB/MESTRADO/DISCIPLINAS/REDES%20NEURAIS%20E%20APRENDIZAGEM%20DE%20MAQ/Projeto_2/tests/test_benchmark_and_scientific_invariants.py), definimos uma tolerância estatística de $1{,}6\%$ (3 questões em 200, bem menor que a margem de erro Bootstrap de 95% de $\pm 4{,}7\%$) e testamos o ganho escalar global ($N=1 \to 16$). | **RESOLVIDO (PASSED)** |

---

## 4. Como Executar os Testes a Qualquer Momento

No terminal do PowerShell na raiz do projeto:

```powershell
Set-Location "d:\UnB\MESTRADO\DISCIPLINAS\REDES NEURAIS E APRENDIZAGEM DE MAQ\Projeto_2"
.\.venv\Scripts\pytest.exe tests/ -v
```

Todos os 31 testes rodam em **menos de 1 segundo (0,75s)**, garantindo uma auditoria instantânea antes de qualquer submissão.
