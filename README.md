# Projeto Individual 2 — Redes Neurais e Aprendizado Profundo (PPGI0034 / UnB — 2026/2)

**Título:** Controle Dinâmico de "Overthinking" em Modelos de Linguagem via Parada Precoce Calibrada por Entropia em Cadeias de Pensamento (*Chain-of-Thought*)  
**Autor:** Kelvin Rafael Paulino Buss (`kelvin.buss@aluno.unb.br`)  
**Professor:** Prof. Dr. Díbio Leandro Borges (CIC/UnB)  
**Repositório GitHub Alvo:** `https://github.com/kelvinbuss/ProjetoFinalRNAP_UnB2026`

---

## Estrutura Organizacional do Projeto 2

```text
Projeto_2/
├── GUIA_PASSO_A_PASSO_PROJETO_2.md       # <--- COMECE POR AQUI: Explicação didática completa, tempos por fase e plano paralelo
├── opcoes_projeto_2_iclr2026.txt         # Documento de mapeamento inicial das 5 opções de temas ICLR 2026
├── projeto2-rnap-2026_2 (1).pdf          # Especificação oficial do Projeto Individual 2 (Prof. Díbio)
│
├── entrega_etapa1_04nov/                 # [ENTREGA 1 - PRAZO 04/11/2026 ÀS 23h59] (CONCLUÍDA!)
│   ├── proposta_projeto2_04nov_Kelvin_Buss.pdf  # Documento oficial de 1 página exata pronto para submissão
│   ├── proposta_projeto2_04nov_Kelvin_Buss.tex  # Código-fonte LaTeX da proposta de 1 página
│   ├── proposta_projeto2_04nov_Kelvin_Buss.md   # Versão em Markdown
│   └── proposta_projeto2_04nov_Kelvin_Buss.txt  # Versão em texto puro (ASCII)
│
├── referencias/                          # Acervo completo do ICLR 2026 (PDFs + Códigos-Fonte GitHub)
│   ├── README.md                         # Índice detalhado dos 4 artigos e 4 repositórios
│   ├── artigos_iclr2026_pdfs/            # PDFs completos de Wu (2026), Huang (2026), Hao (2026) e Feng (2026)
│   └── repos_iclr2026/                   # Repositórios GitHub oficiais extraídos localmente
│
├── slides_aula/                          # Slides oficiais das Aulas 10, 11 e 12 (Prof. Díbio)
│   ├── aula_10_rnap_ciber2026_2.pdf
│   ├── aula_11_rnap_ciber2026_2.pdf      # Base teórica de CoT e Self-Consistency (Slide 17)
│   └── aula_12_rnap_ciber2026_2.pdf
│
├── src/                                  # Implementação em Python/PyTorch (CUDA) do método proposto LC-SC
├── scripts/                              # Scripts de download de datasets, inferência GPU, benchmarks e figuras
├── tests/                                # Suíte de 31 testes automatizados (pytest):
│   ├── test_decoders_and_metrics.py      # 16 testes unitários dos decodificadores e métricas de calibração
│   ├── test_data_and_traces_integrity.py # 4 testes de zero data leakage e integridade dos Parquets
│   ├── test_benchmark_and_scientific_invariants.py # 6 testes de invariantes científicos e Pareto
│   └── test_deliverables_and_cross_doc_consistency.py # 5 testes de conformidade com edital (páginas e ABNT)
├── data/                                 # Datasets GSM8K e MATH-500 (salvos localmente em Parquet/JSONL)
├── results/                              # Trilhas CoT cacheadas, métricas JSON/CSV e figuras de 300 DPI
│   └── figures/
│
├── artigo_ieee/                          # [ENTREGA FINAL] Artigo formato IEEE Conference (6 a 8 páginas, ABNT)
│   ├── IEEEtran.cls
│   ├── IEEEtran.bst
│   └── figures/
├── relatorio_ia/                         # [ENTREGA FINAL] Relatório de Uso de IA (exatas 2 páginas)
├── apresentacao/                         # [ENTREGA FINAL] Apresentação Oral (exatos 4 slides para 5 minutos)
├── docs/                                 # Documentação técnica, diário de prompts e relatório de testes:
│   ├── 01_Relatorio_Evolucao_e_Prompts.md
│   └── 02_Relatorio_de_Testes_e_Validacao.md # <--- Relatório completo dos 31 testes e correções aplicadas
└── bin/                                  # Compilador LaTeX autocontido (tectonic.exe)
```
