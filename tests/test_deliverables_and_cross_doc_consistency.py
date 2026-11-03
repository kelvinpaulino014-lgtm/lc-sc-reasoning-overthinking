#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suíte de Testes 04: Conformidade dos Entregáveis Finais (PDFs, Contagem de Páginas,
Resolução 300 DPI das Figuras, Referências ICLR 2026 em ABNT e Consistência Cruzada JSON <-> LaTeX).
"""

import json
import re
import struct
import unittest
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_pdf_pages(pdf_path: Path) -> int:
    """Conta o número exato de páginas de um arquivo PDF (incluindo PDF 1.5+ com ObjStm FlateDecode)."""
    data = pdf_path.read_bytes()
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", data)]
    for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
        try:
            dec = zlib.decompress(stream)
            counts.extend([int(m) for m in re.findall(rb"/Type\s*/Pages.*?/Count\s+(\d+)", dec, re.DOTALL)])
        except Exception:
            pass
    if counts:
        return max(counts)
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches)


def get_png_dimensions(png_path: Path) -> tuple[int, int]:
    """Extrai (largura, altura) em pixels diretamente do cabeçalho IHDR de um arquivo PNG."""
    with open(png_path, "rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Arquivo PNG inválido: {png_path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


class TestDeliverablesAndCrossDocConsistency(unittest.TestCase):
    """Testa todos os arquivos finais de entrega, limites de páginas do edital e consistência numérica."""

    def test_pdf_deliverables_exist_and_exact_page_counts(self) -> None:
        """
        Verifica se os 4 PDFs obrigatórios existem e cumprem rigorosamente o limite de páginas:
        1. Etapa 1 (04/11/2026): exata 1 página
        2. Artigo Final IEEE: entre 6 e 8 páginas (exatas 8 páginas)
        3. Relatório de Uso de IA: exatas 2 páginas
        4. Apresentação Oral (5 min): exatos 4 slides (4 páginas)
        """
        etapa1_pdf = PROJECT_ROOT / "entrega_etapa1_04nov" / "proposta_projeto2_04nov_Kelvin_Buss.pdf"
        artigo_pdf = PROJECT_ROOT / "artigo_ieee" / "main.pdf"
        relatorio_ia_pdf = PROJECT_ROOT / "relatorio_ia" / "relatorio_ia.pdf"
        slides_pdf = PROJECT_ROOT / "apresentacao" / "slides_projeto2.pdf"

        for p in (etapa1_pdf, artigo_pdf, relatorio_ia_pdf, slides_pdf):
            self.assertTrue(p.exists(), f"PDF obrigatório ausente: {p}")
            self.assertGreater(p.stat().st_size, 15_000, f"PDF {p.name} parece vazio ou corrompido")

        self.assertEqual(count_pdf_pages(etapa1_pdf), 1, "A proposta da Etapa 1 (04/11) deve ter EXATA 1 página")
        artigo_pages = count_pdf_pages(artigo_pdf)
        self.assertTrue(6 <= artigo_pages <= 8, f"O artigo IEEE deve ter entre 6 e 8 páginas (atual: {artigo_pages})")
        self.assertEqual(count_pdf_pages(relatorio_ia_pdf), 2, "O Relatório de IA deve ter EXATAS 2 páginas")
        self.assertEqual(count_pdf_pages(slides_pdf), 4, "A apresentação deve ter EXATOS 4 slides")

    def test_figures_exist_and_300dpi_dimensions(self) -> None:
        """Verifica se todas as 4 figuras existem em PDF e PNG (300 DPI) em results/figures e artigo_ieee/figures."""
        res_figs = [
            "fig1_inverted_u_curve",
            "fig2_entropy_dynamics",
            "fig3_pareto_frontier",
            "fig4_calibration_ablation",
        ]
        for stem in res_figs:
            for ext in (".pdf", ".png"):
                p = PROJECT_ROOT / "results" / "figures" / f"{stem}{ext}"
                self.assertTrue(p.exists(), f"Figura ausente em results/figures: {p.name}")
                if ext == ".png":
                    w, h = get_png_dimensions(p)
                    self.assertGreaterEqual(w, 1800, f"Resolução horizontal < 300 DPI em {p.name}: {w}px")
                    self.assertGreaterEqual(h, 900, f"Resolução vertical baixa em {p.name}: {h}px")

        ieee_figs = [
            "fig1_inverted_u_curve.pdf",
            "fig2_entropy_dynamics_overthinking.pdf",
            "fig3_pareto_frontier_acc_vs_tokens.pdf",
            "fig4_calibration_reliability_and_ablation.pdf",
        ]
        for fname in ieee_figs:
            p = PROJECT_ROOT / "artigo_ieee" / "figures" / fname
            self.assertTrue(p.exists(), f"Figura vetorial ausente em artigo_ieee/figures: {fname}")

    def test_iclr2026_corpus_pdfs_and_repos_exist(self) -> None:
        """Verifica se os 4 PDFs do ICLR 2026 e os 4 repositórios oficiais do GitHub estão salvos localmente."""
        pdf_dir = PROJECT_ROOT / "referencias" / "artigos_iclr2026_pdfs"
        expected_pdfs = [
            "01_Wu_et_al_2026_When_More_is_Less_CoT_Length.pdf",
            "02_Huang_et_al_2026_CaTS_Calibrated_Test_Time_Scaling.pdf",
            "03_Hao_et_al_2026_RL_of_Thoughts_Navigating_LLM_Reasoning.pdf",
            "04_Feng_et_al_2026_In_Place_Test_Time_Training.pdf",
        ]
        for pdf_name in expected_pdfs:
            p = pdf_dir / pdf_name
            self.assertTrue(p.exists(), f"Artigo ICLR 2026 ausente: {pdf_name}")
            self.assertGreater(p.stat().st_size, 500_000, f"PDF {pdf_name} incompleto")

        repo_dir = PROJECT_ROOT / "referencias" / "repos_iclr2026"
        expected_repos = [
            "01_PKU-ML_CoT-Length",
            "02_Chengsong-Huang_Self-Calibration_CaTS",
            "03_tsinghua-fib-lab_RL-LLM-Reasoning",
            "04_ByteDance-Seed_In-Place-TTT",
        ]
        for r_name in expected_repos:
            r_path = repo_dir / r_name
            self.assertTrue(r_path.is_dir(), f"Repositório GitHub ICLR 2026 ausente: {r_name}")

    def test_abnt_citations_and_github_url_in_latex_files(self) -> None:
        """Verifica citações ABNT (AUTOR, ano), ausência de citações numéricas [1] e presença do link GitHub."""
        main_tex = (PROJECT_ROOT / "artigo_ieee" / "main.tex").read_text(encoding="utf-8")
        etapa1_tex = (
            PROJECT_ROOT / "entrega_etapa1_04nov" / "proposta_projeto2_04nov_Kelvin_Buss.tex"
        ).read_text(encoding="utf-8")
        rel_ia_tex = (PROJECT_ROOT / "relatorio_ia" / "relatorio_ia.tex").read_text(encoding="utf-8")
        slides_tex = (PROJECT_ROOT / "apresentacao" / "slides_projeto2.tex").read_text(encoding="utf-8")

        github_url = "https://github.com/kelvinbuss/ProjetoFinalRNAP_UnB2026"
        for doc_name, content in [
            ("main.tex", main_tex),
            ("proposta_04nov.tex", etapa1_tex),
            ("relatorio_ia.tex", rel_ia_tex),
            ("slides_projeto2.tex", slides_tex),
        ]:
            self.assertIn(github_url, content, f"URL do GitHub ausente em {doc_name}")
            for author in ("WU et al., 2026", "HUANG et al., 2026", "HAO et al., 2026", "FENG et al., 2026"):
                self.assertIn(author, content, f"Citação ABNT ({author}) ausente em {doc_name}")

        # Garante que não há citações IEEE numéricas tipo [1], [2] no corpo do artigo
        numbered_cites = re.findall(r"(?<!\$)(?<!\\)\[[1-9]\d*(?:,\s*[1-9]\d*)*\]", main_tex)
        self.assertEqual(len(numbered_cites), 0, f"Citações numéricas não-ABNT encontradas em main.tex: {numbered_cites}")

    def test_numerical_consistency_between_json_and_documents(self) -> None:
        """
        Verifica se os números reportados nas tabelas de main.tex, slides_projeto2.tex e relatorio_ia.tex
        batem exatamente com results/benchmark_results.json (zero números desatualizados).
        """
        with open(PROJECT_ROOT / "results" / "benchmark_results.json", "r", encoding="utf-8") as f:
            res = json.load(f)

        main_tex = (PROJECT_ROOT / "artigo_ieee" / "main.tex").read_text(encoding="utf-8")
        slides_tex = (PROJECT_ROOT / "apresentacao" / "slides_projeto2.tex").read_text(encoding="utf-8")
        rel_ia_tex = (PROJECT_ROOT / "relatorio_ia" / "relatorio_ia.tex").read_text(encoding="utf-8")

        gsm_lcsc = res["gsm8k"]["methods"]["LC-SC (Proposto)"]
        math_lcsc = res["math500"]["methods"]["LC-SC (Proposto)"]

        # Formata no padrão brasileiro usado no LaTeX (ex.: 86,50% e 1.098,3)
        def fmt_pct(val: float) -> str:
            return f"{val:.2f}%".replace(".", ",")

        def fmt_tok(val: float) -> str:
            s = f"{val:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return s

        gsm_acc_str = fmt_pct(gsm_lcsc["accuracy_pct"])
        math_acc_str = fmt_pct(math_lcsc["accuracy_pct"])
        gsm_tok_str = fmt_tok(gsm_lcsc["avg_tokens"])
        math_tok_str = fmt_tok(math_lcsc["avg_tokens"])

        self.assertIn(gsm_acc_str.replace("%", "\\%"), main_tex, f"Acurácia GSM8K ({gsm_acc_str}) divergente em main.tex")
        self.assertIn(math_acc_str.replace("%", "\\%"), main_tex, f"Acurácia MATH-500 ({math_acc_str}) divergente em main.tex")
        self.assertIn(gsm_tok_str, main_tex, f"Tokens GSM8K ({gsm_tok_str}) divergentes em main.tex")
        self.assertIn(math_tok_str, main_tex, f"Tokens MATH-500 ({math_tok_str}) divergentes em main.tex")

        self.assertIn(gsm_acc_str.replace("%", "\\%"), slides_tex, f"Acurácia GSM8K ({gsm_acc_str}) divergente em slides_projeto2.tex")
        self.assertIn(math_acc_str.replace("%", "\\%"), slides_tex, f"Acurácia MATH-500 ({math_acc_str}) divergente em slides_projeto2.tex")
        self.assertIn(gsm_tok_str, slides_tex, f"Tokens GSM8K ({gsm_tok_str}) divergentes em slides_projeto2.tex")
        self.assertIn(math_tok_str, slides_tex, f"Tokens MATH-500 ({math_tok_str}) divergentes em slides_projeto2.tex")

        # Verifica que relatorio_ia.tex referencia o arquivo de teste real e não nomes antigos
        self.assertIn("test_decoders_and_metrics.py", rel_ia_tex, "relatorio_ia.tex deve citar test_decoders_and_metrics.py")
        self.assertNotIn("test_lc_sc.py", rel_ia_tex, "relatorio_ia.tex contém caminho antigo test_lc_sc.py")


if __name__ == "__main__":
    unittest.main()
