"""
Generate the CARE 2026 submission document.

Contains two self-contained versions so either can be pasted into the submission form:

    * a ~230-word short abstract
    * a ~1,250-word extended abstract with results tables and a reference list

Every number is taken from cgsae/results/. Nothing here is rounded in our favour, and the
scope limitations are stated in the body rather than omitted.

Usage:  python make_abstract_doc.py --out ../CARE2026_Abstract.docx
"""

from __future__ import annotations

import argparse

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x55, 0x55, 0x55)


def shade(hexcolour):
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), hexcolour)
    return el


class Doc:
    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        for s in self.d.sections:
            s.left_margin = s.right_margin = Inches(0.9)
            s.top_margin = s.bottom_margin = Inches(0.8)

    def centre(self, text, size=11, bold=False, colour=None, after=4):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        if colour is not None:
            r.font.color.rgb = colour
        p.paragraph_format.space_after = Pt(after)

    def h(self, text, size=12):
        p = self.d.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = BLUE
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(3)

    def para(self, text, justify=True, size=10.5, italic=False, after=5):
        p = self.d.add_paragraph()
        if justify:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.italic = italic
        p.paragraph_format.space_after = Pt(after)
        return p

    def rule(self):
        p = self.d.add_paragraph()
        r = p.add_run("─" * 78)
        r.font.color.rgb = GREY
        r.font.size = Pt(7)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)

    def table(self, headers, rows, widths=None, size=9):
        t = self.d.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(headers):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(htxt)
            r.bold = True
            r.font.size = Pt(size)
            c._tc.get_or_add_tcPr().append(shade("DCE6F1"))
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def caption(self, text):
        p = self.d.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(8.5)
        r.italic = True
        r.font.color.rgb = GREY
        p.paragraph_format.space_after = Pt(8)

    def save(self, path):
        self.d.save(path)


TITLE = ("Where Should a Concept Live? Latent Capacity Allocation and the Role of a "
         "Neutral Buffer in Supervised Sparse Autoencoders")
AUTHORS = "Varanasi Sathveek, Vanamala Aravind, Pula Naga Manikanta"
SUPER = "Supervisor: Dr. K. Hima Bindu"
AFFIL = "Department of Computer Science & Engineering, National Institute of Technology Andhra Pradesh"

SHORT = (
    "Sparse autoencoders (SAEs) recover interpretable features from language-model "
    "activations, and recent work adds label supervision so that a target concept occupies "
    "a known latent position: Scar conditions a single latent, while SAE-SSV selects latents "
    "post hoc. Neither asks how much latent capacity a concept should receive, or where "
    "features shared between two opposing classes should live. We partition an SAE's latent "
    "space before training into a wanted block, an unwanted block, and an unpenalised "
    "neutral buffer whose width is a tunable parameter, so that unwanted behaviour can later "
    "be removed by address rather than by search. Across GPT-2 and Qwen2.5-1.5B on sentiment "
    "and toxicity, ablating the reserved block damages the target behaviour significantly "
    "more than ablating a size-matched random block in 10 of 16 tests. The neutral buffer is "
    "the mechanism: ablation specificity falls monotonically from 16.6x to ~1x as the buffer "
    "is removed, and is ~1x throughout when the penalty is disabled. We further show that "
    "the cross-group leakage metric used in this literature is uncorrelated with whether the "
    "block encodes anything (r = +0.03), that suppression-style objectives are inert under "
    "realistic sparsity, and that selecting an intervention layer by probe accuracy forfeits "
    "45% of achievable steering efficiency."
)


def build(out: str) -> None:
    D = Doc()

    # ---- header ----
    D.centre(TITLE, size=14, bold=True, colour=BLUE, after=6)
    D.centre(AUTHORS, size=10.5)
    D.centre(SUPER, size=10)
    D.centre(AFFIL, size=9.5, colour=GREY, after=3)
    D.centre("Track: Foundations of AI   |   CARE Conference 2026, IIT Roorkee",
             size=9.5, bold=True, colour=GREY, after=8)
    D.rule()

    # ================= SHORT VERSION =================
    D.h("Short abstract (230 words)")
    D.para(SHORT)
    D.rule()

    # ================= EXTENDED =================
    D.h("Extended abstract")

    D.h("1.  Motivation", 11)
    D.para(
        "Removing a specific behaviour from a language model without disturbing anything "
        "else is hard because features are stored in superposition: a model represents far "
        "more concepts than it has dimensions, so concept directions overlap and any "
        "intervention aimed at one collaterally affects others. Sparse autoencoders (SAEs) "
        "[1,2,3] mitigate this by projecting activations into an overcomplete space under a "
        "sparsity constraint, recovering features that are closer to single concepts.")
    D.para(
        "Standard SAE training, however, imposes no semantic organisation on the recovered "
        "latents. Recent supervised variants address this. Scar [4] conditions one latent on "
        "a concept label during training; SAE-SSV [5] selects concept-relevant latents post "
        "hoc from a pretrained SAE; further variants add probe or contrastive terms [6,7]. "
        "All of them treat the concept as occupying either a single dimension or an "
        "unspecified set discovered afterwards. None asks the capacity question: how many "
        "latents should a concept receive, and where should features genuinely shared "
        "between two opposing classes reside?")
    D.para(
        "This matters practically. If a concept occupies a contiguous block fixed before "
        "training, it can be removed at inference by address - no probing, no ranking, and "
        "no labelled data at intervention time. That property distinguishes the approach "
        "from post-hoc selection, and it is what we set out to test.")

    D.h("2.  Method", 11)
    D.para(
        "We partition the m latents of a TopK SAE [3] into three contiguous blocks before "
        "training, with n_tag = floor(tau * m) latents per side: a G-block for the wanted "
        "class, a B-block for the unwanted class, and an unpenalised neutral buffer between "
        "them. The tag fraction tau sets the buffer width; tau = 0.5 removes it entirely and "
        "serves as the ablation that isolates its contribution.")
    D.para(
        "A natural training objective penalises each block for activating on the opposing "
        "class. We find this fails in two distinct ways. Squaring the mean off-block "
        "activation makes the term vanish under realistic sparsity (at L0 = 32 of 6,144, the "
        "term is 0.5% of the loss even at ten times the intended weight, and leakage moves "
        "only 0.494 to 0.435 against a chance level of 0.5). More seriously, every "
        "suppression-only objective is minimised by silence: the model can empty both tagged "
        "blocks, route all activity through the neutral buffer, and record perfect "
        "separation while encoding nothing. We therefore use a contrastive routing objective "
        "in which an empty block scores zero and correct routing scores -1, so silence is no "
        "longer rewarded:")
    D.para(
        "    P = -1/2 [ (B-share | unwanted  -  B-share | wanted) "
        "+ (G-share | wanted  -  G-share | unwanted) ]", justify=False, italic=True)
    D.para(
        "where share denotes a block's fraction of total post-TopK activation mass. This "
        "raises routing from ~0.01 to ~0.40 with the block remaining active, at a "
        "reconstruction cost of FVU 0.16 to 0.25.")

    D.h("3.  Experimental setup", 11)
    D.para(
        "We evaluate on GPT-2 (124M) and Qwen2.5-1.5B-Instruct, on sentiment (IMDB) and "
        "toxicity (RealToxicityPrompts), using per-token residual activations. Comparison "
        "arms are: the reserved-block ablation; a vanilla TopK SAE with post-hoc selection "
        "(the SAE-SSV recipe); a Scar-style single conditioned latent; difference-in-means "
        "steering [8,9]; a logistic-probe direction; and three null controls (random latents, "
        "a size-matched random block, a random direction).")
    D.para(
        "Three constraints make the comparison fair. All arms use identical sparsity (TopK, "
        "L0 = 32 exactly). All arms intervene on an identical number of latents. And every "
        "arm is scored by its effect at a matched perplexity budget on off-axis held-out "
        "text, swept over intervention strength - so no method can win by intervening less. "
        "Interventions are applied as a difference between decodes, h' = h + (dec(z') - "
        "dec(z)), so the autoencoder's reconstruction error never enters the residual stream "
        "and a null intervention costs exactly zero perplexity. Effects are judged by "
        "external classifiers, with greedy decoding throughout, over 3-5 seeds; a 20-seed "
        "synthetic experiment provides a controlled counterpart.")

    D.h("4.  Results", 11)
    D.para("(a) The reserved block localises the concept. Ablating the B-block damages the "
           "target behaviour significantly more than ablating a size-matched random block in "
           "10 of 16 tests, in all four main runs, on both concept axes and both model "
           "scales. tau = 0.10 survives Bonferroni correction in three independent runs.")
    D.table(["Run", "tau=0.05", "tau=0.10", "tau=0.20", "tau=0.50 (no buffer)"],
            [["Sentiment, GPT-2", "0.010*", "0.229", "0.181", "0.352"],
             ["Toxicity, GPT-2", "0.015*", "0.002*", "0.018*", "0.013*"],
             ["Sentiment, Qwen L15", "0.066", "0.004*", "0.088", "0.144"],
             ["Sentiment, Qwen L18", "0.029*", "0.003*", "0.044*", "0.111"]],
            widths=[1.6, 1.0, 1.0, 1.0, 1.4])
    D.caption("Table 1. p-values, reserved block vs size-matched random block, paired over "
              "seeds at a matched 10% perplexity budget. * p < 0.05.")

    D.para("(b) The neutral buffer is the mechanism. In the controlled synthetic setting "
           "(20 seeds), specificity - the ratio of damage from ablating the reserved block "
           "to damage from ablating a random block of equal width - falls monotonically as "
           "the buffer shrinks, and is ~1 throughout when the penalty is disabled.")
    D.table(["Buffer ratio", "tau", "Penalty off", "Penalty on"],
            [["4.33", "0.10", "1.6x", "16.6x"], ["1.67", "0.20", "1.3x", "4.0x"],
             ["0.78", "0.30", "1.1x", "1.6x"], ["0.33", "0.40", "1.0x", "1.6x"],
             ["0.00", "0.50", "1.1x", "0.9x"]],
            widths=[1.3, 0.8, 1.4, 1.4])
    D.caption("Table 2. Ablation specificity against buffer width. Removing the buffer "
              "returns the method to chance. The same experiment shows a double "
              "dissociation: B-block ablation costs the unwanted class 0.447 accuracy while "
              "gaining 0.184 on the wanted class, G-block ablation mirrors it, and a "
              "size-matched random block does neither.")

    D.para("(c) Scope. Against post-hoc selection at matched intervention width, the "
           "reserved block confers no advantage on GPT-2 (p = 0.24-0.40) but does on "
           "Qwen2.5-1.5B (+0.277 vs +0.127, p = 0.001 at layer 15; p = 0.017 at layer 18). "
           "Two of four tests are significant, so this is suggestive rather than "
           "established. Against difference-in-means steering the picture is "
           "task-dependent: the direction method wins on sentiment at both scales, while "
           "SAE-based methods win on toxicity (+0.194 vs +0.130, p = 0.022), consistent "
           "with the task-dependence reported in [10].")

    D.para("(d) Two methodological findings. First, the cross-group leakage metric used "
           "throughout this literature is uninformative: across twelve configurations its "
           "correlation with whether the block actually encodes the concept is r = +0.03, "
           "and the best-scoring configuration in the sweep (leakage 0.0017) has a 96% dead "
           "block. Leakage is also scale-gameable - multiplying all activations by 0.01 "
           "divides it by 100 while changing nothing. We report a routing measure and a "
           "block-occupancy measure alongside it. Second, selecting an intervention layer by "
           "probe accuracy is suboptimal: on GPT-2 probe AUC peaks at the final layer "
           "(0.816) whereas steering efficiency peaks mid-network, and choosing by "
           "decodability forfeits 45% of achievable efficiency.")

    D.h("5.  Limitations", 11)
    D.para(
        "The method does not improve steering universally; on sentiment a difference-in-means "
        "baseline outperforms every SAE-based arm. The scale-dependence claim rests on two "
        "model sizes with two of four tests significant. The routing objective costs "
        "reconstruction fidelity (FVU 0.16 to 0.25). The approach operates on activations at "
        "inference and does not modify model weights. Results cover two binary concept axes; "
        "multi-concept partitioning is untested.")

    D.h("6.  Conclusion", 11)
    D.para(
        "Reserving a labelled block of SAE latent capacity for an unwanted behaviour, with "
        "an unpenalised buffer separating it from the rest, produces a latent region that "
        "demonstrably contains the concept and can be removed by address rather than by "
        "search. The buffer, rather than the partition itself, is what makes the region "
        "specific. We also show that the standard evaluation metric in this area cannot "
        "distinguish a working partition from an empty one, which suggests existing "
        "supervised-SAE results would benefit from re-examination under an occupancy-aware "
        "measure.")

    D.h("References", 11)
    refs = [
        "Bricken, T., et al. Towards Monosemanticity: Decomposing Language Models With Sparse Autoencoders. Transformer Circuits Thread, 2023.",
        "Cunningham, H., et al. Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600, 2023.",
        "Gao, L., et al. Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093, 2024.",
        "Scar: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs. arXiv:2411.07122, 2024.",
        "SAE-SSV: Supervised Steering in Sparse Representation Spaces. arXiv:2505.16188, 2025.",
        "Graph-Regularized Sparse Autoencoders for LLM Safety Steering. arXiv:2512.06655.",
        "Disentangled Sparse Representations for Concept-Separated Unlearning. arXiv:2605.12122.",
        "Turner, A., et al. Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248, 2023.",
        "Arditi, A., et al. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS, 2024.",
        "A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering. arXiv:2510.01246, 2025.",
        "Elhage, N., et al. Toy Models of Superposition. Transformer Circuits Thread, 2022.",
    ]
    for i, r in enumerate(refs, 1):
        p = D.d.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(f"[{i}]  {r}")
        run.font.size = Pt(8.5)

    D.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../CARE2026_Abstract.docx")
    build(ap.parse_args().out)
