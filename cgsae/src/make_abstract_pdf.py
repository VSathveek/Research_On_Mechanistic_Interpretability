"""
Generate the CARE 2026 extended abstract as a PDF.

The submission form requires PDF (max 10 MB) and states no word or page limit, so this
targets a conventional two-page extended abstract. Built directly with ReportLab rather
than converted from the .docx, so it does not depend on Word or LibreOffice being present.

Usage:  python make_abstract_pdf.py --out ../CARE2026_Abstract.pdf
"""

from __future__ import annotations

import argparse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ACCENT = colors.HexColor("#1F4E79")
GREY = colors.HexColor("#555555")

TITLE = ("Where Should a Concept Live? Latent Capacity Allocation and the Role of a "
         "Neutral Buffer in Supervised Sparse Autoencoders")
AUTHORS = "Varanasi Sathveek, Vanamala Aravind, Pula Naga Manikanta, K. Hima Bindu"
SUPERVISOR = "Supervised by Dr. K. Hima Bindu"
AFFIL = ("Department of Computer Science &amp; Engineering, "
         "National Institute of Technology Andhra Pradesh, India")
TRACK = "Track: Foundations of AI"

ABSTRACT = (
    "Sparse autoencoders (SAEs) recover interpretable features from language-model "
    "activations, and recent work adds label supervision so that a target concept occupies a "
    "known latent position: Scar conditions a single latent, while SAE-SSV selects latents "
    "post hoc. Neither asks how much latent capacity a concept should receive, or where "
    "features shared between two opposing classes should live. We partition an SAE's latent "
    "space before training into a wanted block, an unwanted block, and an unpenalised neutral "
    "buffer whose width is a tunable parameter, so that unwanted behaviour can later be "
    "removed by address rather than by search. Across GPT-2 and Qwen2.5-1.5B on sentiment and "
    "toxicity, ablating the reserved block reduces the target behaviour more than ablating a "
    "size-matched random block in 19 of 20 run&ndash;configuration cells (sign test "
    "p = 2&times;10<super>&minus;5</super>). The neutral buffer is the mechanism: in a "
    "controlled setting, ablation specificity falls from 16.6&times; to chance as the buffer is "
    "removed, and is ~1&times; throughout when the penalty is disabled. We further show that "
    "the cross-group leakage metric used in this literature is uncorrelated with whether the "
    "block encodes anything (r = +0.03), that suppression-style objectives are inert under "
    "realistic sparsity, and that in all four layer scans the most linearly decodable layer is "
    "not the most steerable one."
)

KEYWORDS = ("mechanistic interpretability; sparse autoencoders; activation steering; "
            "representation engineering; model editing; LLM safety")

SECTIONS = [
    ("1.  Motivation", [
        "Removing a specific behaviour from a language model without disturbing anything else "
        "is hard because features are stored in superposition: a model represents far more "
        "concepts than it has dimensions, so concept directions overlap and any intervention "
        "aimed at one collaterally affects others [11]. Sparse autoencoders [1,2,3] mitigate "
        "this by projecting activations into an overcomplete space under a sparsity "
        "constraint, recovering features closer to single concepts.",

        "Standard SAE training imposes no semantic organisation on the recovered latents. "
        "Recent supervised variants address this: Scar [4] conditions one latent on a concept "
        "label during training; SAE-SSV [5] selects concept-relevant latents post hoc from a "
        "pretrained SAE; further variants add probe or contrastive terms [6,7]. All treat the "
        "concept as occupying either a single dimension or an unspecified set discovered "
        "afterwards. None asks the capacity question: how many latents should a concept "
        "receive, and where should features genuinely shared between two opposing classes "
        "reside?",

        "This matters practically. If a concept occupies a contiguous block fixed before "
        "training, it can be removed at inference <i>by address</i> &mdash; no probing, no "
        "ranking, no labelled data at intervention time. That property distinguishes the "
        "approach from post-hoc selection, and it is what we set out to test.",
    ]),
    ("2.  Method", [
        "We partition the <i>m</i> latents of a TopK SAE [3] into three contiguous blocks "
        "before training, with n<sub>tag</sub> = floor(&tau;&middot;m) latents per "
        "side: a G-block for the wanted class, a B-block for the unwanted class, and an "
        "unpenalised neutral buffer between them. The tag fraction &tau; sets the buffer "
        "width; &tau; = 0.5 removes it entirely and serves as the ablation isolating its "
        "contribution.",

        "A natural objective penalises each block for activating on the opposing class. We "
        "find this fails in two ways. Squaring the mean off-block activation makes the term "
        "vanish under realistic sparsity &mdash; at L<sub>0</sub> = 32 of 6,144 it is 0.5% of "
        "the loss even at ten times the intended weight, and leakage moves only 0.495 to "
        "0.435 against a chance level of 0.5. More seriously, <i>every suppression-only "
        "objective is minimised by silence</i>: the model can empty both tagged blocks, route "
        "all activity through the neutral buffer, and record perfect separation while "
        "encoding nothing. We therefore use a contrastive routing objective in which an empty "
        "block scores 0 and correct routing scores &minus;1:",

        "<para alignment='center'><font face='Courier' size='8.5'>"
        "P = &minus;&#189; [ (B<sub>share</sub>|unwanted &minus; B<sub>share</sub>|wanted) + "
        "(G<sub>share</sub>|wanted &minus; G<sub>share</sub>|unwanted) ]</font></para>",

        "where <i>share</i> denotes a block's fraction of total post-TopK activation mass. "
        "This raises routing from ~0.01 to ~0.40 with the block remaining active, at a "
        "reconstruction cost of FVU 0.16 &rarr; 0.25.",
    ]),
    ("3.  Experimental setup", [
        "We evaluate on GPT-2 (124M) and Qwen2.5-1.5B-Instruct, on sentiment (IMDB) and "
        "toxicity (RealToxicityPrompts), using per-token residual activations. Arms: the "
        "reserved-block ablation; a vanilla TopK SAE with post-hoc selection (the SAE-SSV "
        "recipe); a Scar-style single conditioned latent; difference-in-means steering [8,9]; "
        "a logistic-probe direction; and three null controls (random latents, a size-matched "
        "random block, a random direction).",

        "Three constraints make the comparison fair. All arms use identical sparsity (TopK, "
        "L<sub>0</sub> = 32 exactly). All arms intervene on an identical number of latents. "
        "Every arm is scored by its effect <i>at a matched perplexity budget</i> on off-axis "
        "held-out text, swept over intervention strength &mdash; so no method can win by "
        "intervening less. Interventions are applied as a difference between decodes, "
        "h&prime; = h + (dec(z&prime;) &minus; dec(z)), so the autoencoder's reconstruction "
        "error never enters the residual stream and a null intervention costs exactly zero "
        "perplexity. Effects are judged by external classifiers with greedy decoding "
        "throughout, over 3&ndash;5 seeds; a 20-seed synthetic experiment provides a "
        "controlled counterpart.",
    ]),
]

RESULTS_A = ("<b>(a) The reserved block localises the concept.</b> Ablating the B-block "
             "reduces the target behaviour more than ablating a size-matched random block in "
             "<b>19 of 20</b> cells across five runs covering both concept axes and both model "
             "scales. With 3&ndash;5 seeds each cell is underpowered: 10 reach p &lt; 0.05 "
             "individually, so the evidence is the consistency of the direction. The effect "
             "concentrates where the buffer is widest: at &tau; = 0.10 it is significant in "
             "<b>4 of 5 runs</b>, and at &tau; = 0.50, where the buffer is removed, in 1 of 5.")

TABLE1 = [["Run", "τ=0.05", "τ=0.10", "τ=0.20", "τ=0.50 (no buffer)"],
          ["Sentiment, GPT-2", "0.010*", "0.229", "0.181", "0.352"],
          ["Toxicity, GPT-2", "0.015*", "0.002*", "0.018*", "0.013*"],
          ["Sentiment, Qwen L15", "0.066", "0.004*", "0.088", "0.144"],
          ["Sentiment, Qwen L18", "0.029*", "0.003*", "0.044*", "0.111"],
          ["Toxicity, Qwen L10", "0.373", "0.036*", "0.075", "0.423"]]

CAP1 = ("<b>Table 1.</b> p-values, reserved block vs size-matched random block, paired over "
        "seeds at a matched 10% perplexity budget. * p &lt; 0.05. Contrast the &tau; = 0.10 "
        "column with the final one: removing the buffer removes the effect.")

RESULTS_B = ("<b>(b) The neutral buffer is the mechanism.</b> In the controlled synthetic "
             "setting (20 seeds), specificity &mdash; the ratio of damage from ablating the "
             "reserved block to damage from ablating a random block of equal width &mdash; "
             "falls as the buffer shrinks (strictly monotonically at &lambda; = 5), and is ~1 "
             "throughout when the penalty is disabled.")

TABLE2 = [["Buffer ratio", "τ", "Penalty off", "Penalty on"],
          ["4.33", "0.10", "1.6×", "16.6×"],
          ["1.67", "0.20", "1.3×", "4.0×"],
          ["0.78", "0.30", "1.1×", "1.6×"],
          ["0.33", "0.40", "1.0×", "1.6×"],
          ["0.00 (none)", "0.50", "1.1×", "0.9×"]]

CAP2 = ("<b>Table 2.</b> Ablation specificity against buffer width. Removing the buffer "
        "returns the method to chance. The same experiment shows a double dissociation: "
        "B-block ablation costs the unwanted class 0.447 accuracy while <i>gaining</i> 0.184 "
        "on the wanted class; G-block ablation mirrors it; a size-matched random block does "
        "neither.")

REST = [
    "<b>(c) Scope.</b> Under identical post-hoc selection at matched width, a CG-SAE "
    "dictionary steers no better than a vanilla SAE on GPT-2 (0 of 4 tests, p = "
    "0.24&ndash;0.40) but better on Qwen2.5-1.5B, where the difference is positive in 8 of 8 "
    "tests and significant in 6 (3 after Holm correction), across both concept axes. This "
    "rests on two model sizes, so we read it as scale-associated. Against "
    "difference-in-means steering, with SAE arms fixed in advance, the picture is "
    "task-dependent: the direction method wins every sentiment comparison, while on toxicity "
    "no SAE arm is significantly worse and three are significantly better (e.g. GPT-2 SCAR "
    "+0.194 vs +0.130, p = 0.022; Qwen CG-SAE +0.127 vs +0.044, p = 0.007), consistent with "
    "the task-dependence reported in [10].",

    "<b>(d) Two methodological findings.</b> First, the cross-group leakage metric used "
    "throughout this literature is uninformative: across twelve configurations its "
    "correlation with whether the block actually encodes the concept is r = +0.03, and the "
    "best-scoring configuration in the sweep (leakage 0.002) has a 96% dead block. Leakage "
    "is also scale-gameable &mdash; multiplying all activations by 0.01 divides it by 100 "
    "while changing nothing about separation. We report a routing measure and a "
    "block-occupancy measure alongside it. Second, selecting an intervention layer by probe "
    "accuracy is suboptimal: in all four layer scans the two optima differ. On GPT-2 probe "
    "AUC peaks at the final layer (0.816) whereas steering efficiency peaks at layer 7, and "
    "choosing by decodability forfeits 45% of achievable efficiency.",
]

LIMITS = ("The method does not improve steering universally; on sentiment a "
          "difference-in-means baseline outperforms every SAE-based arm. The scale-dependence "
          "claim is consistent across both axes but rests on only two model sizes. "
          "The routing "
          "objective costs reconstruction fidelity (FVU 0.16 &rarr; 0.25). The approach "
          "operates on activations at inference and does not modify model weights. Results "
          "cover two binary concept axes; multi-concept partitioning is untested.")

CONCL = ("Reserving a labelled block of SAE latent capacity for an unwanted behaviour, with "
         "an unpenalised buffer separating it from the rest, produces a latent region that "
         "demonstrably contains the concept and can be removed by address rather than by "
         "search. The buffer, rather than the partition itself, is what makes the region "
         "specific. We also show that the standard evaluation metric in this area cannot "
         "distinguish a working partition from an empty one, which suggests existing "
         "supervised-SAE results would benefit from re-examination under an occupancy-aware "
         "measure.")

REFS = [
    "Bricken, T., et al. Towards Monosemanticity. Transformer Circuits Thread, 2023.",
    "Cunningham, H., et al. Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600, 2023.",
    "Gao, L., et al. Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093, 2024.",
    "Scar: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs. arXiv:2411.07122, 2024.",
    "SAE-SSV: Supervised Steering in Sparse Representation Spaces. arXiv:2505.16188, 2025.",
    "Graph-Regularized Sparse Autoencoders for LLM Safety Steering. arXiv:2512.06655.",
    "Disentangled Sparse Representations for Concept-Separated Unlearning. arXiv:2605.12122.",
    "Turner, A., et al. Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248, 2023.",
    "Arditi, A., et al. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS, 2024.",
    "Comparative Analysis of Sparse Autoencoder and Activation Difference Steering. arXiv:2510.01246, 2025.",
    "Elhage, N., et al. Toy Models of Superposition. Transformer Circuits Thread, 2022.",
]


def styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                                fontSize=14, leading=17, textColor=ACCENT, spaceAfter=6),
        "auth": ParagraphStyle("a", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=10.5, leading=13, alignment=TA_CENTER, spaceAfter=1),
        "affil": ParagraphStyle("af", parent=ss["Normal"], fontName="Times-Italic",
                                fontSize=9, leading=11, alignment=TA_CENTER,
                                textColor=GREY, spaceAfter=1),
        "track": ParagraphStyle("tr", parent=ss["Normal"], fontName="Times-Bold",
                                fontSize=9, leading=11, alignment=TA_CENTER,
                                textColor=GREY, spaceAfter=8),
        "h": ParagraphStyle("h", parent=ss["Normal"], fontName="Times-Bold",
                            fontSize=11, leading=13, textColor=ACCENT,
                            spaceBefore=5.5, spaceAfter=2.5),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=9.1, leading=11.0, alignment=TA_JUSTIFY, spaceAfter=3.5),
        "abs": ParagraphStyle("ab", parent=ss["Normal"], fontName="Times-Roman",
                              fontSize=8.8, leading=11, alignment=TA_JUSTIFY,
                              leftIndent=10, rightIndent=10, spaceAfter=3),
        "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Times-Roman",
                              fontSize=7.8, leading=9.6, alignment=TA_JUSTIFY,
                              textColor=GREY, spaceAfter=5),
        "ref": ParagraphStyle("r", parent=ss["Normal"], fontName="Times-Roman",
                              fontSize=7.2, leading=8.7, leftIndent=11,
                              firstLineIndent=-11, spaceAfter=1),
    }


def mk_table(data, widths):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def build(out: str) -> None:
    S = styles()
    doc = SimpleDocTemplate(out, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=12 * mm, bottomMargin=10 * mm,
                            title="CARE 2026 Extended Abstract", author=AUTHORS)
    f = []
    f.append(Paragraph(TITLE, S["title"]))
    f.append(Paragraph(AUTHORS, S["auth"]))
    f.append(Paragraph(AFFIL, S["affil"]))
    f.append(Paragraph(TRACK, S["track"]))

    f.append(Paragraph("<b>Abstract</b>", S["h"]))
    f.append(Paragraph(ABSTRACT, S["abs"]))
    f.append(Paragraph(f"<b>Keywords:</b> {KEYWORDS}", S["abs"]))

    for head, paras in SECTIONS:
        f.append(Paragraph(head, S["h"]))
        for p in paras:
            f.append(Paragraph(p, S["body"]))

    f.append(Paragraph("4.  Results", S["h"]))
    f.append(Paragraph(RESULTS_A, S["body"]))
    f.append(KeepTogether([mk_table(TABLE1, [38 * mm, 18 * mm, 18 * mm, 18 * mm, 30 * mm]),
                           Spacer(1, 3), Paragraph(CAP1, S["cap"])]))
    f.append(Paragraph(RESULTS_B, S["body"]))
    f.append(KeepTogether([mk_table(TABLE2, [28 * mm, 16 * mm, 26 * mm, 26 * mm]),
                           Spacer(1, 3), Paragraph(CAP2, S["cap"])]))
    for p in REST:
        f.append(Paragraph(p, S["body"]))

    f.append(Paragraph("5.  Limitations", S["h"]))
    f.append(Paragraph(LIMITS, S["body"]))
    f.append(Paragraph("6.  Conclusion", S["h"]))
    f.append(Paragraph(CONCL, S["body"]))

    f.append(Paragraph("References", S["h"]))
    for i, r in enumerate(REFS, 1):
        f.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", S["ref"]))

    doc.build(f)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../CARE2026_Abstract.pdf")
    build(ap.parse_args().out)
