
TransformerLens Practical Examples
===================================

Research-Oriented Documentation
--------------------------------

This file contains a structured series of mechanistic interpretability
experiments using TransformerLens on GPT-2 Small (124M parameters).

The goal is to move from black-box usage of transformers to
white-box mechanistic understanding of internal circuits.

Each example demonstrates a progressively deeper level of analysis.

-------------------------------------------------------------------------------
BACKGROUND: MECHANISTIC INTERPRETABILITY
-------------------------------------------------------------------------------

Mechanistic interpretability aims to reverse-engineer neural networks by
identifying the exact computational circuits that implement behaviors.

Instead of asking:
    "What does the model output?"

We ask:
    "Which internal components cause this output?"
    "How does information flow through the residual stream?"
    "Which attention heads implement specific algorithms?"

TransformerLens provides:
    • Full activation access
    • Hook-based intervention
    • Causal patching mechanisms
    • Residual stream decomposition tools

This file demonstrates these capabilities in increasing depth.

-------------------------------------------------------------------------------
EXPERIMENT 1 — MODEL LOADING & BASELINE INFERENCE
-------------------------------------------------------------------------------

Objective:
    Establish baseline inference behavior and inspect model structure.

Research Relevance:
    Before analyzing circuits, we must confirm:
        • Model architecture (layers, heads, dimensions)
        • Tokenization behavior
        • Output logits and probability distribution

Key Concepts:
    - Logits represent unnormalized token predictions.
    - Softmax converts logits to probabilities.
    - Top-k sampling reveals dominant next-token predictions.

This serves as a sanity check before mechanistic analysis.

-------------------------------------------------------------------------------
EXPERIMENT 2 — ACTIVATION CACHING
-------------------------------------------------------------------------------

Objective:
    Capture every intermediate tensor during forward pass.

Research Relevance:
    Transformers operate via residual stream accumulation.
    To understand computation, we must inspect:

        • Token embeddings
        • Attention patterns
        • MLP outputs
        • Residual stream states

The residual stream acts as a shared communication bus.
Each layer writes additive updates to it.

Caching enables:
    - Circuit tracing
    - Head inspection
    - Residual decomposition
    - Causal patching

Without caching, mechanistic interpretability is impossible.

-------------------------------------------------------------------------------
EXPERIMENT 3 — ATTENTION PATTERN ANALYSIS
-------------------------------------------------------------------------------

Objective:
    Visualize how tokens attend to one another.

Research Relevance:
    Attention heads implement specific algorithms such as:
        • Name copying
        • Induction
        • Syntax tracking
        • Coreference resolution

The attention matrix:
    shape = [query_position, key_position]

Each row shows:
    "Where this token is looking."

Patterns to look for:
    - Diagonals → positional copying
    - Vertical stripes → global token tracking
    - Offset diagonals → induction heads

Visualization transforms raw tensors into interpretable structure.

-------------------------------------------------------------------------------
EXPERIMENT 4 — ACTIVATION PATCHING (CAUSAL INTERVENTION)
-------------------------------------------------------------------------------

Objective:
    Identify which attention heads causally influence prediction.

Method:
    1. Define clean prompt (correct answer).
    2. Define corrupted prompt (incorrect answer).
    3. Replace (patch) one component from clean into corrupted.
    4. Measure logit recovery.

Metric:
    Logit difference:
        logit("Paris") - logit("London")

Recovery Fraction:
    Measures how much patching restores correct behavior.

Research Importance:
    - Correlation ≠ Causation
    - Patching provides causal evidence
    - Identifies functional circuits

This is a standard mechanistic interpretability technique used in
state-of-the-art research.

-------------------------------------------------------------------------------
EXPERIMENT 5 — RESIDUAL STREAM DECOMPOSITION
-------------------------------------------------------------------------------

Objective:
    Decompose final prediction into additive component contributions.

Theory:
    Final residual stream =
        Embeddings
      + Positional embeddings
      + All attention outputs
      + All MLP outputs

We:
    - Stack all residual-writing components
    - Verify exact reconstruction
    - Project each component onto the unembedding direction
      of a target token

Interpretation:
    Contribution = dot(component, W_U[:, token])

This quantifies:
    "Which layer contributed how much to predicting this token?"

This is circuit-level attribution.

-------------------------------------------------------------------------------
EXPERIMENT 6 — INDUCTION HEAD DETECTION
-------------------------------------------------------------------------------

Objective:
    Detect induction heads (pattern-copying heads).

Induction behavior:
    If sequence A B C appears earlier,
    and later A B appears again,
    the head attends to previous C.

Visual signature:
    Offset diagonal stripe in attention matrix.

We define an induction score:
    Average attention weight to token at fixed offset.

High score → potential induction head.

Induction heads are critical for:
    - In-context learning
    - Few-shot generalization
    - Pattern completion

-------------------------------------------------------------------------------
EXPERIMENT 7 — CUSTOM HOOK INTERVENTIONS
-------------------------------------------------------------------------------

Objective:
    Demonstrate forward-pass intervention via hooks.

Two examples:
    1. Print hook → Observational
    2. Ablation hook → Causal removal

Ablation test:
    Zero out a head
    Compare top predictions

If predictions change significantly:
    The head is functionally important.

Hooks enable:
    - Circuit editing
    - Behavioral modification
    - Controlled experiments

-------------------------------------------------------------------------------
OVERALL CONTRIBUTION
-------------------------------------------------------------------------------

This file demonstrates:

    ✓ Structural inspection
    ✓ Full activation tracing
    ✓ Attention visualization
    ✓ Causal patching
    ✓ Residual decomposition
    ✓ Induction head detection
    ✓ Forward-pass intervention

These techniques form the foundation of
mechanistic interpretability research.

-------------------------------------------------------------------------------
INTENDED USE
-------------------------------------------------------------------------------

• Educational demonstration
• Research prototyping
• Circuit discovery experiments
• Interpretability presentations
• Mechanistic debugging

-------------------------------------------------------------------------------
REFERENCES
-------------------------------------------------------------------------------

Primary Library:
    TransformerLens (Neel Nanda)

Key Research Themes:
    - Induction Heads
    - Residual Stream as Communication Bus
    - Activation Patching
    - Circuit Discovery

-------------------------------------------------------------------------------

Author: Varanasi Sathveek
Purpose: Research & Presentation Demonstration
Model: GPT-2 Small (124M)
Framework: PyTorch + TransformerLens

"""
