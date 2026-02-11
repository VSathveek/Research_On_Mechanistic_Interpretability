"""
TransformerLens Practical Examples
Complete runnable code for your presentation

Run these in order to understand each concept
"""

# ============================================================================
# EXAMPLE 1: Basic Setup and Model Loading
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Loading Model and Basic Inference")
print("=" * 70)

import torch
from transformer_lens import HookedTransformer
import numpy as np

# Load GPT-2 Small (124M parameters)
print("\nLoading GPT-2 Small...")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,  # Center the unembedding matrix
    center_writing_weights=True,  # Center weights that write to residual stream
    fold_ln=True,  # Fold layer norm into weights for cleaner analysis
)

print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")
print(f"Hidden dimension: {model.cfg.d_model}")
print(f"Vocabulary size: {model.cfg.d_vocab}")

# Simple inference
text = "The capital of France is"
print(f"\nInput: '{text}'")

# Tokenize
tokens = model.to_tokens(text)
print(f"Tokens shape: {tokens.shape}")
print(f"Tokens: {model.to_str_tokens(text)}")

# Forward pass
logits = model(tokens)
print(f"Logits shape: {logits.shape}")  # [batch, seq_len, vocab_size]

# Get predictions
probs = torch.softmax(logits[0, -1], dim=-1)
top_k = 5
top_probs, top_indices = torch.topk(probs, top_k)

print(f"\nTop {top_k} predictions:")
for i in range(top_k):
    token = model.to_string(top_indices[i])
    prob = top_probs[i].item()
    print(f"  {i+1}. '{token}' - {prob*100:.2f}%")


# ============================================================================
# EXAMPLE 2: Understanding Hooks and Caching
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Activation Caching")
print("=" * 70)

text = "When Mary and John went to the store, John gave a drink to"
print(f"\nInput: '{text}'")

# Run with cache - stores ALL intermediate activations
logits, cache = model.run_with_cache(text)

print(f"\nCache contains {len(cache)} activation tensors")
print("\nSample cache keys:")
for i, key in enumerate(list(cache.keys())[:10]):
    shape = cache[key].shape
    print(f"  {key}: {shape}")

# Access specific activations
print("\n--- Embedding Layer ---")
embed = cache['hook_embed']
print(f"Token embeddings shape: {embed.shape}")  # [batch, seq_len, d_model]

print("\n--- Attention Patterns ---")
# Attention pattern from Layer 5, Head 7
attn_pattern = cache['blocks.5.attn.hook_pattern']
print(f"Attention patterns shape: {attn_pattern.shape}")  # [batch, n_heads, seq_len, seq_len]

# Get specific head
head_pattern = attn_pattern[0, 7]  # batch 0, head 7
print(f"Single head pattern shape: {head_pattern.shape}")  # [seq_len, seq_len]

print("\n--- MLP Activations ---")
mlp_output = cache['blocks.3.hook_mlp_out']
print(f"MLP output (Layer 3) shape: {mlp_output.shape}")

print("\n--- Residual Stream ---")
# The residual stream at different points
resid_pre = cache['blocks.0.hook_resid_pre']  # Before first layer
resid_post = cache['blocks.11.hook_resid_post']  # After last layer
print(f"Residual stream (start) shape: {resid_pre.shape}")
print(f"Residual stream (end) shape: {resid_post.shape}")


# ============================================================================
# EXAMPLE 3: Visualizing Attention Patterns
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Attention Pattern Analysis")
print("=" * 70)

text = "The cat sat on the mat and the dog sat on the"
tokens_list = model.to_str_tokens(text)
logits, cache = model.run_with_cache(text)

print(f"\nTokens: {tokens_list}")

# Choose a layer and head to analyze
layer = 9
head = 9

attn_pattern = cache[f'blocks.{layer}.attn.hook_pattern'][0, head]
print(f"\nAnalyzing Layer {layer}, Head {head}")
print(f"Attention pattern shape: {attn_pattern.shape}")

# Show which token each position attends to most
print("\nMost attended token for each position:")
for i, token in enumerate(tokens_list):
    attended_idx = attn_pattern[i].argmax().item()
    attended_token = tokens_list[attended_idx]
    attention_weight = attn_pattern[i, attended_idx].item()
    print(f"  {token:15s} -> {attended_token:15s} (weight: {attention_weight:.3f})")

# Save visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
plt.imshow(attn_pattern.detach().cpu().numpy(), cmap='Blues', aspect='auto')
plt.xticks(range(len(tokens_list)), tokens_list, rotation=90)
plt.yticks(range(len(tokens_list)), tokens_list)
plt.xlabel('Key Position (what we attend TO)')
plt.ylabel('Query Position (what is attending)')
plt.title(f'Attention Pattern - Layer {layer}, Head {head}')
plt.colorbar(label='Attention Weight')
plt.tight_layout()
plt.savefig('C:/Users/ARAVIND/OneDrive/Desktop/Research_On_Mechanistic_Interpretability/attention_pattern1.png', dpi=150, bbox_inches='tight')
print(f"\nSaved attention visualization to attention_pattern1.png")


# ============================================================================
# EXAMPLE 4: Activation Patching - Finding Important Components
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Activation Patching (Causal Intervention)")
print("=" * 70)

# Setup: Find which heads matter for predicting "Paris"
clean_text = "The Eiffel Tower is in Paris"
corrupted_text = "The Eiffel Tower is in London"

print(f"Clean prompt: '{clean_text}'")
print(f"Corrupted prompt: '{corrupted_text}'")

# Get caches for both
clean_logits, clean_cache = model.run_with_cache(clean_text)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_text)

# Define metric: logit difference between Paris and London
paris_token = model.to_single_token(" Paris")
london_token = model.to_single_token(" London")

def get_logit_diff(logits):
    """Difference between Paris and London logits"""
    return logits[0, -1, paris_token] - logits[0, -1, london_token]

clean_diff = get_logit_diff(clean_logits)
corrupted_diff = get_logit_diff(corrupted_logits)

print(f"\nClean logit difference (Paris - London): {clean_diff:.2f}")
print(f"Corrupted logit difference (Paris - London): {corrupted_diff:.2f}")

# Patch each attention head and measure effect
print("\nPatching attention heads...")
results = {}

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        
        # Create hook that patches this specific head
        def patch_head_hook(activation, hook, head_idx=head):
            # Replace corrupted activation with clean activation for this head
            activation[:, :, head_idx, :] = clean_cache[hook.name][:, :, head_idx, :]
            return activation
        
        # Run corrupted input with this head patched
        patched_logits = model.run_with_hooks(
            corrupted_text,
            fwd_hooks=[(f"blocks.{layer}.attn.hook_result", patch_head_hook)]
        )
        
        # Measure how much the patch recovers the clean behavior
        patched_diff = get_logit_diff(patched_logits)
        recovery_fraction = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
        
        results[(layer, head)] = recovery_fraction.item()

# Find most important heads
sorted_results = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 10 most important heads:")
for i, ((layer, head), recovery) in enumerate(sorted_results[:10]):
    print(f"  {i+1}. Layer {layer:2d}, Head {head:2d}: {recovery:+.2%} recovery")

# Visualize results as heatmap
import matplotlib.pyplot as plt
import numpy as np

# Create heatmap
heatmap_data = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
for (layer, head), recovery in results.items():
    heatmap_data[layer, head] = recovery

plt.figure(figsize=(12, 8))
plt.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
plt.colorbar(label='Recovery Fraction')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.title('Attention Head Importance via Activation Patching')
plt.xticks(range(model.cfg.n_heads))
plt.yticks(range(model.cfg.n_layers))

# Mark top heads
for i, ((layer, head), recovery) in enumerate(sorted_results[:3]):
    plt.text(head, layer, f'{i+1}', ha='center', va='center', 
             color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('C:/Users/ARAVIND/OneDrive/Desktop/Research_On_Mechanistic_Interpretability/attention_pattern2.png', dpi=150, bbox_inches='tight')
print("\nSaved activation patching heatmap to attention_pattern2.png")


# ============================================================================
# EXAMPLE 5: Residual Stream Decomposition
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Residual Stream Decomposition")
print("=" * 70)

text = "The quick brown fox"
tokens_list = model.to_str_tokens(text)
print(f"\nInput: '{text}'")
print(f"Tokens: {tokens_list}")

logits, cache = model.run_with_cache(text)

# Collect all components that write to the residual stream
components = {}
component_names = []

# Embeddings
components['embed'] = cache['hook_embed']
component_names.append('embed')

components['pos_embed'] = cache['hook_pos_embed']
component_names.append('pos_embed')

# Attention and MLP outputs for each layer
for layer in range(model.cfg.n_layers):
    attn_key = f'attn_L{layer}'
    mlp_key = f'mlp_L{layer}'
    
    components[attn_key] = cache[f'blocks.{layer}.hook_attn_out']

    components[mlp_key] = cache[f'blocks.{layer}.hook_mlp_out']
    
    component_names.append(attn_key)
    component_names.append(mlp_key)

# Stack all components
stacked_components = torch.stack([components[name] for name in component_names])
print(f"\nStacked components shape: {stacked_components.shape}")
# Shape: [num_components, batch, seq_len, d_model]

# Verify reconstruction
final_residual = cache[f'blocks.{model.cfg.n_layers-1}.hook_resid_post']
reconstructed = stacked_components.sum(dim=0)
reconstruction_error = (final_residual - reconstructed).abs().max().item()
print(f"Reconstruction error: {reconstruction_error:.2e}")

# Analyze contribution to final token prediction
target_token = model.to_single_token(" fox")
print(f"\nAnalyzing contributions to predicting: '{model.to_string(target_token)}'")

# Get last token's residual stream components
last_token_components = stacked_components[:, 0, -1, :]  # [num_components, d_model]

# Project onto unembedding direction for target token
W_U = model.W_U  # [d_model, d_vocab]
contributions = last_token_components @ W_U[:, target_token]

print("\nTop 10 components contributing to ' fox':")
component_contributions = list(zip(component_names, contributions.tolist()))
sorted_contributions = sorted(component_contributions, key=lambda x: abs(x[1]), reverse=True)

for i, (name, contrib) in enumerate(sorted_contributions[:10]):
    print(f"  {i+1}. {name:15s}: {contrib:+.2f}")

# Visualize
plt.figure(figsize=(14, 6))
contributions_np = contributions.detach().cpu().numpy()
colors = ['green' if c > 0 else 'red' for c in contributions_np]
plt.bar(range(len(component_names)), contributions_np, color=colors, alpha=0.7)
plt.xticks(range(len(component_names)), component_names, rotation=90, fontsize=8)
plt.ylabel('Contribution to " fox" logit')
plt.title('Residual Stream Decomposition')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/ARAVIND/OneDrive/Desktop/Research_On_Mechanistic_Interpretability/residual_decomposition.png', dpi=150, bbox_inches='tight')
print("\nSaved residual decomposition plot to residual_decomposition.png")


# ============================================================================
# EXAMPLE 6: Detecting Induction Heads
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Induction Head Detection")
print("=" * 70)

# Induction heads copy previous occurrences of patterns
# Test with repeated sequence
text = "The cat sat on the mat. The cat sat on the"
tokens_list = model.to_str_tokens(text)
print(f"\nInput: '{text}'")
print(f"Tokens: {tokens_list}")

logits, cache = model.run_with_cache(text)

def calculate_induction_score(attn_pattern, prefix_len=3):
    """
    Induction heads show diagonal stripe pattern offset by prefix length.
    Score how much each position attends to position offset by prefix_len.
    """
    seq_len = attn_pattern.shape[-1]
    score = 0
    count = 0
    
    for i in range(prefix_len, seq_len):
        # Check if position i attends strongly to position i - prefix_len
        score += attn_pattern[i, i - prefix_len].item()
        count += 1
    
    return score / count if count > 0 else 0

# Check all heads for induction pattern
induction_scores = {}

print("\nCalculating induction scores for all heads...")
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern'][0, head]
        score = calculate_induction_score(pattern, prefix_len=3)
        induction_scores[(layer, head)] = score

# Find top induction heads
sorted_scores = sorted(induction_scores.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 potential induction heads:")
for i, ((layer, head), score) in enumerate(sorted_scores[:10]):
    print(f"  {i+1}. Layer {layer:2d}, Head {head:2d}: score {score:.3f}")

# Visualize the top induction head
top_layer, top_head = sorted_scores[0][0]
top_pattern = cache[f'blocks.{top_layer}.attn.hook_pattern'][0, top_head]

plt.figure(figsize=(14, 10))
plt.imshow(top_pattern.detach().cpu().numpy(), cmap='Blues', aspect='auto')
plt.xticks(range(len(tokens_list)), tokens_list, rotation=90, fontsize=10)
plt.yticks(range(len(tokens_list)), tokens_list, fontsize=10)
plt.xlabel('Key Position (attended TO)')
plt.ylabel('Query Position (attending FROM)')
plt.title(f'Top Induction Head - Layer {top_layer}, Head {top_head} (score: {sorted_scores[0][1]:.3f})')
plt.colorbar(label='Attention Weight')

# Highlight the diagonal offset pattern
for i in range(3, len(tokens_list)):
    plt.plot([i-3, i-3], [i-0.5, i+0.5], 'r-', linewidth=2, alpha=0.5)
    plt.plot([i-3-0.5, i-3+0.5], [i, i], 'r-', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('C:/Users/ARAVIND/OneDrive/Desktop/Research_On_Mechanistic_Interpretability/induction_head.png', dpi=150, bbox_inches='tight')
print(f"\nSaved induction head visualization to induction_head.png")


# ============================================================================
# EXAMPLE 7: Hook Functions for Custom Analysis
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: Custom Hook Functions")
print("=" * 70)

# Hook functions allow you to intervene during forward pass
text = "The capital of Germany is"
print(f"\nInput: '{text}'")

# Example 1: Print hook - observe activations
print("\n--- Using a Print Hook ---")
def print_attention_hook(activation, hook):
    """Hook that prints attention pattern statistics"""
    print(f"Hook name: {hook.name}")
    print(f"Activation shape: {activation.shape}")
    print(f"Mean attention: {activation.mean():.4f}")
    print(f"Max attention: {activation.max():.4f}")
    return activation  # Must return activation

# Run with hook on Layer 5, Head 3
model.run_with_hooks(
    text,
    fwd_hooks=[("blocks.5.attn.hook_pattern", print_attention_hook)]
)

# Example 2: Ablation hook - zero out a component
print("\n--- Using an Ablation Hook ---")

def ablate_head_hook(activation, hook, head_to_ablate=3):
    """Hook that zeros out a specific attention head"""
    activation[:, :, head_to_ablate, :] = 0
    return activation

# Run normally
normal_logits = model(text)
normal_probs = torch.softmax(normal_logits[0, -1], dim=-1)
normal_top5 = torch.topk(normal_probs, 5)

# Run with head ablated
ablated_logits = model.run_with_hooks(
    text,
    fwd_hooks=[("blocks.5.attn.hook_result", ablate_head_hook)]
)
ablated_probs = torch.softmax(ablated_logits[0, -1], dim=-1)
ablated_top5 = torch.topk(ablated_probs, 5)

print("\nTop predictions (normal):")
for i in range(5):
    token = model.to_string(normal_top5.indices[i])
    prob = normal_top5.values[i].item()
    print(f"  {i+1}. '{token}' - {prob*100:.2f}%")

print("\nTop predictions (with L5H3 ablated):")
for i in range(5):
    token = model.to_string(ablated_top5.indices[i])
    prob = ablated_top5.values[i].item()
    print(f"  {i+1}. '{token}' - {prob*100:.2f}%")


# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: TransformerLens Capabilities Demonstrated")
print("=" * 70)

print("""
✓ Model Loading - Load any HuggingFace transformer
✓ Activation Caching - Store all intermediate computations
✓ Attention Analysis - Visualize what tokens attend to what
✓ Activation Patching - Find causally important components
✓ Residual Stream Decomposition - Track information flow
✓ Induction Head Detection - Find pattern-copying behavior
✓ Custom Hooks - Intervene and modify computations

Generated Files:
  1. attention_pattern.png - Attention pattern visualization
  2. activation_patching.png - Head importance heatmap
  3. residual_decomposition.png - Component contributions
  4. induction_head.png - Induction head pattern

Key Takeaways:
  • TransformerLens makes transformers transparent
  • Hooks provide access to all internal activations
  • Activation patching reveals causal structure
  • Perfect tool for mechanistic interpretability research
""")

print("=" * 70)
print("Code execution complete!")
print("=" * 70)









"""
======================================================================
EXAMPLE 1: Loading Model and Basic Inference
======================================================================

Loading GPT-2 Small...
`torch_dtype` is deprecated! Use `dtype` instead!
Loaded pretrained model gpt2-small into HookedTransformer
Model loaded: 12 layers, 12 heads per layer
Hidden dimension: 768
Vocabulary size: 50257

Input: 'The capital of France is'
Tokens shape: torch.Size([1, 6])
Tokens: ['<|endoftext|>', 'The', ' capital', ' of', ' France', ' is']
Logits shape: torch.Size([1, 6, 50257])

Top 5 predictions:
  1. ' now' - 4.75%
  2. ' the' - 3.74%
  3. ' a' - 3.55%
  4. ' home' - 3.09%
  5. ' in' - 2.70%

======================================================================
EXAMPLE 2: Activation Caching
======================================================================

Input: 'When Mary and John went to the store, John gave a drink to'

Cache contains 208 activation tensors

Sample cache keys:
  hook_embed: torch.Size([1, 15, 768])
  hook_pos_embed: torch.Size([1, 15, 768])
  blocks.0.hook_resid_pre: torch.Size([1, 15, 768])
  blocks.0.ln1.hook_scale: torch.Size([1, 15, 1])
  blocks.0.ln1.hook_normalized: torch.Size([1, 15, 768])
  blocks.0.attn.hook_q: torch.Size([1, 15, 12, 64])
  blocks.0.attn.hook_k: torch.Size([1, 15, 12, 64])
  blocks.0.attn.hook_v: torch.Size([1, 15, 12, 64])
  blocks.0.attn.hook_attn_scores: torch.Size([1, 12, 15, 15])
  blocks.0.attn.hook_pattern: torch.Size([1, 12, 15, 15])

--- Embedding Layer ---
Token embeddings shape: torch.Size([1, 15, 768])

--- Attention Patterns ---
Attention patterns shape: torch.Size([1, 12, 15, 15])
Single head pattern shape: torch.Size([15, 15])

--- MLP Activations ---
MLP output (Layer 3) shape: torch.Size([1, 15, 768])

--- Residual Stream ---
Residual stream (start) shape: torch.Size([1, 15, 768])
Residual stream (end) shape: torch.Size([1, 15, 768])

======================================================================
EXAMPLE 3: Attention Pattern Analysis
======================================================================

Tokens: ['<|endoftext|>', 'The', ' cat', ' sat', ' on', ' the', ' mat', ' and', ' the', ' dog', ' sat', ' on', ' the']

Analyzing Layer 9, Head 9
Attention pattern shape: torch.Size([13, 13])

Most attended token for each position:
  <|endoftext|>   -> <|endoftext|>   (weight: 1.000)
  The             -> <|endoftext|>   (weight: 0.997)
   cat            -> <|endoftext|>   (weight: 0.994)
   sat            -> <|endoftext|>   (weight: 0.997)
   on             -> <|endoftext|>   (weight: 0.989)
   the            -> <|endoftext|>   (weight: 0.988)
   mat            -> <|endoftext|>   (weight: 0.991)
   and            -> <|endoftext|>   (weight: 0.974)
   the            -> <|endoftext|>   (weight: 0.941)
   dog            -> <|endoftext|>   (weight: 0.960)
   sat            -> <|endoftext|>   (weight: 0.939)
   on             -> <|endoftext|>   (weight: 0.979)
   the            -> <|endoftext|>   (weight: 0.960)

Saved attention visualization to attention_pattern1.png

======================================================================
EXAMPLE 4: Activation Patching (Causal Intervention)
======================================================================
Clean prompt: 'The Eiffel Tower is in Paris'
Corrupted prompt: 'The Eiffel Tower is in London'

Clean logit difference (Paris - London): 2.93
Corrupted logit difference (Paris - London): -3.58

Patching attention heads...

Top 10 most important heads:
  1. Layer  0, Head  0: +0.00% recovery
  2. Layer  0, Head  1: +0.00% recovery
  3. Layer  0, Head  2: +0.00% recovery
  4. Layer  0, Head  3: +0.00% recovery
  5. Layer  0, Head  4: +0.00% recovery
  6. Layer  0, Head  5: +0.00% recovery
  7. Layer  0, Head  6: +0.00% recovery
  8. Layer  0, Head  7: +0.00% recovery
  9. Layer  0, Head  8: +0.00% recovery
  10. Layer  0, Head  9: +0.00% recovery

Saved activation patching heatmap to attention_pattern2.png

======================================================================
EXAMPLE 5: Residual Stream Decomposition
======================================================================

Input: 'The quick brown fox'
Tokens: ['<|endoftext|>', 'The', ' quick', ' brown', ' fox']

Stacked components shape: torch.Size([26, 1, 5, 768])
Reconstruction error: 3.66e-04

Analyzing contributions to predicting: ' fox'

Top 10 components contributing to ' fox':
  1. mlp_L0         : +35.68
  2. attn_L11       : +15.11
  3. embed          : +13.44
  4. attn_L10       : +12.56
  5. attn_L0        : +9.99
  6. mlp_L1         : -6.36
  7. mlp_L9         : +6.28
  8. attn_L6        : +5.92
  9. attn_L7        : +5.41
  10. attn_L9        : +5.35

Saved residual decomposition plot to residual_decomposition.png

======================================================================
EXAMPLE 6: Induction Head Detection
======================================================================

Input: 'The cat sat on the mat. The cat sat on the'
Tokens: ['<|endoftext|>', 'The', ' cat', ' sat', ' on', ' the', ' mat', '.', ' The', ' cat', ' sat', ' on', ' the']

Calculating induction scores for all heads...

Top 10 potential induction heads:
  1. Layer  4, Head  1: score 0.175
  2. Layer 11, Head  3: score 0.166
  3. Layer  0, Head  8: score 0.165
  4. Layer 10, Head  4: score 0.162
  5. Layer  7, Head  9: score 0.162
  6. Layer  1, Head  1: score 0.160
  7. Layer  1, Head  2: score 0.156
  8. Layer  4, Head  9: score 0.155
  9. Layer 11, Head  0: score 0.152
  10. Layer 11, Head 10: score 0.150

Saved induction head visualization to induction_head.png

======================================================================
EXAMPLE 7: Custom Hook Functions
======================================================================

Input: 'The capital of Germany is'

--- Using a Print Hook ---
Hook name: blocks.5.attn.hook_pattern
Activation shape: torch.Size([1, 12, 6, 6])
Mean attention: 0.1667
Max attention: 1.0000

--- Using an Ablation Hook ---

Top predictions (normal):
  1. ' now' - 5.67%
  2. ' the' - 4.45%
  3. ' a' - 4.04%
  4. ' home' - 3.65%
  5. ' in' - 2.85%

Top predictions (with L5H3 ablated):
  1. ' now' - 5.67%
  2. ' the' - 4.45%
  3. ' a' - 4.04%
  4. ' home' - 3.65%
  5. ' in' - 2.85%

======================================================================
SUMMARY: TransformerLens Capabilities Demonstrated
======================================================================

✓ Model Loading - Load any HuggingFace transformer
✓ Activation Caching - Store all intermediate computations
✓ Attention Analysis - Visualize what tokens attend to what
✓ Activation Patching - Find causally important components
✓ Residual Stream Decomposition - Track information flow
✓ Induction Head Detection - Find pattern-copying behavior
✓ Custom Hooks - Intervene and modify computations

Generated Files:
  1. attention_pattern.png - Attention pattern visualization
  2. activation_patching.png - Head importance heatmap
  3. residual_decomposition.png - Component contributions
  4. induction_head.png - Induction head pattern

Key Takeaways:
  • TransformerLens makes transformers transparent
  • Hooks provide access to all internal activations
  3. ' a' - 4.04%
  4. ' home' - 3.65%
  5. ' in' - 2.85%

======================================================================
SUMMARY: TransformerLens Capabilities Demonstrated
======================================================================

✓ Model Loading - Load any HuggingFace transformer
✓ Activation Caching - Store all intermediate computations
✓ Attention Analysis - Visualize what tokens attend to what
✓ Activation Patching - Find causally important components
✓ Residual Stream Decomposition - Track information flow
✓ Induction Head Detection - Find pattern-copying behavior
✓ Custom Hooks - Intervene and modify computations

Generated Files:
  1. attention_pattern.png - Attention pattern visualization
  2. activation_patching.png - Head importance heatmap
  3. residual_decomposition.png - Component contributions
  4. induction_head.png - Induction head pattern

Key Takeaways:
  • TransformerLens makes transformers transparent
  • Hooks provide access to all internal activations
✓ Activation Caching - Store all intermediate computations
✓ Attention Analysis - Visualize what tokens attend to what
✓ Activation Patching - Find causally important components
✓ Residual Stream Decomposition - Track information flow
✓ Induction Head Detection - Find pattern-copying behavior
✓ Custom Hooks - Intervene and modify computations

Generated Files:
  1. attention_pattern.png - Attention pattern visualization
  2. activation_patching.png - Head importance heatmap
  3. residual_decomposition.png - Component contributions
  4. induction_head.png - Induction head pattern

Key Takeaways:
  • TransformerLens makes transformers transparent
  • Hooks provide access to all internal activations
  4. induction_head.png - Induction head pattern

Key Takeaways:
  • TransformerLens makes transformers transparent
  • Hooks provide access to all internal activations
  • TransformerLens makes transformers transparent
  • Hooks provide access to all internal activations
  • Hooks provide access to all internal activations
  • Activation patching reveals causal structure
  • Perfect tool for mechanistic interpretability research

======================================================================
Code execution complete!
======================================================================



"""