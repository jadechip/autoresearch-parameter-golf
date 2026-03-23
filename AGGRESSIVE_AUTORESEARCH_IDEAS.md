# Aggressive Autoresearch Ideas

This file defines the high-aggression campaign for the separate Ralph-style loop.

The loop should treat each item as one campaign story and spend exactly six training
attempts on it before moving to the next story.

The point of this campaign is not tiny local hill-climbing. The point is to use
Codex as a coding agent that can create or reshape architecture surfaces while
still preserving the keep/revert discipline and structured run accounting.

## Story 1: XSA + Neural Cache

- Focus on self-contained XSA or top-layer cross-window KV caching.
- Prefer caching only upper layers at first.
- Preserve causal scoring and evaluation-budget validity.
- Avoid rewriting the entire eval harness in the first six attempts.

## Story 2: Late Selective Compression-Aware Quantization

- Explore coarse tensor groups, not per-row mixed-bit complexity.
- Likely groups:
  - embeddings and head
  - early attention
  - late attention
  - middle MLPs
  - final MLPs
- Prefer later fake quant or post-quant selection over blanket early QAT.

## Story 3: Low-Rank Q Structural Reallocation

- Use factored or low-rank Q to buy more useful carrier capacity.
- Spend savings on depth, stronger local modules, or more steps.
- Do not stop at "low-rank Q alone" if a stronger reallocation is feasible.

## Story 4: Split Local-Context Families

- Keep two separate families instead of stacking everything:
  - XSA + EMA + partial RoPE + late QAT, with no TTT
  - SmearGate + smarter bigram + mixed low-bit quant + meta-TTT or causal TTT, with no XSA
- The loop should not blur these families into one hybrid unless earlier attempts prove the interaction is safe.

## Story 5: Compute-Aware Context Curriculum

- Default idea: shorter context early, then longer context later.
- Favor 1024 or 1536 to 2048 curricula over jumping to very long context from step 1.
- Reclaim compute through batch or carrier changes instead of brute force.

## Story 6: Smarter Local-Token Module

- Replace crude raw BigramHash scaling with better byte-quality tradeoffs.
- Good candidates:
  - top-N exact bigram table plus hashed tail
  - two-scale local features
  - factorized bigram projection
- Keep the implementation compact and submission-plausible.

## Story 7: Quantization-Optimized Checkpoint Soup

- Treat post-quant quality as the selection target.
- A viable branch is:
  - no QAT
  - denser checkpoint saves in warmdown
  - quantize snapshots
  - select or average the best post-quant snapshots

## Story 8: Localized Canon Inserts

- Use one or two localized Canon-style neighbor-mixing inserts.
- Prefer bottom or pre-final-XSA placements.
- Do not pivot the whole architecture to Canon in one shot.

## Story 9: Selective INT4 MLP Moonshot

- Only test selective INT4 on less-sensitive middle MLPs.
- Keep boundary and final MLPs at safer precision.
- Treat this as a late high-risk branch, not the first branch to refine.

## Campaign Default

The default serious branch shape is:

- artifact target around 15.2 MB to 15.9 MB
- 11 to 12 layer carrier
- d_model 512
- about 3x MLP
- seq_len 2048
- batch around 524K tokens

Default branch families:

- XSA branch
- XSA + neural cache branch
- low-rank Q branch
- Smear or meta-TTT branch
- context curriculum branch
