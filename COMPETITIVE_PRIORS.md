# Competitive Priors

Use this file as a compact research brief before starting a new autoresearch search block.

## Recovered 5090 Baseline

The last trusted accepted 5090 state from the March 21 handoff was:

- commit: `dddc4c6`
- run: `ar5090-20260321-073916`
- `val_bpb = 1.563742695`
- `artifact_bytes = 7,257,739`

That recovered line is a compact `seq_len=768` model with:

- `stem_layers=0`
- `shared_layers=1`
- `recurrence_loops=1`
- `tail_layers=3`
- `mlp_mult=2`
- `shared_mlp_hidden_bonus=0`
- `non_recurrent_mlp_hidden_bonus=3072` (`8x` unique-tail MLP total width at `d_model=512`)
- broad rank-8 adapters on all allowed targets
- `fake_quant_start_step=20`
- `clip_percentile=96.5`
- MLP-only int6 export

The search reached this by:

1. moving from a heavy recurrent deep-tail line into a much faster compact line
2. spending capacity on unique tail MLP width instead of more repeated recurrence
3. extending context from `640` to `768`
4. rebalancing `1x2 + tail=2 + 12x tail` into `1x1 + tail=3 + 8x tail`

## What Seems Repeatedly Good

- Deliberate depth-to-width reallocation can win when it also improves fixed-budget throughput.
- Larger unique-tail MLPs matter more than broad global `mlp_mult` changes.
- Compact models that fit more steps into the fixed 5090 budget can dominate heavier lines.
- Longer context helped up to `seq_len=768`.
- The current accepted line is still small relative to the hard `16 MB` cap, so there is room to spend bytes deliberately.

## What Seems Repeatedly Bad

- Blind `d_model` widening on the compact line.
- Full `num_kv_heads=8` on the compact line.
- Global `mlp_mult=3` on the compact line, even when compensated.
- Near-neighbor context increases above `768` without reclaiming compute first.
- Pure tail-width nudges after the `1x1 + tail=3 + 8x` rebalance.
- Long streaks of fake-quant or optimizer micro-tunes without a larger structural hypothesis.

## Frontier Gap Versus The Leaderboard

The current leaderboard frontier is much better than this line and clusters around:

- 10-11 layers
- MLP about `2.6x` to `3x`
- int6 or mixed int5/int6 quantization
- sliding-window eval
- fp16 or selective higher-precision embeddings / head
- tuned Muon / weight decay / SWA schedules
- in several entries: BigramHash, SmearGate, OrthoInit, or related train-time architectural helpers

Our search has covered some of this, but not enough. In particular, we have not yet seriously explored:

- mixed low-bit quantization beyond MLP-only int6 export
- a clean leaderboard-style `~3x` MLP family that still fits the 5090 budget
- sliding-window eval
- stronger optimizer bundles with WD / SWA after choosing a structural candidate
- train-time init / gating ideas like OrthoInit or smear-like residual gating

## Next Search Directions

Prioritize larger, leaderboard-informed experiments over more local hill-climbing:

1. Spend bytes upward from the recovered `~7.3 MB` baseline toward roughly `9 MB` to `14 MB`, but only with a clear structural hypothesis.
2. Explore mixed low-bit quantization on more matrices so saved bytes can be reinvested into width or unique depth.
3. Revisit wider MLPs, but selectively:
   - unique-tail-only
   - specific repeated/shared blocks
   - compensated depth/width reallocations
   Do not just set global `mlp_mult=3` on the whole stack again.
4. Explore context/eval as a first-class axis:
   - longer `seq_len` if compute is reclaimed elsewhere
   - sliding-window eval if it remains competition-valid
5. Explore selective higher precision for embeddings / head if the byte cost is justified.
6. Explore frontier-style train-time helpers that still fit in `train.py`, such as better initialization, residual mixing, or smear-like gating.
7. Only after a stronger structural candidate exists, run optimizer bundles with Muon momentum, weight decay, warmdown, and possibly SWA.

## Search Guardrails

- Do not spend the next search block on tiny hidden-bonus retunes.
- Do not assume the smallest artifact is the best model. The recovered baseline materially under-spends the hard cap.
- Prefer experiments that could plausibly matter on `8xH100`, not just on the 5090 proxy.
