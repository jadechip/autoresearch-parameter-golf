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

## Regime Change

This compact line is a useful local optimum, but it is now in the wrong regime for a likely win.

The current frontier is concentrated much closer to the hard artifact ceiling with:

- near-full-budget carriers
- more unique depth
- more aggressive low-bit compression
- longer context and eval-time tricks

So the goal of the next 5090 campaign is not to keep polishing or shrinking this compact line. The goal is to move onto a stronger carrier family while keeping the fixed-budget 5090 loop honest.

## What Seems Repeatedly Good In This Repo

- Deliberate depth-to-width reallocation can win when it also improves fixed-budget throughput.
- Larger unique-tail MLPs matter more than broad global `mlp_mult` changes.
- Compact models that fit more steps into the fixed 5090 budget can dominate heavier lines.
- Longer context helped up to `seq_len=768`.
- The current accepted line is still materially below the hard `16 MB` cap, so there is room to spend bytes deliberately.

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

Our search has covered some of this, but not enough. In particular, we have not yet seriously explored enough of the following clean branch families:

- packed mixed low-bit quantization beyond MLP-only int6 export
- a clean leaderboard-style near-full-budget carrier that still fits the 5090 budget
- low-rank Q as a structural reallocation tool
- compute-aware context curricula
- sliding-window eval
- stronger optimizer bundles with WD / SWA after choosing a structural candidate
- train-time init / gating ideas like OrthoInit or smear-like residual gating
- local activation / RoPE / residual-scale variants that remain self-contained inside `train.py`
- explicit `config_transform_profile` control so creative branches are not silently rewritten into the old lineage
- per-pattern clip percentile sweeps for post-quant robustness

## Actionable Campaign Branches

Do not stack every local-context trick into one line. Split the next campaign into a few clean branches.

### Branch A: Near-Full-Budget Carrier

Default target:

- artifact band roughly `12 MB` to `15.5 MB`
- low recurrence (`0` or `1` loop)
- more unique depth
- `d_model=512`
- meaningfully wider MLPs
- `seq_len` trending toward `1024`, `1536`, then `2048` if compute allows

Primary knobs:

- unique depth vs repeated depth
- selective MLP width allocation
- batch shrink for more steps
- context curriculum

### Branch B: Late Selective Quantization

Focus on coarse, compression-friendly groups rather than fancy heterogeneous precision:

- embeddings / head
- early attention
- late attention
- middle MLPs
- final MLPs

The promising direction is:

- later fake quant, not earlier
- selective groups, not blanket QAT
- coarse schemes that still compress well under zlib / zstd-style packing

Also consider a no-QAT branch with checkpoint soup / PTQ selection under post-quant proxy loss.

### Branch C: Low-Rank Q Reallocation

This is the cleanest structural differentiation axis that still looks plausible:

- factor Q to buy either more unique depth, a better local module, or more steps
- combine with a stronger carrier rather than using it alone

### Branch D: Smarter Local-Token Module

Treat this as a separate branch, not something to stack on everything else immediately:

- top-N exact bigram plus hashed tail
- factorized bigram projection
- simple two-scale local token features

Do not assume raw BigramHash scaling is the best byte/quality trade.

## Feasibility Tiers In This Repo

Immediate / current `train.py` scope:

- low-rank Q
- late selective coarse-group quantization
- selective embedding / head precision
- compute-aware batch / context curricula
- checkpoint soup or selective post-quant snapshot choice
- simpler local-token modules

Medium complexity:

- sliding-window eval
- partial RoPE variants
- train-time init or gating changes

Module-writing campaigns that the loop may now seed deliberately when recent history is too narrow:

- XSA
- cross-window neural cache / top-layer KV cache
- SmearGate + meta-TTT branch
- Canon inserts

These are no longer "never touch" ideas. They are branch stories that should be attempted one at a time, with minimal self-contained implementations rather than blended stacks.

## Next Search Directions

For the next serious 5090 restart:

1. Spend bytes upward from the recovered `~7.3 MB` baseline toward roughly `12 MB` to `15.5 MB`, not `8 MB`.
2. Stop treating recurrence or tiny-model compression as the main axis.
3. Prefer moving onto a stronger carrier before stacking eval-time tricks.
4. Explore low-rank Q, selective wider MLPs, compute-aware context, and train.py-local activation or RoPE variants, but one branch at a time.
5. Use late selective quantization as a second-stage optimization on top of a better carrier, not the first move.
6. Treat sliding eval and cache ideas as separate branches, not automatic add-ons.

## Search Guardrails

- Do not spend the next search block on tiny hidden-bonus retunes.
- Do not assume the smallest artifact is the best model. The recovered baseline materially under-spends the hard cap.
- Prefer experiments that could plausibly matter on `8xH100`, not just on the 5090 proxy.
- Do not stack every local-context trick into one line.
- Do not keep revisiting recurrence, shared-core looping, or tiny-model compression as a main line of attack.
- Avoid doc-isolated eval at stride 64 as a standalone trick.

## Current Loop Failure Mode

The current Codex loop can drift into narrow exploitation of one accepted carrier, especially:

- selective float / fake-quant toggles on one or two late tensors
- repeated narrowing of which final-tail tensors stay float
- near-neighbor batch tweaks without a carrier change

Treat those as micro-tunes, not structural exploration.

If the recent git history is dominated by one family for 3 or more consecutive runs, the next run should be a forced branch switch into one of:

- carrier repartition / depth / width
- low-rank Q placement or strength
- batch / context curriculum
- local-token module
- a qualitatively different selective-quantization branch
- XSA or top-layer cache
- SmearGate / TTT
- Canon or neighboring-token mixer

## Ralph-Style Campaign Rule

Treat the search as a story board, not as free-form mutation roulette:

- Pick exactly one named campaign story per iteration.
- If recent history is narrow, switch stories before refining.
- If a story requires a missing self-contained module in `train.py`, implement the module instead of downgrading the story to a precision micro-tune.
- Keep module-writing stories isolated so the result is interpretable and easy to revert.


## Discovery Versus Refinement

Use the aggressive loop to discover new basins and the ordinary loop to refine only those branches that are already ranking-valid and plausibly contender-like. Do not spend the aggressive loop polishing a branch that should instead move into post-quant, checkpoint-choice, or carrier-local refinement.
