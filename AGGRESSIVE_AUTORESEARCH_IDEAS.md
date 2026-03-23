# Aggressive Autoresearch Ideas

This file defines the high-aggression campaign for the separate Ralph-style loop.

The loop should treat each item as one campaign story and spend exactly six training
attempts on it before moving to the next story.

The six attempts are not six polish steps on one branch. They are six materially
different architecture variants of the same story.

The point of this campaign is not tiny local hill-climbing. The point is to use
Codex as a coding agent that can create or reshape architecture surfaces while
still preserving the keep/revert discipline and structured run accounting.

## Architecture Sprint Rule

For every story:

- Attempts 1 through 6 should be treated as variants A through F, not as a local refinement ladder.
- A valid attempt must either:
  - introduce the named module or mechanism if it is not already present, or
  - change at least two macro axes from the accepted aggressive branch.
- Useful macro axes include:
  - shared vs tail partition
  - effective depth
  - `d_model`
  - `seq_len` bucket or curriculum shape
  - batch or grad-accum regime
  - local-module family
  - Q-rank topology
  - coarse quantization grouping or schedule

Disallowed attempt types unless they are only fixing a crash blocker in an otherwise valid new variant:

- canonicalize, cleanup, restore, repair, or path-only commits
- shifting float or QAT budget around within the same carrier
- funding a same-family width tweak from a tiny Q compression
- adding a second copy of the same local module on the same outer carrier
- branch-tip "restore" attempts that merely move back toward the current winner

The first four attempts of a story should be the most different variants, not the safest ones.

## Story 1: XSA + Neural Cache

- Focus on self-contained XSA or top-layer cross-window KV caching.
- Prefer caching only upper layers at first.
- Preserve causal scoring and evaluation-budget validity.
- Avoid rewriting the entire eval harness in the first six attempts.
- At least three attempts should implement real XSA or cache behavior, not just prepare the carrier.

## Story 2: Late Selective Compression-Aware Quantization

- Explore coarse tensor groups, not per-row mixed-bit complexity.
- Likely groups:
  - embeddings and head
  - early attention
  - late attention
  - middle MLPs
  - final MLPs
- Prefer later fake quant or post-quant selection over blanket early QAT.
- Do not spend multiple attempts only slicing finer within the same late-tail float set.

## Story 3: Low-Rank Q Structural Reallocation

- Use factored or low-rank Q to buy more useful carrier capacity.
- Spend savings on depth, stronger local modules, or more steps.
- Do not stop at "low-rank Q alone" if a stronger reallocation is feasible.
- Do not spend attempts only on same-carrier tail-width tweaks funded by small Q cuts.
- Prefer depth, partition, batch/context, or local-module-family changes over "more of the same carrier."

## Story 4: Split Local-Context Families

- Keep two separate families instead of stacking everything:
  - XSA + EMA + partial RoPE + late QAT, with no TTT
  - SmearGate + smarter bigram + mixed low-bit quant + meta-TTT or causal TTT, with no XSA
- The loop should not blur these families into one hybrid unless earlier attempts prove the interaction is safe.
- Do not spend attempts on repair or canonicalization work that does not materially move one of the two families.

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
- Each attempt should introduce a distinct local-token module family or placement, not just another neighbor-mixer polish.

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
- Do not spend attempts on canonicalizing an already-existing neighbor path unless it is a crash fix for a new Canon placement.

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
