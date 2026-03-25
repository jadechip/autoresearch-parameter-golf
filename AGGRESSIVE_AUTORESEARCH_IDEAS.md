# Aggressive Autoresearch Ideas

This loop is for discovery, not for final polishing.

The right mindset is:

- establish viable basins quickly
- make each run interpretable
- preserve creativity, but spend it where the 5090 proxy can still teach you something

## Main Rule

Use a stable frontier seed and test **one primary novelty at a time**.

Allowed support change:

- one small supporting move that keeps the story honest, such as a clip override, a checkpoint-selection rule, or a tiny width repair

Not allowed by default:

- topology + quantization + curriculum + module-family changes in one 5-minute proxy
- rebuilding the carrier from scratch when the lane already provides a frontier seed config

## When A Family Is Alive

A family is alive when at least one run is:

- ranking-valid, or
- invalid but close enough that one obvious budget repair could plausibly save it

If a family is alive, the next attempt should usually be:

- a repair, or
- one adjacent refinement that preserves attribution

## When A Family Is Dead

A family should be cut early when it shows one of these patterns:

- repeated valid-but-weak runs
- a catastrophic score gap
- repeated invalid runs without a near-miss story
- the only way to improve it seems to be stacking several extra changes at once

## Current Search Priorities

High priority:

- stable seed + activation / partial-RoPE / residual-scale changes
- late selective quantization
- checkpoint choice and post-quant soup
- low-rank-Q with one clear spend
- localized Canon-style inserts

Lower priority:

- XSA or cache branches
- tiny local-token modules

Deprioritized for now:

- blanket selective INT4 MLP as the headline idea
- heavyweight local modules
- broad many-axis moonshots in one run

## Budget-Repair Rule

If a branch is numerically good but invalid:

- do not immediately switch ideas
- do not treat the invalid run as the frontier
- spend the next attempt on one clean budget repair if the branch still looks promising

## What Creativity Should Look Like Here

Good creativity:

- a new local activation family on a stable carrier
- a new coarse quant grouping on the same carrier
- low-rank-Q buying one extra layer or one cautious context step
- a single localized insert with clear placement

Bad creativity:

- changing everything at once so the result is unreadable
- cargo-culting a public winner exactly
- using novelty as an excuse to dodge what the recent results already ruled out
