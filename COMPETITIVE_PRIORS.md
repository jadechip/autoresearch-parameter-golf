# Competitive Priors

Use this file as the compact research brief before starting a new search block.

## Current Working View

The short 5090 proxy is not currently bottlenecked by basic training correctness. It is bottlenecked by search discipline.

Recent evidence points to four main process failures:

1. the harness was still anchoring from an old compact base config even when the lane policy talked about frontier-style search
2. many aggressive runs changed too many axes at once, so losses were hard to interpret
3. some families consumed too many attempts despite clearly weak early evidence
4. invalid or oversize runs were too easy to mistake for the frontier

## First-Principles Proxy Guidance

The 5-minute 5090 loop should act like a fast model-selection instrument, not like a full submission rehearsal.

That means:

- use a stable seed carrier that already lives near the intended regime
- test one primary novelty at a time
- compare branches on throughput, byte use, and final score
- repair near-miss invalid branches before inventing a new story
- kill obviously weak families early

## Practical Regime For This Repo

Treat this as the default practical 5090 frontier-search regime unless a branch explicitly says otherwise:

- 11-12 effective layers
- `d_model=512`
- near-cap artifact target, not tiny-model under-spend
- selective `~2.5x-3x` MLP allocation rather than blunt global widening
- `seq_len=768` as the stable anchor
- `seq_len=1024-1536` only when throughput is protected
- train-time PTQ/no-QAT or late coarse-group QAT as higher-priority than blanket mixed-bit experiments

## What Currently Looks Promising

Based on recent runs and failure modes in this repo, the best next families are:

- stable frontier-seed carriers plus one local novelty
- post-quant checkpoint choice and checkpoint soup
- late selective quantization on stable carriers
- low-rank Q when it funds one obvious spend and stays under the cap
- localized Canon-style inserts on otherwise stable carriers

## What Currently Looks Weak

These directions are currently lower EV on this proxy and should be attempted later or with fewer tries:

- selective INT4 MLP as the headline story
- heavyweight local-token modules
- blended topology + quantization + curriculum + module experiments
- aggressive context jumps without reclaimed compute

## Search Rules Of Thumb

- If a branch is clearly alive but invalid, spend the next attempt on budget repair.
- If a branch is valid but still clearly weak twice in a row, cut it.
- If a branch is catastrophic on the proxy, do not give it a long runway.
- If a branch needs too many train.py edits to explain what changed, it is probably testing too many hypotheses at once.
- If a preflight projects the branch over the hard cap, do not pay for a full run.

## Creativity Guardrail

Do not copy public winning solutions verbatim.

Instead:

- use public frontier structure as a prior about regime
- preserve at least one material difference in carrier shape, schedule, placement, quant grouping, or local mechanism
- look for edges this repo can test cleanly inside `train.py`: activation family, partial RoPE, residual scaling, low-rank-Q spends, coarse-group quant rules, localized inserts, and post-quant selection

## Story Board For The Next Search Block

Order matters. Start with the highest-information, highest-EV stories:

1. stable frontier seed + one local train.py novelty
2. late selective quantization / PTQ
3. checkpoint choice / checkpoint soup
4. low-rank-Q reallocation with one spend
5. localized Canon-style insert
6. XSA or top-layer cache only after the cheaper stories are exercised
7. tiny local-token modules only if there is still remaining budget for creativity

## Success Condition For The Loop

A good search system here is not one that makes the wildest carriers.
A good search system is one that:

- discovers at least one ranking-valid contender family
- learns quickly which families are dead
- keeps the frontier pointer clean and valid
- hands off alive branches from discovery to refinement without wasting attempts
