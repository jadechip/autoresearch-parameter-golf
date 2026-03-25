# Parameter Golf agent context capsule

Use this as a prior, not a recipe.

## Mission
Lower **frozen-protocol post-quant val_bpb** under:
- **16,000,000 total bytes** = counted code + compressed model
- **10 minutes training** on 8×H100 SXM
- evaluation must also stay within budget and must not leak training/validation information.

In `minimal_autoresearch`, protected launch fields are frozen by state. Do **not** try to win by changing evaluation budget, tokenizer inputs, dataset inputs, or other protected fields.

## What the records really say

The winning pattern was:

1. smarter evaluation/context,
2. lower quantization/compression damage,
3. spend saved bytes on width/depth,
4. add one small local inductive bias,
5. then add tail-only specialists or legal TTT.

The records did **not** mostly win by giant new architectures.

## Strongest durable ideas

### 1) Sliding-window evaluation is enormous
Sliding window gave about **0.03–0.034 bpb** improvement by ensuring scored tokens get near-full context. Document isolation matters too. If the protocol is unfrozen, sliding eval is table stakes.

### 2) This benchmark is about the artifact, not the float model
Repeated wins came from:
- fp16 tied embedding passthrough,
- STE/QAT on block weights,
- mixed precision by tensor family,
- zstd-22 in the int6 regime,
- int5 MLP when it compresses better than int6,
- long warmdown / denser late checkpoints,
- EMA / SWA / post-quant checkpoint selection,
- GPTQ-lite / clip search.

Always track:
- float bpb,
- roundtrip bpb,
- artifact bytes,
- quant gap,
- ms/step,
- eval time.

### 3) Spend byte savings on MLP width first, then depth
The best mainstream carrier became roughly:
- **10–11 layers**
- **512d**
- **MLP ≈ 2.6× to 3×**
- tied embeddings
- GQA
- U-Net-like skips
- small local-context helpers

The alternate ternary frontier also said **width beats excess depth** when step-time is counted.

### 4) Tiny local modules can pay
High-ROI small modules:
- SmearGate
- BigramHash
- U-Net skips / residual mixing
- top-layer value-feature sharing

These worked because they were cheap and ablatable.

### 5) Tail-only specialists are high EV
Best examples:
- XSA only on deepest 3–4 layers
- Partial RoPE only on first 16/64 head dims
- layerwise norm scaling
- top-layer value embedding
- score-first legal TTT at evaluation

Try new module families in the **last 2–4 layers first**, not globally.

### 6) Short-run stability hacks matter
Repeated winners:
- lower LR than the original baseline,
- momentum warmup,
- orthogonal init,
- Muon weight decay in the int6 regime,
- EMA or late SWA,
- longer warmdown,
- layerwise scaling,
- LeakyReLU(0.5)^2.

A near-frontier local knob can beat a larger redesign.

### 7) Systems speedups are real quality
FlashAttention-3, fused projections, zero-allocation XSA, better Muon plumbing, and reduced recompile churn all buy extra steps. In a 600-second contest, that matters.

## Best experiment policy for the loop

Keep a **stable carrier** and test **one primary novelty** at a time.

### Priority order
1. **Local frontier knobs**
   - activation family tweaks
   - partial RoPE fraction
   - layerwise scale
   - warmdown / checkpoint cadence
   - EMA vs SWA vs checkpoint soup
   - GPTQ-lite / clip-selection / export-selection

2. **Selective quantization / byte reallocation**
   - safe precision exemptions (embedding, final K, head)
   - selective int5/int6 groups
   - late selective QAT
   - spend recovered bytes on one extra layer or cleaner MLP widening

3. **Tiny module families**
   - SmearGate
   - factorized or hashed bigram feature
   - XSA only in top layers
   - small top-layer shared-value mechanism

4. **Large regime shifts**
   - tokenizer change (e.g. 8k BPE)
   - ternary/binary/fp8 storage
   - width/depth repartition
   - only after the loop is already stable

## Avoid these traps

- bundling topology + quantization + schedule + new module family in one run
- ranking by float loss instead of roundtrip bpb
- aggressive low precision on tied embeddings by default
- full-model heavy modules
- sequence/batch schedules that cut step count or trigger recompiles
- recurrence in the 10-minute regime
- assuming tricks transfer across int6 and ternary/binary regimes

## Specific negative evidence from the frontier writeups

In the ternary/binary regime, these were repeatedly bad or incompatible:
- EMA
- FP4 storage
- grouped MLP
- Tversky layers
- BigramHash
- LoRA/TTT
- heavy local modules
- over-deep narrow stacks

## Minimal pseudocode mental models

```python
# rank by post-quant metric, not float metric
score = evaluate_roundtrip(quantize_for_export(checkpoint))
```

```python
# tail-only insertion
if layer_idx >= n_layers - 4:
    x = tail_specialist(x)
```

```python
# spend saved bytes on one macro axis
if bytes_saved > layer_cost:
    num_layers += 1
elif bytes_saved > mlp_cost:
    mlp_mult += 0.25
```

```python
# legal TTT: score first, adapt after
for chunk in chunks:
    score(chunk)      # no grads, no mutation
    adapt_on(chunk)   # only after scoring
```

## Bottom line
The highest-EV generic strategy is:

- start from a stable strong carrier,
- freeze the protocol,
- improve the artifact more than the float model,
- use one clear hypothesis per run,
- prefer byte-efficient width/MLP spending,
- prefer tail-only mechanisms,
- keep systems/eval honest,
- only do frontier regime changes deliberately.
