# The TP=3 pad kit

**Why this exists.** MiniMax M3.1 Flash publishes an attention geometry of
**64 query heads / 4 KV heads / 4 index heads**. The engine requires each of
those to divide by the tensor-parallel size, and 64 % 3 ≠ 0 — the model
cannot be sharded across three ranks as published. It is a four-way model.

The DSpark draft head inside the checkpoint carries the **same** geometry, so
a speculative-decode configuration meets the same wall a second time. Both
are fixed here, by the same kit.

## The answer: a ratio-preserving, append-only pad

| | published | padded to | why |
|---|---|---|---|
| query heads | 64 | **96** | 16 query heads per KV head, before and after |
| KV heads | 4 | **6** | |
| index heads | 4 | **6** | the indexer shards on the KV path, so it pads in lockstep |
| head dim | 128 | 128 | unchanged |
| vocab | 200,064 | 200,064 | already divisible by 3 |

**Ratio-preserving is the whole trick.** 96 = 16 × 6, so the GQA mapping is
unchanged: each rank's padded query heads land on that rank's padded KV
heads. The padded heads carry **zero** weights, so their contribution to the
output is exactly zero — the model is arithmetically the published model.

**Append-only means no interior surgery.** Every operation is an append at
the end of an axis: no element is moved, so no row or column index shifts and
the operation is idempotent (running it twice is the same as running it
once).

**The pads, exactly:**

* `q_proj` — **rows**: +32 zero query heads (64 → 96).
* `k_proj`, `v_proj` — **rows**: +2 zero KV heads (4 → 6).
* `index_q_proj` — **rows**: +2 zero index heads (4 → 6), in lockstep with
  KV, because the engine's indexer requires the total index heads to equal
  the total KV heads and shards the index query on the KV path.
* `o_proj` — **columns**: +32 zero columns for the padded query heads (their
  contributions are exactly zero). Note the axis: padding `o_proj` on rows
  would be wrong.
* **FP8 block scales pad 1:1 with their weights.** The rowwise scales pad on
  rows; `o_proj`'s column-block scales (block size 32) pad on columns:
  +4,096 columns is +128 scale columns.

**What is never padded, and why:**

* `index_k` is a single replicated head — it is not sharded, so it is not
  padded.
* `q_norm`, `k_norm` and the index norms are per-head broadcast vectors of
  length 128 — they are head-count-independent.
* The MoE (3,072) and dense (12,288) intermediates already divide by 3.
* The vocabulary already divides by 3 and by 64.

**The vision tower cannot be padded this way.** It runs data-parallel, so its
hidden size (1,280) does not divide by 3 and a tensor pad would cascade.
The tower is left alone and the collective dtype is fixed instead (below).

**The draft's rule scope is disjoint from the target's.** The target's rules
match `language_model.model.layers.` and the draft's match
`language_model.model.dspark.layers.` — neither is a substring of the other,
so the two rule sets never overlap and neither can pad the other's tensors.
The draft's MLP needs no pad (12,288 % 3 = 0) and its `fc` and confidence
head are replicated (never sharded), so their wide inputs do not need to
divide by 3.

## The four edits

`apply_mm31_tp3pads.py` makes four edits to the engine's tree:

1. **the config pads** — the head counts, in the *nested* text config (and in
   the draft's promoted config), plus the vocab check;
2. **the weight pads** — the rules in `tp3_pad_table.json`, applied to the
   checkpoint's tensors;
3. **the vision collective dtype** — the DP-sharded vision all-gather has an
   empty branch and a populated branch that disagree on the collective's
   dtype, which deadlocks the ring. The two branches are made to agree.
4. **the reasoning-parser scan scope** — the parser decides that a response has
   finished reasoning by scanning the WHOLE prompt for the marker pair, and the
   chat template writes those markers into the system turn as documentation, so
   the scan always finds a closed pair and declares the reasoning finished
   before the first token. The streaming path then emits the raw markers as
   content. The scan is scoped to the final assistant turn instead.

The applier is idempotent and has a `verify` mode:

```sh
python3 apply_mm31_tp3pads.py /usr/local/lib/python3.12/dist-packages
python3 apply_mm31_tp3pads.py /usr/local/lib/python3.12/dist-packages verify
```

**The applier is applied inside the container at every boot**, before anything
imports vllm — so the engine's tree is always patched, and a boot whose pad
does not verify fails instead of running on a half-patched tree.

## The draft-pad gate

`verify-draft-pad.py` proves the draft head is padded to TP=3 **from the real
engine path**: the real config loader, the real speculative-config promotion,
the real pad hook, and the real checkpoint's tensor headers. It fails the
boot if any of that does not line up.

It also has a `--selftest` mode that mutates the inputs — drops the whitelist
entry, uses an unpadded config, corrupts a rule, merges the two rule scopes —
and requires each mutation to be **rejected**. A gate that cannot fail
proves nothing, so the selftest runs at every boot too.

## Re-pinning the kit

The pad rules are expressed in terms of the *published* geometry (64 / 4 / 4)
and the TP size, so a different TP is a table change, not a code change.
Before re-pinning against a different engine build, re-run the applier's
`verify` mode and the draft-pad gate against that build: the rules assume
the tensor names and the block-scale layout of the pinned engine, and a new
engine build can move either.
