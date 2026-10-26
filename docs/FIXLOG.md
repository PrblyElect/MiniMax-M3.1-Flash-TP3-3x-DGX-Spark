# FIXLOG — the ladder from nothing to a serving ring

The bring-up and tuning campaign behind this recipe, recorded so that nobody
repeats the grind. Every rung is one variable, decided by a rule that was
written down before the boot.

This ledger does two things most do not: it records **rejections** (the rungs
that measured beautifully and still lost), and it labels every memory claim
**shipped** (the state the lane is left in) rather than only the state it was
measured in.

## The first ladder: bring-up (a two-day window)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | the multi-node executor cannot start | the upstream day-0 image for this model ships **no ray** | build one extra image layer (`launch/Dockerfile`) and distribute the build |
| 2 | the model refuses to load: "Total number of attention heads (64) must be divisible by tensor parallel size (3)" | the published geometry is a **four-way** model: 64 query heads, 4 KV heads, 4 index heads | the TP=3 pad kit: ratio-preserving append-only pads to 96 / 6 / 6 (`patches/`) |
| 3 | the pad kit's own config code coerces the model's text config into a different architecture family | the checkpoint's config loader is opinionated about the config shape | the whitelist has to target the *nested* text config when the parent is whitelisted |
| 4 | the draft head is a second copy of the same wall | in-checkpoint DSpark draft carries the same 64/4 geometry and the engine asserts divisibility in the draft's own init | pad the draft with the same kit, with a disjoint rule scope, and prove it at boot from the real engine path |
| 5 | the vision path deadlocks the ring | a dtype split in the DP-sharded vision all-gather: the empty branch and the populated branch disagree on the collective's dtype | patch the stock file to make the two branches agree (`patches/`, edit 3) |
| 6 | the ring dies in the collective, on the fabric | the ray executor's **all-pairs** mesh attempts non-cabled port pairs on a triangle | the `mp`/TCPStore rendezvous instead of ray's mesh |
| 7 | a bare three-rank all-reduce fails with plausible-looking settings | the per-link /24 triangle needs subnet-aware rail matching | `NCCL_IB_SUBNET_AWARE_ROUTING=1` + a matching prefix length + `GID_AUTO=1` + `MERGE_NICS=0` — and a **bare collective probe** to separate fabric from engine |
| 8 | the ranks hang at rendezvous | ray workers die with their head and do not rejoin | relaunch the ranks after any head restart |

## The second ladder: the fence (capacity, collectives, KV)

| # | Symptom / goal | Root cause / finding | Outcome |
|---|---|---|---|
| 9 | decode is ~4× slower than the model should be | `CUDA_LAUNCH_BLOCKING=1` survived in the container environment from an earlier diagnostic baseline; it makes every launch synchronous | removed — the single cheapest win in the campaign, and nothing looked broken |
| 10 | should CUDA-graph capture be enabled? | under `--enforce-eager` the engine disables compile, graphs and JIT warmup together and **overrides** the requested compilation config | eager off: capture on, and the recipe's own compilation config finally takes effect |
| 11 | capture on cost the KV pool | the capture pool's *estimate* is subtracted from the budget before the KV cache is sized, and the real pool is smaller than the estimate | a **budget** change is the obvious lever, and it is priced in the third ladder |
| 12 | speculative decode, first attempt | the draft head is TP4-native, so the boot died in the draft's init on every rank | rejected on the arithmetic: the alternative (replicate the draft on every rank) costs ~2 GiB per rank and leaves the tightest box no margin |
| 13 | speculative decode, second attempt (padded draft) | the padded draft boots | **rejected**: the battery's time-to-first-token ceiling was missed, and the guard then killed the lane minutes later |
| 14 | why did the guard kill it? | spec decode spends host memory three ways at once: the draft's weights, the draft's own KV group, and the capture-pool growth (the decode query length grows, so the graph list grows) — on a unified-memory box that comes straight out of the shared pot | **the lesson, not a fix**: a configuration can be right and still be unshippable. The budget moved down; the lane came back |
| 15 | spec decode at the lower budget | — | **kept**: the throughput win is real, and it now fits |

## The third ladder: the last levers (measured, priced, closed)

| # | Lever | Verdict | Why |
|---|---|---|---|
| 16 | the capture grid, capped at 128 | **kept** | the capture estimate and the real pool fall together, and the freed estimate becomes KV capacity at approximately zero price in host memory: the pool grows by about 15,000 tokens (roughly 8% of the pool it had before the cap) with the host budget unmoved, and the decode hot path is untouched |
| 17 | the memory budget, one step up | **rejected on the arithmetic** | on a unified-memory box the budget is host memory, and the tightest box does not have the headroom. In a measurement window it would have passed; in the shipped state the guard would fire. The price is written out per box in this file's last section |
| 18 | the capture-estimate flag (turn the estimate off) | **rejected on the arithmetic** | it frees the whole estimate but the real pool then lands on top of the budget, so it costs real memory the lane does not have |
| 19 | the dspark verification budget (adaptive verification) | **closed by the engine** | the physics says it should matter — on prose the third draft position is worth about 0.03 tokens per step — but this lane's attention backends declare uniform-batch cudagraph support and cannot replay variable-length verification, so the engine refuses it at startup. The refusal is the engine's own; the rung was never booted |
| 20 | a lower capture ceiling still | **priced and declined** | worth roughly 3,000 tokens, which does not move the lane off a single full-length request, and it spends the draft path's entire safety margin |
| 21 | synthetic rejection sampling | **rejected on correctness** | it accepts draft tokens on a calibrated probability instead of verifying them, so the output is no longer sampled from the model being served |
| 22 | greedy draft sampling, parallel drafting | **not taken / closed** | greedy matches a greedy target and this lane serves a stochastic one; the draft is not trained for parallel drafting |
| 23 | the served context, 131,072 → 196,608 | **kept** | the pool is sized by the memory budget, not by the context, so a longer context costs CONCURRENCY, not pool: the engine's own figure goes from ~1.6x to ~1.06x and the lane still holds one full-length request. The rule and the capture gate were re-based with it — the concurrency figure is now read together with its denominator, and the gate checks the engine's own `max_seq_len` against the pinned context, which it did not do before |
| 24 | a served context of 262,144 | **rejected on the arithmetic — do not retry** | it does not fit the pool: one request would need more tokens than the pool holds, and the engine refuses at request time rather than warning at boot. The context that fits is bounded by the pool, and the pool is bounded by the model |
| 25 | quantising the KV cache to 4 bits to buy context | **not taken / closed** | the KV cache is the one thing that must not lose fidelity: a quantised cache is a different model. It was never tried, because the quality cost of a 4-bit cache was never shown to be zero, and "not visible in the acceptance numbers" is not "zero". The context was bought with concurrency instead, which costs no quality at all |
| 26 | the capture gate proved the served context from the API server's echo of what was ASKED for | the echo is the REQUEST; the engine's own resolved config is the FACT IN FORCE, and they can disagree | **fixed**: the gate reads the engine's own `max_seq_len` and NAMES a disagreement, so a lane serving the wrong context is rejected by name rather than accepted |

## The ladder's shape, stated plainly

The rung that mattered most was not a speed knob. It was the **memory
arithmetic**: on a unified-memory box, every capacity decision is a host-memory
decision, and the host memory is shared with the OS and with whatever else the
box carries. Two of the three best-looking configurations in this campaign were
rejected for that reason alone, and one of them was rejected *after* it had
measured green.

The second rung that mattered was **the pad**: a 428B model whose published
geometry is four-way cannot be sharded three ways, and the same wall sits
behind the draft head.

## The shipped state, priced

Every box's shipped free memory against the guard floor, and what one step of
the budget would cost. This is the table that decides whether a budget change
is shippable at all.

| box | shipped free memory | guard floor | free memory LEFT after one budget step (≈1.2 GiB) |
|---|---|---|---|
| head | ~7–8 GB | 4.0 GB | ~6–7 GB — above the floor ✓ |
| rank 1 | ~5–6 GB | 3.5 GB | ~4–5 GB — above the floor ✓ |
| rank 2 (the busiest) | ~4–5 GB | 2.4 GB | ~2.8–3.8 GB — **straddles the rule's 3.4 GB requirement for the tightest box** |

**WHAT "THE REQUIREMENT" IS, EXACTLY.** The rule asks two different things of the
tightest box, and conflating them is how a table starts contradicting itself.
Every box must stay above its own **floor** (rank 2: 2.4 GB). The tightest box
must additionally keep **1,000 MB of margin over that floor** — 2,432 + 1,000 =
**3,432 MB ≈ 3.4 GB** — because that is the box whose margin the price is paid
from. One step of ~1.2 GiB against a shipped free memory of ~4–5 GB leaves
**~2.8–3.8 GB**, and that range **STRADDLES the 3.4 GB requirement**: the high end
clears it, the low end does not. So the step is shippable only when the box's
actual free memory sits in the upper part of the band, and a table that states
only "~3.4 GB" hides which end it means — that is how a table starts
contradicting itself.

⇒ **The tightest box decides the budget, and one step up is a
service-availability decision, not a configuration change.** Raising the budget
means taking memory from the services that share the box, and that is the
operator's call, not the recipe's.
