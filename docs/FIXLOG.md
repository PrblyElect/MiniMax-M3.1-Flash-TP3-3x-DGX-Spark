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
| 16 | the capture grid, capped at 128 | **kept** | the capture estimate and the real pool fall together, and the freed estimate becomes KV capacity at approximately zero price in host memory: the pool grows by about 15,000 tokens (roughly 8% of the pool it had before the cap (measured at the then-0.78 budget)) with the host budget unmoved, and the decode hot path is untouched |
| 17 | the memory budget, one step up | **rejected on the arithmetic** | on a unified-memory box the budget is host memory, and the tightest box does not have the headroom. In a measurement window it would have passed; in the shipped state the guard would fire. The price is written out per box in this file's last section |
| 18 | the capture-estimate flag (turn the estimate off) | **rejected on the arithmetic** | it frees the whole estimate but the real pool then lands on top of the budget, so it costs real memory the lane does not have |
| 19 | the dspark verification budget (adaptive verification) | **closed by the engine** | the physics says it should matter — on prose the third draft position is worth about 0.03 tokens per step under the then-current mixed-acceptance measurement (predating the per-workload split in the shipped numbers) — but this recipe's attention backends declare uniform-batch cudagraph support and cannot replay variable-length verification, so the engine refuses it at startup. The refusal is the engine's own; the rung was never booted |
| 20 | a lower capture ceiling still | **priced and declined** | worth roughly 3,000 tokens, which does not move the lane off a single full-length request, and it spends the draft path's entire safety margin |
| 21 | synthetic rejection sampling | **rejected on correctness** | it accepts draft tokens on a calibrated probability instead of verifying them, so the output is no longer sampled from the model being served |
| 22 | greedy draft sampling, parallel drafting | **not taken / closed** | greedy matches a greedy target and this recipe serves a stochastic one; the draft is not trained for parallel drafting |
| 23 | the served context, 131,072 → 196,608 | **kept** | the pool is sized by the memory budget, not by the context, so a longer context costs CONCURRENCY, not pool: on the pool this rung was measured against (the 0.78-era budget, roughly 209k tokens) the engine's own figure goes from ~1.6x to ~1.06x and the lane still holds one full-length request. The rule and the capture gate were re-based with it — the concurrency figure is now read together with its denominator, and the gate checks the engine's own `max_seq_len` against the pinned context, which it did not do before |
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

## The shipped state, priced (at the 0.78 budget, before the tenant change below)

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

**WHERE THE TWO NUMBERS LIVE.** The floor and the margin are the **profile of
record's** — `GUARD_MIN_R2_MB` and `GUARD_MARGIN_MB` in `config/cluster.env`.
The rule reads both from there and carries no second copy, so a floor or a
margin is edited in the profile and nowhere else.

⇒ **The tightest box decides the budget, and one step up is a
service-availability decision, not a configuration change.** Raising the budget
means taking memory from the services that share the box, and that is the
operator's call, not the recipe's.

## The tenant change: the budget un-frozen

The 0.78 budget was never an engine limit; it was a **co-tenant limit**. The
pricing that fixed the budget recorded it plainly: one more step (about
1.2 GiB of host memory per box) put the ring's tightest box under its guard
floor *in the layout where the model boxes also carried heavier co-tenant
services* — a lane-killing shortfall measured in the field, when a
co-tenant's memory growth pushed the tightest box under its floor and the
guard killed the lane. The rejection was correct for that layout.

The layout then changed. The heavier co-tenant services moved to a dedicated
host of their own, off the model boxes entirely; the model boxes now carry
only their light GPU tenants (one embedder, one small service, the head
bare). Every box's shipped free memory rose by multiples of the step the
budget was asking for, and the rejection's premise — the only reason 0.80
was refused — is void.

So the profile steps: **GMU 0.78 → 0.80**, with the capture-grid cap
unchanged. A clean-head measurement puts the 0.80 pool at roughly **224,000
tokens**: about **1.71x** concurrency at a 131,072-token request, about
**1.14x** at the served 196,608 — a larger pool than the one the context rung
was priced against (rung 23 measured the 0.78-era pool), read against the
same denominators. The number is lower than an unprofiled 0.80 would suggest
because the engine takes its CUDA-graph memory-profiling estimate out of the
budget before sizing the KV cache: it reports 0.80 as equivalent to 0.7819
without that profiling, and 0.8181 would be needed to keep the same effective
KV. A 262,144-token request does **not** fit at ~224K — it is refused at
request time — so the served context stays 196,608.

The profile is arithmetic-plus-guard, not a new magic number: the boot-time
capture gate pins 0.80 and verifies the budget on all four surfaces (the
pin, the recipe default, the container env, and the engine's own resolved
value on every rank), the guard enforces the floors on every box for the
whole session, and the acceptance rule judges the run from the boot's own
evidence. A future layout that puts heavy tenants back on the boxes owes
this file the same honesty: re-price, lower `GMU`, and let the guard be the
arbiter.

## After the ladder: two later rungs the record should carry

Two experiments run after this ladder change what a reader should expect from
the 0.80 profile. Both are recorded here rather than left in the internal
ledger.

**The allocator guard is boot-fatal.** The usual fragmentation countermeasure,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, makes the head's guard
kill the boot: on this unified-memory stack the allocator commits physical
memory for its segments, and the head collapsed to 436 MB available against a
4,096 MB floor. The identical profile without the variable boots in about
seven minutes. The guard is a per-box host-memory floor, not an allocator
hint — do not retry it.

**An NVFP4 KV cache is a fork swap, not a flag.** The day-0 image refuses it
on this silicon: the main attention raises `ValueError` off SM100, and the
`nvfp4` indexer raises `NotImplementedError` ("needs the (not-yet-added)
CuteDSL indexer impl"). The community pattern that does work on GB10 is a
Triton inline-dequant mod — **+36% pool, 1M context** — at a lower
single-stream decode (it ships eager; CUDA graphs regress on that build). It
is a fork change; FlashInfer's SM120/121 NVFP4 backends may close the gap
natively in a later image.

## The desk ladder: what a reader inherits (no boot, no window)

These rungs were not booted and did not need to be: each one is a place
where the recipe as published could mislead, mislead silently, or fail
closed on something that was never wrong. They are recorded here in the
same shape as the boot ladder — symptom, root cause, fix — because a
reader who finds one of them in the wild deserves to know it was found on
paper.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 27 | a reader on a different mesh cannot change the collective's HCA pair | `config/cluster.env` assigned `NCCL_IB_HCA` as a plain value, so it clobbered an exported pair before the launcher ever read it — and the launcher was already reading the overridable form | the assignment takes the overridable form, with the shipped pair as its default: an exported pair wins, and a same-mesh reader still gets the shipped one |
| 28 | a reader on a different mesh fails the all-reduce with no clue | the README's "edit the profile for your site" list never mentioned the fabric, and `ibdev2netdev` appeared nowhere in the recipe | the fabric is named in that list and in the landmines, with the command that discovers the device names |
| 29 | the serve knobs could not be changed without editing the script | the block size, the KV dtype, the sequence count and the batched-token budget were literals on the serve line, so the profile of record could not reach them | each is an environment knob with the validated value as its default, and the profile of record publishes all four |
| 30 | a boot whose overlay silently failed is a boot that serves the wrong tree | the pad kit is applied in-container before anything starts, and nothing proved it landed — the kit's own verify checks its edits semantically, not byte for byte | the manifest gained a container view: the installed package is hashed against the recipe's own bytes, and every patched stock file against the kit's own re-derived edit. It runs at boot, and a selftest proves the check can fail |
| 31 | a short prompt can be answered with deterministic garbage and nothing notices | the upstream dispatch check classifies a prefill as a uniform-decode step by shape alone; with k=3 the trap window is four tokens, and the lane serves short chat turns | a gate sends a prompt of exactly the engine's own `uniform_decode_query_len` tokens, and the same probe one token longer, and requires both to answer |
| 32 | a long prompt can differ from the same prompt served from cache, and nothing notices | the validation flow exercised no long prompt at all | a gate sends a prompt that crosses the batched-token budget three times — cold, then twice with the prefix cache warm — and compares the generated token sequences |
| 33 | the reader cannot see what the speculative-decode posture inherits | the posture itself (dspark k=3, a Triton attention backend, an fp8 KV cache) lived in the config and not in the prose | the two upstream defects that sit on that posture are documented, with what this recipe inherits and what the available evidence says it does not |
| 34 | the container command could only be seen by starting it | the `docker run` was built in place, so the only way to read it was to run it | the command is built in one array and executed from it, and `PRINT_CMD=1` prints it — one argument per line, then the whole command — and starts nothing |
| 35 | the short-prompt gate could pass without ever asking the question | the nonce was eight hex characters — several tokens — and the trap window is four, so the question was pushed out of the prompt; the control prompt was built by slicing a shorter list, so it was not one token longer than the trap either | the probe now keeps the question whole and is padded to the exact length with a neutral token, the control is the trap plus one appended token, and a question that does not fit is a hard failure of the gate rather than a note |
| 36 | the documented command for reading the container command could not run | the launcher needs a role, a rank and an address, and the documented line supplied neither | the wrapper that resolves them from the hostname map is documented, and the direct invocation is documented with the three values the start path passes |
| 37 | an exported serve knob was silently discarded | the profile assigned the four serve knobs as plain values, so sourcing it overwrote the environment, while the same profile used the overridable form for the fabric | the four knobs use the same overridable form as the fabric, so an exported value survives the profile |
| 38 | the upstream long-context citation read as if the defect applied here | the citation gave the class but not the scope, and the scope is what decides whether it applies | the citation now names the upstream model, topology, attention backend and the request-length regime, and the one axis this recipe does share is stated |
| 39 | three figures in the acceptance rule and the ladder disagreed with the shipped profile | the pool was re-based from the 0.78 budget to 0.80 and the older numbers were left in the comments and in one rung | the figures are reconciled: the rung that was measured at the older pool says so, and the comments carry the shipped pool |
| 40 | the capture gates could only be exercised against a live lane, and the documents did not say which floors the rule held for itself | the two capture gates are fail-closed and are the first thing a reader runs, but a reader with no lane — and a reviewer checking the published artifact — had no way to ask "does this gate actually refuse a bad boot?" without one; and the acceptance rule's floors were published as a list of numbers, which read as the rule's own constants rather than as the profile's | both capture gates carry a `--selftest` entry point: it runs the gate's OWN logic over synthetic engine logs (no container, no lane, no network), prints a PASS/FAIL line per check, and exits non-zero if any check fails. It requires a healthy synthetic boot to PASS and requires the malformed thousands grouping, the below-floor pool, the interrupted capture bar, the still-eager boot and the captured encoder to be REFUSED and NAMED — so the gate is shown to be able to fail, which is the only evidence that a PASS means anything. The gates' assertions are unchanged — the only production-path change is that an unrecognised argument now says so on stderr instead of being dropped in silence — and the self-test drives the shipped script rather than a copy of it. The documents now also say, per floor, whether it is the profile of record's (`CTX`, `EXPECTED_CG_*`, `GUARD_MIN_*_MB`, `GUARD_MARGIN_MB`, `GMU`) or the rule's own criterion — one place to change each |
