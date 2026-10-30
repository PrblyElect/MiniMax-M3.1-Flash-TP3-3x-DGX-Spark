---
license: mit
base_model: MiniMaxAI/MiniMax-M3.1-Flash
library_name: vllm
tags:
- serving-recipe
- vllm
- dgx-spark
- tensor-parallel
- speculative-decoding
---

# MiniMax M3.1 Flash at TP=3 across 3× NVIDIA DGX Spark — with DSpark speculative decode

**TL;DR.** MiniMax M3.1 Flash (~428B total / ~23B active, multimodal MoE) on
three DGX Spark GB10 boxes, one rank per box, one OpenAI-compatible API on the
head. The model ships with a DSpark draft head inside the checkpoint, so
speculative decode is available with no second download. This recipe is the
working configuration: the TP=3 pad kit that makes a 64-head model shardable
three ways, the serving profile that fits the KV pool inside the memory the
boxes actually have, and the tools that make the claims checkable instead of
asserted.

Everything here was measured on the hardware named below. Numbers are rounded
deliberately: they are the recipe's spec claims, not a benchmark report.

## What this is

* A **three-box TP=3 ring** for a model that does not natively shard three
  ways. The published geometry is 64 query heads / 4 KV heads / 4 index
  heads, and the engine requires each to divide by the tensor-parallel size.
  64 % 3 ≠ 0, so the model cannot be split across three ranks as published.
  The fix is a **ratio-preserving append-only pad** to 96 / 6 / 6 — 16 query
  heads per KV head before and after — applied at every boot by
  `patches/apply_mm31_tp3pads.py`. The padded heads carry zero weights, so
  their contribution is exactly zero and the GQA mapping still lands each
  rank's padded query heads on that rank's padded KV heads.
* **Speculative decode** with the in-checkpoint DSpark draft head, padded to
  TP=3 by the same kit. The draft carries the same published geometry as the
  target, so it hits the same wall and needs the same answer.
* **A capacity-honest profile.** On a unified-memory box the GPU reservation
  and the request-side CPU staging come out of one pot, so the KV pool is
  whatever is left after the weights, the OS, the other services, the
  activation peak and the capture pool. This recipe's budget is chosen so
  that a memory guard still has room on every box, and the capture grid is
  capped so that the pool is not paying for graphs the lane never uses.
* **The tools** that turn "it works" into "it is proven": a capture tool that
  reads the engine's *resolved* configuration and gates it, an acceptance
  rule that fails closed, a bench harness with the anti-contamination
  discipline built in, a capture-grid resolver, and a memory guard.

## The numbers (measured, labeled, rounded)

> Every row below is one measurement set at the shipped **196,608-token served
> context**.

| | |
|---|---|
| 1-way decode, single stream | **~26 tok/s** |
| 16-way decode, aggregate | **~130 tok/s** |
| prefill (a ~8K-token prompt) | **~1,500 tok/s** |
| time to first token | **~0.5 s** |
| per-token latency | **~38 ms** |
| KV pool | **~266K tokens** — **1.35×** a 196,608-token request (the same pool is **2.0×** a 131,072-token one) |
| capture pool per rank | **under 1 GiB**, against the engine's own estimate of ~2 GiB |
| draft acceptance, prose | **~0.60 / ~0.30 / ~0.14** per position |
| draft acceptance, long reasoning chains | **~0.99 / ~0.96 / ~0.94** per position |

**Read the acceptance row twice.** On prose the acceptance decays by about half
per position, so the third draft position is worth very little; on long
reasoning chains it barely decays at all, and the same configuration is worth
far more. **A single "spec decode is worth N%" number is meaningless on this
model** — the honest statement is that the draft budget is worth very
different things on different workloads, which is why the profile is priced
per workload in `docs/FIXLOG.md`.

**Which of these numbers the rule actually gates.** The 4-way cliff (the rate
the lane holds at the concurrency it was bought for), the merged position-0
acceptance, and each workload class's own position-0 acceptance are floors —
a run that misses one fails the rule by name. The 8- and 16-way aggregates
and the per-position decay inside a class are **published, not floored**: they
are context for a reader, and the rule asks of them only that they are
present and readable.

**The pool is the capacity story, not the throughput story.** The engine's own
concurrency figure is a property of **(pool / request length)**: the same
pool reads ~2.0× at a 131,072-token request and ~1.35× at 196,608, because a
request at the longer context uses almost all of it. At 196,608 the lane
holds **one full-length request, not two** — that is the price of the longer
context, and it is paid in concurrency, not in pool: the pool itself did not
shrink to pay for it. Raising the pool further is not a configuration
question on this stack — see "What the ceiling actually is" below.

## Quick start

```sh
# 0. get the recipe. Every command below runs from the recipe root.
git clone https://github.com/PrblyElect/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark.git
export RECIPE_DIR=~/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark   # LOCAL path (expanded here)
cd "$RECIPE_DIR"
#    (the upstream recipe for this model, with its mandatory serve flags, is
#    at https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash)

# 0b. THE SAME DIRECTORY ON THE BOXES, AS THE REMOTE SHELL MUST SEE IT.
#     The tilde is LEFT QUOTED on purpose: a remote path is expanded by the
#     REMOTE shell, so `ssh box "ls ~/x"` finds the box's own home and
#     `ssh box "ls $RECIPE_DIR/x"` would look in YOUR home on the box. Keeping
#     one literal for the remote side and one expanded path for the local
#     side is the only way both can be right.
export REMOTE_DIR='~/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark'

# 1. EDIT config/cluster.env for your site: the head's address, the
#    checkpoint path, the guard floors, the image digest, the served context.
#    EDIT IT ONCE, HERE -- the copy you ship to the boxes in step 5 is the
#    one they read. Then source it in every shell that starts anything:
set -a; . "$RECIPE_DIR/config/cluster.env"; set +a

# 2. stage the checkpoint (~233 GiB) with the profile's own coordinates, on
#    one box, then copy it to the other two. The DSpark draft head rides
#    along under <model>/dspark and needs no second download.
huggingface-cli download "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"

# 3. build the runtime image ONCE and load it on the other two boxes.
#    The upstream day-0 build ships no ray; this adds it as one layer.
docker build -t vllm/vllm-openai:"$VLLM_TAG" \
  --build-arg BASE_IMAGE="$VLLM_BASE_IMAGE" --build-arg RAY_VERSION="$RAY_VERSION" \
  launch/
docker save vllm/vllm-openai:"$VLLM_TAG" | ssh spark-b.local docker load

# 4. put this checkout on all three boxes. CHECK THE MANIFEST OVER THE
#    SECOND CHANNEL FIRST. A manifest cannot contain its own hash, so a box
#    carrying a stale kit AND a stale manifest would agree with itself -- and
#    would serve an older recipe without saying so. Read each box's digest
#    BEFORE copying anything: after the copy every box agrees with the repo
#    BY CONSTRUCTION, so a comparison made afterwards would prove nothing.
for h in spark-head.local spark-a.local spark-b.local; do
  ssh "$h" "md5sum $REMOTE_DIR/kit-md5.txt" 2>/dev/null || echo "$h: no kit on this box yet"
done
md5sum kit-md5.txt        # the repo's own digest -- the one every box must show
                          # (on macOS: md5 -q kit-md5.txt)

# 5. now copy the kit out, and confirm the copy landed byte-exactly.
#    `config/` comes along because config/spec-config.json is a kit file and
#    config/cluster.env carries the digest the boot checks against -- so
#    this copy IS how your step-1 edits reach the boxes. Re-running it
#    replaces them with whatever the file says now.
for h in spark-head.local spark-a.local spark-b.local; do
  ssh "$h" "mkdir -p $REMOTE_DIR"
  scp -r launch tools patches config kit-md5.py kit-md5.txt "$h":"$REMOTE_DIR"/
  ssh "$h" "md5sum $REMOTE_DIR/kit-md5.txt"
done

# 6. head first, then the two ranks (headless). THE BOOT RUNS ON EACH BOX.
#    The boot applies the pad kit and proves the draft pad before the engine
#    starts. A box whose hostname is spark-head*/spark-a*/spark-b* knows its
#    own role, rank and address; a box with any other name is told so, and
#    told what to do instead.
bash launch/boot-tp3.sh               # on the head
bash launch/boot-tp3.sh               # on each rank (the hostname decides)

# 7. arm the memory guard on EACH box, before any load. It kills ONLY this
#    lane's container when the box's free memory stays under ITS floor --
#    the floor comes from config/cluster.env, so source it first (step 1).
#    ARMING STARTS A FRESH LOG: the previous cycle's guard.log is kept as
#    guard.prev-<timestamp>.log (nothing is deleted or overwritten), so the
#    log the rule reads in step 10 covers THIS cycle alone -- which is what
#    lets the rule judge the window instead of guessing where it begins.
bash tools/guard-memory.sh --container minimax31-flash-tp3 &

# 8. ON EACH BOX, from the recipe root, check that the engine resolved what
#    you think it did. THE CAPTURE GATE RUNS ON EACH BOX, with the same three
#    lane labels every time, and writes THIS box's section under ITS label.
curl -s localhost:8100/v1/models
bash tools/capture-config.sh --head spark-head.local --r1 spark-a.local \
  --r2 spark-b.local --out logs/capture

# 9. ON THE HEAD -- THE ENGINE SERVES ONLY THERE -- measure the lane and
#    write the evidence set. The bench comes FIRST: it is the only producer
#    of bench.txt and manifest.txt, and the rule reads both. It measures
#    BOTH workload classes and a long-context prefill in one run, which is
#    what the rule judges. THE PREFILL BUDGET IS A TOKEN BUDGET: the bench
#    asks the engine for the prompt's own token count and grows the prompt
#    until that count reaches the budget below. The declared number is a
#    FLOOR on the measured length, not the size of the prompt that is sent:
#    the engine counts the prompt it was given, and the row publishes THAT
#    count. A budget the engine cannot reach publishes no prefill row at
#    all, and so does a prompt the engine refuses.
python3 tools/bench-decode.py --lane-url http://127.0.0.1:8100 \
  --tokens 800 --rounds 2 --streams 1,4,8,16 --workload prose,high-draft \
  --prefill-tokens 8192 --evidence-dir logs/evidence

# 10. ON THE HEAD, collect the OTHER boxes' evidence into that same
#     directory, assemble the aggregates, and judge it. THE HEAD'S OWN
#     FILES ARE COPIED IN: the bench above wrote bench.txt and
#     manifest.txt here, and the capture gate and the guard wrote their
#     files here too.
#     (docs/VALIDATION.md §2-§4 has the full flow and what each file is for)
mkdir -p logs/evidence
cp logs/capture/capture-gate-spark-head.local.txt logs/evidence/
cp logs/guard/guard.log     logs/evidence/guard-spark-head.local.log
cp logs/guard/guard_mem.log logs/evidence/guard-spark-head.local-mem.log
for h in spark-a.local spark-b.local; do
  scp "$h":"$REMOTE_DIR"/logs/capture/capture-gate-$h.txt logs/evidence/
  scp "$h":"$REMOTE_DIR"/logs/guard/guard.log     "logs/evidence/guard-$h.log"
  scp "$h":"$REMOTE_DIR"/logs/guard/guard_mem.log "logs/evidence/guard-$h-mem.log"
done
python3 tools/assemble-evidence.py --evidence-dir logs/evidence \
  --head spark-head.local --r1 spark-a.local --r2 spark-b.local
python3 tools/acceptance-check.py --lane-url http://127.0.0.1:8100 \
  --head spark-head.local --r1 spark-a.local --r2 spark-b.local \
  --evidence-dir logs/evidence
```

**THE RULE IS AN OFFLINE JUDGE.** It reads the evidence directory and makes
no network call at all, so the directory can be copied to another machine
and judged there — but `--lane-url` must still match the URL the bench
recorded in `manifest.txt`, because that record is what the rule checks the
measurement against.

`boot-tp3.sh` resolves this box's role in the ring and calls `start-tp3.sh`.
The **environment wins over the hostname map**: `ROLE=worker NODE_RANK=1
NODE_IP=… bash launch/boot-tp3.sh` makes a worker whatever the box is called,
and a box the map does not recognise is refused with the remedy rather than
guessed at. Every knob is an environment override; `config/cluster.env` is the
profile of record, and it publishes the digest of the manifest that ships
with this recipe (`KIT_MANIFEST_MD5`) so a box can be checked against the repo
rather than against itself. If you change a kit file, regenerate the manifest
(`python3 kit-md5.py --write`), update that one line, and re-check the three
boxes as in step 4.

## The hardware this was proven on

| | |
|---|---|
| boxes | 3 × NVIDIA DGX Spark GB10 (unified CPU/GPU memory, 128 GB) |
| interconnect | dual-port 200 Gb/s RoCE per box, full triangle |
| host OS | Ubuntu 24.04, driver 580.x |
| model | MiniMax M3.1 Flash, MXFP8 dense + NVFP4-packed MoE experts, ~428B/~23B active |
| context served | 196,608 (the model's native class is 1M — not enabled here) |

**The triangle, not the chain.** Every rank must reach both peers. A chain
works for two boxes and quietly fails for three.

## The stack (pinned)

| | |
|---|---|
| engine | `vllm 0.1.dev23341+g8744dd2fb.d20261022.cu134` (upstream day-0 build, digest-pinned) |
| torch | 2.14.0+cu130 |
| transformers | 5.19.0 |
| ray | 2.59.0 (the upstream image ships none) |
| patches | this recipe's `patches/` (four edits, applied at boot) |

## The landmines (read before booting)

**1. The divisibility wall, and it applies twice.** 64 query heads and 4 KV
heads do not divide by 3. The published model is a four-way model. The pad
kit is the answer, and the *draft* head has the same geometry, so a
speculative-decode configuration hits the wall a second time. Both are padded
by the same kit, and the boot proves the draft pad before the engine starts.

**2. The pad must be ratio-preserving.** 96 = 16 × 6: the query-per-KV ratio
must not change, or the GQA mapping sends real query heads to padded KV
heads and the model quietly produces garbage. `o_proj` pads on **columns** (the
padded heads contribute exactly zero) and `k_proj`/`v_proj` on rows, and the
FP8 block scales pad 1:1 with their weights — `o_proj`'s scale pads on
columns at block size 32, which is the one that is easy to get wrong.

**3. Unified memory is one pot.** The GPU reservation and the request-side
CPU staging draw from the same memory. A GPU-side tenant (anything that
allocates device memory) is **not** free, and `--gpu-memory-utilization` is a
host-memory budget in disguise. This is the single most expensive lesson of
the bring-up: a speculative-decode configuration that measured beautifully in
a window left the tightest box with a few hundred MB of headroom, and the
guard — correctly — killed the lane minutes later.

**4. Measure the state you will SHIP, not the state you measured in.** A
configuration can pass a measurement window and still be unshippable, because
the services you parked for the measurement come back. Every memory claim in
`docs/FIXLOG.md` is priced against the *shipped* state.

**5. The pool is sized by the tightest rank.** The head's memory does not
decide the pool; the most-loaded box does. Per-rank budget forks cannot raise
it — the minimum binds.

**6. The capture-pool estimate is subtracted from your budget before the KV
cache is sized.** The engine reserves a conservative estimate of the CUDA
graph pool and takes it out of the memory budget; the pool that is actually
kept is smaller. The difference is capacity you pay for and never use. The
engine prints both numbers ("actual" and "estimated") — read them. Capping
the capture grid makes the two converge and turns the difference into KV
capacity at approximately zero price in host memory.

**7. A CUDA-graph cap has to be proved against the graph DERIVATION, not by
eyeballing the grid.** The decode graphs are derived from the capture grid,
rounded up to a multiple of the uniform decode query length and dropped
beyond the maximum decode tokens. The sizes above 128 never produced a decode
graph, so removing them cannot touch the decode hot path. `tools/resolve-capture-grid.py`
proves this in the pinned image, by building the engine's own config, rather
than by reading the code.

**8. The context must fit the pool.** The native class is 1M; the served
context is what the KV pool can hold. Ask for twice the pool and the engine
refuses the request rather than degrading. The capture tool checks the
engine's **resolved** `max_seq_len`, not what you intended.

**9. `CUDA_LAUNCH_BLOCKING=1` makes every kernel launch synchronous.** It is
the kind of environment variable that survives from a diagnostic baseline
into production. It costs roughly 4× on decode and nothing else, so nothing
looks broken — the lane is just slow. Check the container's environment
before blaming the model.

**10. One low memory sample must not kill a serving lane.** The guard needs
a debounce count, and it must kill only *this* lane's container.

**11. The ray executor's all-pairs mesh mis-wires a triangle.** Use the
`mp`/TCPStore pattern (`--nnodes`/`--node-rank`/`--master-addr`/`--master-port`,
workers `--headless`). Also: ray workers die with their head and do not
rejoin — relaunch them after a head restart.

**12. The collective needs four NCCL variables, not six.** On a per-link /24
triangle, `NCCL_IB_SUBNET_AWARE_ROUTING=1` with a matching
`NCCL_IB_SUBNET_PREFIX_LEN`, plus `NCCL_IB_GID_AUTO=1` and
`NCCL_IB_MERGE_NICS=0`. A bare three-rank all-reduce probe is the fastest way
to tell a fabric problem from an engine problem.

**13. Measurement hygiene: a measured prefix is not a cold prefix.** Prefix
caching is on, the first request pays a JIT tax, and a cached prefix makes the
next measurement meaningless. The bench harness puts a nonce at the FRONT of
the prompt, runs a sacrificial warmup that never warms the measured prefix,
and splits the workload.

**14. The engine prints the graph bars on the first rank only.** Per-rank
symmetry has to be proven from each rank's own capture-pool actual-vs-estimate
pair, not inferred from the head's bars.

## What the ceiling actually is

This is the part people want to skip, and it is the part that decides your
plan. **The KV pool on this stack is bounded by the model's own size, not by
a knob.** The checkpoint is already quantized (MXFP8 dense + NVFP4-packed MoE
experts, ~233 GiB of tensors), and TP=3 shards that across the three boxes —
so each box spends roughly 80 GiB of its 128 GB on one third of the weights
before a single KV token exists. What is left is shared with the OS, the
other services on the box, the activation peak and the capture pool.

The engine says so itself, in its own boot log: it prints the KV size that
would "fully utilize gpu memory", which on the head is a little over twice
what this profile serves. What bounds the budget is the HOST headroom every
box keeps above its guard floor, and that is a property of the **co-tenant
layout**, not of the engine. This profile (0.80) assumes the model boxes
carry only light GPU tenants — an embedder on one rank, one small service
on another, the head bare. In a layout where the model boxes also carry
heavier co-tenant services (a chat UI, document parsing, speech services),
the same 0.80 puts the tightest box under its floor and the guard kills the
lane; drop `GMU` in `config/cluster.env` until every box clears its floor.
The guard converting a too-greedy budget into a kill is the designed
failure, not a bug.

The honest summary: **on this model, a three-box Spark ring buys you
throughput and a modest KV pool; it does not buy you a large-context server.**
For large contexts, one box with more memory is a better shape.

## The methodology (this recipe's other product)

Every claim here came from a ladder: one variable per boot, a pre-registered
decision rule *before* the boot, and a verdict recorded either way.
`docs/FIXLOG.md` is that ladder, including the rungs that were rejected and
why — including the two that were rejected *after* they looked like wins.

Three rules did most of the work:

* **One variable per boot.** A rung that changes two things teaches you
  nothing, and the tempting bundle (eager off + capture on + compile on) is
  really three.
* **Pre-register the decision rule.** The threshold, the failure mode, and
  what would make you revert are written down *before* the boot. A threshold
  that moves after the number arrives is not a threshold.
* **A measurement that is missing is a failure, never a skip.** Same for a
  gate that cannot fail: prove it can fail, or it proves nothing.

## Not tested / not claimed

* **The 1M native context is not enabled.** The served context is 196,608,
  because that is what the KV pool holds at this budget. The knob is
  `CTX` in `config/cluster.env`; the capture gate checks the engine's own
  `max_seq_len` against it and checks the denominator of the engine's own
  concurrency figure, so a boot that served a different context is caught
  rather than narrated.
* **Video is disabled** in this profile (`--limit-mm-per-prompt '{"video": 0}'`).
  Images work; the acceptance rule judges the vision path from the
  operator-collected `vision-smoke.txt` (see `docs/VALIDATION.md` §3 — the
  shipped kit does not contain a vision probe, and this recipe does not
  pretend otherwise).
* **No single-box configuration.** This recipe is a three-box ring; a
  one-box or two-box variant is a different recipe with different numbers.
* **The numbers are this stack's.** Another fabric, another driver, another
  co-tenant layout changes the memory arithmetic, and the memory arithmetic
  is what decides the profile.
* **The draft head's quality is the checkpoint's**, not this recipe's. The
  recipe only decides how much of it to spend.

## Repo layout

```
LICENSE                    the licence this recipe ships under
.gitignore                 what a checkout must not carry
config/cluster.env         the serving profile of record (edit this)
config/spec-config.json    the speculative-decode config (read once, used twice)
launch/boot-tp3.sh         resolve this box's role, check the kit, call start-tp3.sh
launch/start-tp3.sh        the docker run + the in-container bootstrap
launch/Dockerfile          base image + the ray layer
kit-md5.py                 the repo<->box identity manifest (--write / --check)
kit-md5.txt                the manifest itself: the digest every box must show
patches/                   the TP=3 pad kit (see patches/README.md)
tools/capture-config.sh    capture + gate the engine's RESOLVED configuration
tools/capture-base-config.sh  the base capture gate it inherits
tools/acceptance-check.py  the acceptance rule (fails closed)
tools/assemble-evidence.py turn the per-box evidence into the rule's aggregates
tools/bench-decode.py      the bench harness (nonce-isolated, warmup-aware)
tools/resolve-capture-grid.py  prove the capture cap against the derivation
tools/guard-memory.sh      the unified-memory guard
docs/FIXLOG.md             the ladder, including the rejections
docs/VALIDATION.md         how to validate a boot, and what each gate proves
```

## Credits & prior art

* MiniMaxAI for **MiniMax M3.1 Flash** and the in-checkpoint **DSpark**
  draft head — <https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash>
* The vLLM team for the engine, the TP executor and the sparse-attention
  backends this recipe runs on, and for the public upstream recipe for this
  model, whose mandatory serve flags this profile follows —
  <https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash> and
  <https://github.com/vllm-project/vllm>
* NVIDIA for the DGX Spark platform and the GB10 memory model this recipe's
  memory arithmetic depends on — <https://developer.nvidia.com>
