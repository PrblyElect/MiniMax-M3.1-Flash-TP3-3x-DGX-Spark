# VALIDATION — how to prove a boot, and what each gate actually proves

A serving lane that has *booted* is not a serving lane that is *correct*.
This is the order of the gates, what each one proves, and what it costs to
run. Every gate here fails closed: a missing measurement is a failure, never
a skip.

The discipline underneath all of it: **one variable per boot, and a
pre-registered rule before the boot.** If you change a setting, write down
what would make you revert it *before* you see the number.

## 0. Before the boot: the environment

A window on a unified-memory box is not a measurement if the box is also
doing something else.

* **Park what conflicts.** Any service on the tightest box that holds device
  memory or a large host working set competes for the same pot. Stop it,
  **verify it actually stopped**, and restore it afterwards.
* **Verify the headroom you are about to assume.** Compare the box's free
  memory against the floor your guard is armed at, and refuse to continue if
  it is short.
* **No transfers during the window.** A concurrent download or copy dirties
  the page cache and changes the memory arithmetic under the measurement.
* **Drop the page cache on every rank before the boot.** On a unified-memory
  box the page cache competes with the KV pool directly, and a box that has
  just staged hundreds of gigabytes will report a memory profile that does
  not match the one you are about to measure.
* **Record the baseline** — the free memory of each box before the boot.
* **Read the command before you start it.** `PRINT_CMD=1 ./launch/boot-tp3.sh`
  prints the exact `docker run` this box would execute and starts nothing.
  It is the cheapest check there is for the class of mistake that costs a
  window: a value that is in the profile of record but not in the command
  is a value that never reached the container. Run it on the box being
  started. `boot-tp3.sh` is the wrapper that resolves the box's role and
  address from the hostname map; if you invoke `start-tp3.sh` directly
  instead, give it the same values the start path gives it
  (`ROLE`/`NODE_RANK`/`NODE_IP`).

## 0b. Three measurement traps that cost real time

* **`/health` green proves only that the API process is alive**, not that the
  engine loop is healthy. A fresh boot's memory trough can let the kernel
  kill a worker while the endpoint keeps answering 200. Check the engine's
  own state, not the endpoint's.
* **The first deep run pays a JIT tax** that grows with depth, and it is not
  a regression. Pay it once, on a sacrificial run, outside the measurement
  window — and label any number taken before the plateau as cold.
* **Idle-gate every measurement.** Before you measure, read the engine's own
  request counters and require **zero running and zero waiting**. A reading
  taken with traffic in flight — your own warmup replay, a scheduled job, a
  co-tenant's requests, another client — is an unlabeled *mixed* number, not
  a property of the boot, and acting on it sends you chasing a problem that
  does not exist. `tools/bench-decode.py` does this check before every
  measured window and warns when the engine is not idle; `--require-idle`
  makes it fatal.

## 1. The boot gates (inside the container, before the engine starts)

These run at every boot, because they are cheap and they are the difference
between a failed boot and a failed *window*.

| gate | what it proves |
|---|---|
| `python3 patches/apply_mm31_tp3pads.py <site>` | the pad kit applied cleanly to the engine's own tree |
| `python3 patches/apply_mm31_tp3pads.py <site> verify` | the applied tree matches the pad table — the kit is idempotent, and a half-applied pad is worse than none |
| `python3 patches/verify-draft-pad.py --model-dir ... --spec-config ... --served-spec-json ...` | the **draft head** is padded to TP=3, proved from the real engine path: the real config loader, the real speculative-config promotion, the real pad hook, the real checkpoint's tensor headers |
| `python3 patches/verify-draft-pad.py --selftest` | that the gate above **can fail**. It mutates the inputs — drops the whitelist entry, uses an unpadded config, corrupts a rule, merges the two rule scopes — and requires each mutation to be rejected. A gate that cannot fail proves nothing |
| `python3 /kit-md5.py --check-container --site <site>` | the pad overlay **landed in the container**: the installed package is byte-for-byte the recipe's own, and every patched stock file is byte-for-byte what the kit's edit produces. The kit's own `verify` above proves its edits SEMANTICALLY; this proves the BYTES, which is the half a half-applied overlay gets wrong |
| `python3 /kit-md5.py --selftest` | that the container check **can fail**. It builds a synthetic tree from the kit's own anchors, applies the kit to it, and requires the check to accept the tree it produced and to reject a tampered file, a tampered overlay and a missing overlay |

## 2. The capture gate: what the engine actually resolved

**THE GATE RUNS ON EACH BOX, NOT ONCE.** It reads the running engine and the
capture-graph facts out of the container **on the box it runs on**, and it
writes that box's findings under that box's **lane label** — the one passed
to `--head`/`--r1`/`--r2`, not the raw hostname, because `hostname -s` trims
the domain and the acceptance rule keys its sections by the lane label.

```sh
# on EACH box, with the SAME three lane labels every time:
bash tools/capture-config.sh --head spark-head.local --r1 spark-a.local \
  --r2 spark-b.local --out logs/capture
```

Then collect the per-box evidence into one directory on the machine that
runs the rule (the guard's two logs live under the checkout's `logs/guard/`).
`$REMOTE_DIR` below is the quoted-tilde remote path from the README's step
0b -- the remote shell expands it, so it must not be expanded locally:

```sh
mkdir -p logs/evidence
for h in spark-head.local spark-a.local spark-b.local; do
  scp "$h":"$REMOTE_DIR"/logs/capture/capture-gate-$h.txt logs/evidence/
  scp "$h":"$REMOTE_DIR"/logs/guard/guard.log          "logs/evidence/guard-$h.log"
  scp "$h":"$REMOTE_DIR"/logs/guard/guard_mem.log      "logs/evidence/guard-$h-mem.log"
done
```

This reads the **resolved** configuration out of the running engine and the
capture-graph facts out of the containers, and it checks:

* the memory budget, the served context and the capture grid are the ones the
  profile says they are (the engine's own resolved arguments, not your
  intent);
* the capture pool's **actual** size and the engine's **estimate** of it, and
  the gap between them;
* the capture-graph counts — the PIECEWISE set, the FULL decode set, and the
  draft descriptor count;
* each rank's own capture-pool actual-versus-estimate pair, which is how
  **per-rank symmetry** is proved: the engine prints the graph bars on the
  first rank only, so the head's bars are not evidence about the other two;
* the KV pool, the engine's concurrency factor, the KV cost per token and
  the speculative-decode configuration (including the engine's own field-name
  spelling for the number of speculative tokens).

**The capture gate reads the ENGINE. The host memory comes from the guard.**
Each box's free memory over the window is the guard's evidence, and the
acceptance rule is what checks it against the floors — the capture gate has
no opinion about the boxes' memory, because the engine cannot see it.

**Why the resolved config and not the intent:** a default that disagrees with
the profile is invisible until the engine prints it. The one that matters
most is the served context: asking for twice what the pool holds produces a
request-time refusal, not a warning. **The concurrency figure is read
together with its denominator**, because the engine prints it *for the
request length it was asked about*: the same pool reads ~1.71x at 131,072
tokens and ~1.14x at 196,608, and comparing those two numbers as if they
measured the same thing would be wrong in both directions.

**Quote the engine's own pool**, from its boot log's KV-size line — never an
estimate of it — and band it: the pool is boot-labile.

### 2b. The capture gates' offline self-tests

Both capture gates carry a **`--selftest` entry point**. It runs the gate's
OWN logic — the same code the live run executes — over synthetic engine logs:
no container, no lane and no network. It prints a PASS/FAIL line per check
and exits non-zero if any check fails.

```sh
# on any machine -- no box, no lane, no engine:
bash tools/capture-config.sh --selftest
bash tools/capture-base-config.sh --selftest
# expected: a PASS line per check, then
#   capture-config: SELFTEST PASS -- ... (rc=0)
#   capture-base-config: SELFTEST PASS -- ... (rc=0)
```

| gate | what its self-test proves |
|---|---|
| `bash tools/capture-config.sh --selftest` | that the capture gate **can fail**. It requires a healthy synthetic boot to PASS and to publish its machine-readable facts, and requires a malformed thousands grouping, a below-floor pool and an interrupted capture bar to be REFUSED — each NAMED, the malformed figure not laundered into a clean integer, and the interrupted bar carried as the raw pair rather than as a count |
| `bash tools/capture-base-config.sh --selftest` | that the base capture gate **can fail**. It requires a healthy synthetic boot to PASS, and requires a malformed thousands grouping, a below-floor pool, a boot that still logs `Enforce eager set`, and an encoder that IS captured to be REFUSED |

The self-test drives the shipped script rather than a copy of it: it writes a
synthetic log and runs the gate over it, so the assertions it exercises are
the gate's own and cannot drift away from them.

## 3. The acceptance rule

**FIRST, ASSEMBLE THE AGGREGATES.** The rule reads three files that are
aggregates over the three boxes, and the shipped tools produce the per-box
facts those aggregates are made of:

```sh
python3 tools/assemble-evidence.py --evidence-dir logs/evidence \
  --head spark-head.local --r1 spark-a.local --r2 spark-b.local
```

It writes `capture-gate.txt`, `kit-check.txt` and `guard.txt` into the
evidence directory, in the section-and-key contract the rule reads, and it
**refuses to write anything** if a box's evidence is missing — a box with no
evidence is an error, not an empty section. **And it removes those three
files if they are already there**, so a refusal cannot leave the previous
cycle's aggregates in the directory for the rule to judge as this cycle's
evidence: a stale `capture-gate.txt` would otherwise be read as a box's
evidence that was never collected.

Then run the rule:

```sh
python3 tools/acceptance-check.py --lane-url http://127.0.0.1:8100 \
  --head spark-head.local --r1 spark-a.local --r2 spark-b.local \
  --evidence-dir logs/evidence
```

### What the evidence directory has to contain

The rule reads **nine** files, and it is fail-closed about every one: a
missing measurement is a FAIL, never a skip.

| file | who writes it | what the rule looks for |
|---|---|---|
| `bench.txt` | `tools/bench-decode.py --evidence-dir` | the measured rows, labelled by the path they were measured over |
| `manifest.txt` | the same | `rule_md5=`, `lane_url=`, and `relay_url=` when the deployment has a second endpoint |
| `capture-gate.txt` | `tools/assemble-evidence.py` | one `=== <lane label> ===` section per box, and inside it the gate's own `[GATE] EXIT=0` — the gate's VERDICT, machine-readable. An `--extract-only` section says `EXIT=extracted` instead, because facts are not a verdict |
| `kit-check.txt` | `tools/assemble-evidence.py` | `kit-md5: OK` inside **each box's own section** — a box carrying a different kit must fail here, not be masked by another box's OK |
| `guard.txt` | `tools/assemble-evidence.py` | `armed_threshold_mb=`, `guard_start_lines=`, `guard_log_breaches=`, `guard_window_breaches=`, plus the guard's own `low_mb=` / `avail_mb=` / `breaches=` samples |
| `probes.txt` | **you** | the literal `all 7 probes PASS` |
| `vision-smoke.txt` | **you** | an answer that names the colour Red — the whole word, read from the model's answer (its `content:` field, or the file's text if your capture writes no field) — and does not DENY it: an answer that negates the colour in front of it ("not Red", "isn't red") is a broken vision path, not a red answer |
| `c1-proof.txt` | **you** | the literal `C-1 PROOF PASS` (the byte-identity proof over the three ranks) |
| `ring-proof.txt` | **you** | the literal `every rank issued the same collective sequence` |

**THE FOUR YOU WRITE ARE MEASUREMENTS, NOT FORMATTING.** The shipped kit
contains no probe, vision, byte-proof or collective-ring harness, so those
four are yours to measure and yours to record; the assembler deliberately
writes nothing for them, because a rule that can be satisfied by writing a
template is not a rule. Their exact requirements are in the table above.

The rule checks the lane's behaviour, not just its configuration:

* the KV pool is at or above the floor — one full-length request at the
  served context — and the engine's own concurrency factor, read **at the
  denominator the engine printed**, is at or above its floor;
* the observed per-token latency and the time to first token are within their
  ceilings;
* the draft acceptance at position 0 is at or above its floor;
* the two token-accounting accounts agree: the server's usage token count
  against the engine's decode steps times its tokens per step, and the
  server's counted tokens per step against the engine's draft-accounted
  tokens per step (a mismatch means the measurement is wrong, not the
  model);
* every box's free memory stayed above its floor for the whole window, and
  the margin on the tightest box is itself checked.

**Where each floor comes from.** The rule reads the **profile of record**
(`config/cluster.env`) for everything the profile publishes, and holds the
rest as its own acceptance criteria. A floor the profile publishes is
changed **in the profile and nowhere else** — the profile's value wins,
and the rule keeps a literal only as a fallback for a key the profile
does not publish.

A floor that is the rule's own criterion is a **literal in
`tools/acceptance-check.py`**. Most of them are not environment-
overridable at all; the exceptions are named in the rule (for example
`ACCEPT_CONCURRENCY_NO_LOSS_X`), and an override is a decision, not a
convenience.

| criterion | floor | where the floor lives |
|---|---|---|
| KV pool | one full-length request at the served context (196,608 tokens) | the profile: `CTX` |
| concurrency at the served context | 1.00x | the rule's own criterion |
| capacity, read like-for-like at 131,072 tokens | 1.40x | the rule's own criterion (`ACCEPT_CONCURRENCY_NO_LOSS_X`) |
| draft acceptance, position 0 (merged, and each workload class) | 0.5 | the rule's own criterion |
| per-token latency | within the rule's own band | the rule's own criterion |
| time to first token | 0.60 s | the rule's own criterion |
| long-context prefill | 8,192 tokens as a FLOOR: the engine's own measured length must reach it | the rule's own criterion |
| capture coverage | the engine's capture ceiling must reach 128, and its list must contain 128 | the rule's own criterion; the profile separately pins the exact grid the gate **and the rule** check (`EXPECTED_CG_MAX`, `EXPECTED_CG_LIST`, `EXPECTED_CG_GRAPHS`) |
| guard floor, head / rank 1 / rank 2 | 4,096 / 3,584 / 2,432 MB | the profile: `GUARD_MIN_HEAD_MB`, `GUARD_MIN_R1_MB`, `GUARD_MIN_R2_MB` |
| margin over the floor on the tightest box | 1,000 MB | the profile: `GUARD_MARGIN_MB` |
| memory budget, asserted on every rank | the profile's budget (0.80) | the profile: `GMU` |

So there is **one place to change each** profile-published floor. A value
that the profile publishes but that is present and unparseable is a hard
error — the rule refuses to run rather than silently falling back to its
own literal.

**The rule is a set of criteria, not a report.** It exits non-zero when a
criterion fails. A criterion whose input is missing fails; it does not skip.

**The pool and the concurrency are judged on the HEAD's figures.** The
engine's resolved configuration is the head's, and the capture gate publishes
the engine's own concurrency figure for the head alone. The KV pool's token
count is read from each box's own log, so a box whose log carries it
publishes it and a box whose log does not publishes an empty value — **a
rank's empty capacity figure is the contract, not a missing measurement.**
The rule judges the head's figures, says whose figures they are, and treats a
rank that publishes one as something worth naming — a figure it cannot read
is a damaged fact, and a figure that disagrees with the head's is a ring
that disagrees with itself. The memory budget, by contrast, is a per-rank
fact, and it is checked on every rank.

## 4. The bench harness

```sh
python3 tools/bench-decode.py --lane-url http://127.0.0.1:8100 --tokens 800 \
  --rounds 2 --streams 1,4,8,16 --workload prose,high-draft \
  --prefill-tokens 8192 --evidence-dir logs/evidence
```

**ONE RUN, ONE `bench.txt`, EVERY CLASS.** `--workload` takes a comma list,
and the default is `prose,high-draft`: the rule judges the SPLIT — a
free-prose class and a high-draftability class, each with its own acceptance
histogram — so a bench that measured one class per run produced an evidence
set the rule could never accept, and overwrote the previous run's
`bench.txt` besides. `--prefill-tokens` defaults to 8,192, because the rule
also judges that a long-context prefill IS measured.

**THE PREFILL BUDGET IS A TOKEN BUDGET, NOT A COPY COUNT.** The bench asks the
engine for the prompt's own token count, grows the prompt until that count
reaches the declared budget, and publishes **both** numbers: the DECLARED budget
and the MEASURED length. The measured length can sit above the declared one —
the row carries the number that was really sent, which is the only number worth
reading. The rule then refuses a row whose measured length does not reach the
budget it declares, and refuses a budget below the recipe's own 8,192; a bench
that cannot reach the budget publishes **no** prefill row at all rather than a
short one.

**THE DECLARED NUMBER IS A FLOOR, NOT A SIZE.** `--prefill-tokens 8192` says
"the measured length must reach 8,192 tokens", not "the prompt is 8,192 tokens
long". The engine counts the prompt it was given, and the bench cannot ask for
a length the engine has not measured, so the prompt that is sent may be
slightly longer than the floor. That is why the published row carries the
engine's own count and why the rule judges that count: the floor is what the
rule enforces, and the count is what was measured.

**A REFUSED PROMPT FAILS CLOSED, AND IT COSTS THE RUN NOTHING.** If the engine
refuses the prefill request — a context length, a payload too large — the
bench names the refusal, publishes **no** prefill row, and exits non-zero. It
does not take the whole run down with it: the decode rows, the histogram and
the manifest are still written, because one refused prefill must not cost the
lane its measurement. The refusal is not retried with a shorter prompt: the
declared number is a floor, so every prompt that could satisfy it is at least
that long, and a shorter one could only produce a row the rule has to reject.

Three things make its numbers mean something, and all three are load-bearing:

* **A nonce at the FRONT of the prompt.** Prefix caching is on, so a
  measured prefix that has been seen before is a different measurement. The
  nonce goes first, so the cached prefix is never the measured one.
* **A sacrificial warmup.** The first request pays a JIT tax. The warmup
  runs on a different prefix and never warms what is measured.
* **A workload split.** The acceptance rate — and therefore the value of the
  draft budget — depends on the content. Report the workload with the
  number, and never quote a spec-decode throughput figure without one.

The concurrency sweep is the interesting part: aggregate throughput is not
the single-stream number times the stream count, and the gap between them is
where the memory pressure shows up.

**`--evidence-dir` writes the evidence set the rule reads** — `bench.txt`
(the measurement rows, the token-accounting cross-check, the inter-token
figure and the acceptance histograms) and `manifest.txt` (the driver's own
record of the endpoint it was pointed at, and the digest of the rule the
evidence was collected for). The rule selects the rows it judges by the
endpoint they were measured over, so every row carries it.

**The token accounting cross-check is the instrument's own proof.** Two
accounts have to agree, and they come from different machinery: the
engine's draft accounting (the draft rounds it ran and the draft tokens it
accepted give the tokens it emitted per decode step) and the server's own
count (the usage block's token count over **the engine's own decode-step
count**). The two share that denominator and have independent numerators —
the engine's accepted-draft counter on one side, the server's usage block on
the other — so neither can be derived from the other, and a disagreement
between them is a real disagreement. **The client's SSE chunk count is
published beside them as context only**: the engine coalesces a decode
step's tokens into one chunk, so the chunk count is a step count only
because of that engine behaviour, and a criterion that depended on it would
break on an engine that streamed one delta per token. On a healthy lane the
two accounts agree to within a fraction of a percent; if they do not, every
other number in the file is suspect, and the rule says so.

**The rule recomputes both comparisons itself.** It derives them from the
published counters rather than taking the bench's percentages on trust, and
it requires the server-side account to be published at all. A cross-check
whose "deviation" was computed against its own definition — a quantity
compared with itself, which is identically zero and can never fail — is not
accepted as agreement.

**One endpoint, or two.** The bench measures the lane. If your deployment
puts a proxy in front of it, measure that too — the bench takes the second
endpoint as an argument and measures it in the same rounds, over its own
path, with its own nonce and its own sacrificial warmup:

```sh
python3 tools/bench-decode.py --lane-url http://127.0.0.1:8100 \
  --relay-url http://127.0.0.1:8001 --tokens 800 --rounds 2 \
  --evidence-dir logs/evidence
```

The second endpoint's rows carry `path=relay` and are labelled with the
endpoint they were measured over, and the evidence manifest records
`relay_url=`. That record is what makes the rule judge the second endpoint
at all: it has its own throughput floor, and the rule asserts that the
recorded endpoint is the one the floor was calibrated on
(`ACCEPT_BASELINE_RELAY_PATH`, whose default is `127.0.0.1:8001`).

A deployment that serves a single endpoint has no such floor, and the rule
reports that explicitly rather than skipping it quietly. Which is why the
second endpoint has to be **measured**, not merely named: a manifest that
names an endpoint the bench never measured has no rows for it, and the rule
treats that as a FAIL, never a skip.

## 4b. The correctness gates (in a window, on the head)

These two are not measurement gates: they ask the engine whether it is
**correct**, not how fast it is. Both run against the live lane on the
head, in a boot window, and both are **load-bearing** — a FAIL is a reason
not to ship the lane, and neither of them can be skipped silently.

| gate | what it proves |
|---|---|
| `python3 tools/check-spec-decode-short-prompt.py` | a prompt of **exactly** the engine's own `uniform_decode_query_len` tokens (four, at k=3) produces a sane answer at temperature 0, and the same probe one token longer — a shape the upstream dispatch check does NOT misread — produces the same answer. The upstream defect dispatches such a prefill as a uniform-decode step by shape alone, and the symptom is deterministic garbage from the first generated token |
| `python3 tools/check-long-prompt-determinism.py` | a prompt that crosses the batched-token budget produces the **same token sequence** cold and with the prefix cache warm. Long prompts take the chunked prefill path, and that is where a serving engine stops being deterministic |

Both build their prompts from the engine's own `/tokenize` endpoint and
assert the engine's own `usage.prompt_tokens`, because the token count has
to be exact and a character-level search cannot make it exact: byte-level
BPE counts are not monotonic in the prefix length. The long-prompt gate
puts a fresh nonce at the HEAD of its prompt, so the first call is genuinely
cold even when the same gate is run again on the same lane.

The short-prompt gate resolves the engine's own
`uniform_decode_query_len` from three sources, in order: a value stated
with `--query-len`, then the engine's boot log (the line the draft-pad
gate prints), and finally — if neither is reachable — a value **derived**
from the recipe's own speculative config, where the window is 1 + k. It
always says which source it used, and a derived value is labelled as
such rather than presented as a measurement.

## 5. The memory guard

```sh
# with config/cluster.env sourced, the floor comes from the PROFILE --
# the right floor for this box, resolved by role and then by hostname:
bash tools/guard-memory.sh --container minimax31-flash-tp3

# or, to pin it by hand:
bash tools/guard-memory.sh --floor 2432 --container minimax31-flash-tp3
```

On a unified-memory box the guard is part of the serving contract, not an
accessory. It watches free host memory and kills **only this recipe's**
container if it stays under the floor for the debounce count.

* **A single low sample must not kill a serving lane** — hence the debounce.
* **The floor is per box**, and the tightest box sets it.
* **The floor is a profile value, not a kit edit.** `tools/guard-memory.sh`
  takes its floor from the profile of record (`GUARD_MIN_HEAD_MB`,
  `GUARD_MIN_R1_MB`, `GUARD_MIN_R2_MB`) -- resolved by `ROLE` first and then
  by hostname, the same way the boot resolves the box's role -- so the kit's
  manifest stays untouched by a site's floors. An explicit `--floor` still
  wins. And if NO floor resolves, the guard **refuses to arm**: the built-in
  fallback is weaker than every floor this recipe pins, and a guard that
  cannot hold is worse than no guard, because it looks like one.

## 6. What to do when a gate fails

**Diagnose forward.** A failed gate is evidence, not a verdict: preserve the
failed state (the container, the logs, the capture), and read it. Reverting a
configuration is the operator's decision, and a failed gate is not by itself
an instruction to revert.

Two failure modes are worth naming, because they look identical on the
outside and mean opposite things:

* **The configuration is wrong.** The engine says what it resolved, and it
  differs from the profile. Fix the profile.
* **The configuration is right and the box is different.** The engine
  resolved exactly the profile, and a gate still failed on the tightest box.
  That is a tenant question, not a configuration question, and no amount of
  knob-turning fixes it.
