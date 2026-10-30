#!/usr/bin/env python3
"""acceptance-check.py -- the acceptance rule for this serving lane.

WHAT IT IS
  One rule, one verdict. It reads an EVIDENCE DIRECTORY produced by a
  serving cycle, prints one line per criterion -- [PASS], [FAIL] or
  [NOTE] -- and exits 0 only when every criterion is met.

WHAT IT CHECKS
  1. The bench evidence exists, every stream row counts its tokens from the
     engine's own usage block, and that usage block agrees with the
     chunk x tokens-per-step accounting and with the engine's own
     tokens-per-step ratio. EVERY MEASUREMENT ROW IS ALSO ATTRIBUTABLE to a
     class the rule judges: a row the rule cannot attribute is named and
     FAILS, because a row the rule cannot judge is a missing measurement.
  2. The measurement path is proven. The speed thresholds are calibrated on
     two pinned endpoints, and a threshold calibrated on one path cannot be
     applied to another -- so the rule asserts that the driver recorded the
     endpoint it was actually pointed at (the evidence manifest is the
     machine-written record of that).
  3. The rule that produced the verdict is the rule the evidence was
     collected for (the digest self-check; see USAGE).
  4. The speed criteria: the single-stream lane rate, the per-token
     inter-token latency, the time to first token, the 4-way concurrency
     floor, the draft acceptance at position 0, and the workload split (a
     free-prose class, a high-draftability class, the 4-, 8- and 16-way
     aggregates, a long-context prefill, and the per-position acceptance
     histogram). A SECOND ENDPOINT in front of the lane (a proxy) has its
     own floor, and those criteria apply only when the driver's manifest
     records one -- a deployment that serves one endpoint has no such
     floor, and the rule says so explicitly rather than skipping it
     silently. An acceptance window published MORE THAN ONCE is merged
     worst-case over its publications, so the verdict cannot depend on
     which publication came last.
  5. The capture gate: it ran clean on all three ranks; the speculative
     config reached every rank; the memory budget is the pinned one on
     every rank; the capture coverage reaches the lane's shape; and the
     resolved capture grid is the pinned grid, with the pinned number of
     piecewise, full and draft graphs. Each rank's section is published
     EXACTLY ONCE: a section published twice is an ambiguous measurement,
     and the rule cannot tell which publication is this cycle's.
  6. The capacity floor: the KV pool and the engine's own concurrency
     factor, and the capture pool's per-rank actual and estimate.
  7. The kit identity on every rank, and each rank's kit-check section
     published exactly once.
  8. The correctness battery: the thinking probes, the vision smoke test,
     the byte-identity proof and the collective-ring proof.
  9. The memory guard: it never fired on any rank, it ran with the pinned
     floor, its window covers this cycle alone, the running free-memory
     minimum stayed above that rank's floor for the whole window, and each
     rank's guard section is published exactly once.

THE RULE OF THUMB
  A MISSING MEASUREMENT IS A FAILURE, NEVER A SKIP. If a number the rule
  needs was not recorded -- an absent evidence file, an absent line in a
  log, an absent figure in a fact block -- the criterion that needs it
  FAILS. Nothing is skipped for being unavailable, and nothing is reported
  as a pass because the measurement was not taken. AND AN AMBIGUOUS
  MEASUREMENT IS WORSE THAN A MISSING ONE: where the evidence publishes
  two readings of the same thing, the rule takes the worst of them, or --
  where no merge can be right -- names the ambiguity and FAILS.

THE VERDICT
  ACCEPT     every criterion is met.
  NO-EFFECT  the profile applied and the capacity did not move at all (the
             open band between the reject line and the keep line) -- a
             rejection, not a pass.
  REJECT     any criterion failed.
  The rule exits 0 on ACCEPT and 1 on NO-EFFECT or REJECT.

EVIDENCE DIRECTORY
  bench.txt          the bench output: the measurement rows, the token
                     accounting cross-check, the inter-token line and the
                     per-position acceptance histograms. BOTH workload
                     classes are in it, because the rule judges the SPLIT.
  manifest.txt       the driver's own record: rule_md5=, lane_url= and
                     relay_url=.
  capture-gate.txt   the capture gate's output, one '=== <label> ===' section
                     per rank, each with a machine-readable fact block AND
                     the gate's own verdict, '[GATE] EXIT=0'. An
                     extract-only section says 'EXIT=extracted', because
                     facts are not a verdict. A rank's section published
                     MORE THAN ONCE is a named failure: the rule keeps the
                     last publication, and it cannot tell which one is this
                     cycle's.
  kit-check.txt      the kit identity check, one section per rank.
  probes.txt         the thinking probes.
  vision-smoke.txt   the vision smoke test.
  c1-proof.txt       the byte-identity proof.
  ring-proof.txt     the collective-ring proof.
  guard.txt          the memory guard's log, one section per rank.

  THE SECTIONS ARE KEYED BY LANE LABEL, NOT BY HOSTNAME. The label is what
  --head/--r1/--r2 carry, and it is what the capture gate writes into each
  box's own section, so the rule can match a section to a box without
  knowing what that box is called.

  A SECTION PUBLISHED TWICE IS AN AMBIGUOUS MEASUREMENT, AND THE RULE
  NAMES IT AS A FAILURE. The section parser keeps the last publication of a
  name, so a failing publication that a clean one follows would otherwise
  be invisible -- and a failing capture gate published before a clean one
  was accepted. The check is scoped to the ranks the rule judges, because
  the capture gate's own sub-sections ("=== log source: ... ===", "=== role:
  ... ===") are not rank sections.

USAGE
  python3 acceptance-check.py [OPTIONS] EVIDENCE_DIR [EXPECTED_RULE_MD5]

    --evidence-dir DIR, --logs DIR   the evidence directory (the positional
                                    form does the same)
    --lane-url URL                   the pinned lane endpoint
    --relay-url URL                  the pinned relay endpoint
    --head HOST --r1 HOST --r2 HOST  the lane's three box labels
    --expect-rule-md5 DIGEST         the digest of the rule file itself

  The optional digest pins the rule file itself: CI supplies the digest of
  the rule it expects, and a rule file that changed under the evidence
  fails the check instead of reporting a verdict over it. Unset, the
  manifest's own record decides -- which is still fail-closed, because a
  missing record is a failure.

  The expectations below are overridable, so the same rule can be pointed
  at a different profile or a different engine build:

    ACCEPT_BASELINE_PATH              the pinned lane endpoint
    ACCEPT_BASELINE_RELAY_PATH        the pinned relay endpoint
    ACCEPT_POOL_NO_LOSS_TOKENS        the pool floor for "no capacity lost"
    ACCEPT_CONCURRENCY_NO_LOSS_X      the concurrency floor for the same
    ACCEPT_CAPTURE_POOL_ACTUAL_MAX_GIB    the per-rank capture-pool band
    ACCEPT_CAPTURE_POOL_ESTIMATE_MAX_GIB  the per-rank estimate band
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    """A float from the environment, falling back to the default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """An int from the environment, falling back to the default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ══ THE ACCEPTANCE CRITERIA (this recipe's own thresholds) ═══════════════
# Every number in this block is the recipe's acceptance criterion -- the
# floor the shipped serving profile has to clear. They are NOT derived from
# a measurement, and the RULE does not re-weight them per run: they are the
# profile's contract, and a boot that misses one of them fails the rule.
#
# FOUR OF THEM CAN BE OVERRIDDEN FROM THE ENVIRONMENT, and an override
# changes the CONTRACT rather than re-weighting it: SERVED_CONTEXT
# (ACCEPT_SERVED_CONTEXT), KV_FLOOR_TOKENS (ACCEPT_KV_FLOOR_TOKENS),
# KV_NOTE_TOKENS (ACCEPT_KV_NOTE_TOKENS) and COMPARISON_CONTEXT
# (ACCEPT_COMPARISON_CONTEXT). They are the operator's escape hatch for a
# lane whose served context differs from the shipped one; the speed floors,
# the acceptance floor, the round count and the concurrency levels in this
# block are the profile's own and are not overridable. THREE of the four
# name a DIFFERENT criterion -- the criterion's own NAME carries the
# number it is judged against -- so a reader can see which contract was
# applied. The FOURTH, KV_NOTE_TOKENS, names NO criterion at all: it is
# the figure the CAPACITY NOTE compares the pool against, so an override
# moves the line the NOTE draws and renames no criterion.
#
# Speculative decoding buys speed and pays for it in capacity (the draft's
# weights and the draft's own KV group) and in host-memory headroom (the
# capture pool grows). This recipe pays for the speed in KV CAPACITY
# instead, and buys the headroom back by LOWERING the memory budget. So the
# bar is: clear the speed floors by enough to pay for that, and do not lose
# the baseline's concurrency throughput.
LANE_STREAM_KEEP = 24.0        # the single-stream lane floor: a real effect
                               #   over the baseline, not noise
LANE_STREAM_REJECT = 22.5      # at or below this the profile gained nothing
                               #   (22.5, 24.0) is NO-EFFECT -> REJECTED
RELAY_STREAM_MIN = 23.5        # the relay floor
INTER_TOKEN_MAX_MS = 42.0      # per TOKEN, not per step
INTER_TOKEN_REJECT_MS = 52.0   # a regression, not a win
TTFT_MAX_S = 0.60              # time to first token
CONC4_AGG_MIN = 54.5           # spec decode may not COST concurrency
                               #   throughput
CONC4_PER_STREAM_MIN = 13.5    # the per-stream minimum under that load
ACCEPT_POS0_MIN = 0.5          # the draft acceptance at position 0
# THE LONG-CONTEXT PREFILL BUDGET. THIS IS NOT A NEW MAGIC NUMBER: it is
# the recipe's OWN declared budget, already published in THREE places in
# this artifact --
#
#   * tools/bench-decode.py -- the `--prefill-tokens` flag's DEFAULT;
#   * README.md -- the documented invocation;
#   * docs/VALIDATION.md -- the documented invocation.
#
# The rule reads the budget the EVIDENCE carries (`target_tokens`, the
# budget the bench declared when it measured) and checks it against this
# one, so a prefill of a shorter budget is not the long-context prefill
# the criterion names.
PREFILL_BUDGET_TOKENS = 8192
# THE CRITERION'S NAME STATES THE FLOOR IT APPLIES, the way "acceptance at
# position 0 >= 0.5" does. A criterion named "long-context" while it
# accepts a one-token prompt is not publishable: the name is the published
# claim, and this one now claims a floor and proves it.
#
# THE NUMBER IS THE FLOOR ON THE MEASURED LENGTH. The bench sends a prompt
# that reaches the floor -- it may send a slightly longer one, because the
# engine counts the prompt it was given and the bench cannot ask for a
# length the engine has not measured -- so the row's MEASURED length is
# what the criterion judges, and the floor is what the name states.
PREFILL_CRIT = (f"the workload split: a long-context prefill of at least "
                f"{PREFILL_BUDGET_TOKENS} tokens is measured")
# THE SERVED CONTEXT THIS RECIPE PINS. The engine prints its concurrency
# FOR THE REQUEST LENGTH IT WAS ASKED ABOUT, so the denominator has to be
# part of the criterion -- a figure written for one context compared against
# a boot that served another compares two different things.
SERVED_CONTEXT = _env_int("ACCEPT_SERVED_CONTEXT", 196_608)
# THE CAPACITY FLOOR FOLLOWS THE CONTEXT: the lane must be able to hold ONE
# full-length request AT THE CONTEXT IT SERVES. (It used to be a flat
# 150,000 tokens, which was the same idea for a 131,072-token context.)
KV_FLOOR_TOKENS = _env_int("ACCEPT_KV_FLOOR_TOKENS", SERVED_CONTEXT)
# A POOL BETWEEN THE FLOOR AND THIS IS AN ACCEPT WITH AN EXPLICIT CAPACITY
# NOTE. Twice the served context is TWO full-length requests: below it, the
# lane holds one request and not two, and that price is named rather than
# hidden.
KV_NOTE_TOKENS = _env_int("ACCEPT_KV_NOTE_TOKENS", 2 * SERVED_CONTEXT)
CONCURRENCY_MIN = 1.00         # the lane can still hold one full-length
                               #   request
# THE CONTEXT THE RECIPE'S CAPACITY WAS MEASURED AT. The engine's own
# concurrency figure is a property of (pool / request length), so "the
# concurrency did not fall below X" is only comparable at ONE denominator.
# The like-for-like form is used: the pool is read at THIS request length.
COMPARISON_CONTEXT = _env_int("ACCEPT_COMPARISON_CONTEXT", 131_072)
MIN_ROUNDS = 2                 # rounds over the pinned baseline path
REQUIRED_CONC_LEVELS = (4, 8, 16)
# THE WORKLOAD VOCABULARY IS CLOSED. The producer publishes exactly two
# classes (tools/bench-decode.py: WORKLOADS = {"prose": ..., "high-draft": ...}).
# A LABEL THAT IS NOT ONE OF THEM IS NOT A CLASS THIS RULE KNOWS: it is
# neither the free-prose class nor the high-draft class, it cannot be
# counted as prose, and it cannot be quietly dropped -- it is NAMED.
KNOWN_WORKLOADS = ("prose", "high-draft")
# THE WINDOWS WHOSE PER-POSITION DECAY THE RECIPE PUBLISHES. These are the
# rule's OWN known windows -- they are NOT excludable, so the declaration
# below cannot be used to excuse a window that lost its histogram.
CORE_PER_POSITION_WINDOWS = ("overall", "prose-1way", "high-draft-1way")

# THE WINDOWS THAT PUBLISH NO PER-POSITION HISTOGRAM, BY NAME. The prefill
# window measures a prefill, not a draft acceptance: it carries no
# per-position acceptance numbers. The exclusion is a SHORT PUBLISHED LIST,
# and it can only cover windows that are NOT core windows.
NO_PER_POSITION_WINDOWS = ("prefill",)
# THE NEGATIONS THAT CAN DENY A COLOUR. "n't" IS LISTED ON ITS OWN because
# it is a SUFFIX: "isn't" and "doesn't" are negations too. The list is used
# only in the narrow window in front of the colour (see the vision check),
# where a negation in the SAME CLAUSE as the colour means the answer DENIES
# the colour it names.
NEGATION_TOKENS = ("not", "n't", "no", "never", "nor", "without", "neither")
# THE LEVEL TOKEN, AND ONLY THE LEVEL TOKEN. The bench writes the
# concurrency level as "x4" -- "concurrent x4" in the row's mode, "lane
# concurrent x4 <endpoint> [r1]" in its label. The pattern REQUIRES the "x",
# so a label whose token was stripped cannot hand the rule the ENDPOINT's
# digits instead: "lane concurrent 8.8.8.8:8100 [r1]" would otherwise read
# as the level 8, and 8 IS a level the rule judges -- so the row would be
# filed under a criterion that has no floor, escaping both judgement and
# the naming.
#
# THE BOUNDARY IS WHAT MAKES "THE TOKEN" TRUE. The digits must not be
# followed by a WORD character or a dot: that rejects "x4.5" and "x45", and
# it also rejects a HOSTNAME that happens to start with "x" and digits --
# "lane concurrent x16host:8100 [r1]" would otherwise read as the level 16
# out of the middle of a host name. The token is the token, so the token
# must END there: a space, or the end of the string.
#
# The match is case-insensitive because the lane's earlier bench wrote the same
# token in caps ("lane CONCURRENT x4 [r1]").
_LEVEL_TOKEN_RE = re.compile(r"concurrent\s+x(\d+)(?![.\w])(?![,]\d)",
                             re.IGNORECASE)
# ══ THE GUARD'S OWN MARKERS ═════════════════════════════════════════════
# THE GUARD SAYS THINGS WITH MARKERS, AND A MARKER IS NOT A FIGURE.
#
# "I COULD NOT READ IT". guard-memory.sh:234 writes `avail_mb=${avail:-?}`
# when /proc/meminfo could not be read on that tick, so the '?' is the guard
# SAYING SO: it is not a figure, and it is not a damaged one either. A rule
# that called it DAMAGE would FAIL a box for a tick the guard itself flagged,
# and a rule that read it as a figure would publish a measurement of nothing.
_GUARD_NO_READING = "?"
#
# "I HAVE NOT SAMPLED YET", IN BOTH FORMS THE SHIPPED GUARD WRITES: the DASH
# on the sample line (guard-memory.sh:225-227, while the container is not up
# yet) and the bare 999,999 on the exit line (guard-memory.sh:238). The dash
# is there ON PURPOSE -- "the sentinel is not a measurement ... so the
# acceptance rule would PASS a box on which nothing was ever measured" --
# so it is the guard SAYING SO, not corruption.
#
# IT IS NOT A FIGURE, AND IT HIDES NO MEASUREMENT: the guard moves its
# ratchet only when it has read an avail, so a tick that carries the marker
# carries the ratchet's value as it stood BEFORE the first reading. The
# ticks that DID carry a reading are therefore what the running minimum is
# judged on, and the marker ticks are named.
#
# THE TWO MARKERS ARE NOT THE SAME MARKER. The dash and the '?' are both
# written BY THE GUARD, and both have a KNOWN meaning. A token the guard
# never writes -- a '?' on `low_mb=`, which the shipped guard does not do --
# has an UNKNOWN meaning, and a counter that carries one is judged
# fail-closed: its state is UNKNOWN and the criterion that depends on it
# FAILS BY NAME.
_GUARD_NO_SAMPLE = ("-", "999999")
CAPTURE_COVERAGE_MIN = 128
REQUIRED_BOXES = ("spark-head.local", "spark-a.local", "spark-b.local")

# ══ THE CAPTURE GRID (the shipped profile's pinned grid) ═════════════════
# An explicit capture list resolves to exactly that list, with
# max_cudagraph_capture_size == 128. The engine's own default grid is
# larger (35 sizes, ceiling 256): the sizes above 128 never produced a
# decode graph, so the lane pays for them and never uses them.
#
# The uniform-decode sizes the FULL (decode) graphs are built from must all
# survive the cap: they ARE the decode hot path. The engine derives the
# full-graph descriptors FROM the capture grid (each grid size is rounded up
# to a multiple of uniform_decode_query_len (4) and dropped if it passes
# max_decode_tokens = max_num_seqs * query_len = 128), so the cap has to
# leave the decode descriptor set untouched.
EXPECTED_CG_MAX = 128
EXPECTED_CG_LIST = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                    104, 112, 120, 128)
UNIFORM_DECODE_SIZES = (4, 8, 16, 32, 64, 96, 128)
# The graph counts the pinned grid implies: 19 piecewise (the pinned grid),
# 17 full (the decode coverage, unchanged by the cap), 14 dspark (the
# speculator's own graphs, untouched).
EXPECTED_CG_GRAPHS = {"piecewise": 19, "full": 17, "dspark": 14}

# ══ THE CAPTURE-POOL EXPECTATIONS (per rank) ═════════════════════════════
# The shipped profile's capture pool is under 1 GiB per rank, and the
# engine's own estimate for it stays under 3 GiB per rank. Both are
# EXPECTATIONS -- rounded bands, not somebody's measurement -- and both are
# overridable, so a different profile (or a different engine build) can be
# checked against its own numbers.
#
# WHY THE BANDS ARE PER RANK: the engine's capture-pool estimate is a
# MEASURED number, so it differs from rank to rank. Each rank is therefore
# compared with ITS OWN expectation -- one rank's figure is never used as
# another rank's threshold, which would fail a sound boot for a reason that
# has nothing to do with the boot.
#
# THE PER-RANK PROOF: each rank's capture pool ACTUAL stays under the
# engine's own ESTIMATE for that rank. The estimate is subtracted from the
# memory budget BEFORE the KV cache is sized, so the gap between the two is
# capacity the lane pays for and never uses; a rank whose actual pool has
# caught up with its estimate has no gap left to give back.
CAPTURE_POOL_ACTUAL_MAX_GIB = {
    box: _env_float("ACCEPT_CAPTURE_POOL_ACTUAL_MAX_GIB", 1.0)
    for box in REQUIRED_BOXES
}
CAPTURE_POOL_ESTIMATE_MAX_GIB = {
    box: _env_float("ACCEPT_CAPTURE_POOL_ESTIMATE_MAX_GIB", 3.0)
    for box in REQUIRED_BOXES
}

# ══ THE CAPACITY EXPECTATIONS (no capacity lost) ═════════════════════════
# The shipped profile's KV pool sits in the 2.0-2.1e5-token class and the
# engine's own concurrency factor a little above 1.5x at a 131,072-token
# request. Both are rounded EXPECTATIONS, not measurements: the criterion is
# that the pool did not fall below the floor and that the concurrency did
# not fall below the factor. Override them for a different profile.
#
# THE POOL EXPECTATION FOLLOWS THE SERVED CONTEXT: the pool must not have
# shrunk below ONE full-length request at the context this profile serves.
# A pool that had shrunk to 0.9x of the served context would hold NO
# full-length request at all, and the expectation would have called that
# "no capacity lost".
POOL_NO_LOSS_TOKENS = _env_int("ACCEPT_POOL_NO_LOSS_TOKENS", SERVED_CONTEXT)
CONCURRENCY_NO_LOSS_X = _env_float("ACCEPT_CONCURRENCY_NO_LOSS_X", 1.40)

# ══ THE TOKEN-ACCOUNTING FLOORS ═════════════════════════════════════════
TOKEN_SOURCE_REQUIRED = "usage"     # the engine's own usage block
TOKEN_CROSSCHECK_MAX_PCT = 10.0     # the server's usage vs the engine's steps x tokens-per-step
TOKENS_PER_STEP_MAX_PCT = 10.0      # the server's tokens-per-step vs the engine's
# THE DRIFT THE RULE TOLERATES between the bench's own published percentage
# and the rule's own recomputation of it. It is the criterion's OWN limit,
# named so the DISPLAY predicate and the VERDICT cannot drift apart.
CROSSCHECK_DRIFT_MAX_PCT = 1.0

# ══ THE MEMORY GUARD FLOORS ══════════════════════════════════════════════
# The guard is armed at a per-rank floor, and its running minimum has to
# stay above that floor for the whole window. The floors are NOT lowered to
# make a boot survive: the co-tenant change (heavier services off the model
# boxes, the 0.78 -> 0.80 step in docs/FIXLOG.md) gives every rank headroom
# back in multiples of the step, so each floor now sits well below that
# rank's expected drifted minimum instead of a hair above its measured one.
# Lowering the fence to make a boot survive would be the wrong repair; the
# budget was the thing that was wrong. Why each floor carries the value it
# does: docs/FIXLOG.md.
GUARD_FLOORS = {
    "spark-head.local": 4096,
    "spark-a.local": 3584,
    "spark-b.local": 2432,
}

# ══ THE PRICE ════════════════════════════════════════════════════════════
# The last rank's pinned guard floor is 2,432 MB; the profile requires at
# least 1,000 MB of margin above it, because the price of the capture-grid
# cap is paid in exactly that margin. The margin has to be there, so the
# criterion below says so out loud.
BOX3_MIN_AVAILABLE_MB = 3_432
BOX3 = "spark-b.local"

# The head box: the only rank that prints the engine's resolved config and
# the per-mode capture bars.
HEAD_BOX = "spark-head.local"

# ══ THE PINNED FACTS OF THE SHIPPED PROFILE ══════════════════════════════
# 1. THE BUDGET, asserted on EVERY rank from the engine's own resolved
#    config. A rank that booted with a different budget would have a
#    different KV pool size inside the TP ring -- a rank asymmetry that
#    looks like the change itself.
EXPECTED_GMU = "0.80"
# 2. THE MEASUREMENT PATH: the baseline path. The TTFT criterion is judged
#    on rows measured over this path, and the rule FAILS if no such row
#    exists (a missing measurement is a FAIL). AND the DRIVER'S OWN RECORD
#    of the endpoint it used must agree: the evidence manifest writes
#    `lane_url=`, and that is the machine-written fact. A row label is a
#    substring (the same label shape is produced by a box-loopback run), so
#    the label alone cannot prove the path -- the manifest can, and it must
#    be present.
# The default IS the profile's own port (PORT=8100 in config/cluster.env),
# not an invented one: a rule that defaults to a port the recipe does
# not serve judges a correct manifest as the wrong endpoint.
BASELINE_PATH = os.environ.get("ACCEPT_BASELINE_PATH", "127.0.0.1:8100")
BASELINE_URL = f"http://{BASELINE_PATH}"
# 3. THE RELAY'S MEASUREMENT PATH, on the SAME argument as the lane's. The
#    relay floor is calibrated on the relay's own endpoint reading, and the
#    rule must ASSERT it: the driver records `relay_url=` in the same
#    machine-written manifest, and a missing relay_url is a FAIL, never a
#    skip. Without this the relay criterion would be the ONE control
#    criterion whose path could not be proven -- a wrong path would have
#    produced a false green, or a false reject for transport reasons.
BASELINE_RELAY_PATH = os.environ.get("ACCEPT_BASELINE_RELAY_PATH",
                                      "127.0.0.1:8001")
BASELINE_RELAY_URL = f"http://{BASELINE_RELAY_PATH}"

# The rule's own digest, as an OPTIONAL expectation. The driver writes
# `rule_md5=` into the manifest BEFORE the rule runs and pins it against
# the digest it expects. The rule then checks ITSELF against that record,
# so a rule run DIRECTLY -- bypassing the driver -- cannot report a
# verdict over evidence collected for a DIFFERENT rule. A missing record is
# a FAIL, never a skip. The pinned expectation itself is normally supplied
# by CI (see USAGE); unset means the manifest's own record decides.
EXPECTED_RULE_MD5 = os.environ.get("ACCEPT_EXPECTED_RULE_MD5", "").strip() or None

fails: list[str] = []
no_effect: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str, verdict: str = "FAIL") -> bool:
    """A criterion. `verdict` is only used when the criterion FAILS: "FAIL"
    for a plain failure, "NO-EFFECT" for the open band (which is a REJECT)."""
    print(f"[{'PASS' if ok else verdict}] {name} -- {detail}")
    if not ok:
        if verdict == "NO-EFFECT":
            no_effect.append(name)
        else:
            fails.append(name)
    return ok


def note(text: str) -> None:
    notes.append(text)
    print(f"[NOTE] {text}")


def _read(path: str) -> str:
    """Read an evidence file. A MISSING or EMPTY file is a failure, never a
    silent pass (a vacuous pass)."""
    if not os.path.exists(path):
        return ""
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def _bench_rows(text: str) -> tuple[list[tuple[str, dict]], list[str]]:
    """Every 'label: {json}' measurement line in the bench output.

    A line that LOOKS like a measurement row but does not PARSE is returned
    as MALFORMED, not dropped. A corrupt row used to vanish silently, so a
    run could ACCEPT on the survivors while the row that would have been
    judged -- and would have failed -- was never reported at all.
    DROPPING A MEASUREMENT CAN ONLY FLATTER THE EVIDENCE, so a row the
    rule cannot read is a named failure.
    """
    rows = []
    malformed = []
    for line in text.splitlines():
        if ": {" not in line:
            # A LINE THAT CARRIES A JSON OBJECT BUT NOT THE ": {"
            # SEPARATOR IS A DAMAGED ROW, NOT A NOTE. The bench writes
            # every measurement row as "<label>: {json}" and its only
            # other lines carry no braces at all, so a line with a brace
            # and no separator is a row whose separator broke. It used
            # to be skipped in silence, and a row that vanishes is a
            # measurement the rule never saw.
            if "{" in line:
                malformed.append(line.strip()[:80])
            elif _XCHECK_LABEL_RE.search(line):
                # A RECORD LINE WHOSE BODY IS NOT A JSON OBJECT IS A
                # DAMAGED ROW TOO. The rule READS this label's body as a
                # JSON object -- the cross-check scanner does -- so a body
                # that is not one is a measurement row the rule cannot
                # read, and a row it cannot read is NAMED, not dropped.
                # (The bench's own non-row lines -- the inter-token
                # figures, the nonce, the workload -- carry no record
                # label and are left alone.)
                malformed.append(line.strip()[:80])
            continue
        label, _, js = line.partition(": {")
        try:
            row = json.loads("{" + js)
        except Exception:
            malformed.append(label.strip())
            continue
        if isinstance(row, dict):
            rows.append((label.strip(), row))
        else:
            malformed.append(label.strip())
    return rows, malformed


def _workload(row: dict) -> str:
    """The row's class, or "" when its label is not a class this rule knows.

    A MISSING LABEL IS THE PROSE CLASS (the producer's default). A LABEL
    THAT IS NOT ONE OF THE TWO CLASSES IS NOT A CLASS THIS RULE KNOWS, and
    a row the rule cannot classify cannot be counted as prose.

    THE OLD VERSION WAS `row.get("workload") or "prose"`, and it ROUTED
    EVERY LABEL THAT WAS NOT "high-draft" INTO THE PROSE LISTS -- so a row
    labelled "code" was counted as a free-prose row, and an evidence set
    with NO free-prose class at all PASSED the criterion that names one,
    while the 1-way rate floor was judged over the wrong class.
    """
    w = row.get("workload")
    if w is None:
        return "prose"
    if not isinstance(w, str):
        return ""
    w = w.strip()
    if w == "":
        return "prose"
    return w if w in KNOWN_WORKLOADS else ""


def _per_position_readable(hist) -> bool:
    """True when the histogram is PRESENT and READABLE.

    A non-empty mapping of positions to usable numbers. An all-null mapping
    is not readable, and an absent mapping is not readable.

    READABLE IS NOT "THREE POSITIONS". The rule asks that the per-position
    decay be published and legible, not that it carry a fixed number of
    positions: a publication that carries one position is a publication, and
    demanding three of them would fail a sound evidence set over the shape of
    its histogram rather than over the decay it publishes.
    """
    if not isinstance(hist, dict) or not hist:
        return False
    for v in hist.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and _is_finite(v):
            return True
    return False


def _path(label: str, row: dict) -> str:
    if row.get("path"):
        return row["path"]
    return "relay" if "relay" in label else "lane"


# ══ WHAT KIND OF ROW IS THIS? ═════════════════════════════════════════════
#
# THE DISPATCH DECIDES WHICH ROWS ARE JUDGED AT ALL, SO IT MUST NOT BE THE
# ONE PLACE WHERE A ROW CAN VANISH.
#
# IT USED TO BE. The dispatch was `str(mode).startswith("concurrent")` --
# case-SENSITIVE and ANCHORED -- while the level regex it guarded is
# case-INSENSITIVE. So a row whose mode read "CONCURRENT x4", or
# " concurrent x4", or "\tconcurrent x4" was NOT recognised as a
# concurrency row: it was DROPPED. Silently. Neither judged nor named.
# And a 4-way row carrying 1.0 tok/s against a floor of 54.5 rode into no
# criterion at all, so the rule ACCEPTED the lane.
#
# THAT IS THE HARM THIS RULE EXISTS TO PREVENT. The recipe's own claim is
# "a measurement that is missing is a failure, never a skip", and here a
# measurement was missing AND skipped. A one-character case edit in one
# JSON field turned REJECT into ACCEPT.
#
# SO A ROW IS RECOGNISED BY EVERY PLACE THAT CAN CARRY ITS KIND:
#
#   * its MODE, normalized (case and surrounding whitespace);
#   * its FIELDS -- a row that carries the concurrency fields IS a
#     concurrency row, whatever its mode says;
#   * its LABEL -- the same word the level is read from.
#
# The three sources that carry the LEVEL are the three sources that carry
# the KIND. A row that says it is an acceptance window or a prefill row
# says so with its "kind", and that statement wins: the legacy bench
# published per-position acceptance windows named "concurrent-x4-r1", and
# they are acceptance rows, not concurrency rows.
_CONC_ROW_FIELDS = ("aggregate_tok_s", "min_stream_tok_s",
                    "max_stream_tok_s", "streams")
_LABEL_CONC_RE = re.compile(r"concurrent\b", re.IGNORECASE)

# ══ THE FALL-THROUGH: WHAT A ROW THE DISPATCH CANNOT FILE IS ═════════════
#
# IT USED TO BE. THE DISPATCH HAD NO FINAL `else`, SO A ROW THAT MATCHED
# NONE OF ITS BRANCHES WAS APPENDED TO NO LIST AT ALL: it was not judged, it
# was not named, and it did not even appear in the "every measurement row is
# readable" count as anything but a readable line. A row carrying a rate of
# 1.0 tok/s, against floors of 24.0 and 54.5, therefore rode into no
# criterion at all -- and the rule ACCEPTED.
#
# A CATCH-ALL `else` IS NOT THE REPAIR, THOUGH: a FALSE FAIL IS WORSE THAN
# THE DEFECT IT CLOSES, because it stops the rule being usable on the
# evidence the lane actually produces. The recipe's own bench publishes a
# cross-check record that the dispatch dropped (the rule reads it with its own
# scanner, by label), and the lane's earlier bench published a non-streaming
# side check and a warmup record, which the dispatch dropped too. A rule that
# failed on those would fail a sound boot.
#
# SO AN UNMATCHED ROW IS CLASSIFIED BY FACTS THE RULE CAN EVALUATE FROM THE
# ROW ITSELF, and each class gets its own explicit treatment:
#
#   1. A CROSS-CHECK RECORD -- a reconciliation the bench publishes beside
#      the measurements. The rule reads it with its own scanner, not with
#      the dispatch, so it is NAMED as read-elsewhere.
#   2. A BENCH RECORD -- a row that carries NONE of the quantities the
#      rule's criteria are calibrated on (a warmup call, for instance).
#      There is nothing to judge, so it is NAMED and not judged.
#   3. A MEASUREMENT OVER A CLASS THE RULE'S FLOORS ARE NOT CALIBRATED ON
#      -- the non-streaming call. Its RATE IS READ but NOT compared with a
#      floor, because there is no floor for that path -- but it must still
#      be a READABLE measurement: a rate the rule cannot read is a MISSING
#      measurement, and a missing measurement is a FAIL.
#   4. A MEASUREMENT THE RULE CANNOT ATTRIBUTE -- anything else that
#      carries a quantity the rule judges. There is no calibrated criterion
#      for it, so it is NAMED AND FAILED.
#
# THE CRITERION IS STRUCTURAL: "every measurement row is attributable to a
# class the rule judges". It introduces NO NUMBER and NO new pre-registered
# threshold, exactly like the criterion the rule already publishes for the
# concurrency rows ("every concurrency row says which level it measured").
#
# THE QUANTITIES THE RULE'S CRITERIA ARE CALIBRATED ON. A row that carries
# one of them is making a claim this rule is supposed to judge; a row that
# carries NONE of them is a bench RECORD, not a measurement.
_JUDGED_QUANTITIES = ("tok_s", "ttft_s", "per_token_ms", "aggregate_tok_s",
                       "min_stream_tok_s", "max_stream_tok_s", "prefill_tok_s",
                       "acceptance_position0", "counters_present",
                       "tokens_source")
# THE CLASSES THE RECIPE'S BENCH MEASURES AND THIS RULE DOES NOT JUDGE, and
# the field each one publishes its rate in. The non-streaming call is a SIDE
# CHECK: the floors are calibrated on the STREAMING measurement, and a
# threshold calibrated on one path cannot be applied to another -- so its
# rate is READ but NOT judged, and it is NAMED by its label only: it is NOT
# printed, because the rule has no floor for it. A NEW class here is a
# POLICY change: it has to be written down, not inferred from an unknown
# mode.
_UNJUDGED_MEASUREMENT_CLASSES = {"non-stream": "tok_s"}
# THE RECORDS THE RULE READS WITH ITS OWN SCANNER, NOT WITH THE DISPATCH.
# The pattern is the one the cross-check scanner itself uses, so the two can
# never disagree about which record the scanner owns.
_XCHECK_LABEL_RE = re.compile(r"token accounting cross-check \(1-way lane\)")


def _is_conc_row(row: dict) -> bool:
    """A row that carries the concurrency fields is a concurrency row."""
    return any(k in row for k in _CONC_ROW_FIELDS)


def _row_is_conc(label: str, row: dict) -> bool:
    """Is this row a concurrency row? Every place that can say so, says so."""
    if row.get("kind") in ("acceptance", "prefill"):
        return False
    mode_key = str(row.get("mode") or "").strip().lower()
    if mode_key.startswith("concurrent"):
        return True
    if _is_conc_row(row):
        return True
    return bool(_LABEL_CONC_RE.search(str(label)))


# ══ TWO PUBLICATIONS OF ONE ACCEPTANCE WINDOW ═════════════════════════════
#
# LAST WRITE WINS IS THE DEFECT. `acceptance[str(window)] = row` kept the
# LAST publication of a window and discarded every earlier one, so a
# FAILING publication followed by a healthy one vanished: the rule reported
# the healthy figure and ACCEPTED.
#
# THE MERGE BELOW IS FIELD-WISE, AND THE VERDICT IT PRODUCES IS
# ORDER-INDEPENDENT, so the verdict cannot depend on which publication came
# last -- and it cannot LOSE a field the rule reads (counters_present,
# acceptance_position0, acceptance_per_pos, note): the merged row carries the
# UNION of the published rows' fields.
#
# THE PER-KIND RULE IS THE WORST CASE OF THE FIELD, because the acceptance
# criteria are MINIMUMS (ACCEPT_POS0_MIN) and one must-be-True flag: the
# lowest published figure is the one that must not be lost, and the worst
# flag is the one that is not True.
#
#   * a FLAG is True only if EVERY publication says so;
#   * a NUMBER is the minimum, and a missing or unusable figure is None --
#     an unknown figure is unknown, and it cannot be assumed to clear a
#     floor;
#   * a MAP of numbers is the per-position minimum, and a position
#     published by only some of the publications is NOT published for the
#     window;
#   * a STRING is the distinct published values, sorted and concatenated --
#     a missing string is "" and contributes nothing to a concatenation, so a
#     note published by only one of the publications is never LOST;
#   * a LIST is the union, order-preserving.
#
# A MISSING VALUE IS THE WORST VALUE OF ITS KIND -- False for a flag, None
# for a number, "" for a string, {} for a map -- so a field published by
# only one of the publications cannot look healthier than it is. TWO kinds
# are exempt, because no criterion reads them as a verdict: a STRING (no
# criterion compares a note with a threshold) and a LIST (the union keeps
# every published item, so a list is not emptied by a merge).
#
# NO NEW THRESHOLD: the merge is structural. It decides WHICH of the
# published figures the rule reads, never what the floor is.
#
# THE MERGE'S PER-KIND TABLE MUST BE REVIEWED whenever a criterion starts
# reading a NEW acceptance field: the generic rule still carries such a
# field (the merged row is a superset), but its worst case may not be the
# right one for it -- a future "at most Y" criterion would want a maximum.
_MISSING = object()


def _missing_of(sample):
    """The WORST value of a kind: what a MISSING field is worth."""
    if isinstance(sample, bool):
        return False
    if isinstance(sample, (int, float)):
        return None
    if isinstance(sample, dict):
        return {}
    if isinstance(sample, list):
        return []
    if isinstance(sample, str):
        return ""
    return None


def _merge_value(a, b):
    """The WORST case of two published values of the same field.

    A MISSING value is the worst value of its kind, so a field published by
    only one of the two publications cannot look healthier than it is.
    """
    if a is _MISSING and b is _MISSING:
        return None
    if a is _MISSING:
        a = _missing_of(b)
    if b is _MISSING:
        b = _missing_of(a)
    if isinstance(a, bool) or isinstance(b, bool):
        # A FLAG IS TRUE ONLY IF EVERY PUBLICATION SAYS SO. The rule compares
        # `counters_present` with `is not True`, NOT for truthiness: a
        # window whose counters were ABSENT once published the flag as the
        # STRING "false", and a truthiness test read that as PRESENT.
        return a is True and b is True
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        # THE MINIMUM, AND A MISSING FIGURE IS None. An unusable number
        # (NaN, an infinity, an integer too large for a float) is a MISSING
        # measurement, not a figure.
        if not (_is_finite(a) and _is_finite(b)):
            return None
        return min(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        # PER-POSITION MINIMUM. A position published by only SOME of the
        # publications is NOT published for the window: a position the
        # window cannot vouch for is not a position of the window.
        ad = a if isinstance(a, dict) else {}
        bd = b if isinstance(b, dict) else {}
        return {k: _merge_value(ad[k], bd[k]) for k in set(ad) & set(bd)}
    if isinstance(a, list) or isinstance(b, list):
        # THE UNION, ORDER-PRESERVING: no element is lost.
        al = a if isinstance(a, list) else []
        bl = b if isinstance(b, list) else []
        out: list = []
        for item in al + bl:
            if item not in out:
                out.append(item)
        return out
    if isinstance(a, str) or isinstance(b, str):
        # THE DISTINCT VALUES, CONCATENATED. A missing string is the empty
        # string, which contributes nothing -- so no published reason is
        # lost. The values are sorted so the merged string does not depend
        # on the publication order either.
        vals = sorted({v for v in (a, b) if isinstance(v, str) and v != ""})
        if not vals:
            return ""
        return "; ".join(vals)
    return a if a is not None else b


def _merge_acceptance(a: dict, b: dict) -> dict:
    """Two publications of ONE acceptance window, merged worst-case.

    LAST WRITE WINS IS THE DEFECT: a failing publication followed by a
    healthy one was discarded, and the rule then reported the healthy
    figure and ACCEPTED. The merge is FIELD-WISE and the VERDICT is
    ORDER-INDEPENDENT (min, AND and concatenation are; a list's union keeps
    order and no criterion reads a list), so the verdict cannot depend on
    publication order -- and it cannot lose a field the rule reads.
    """
    out: dict = {}
    for k in set(a) | set(b):
        out[k] = _merge_value(a.get(k, _MISSING), b.get(k, _MISSING))
    return out


def _sections(text: str) -> tuple[dict[str, str], dict[str, int]]:
    """Split '=== <host> ===' sections of an aggregated evidence file.

    THE DUPLICATE SECTIONS ARE RETURNED TOO, AND THE DICT STILL KEEPS THE
    LAST SECTION OF A NAME: `out[current] = ...` is unchanged, so the
    earlier publications are still discarded from the BODY. That is why the
    COUNT is needed. A section published TWICE -- the normal shape of an
    interrupted-then-resumed capture -- silently lost the first
    publication's body: a FAILING head capture-gate section (EXIT=1)
    published BEFORE the clean one was INVISIBLE, the rule read the clean
    one, published no FAIL line, and ACCEPTED.

    A MERGE IS NOT THE REPAIR HERE. A section body is a set of facts judged
    by DIFFERENT criteria -- a budget that must EQUAL a pinned value, a
    pool that must be AT LEAST a floor, a capture cap that must be AT MOST a
    ceiling, an exit code that must be 0 -- and no generic per-kind merge
    can be right for all of them at once. So the duplicates are COUNTED,
    and a section for a box the rule judges that is published more than
    once is a NAMED FAILURE: an ambiguous measurement is worse than a
    missing one, because the rule cannot tell which publication is this
    cycle's.

    The second return value maps a section name to the number of times that
    name was published.
    """
    out: dict[str, str] = {}
    counts: dict[str, int] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^=== (\S+) ===$", line.strip())
        if m:
            if current:
                out[current] = "\n".join(buf)
            current = m.group(1)
            counts[current] = counts.get(current, 0) + 1
            buf = []
        elif current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf)
    return out, counts


def _duplicate_box_sections(counts: dict[str, int]) -> dict[str, int]:
    """The REQUIRED BOXES whose section is published more than once.

    ONLY THE BOXES THE RULE JUDGES. The capture gate's own sub-sections
    ("=== log source: ... ===", "=== role: ... ===") are not box sections,
    and an unscoped check would fail a sound evidence set on them.
    """
    return {box: counts[box] for box in REQUIRED_BOXES
            if counts.get(box, 0) > 1}


def _duplicate_detail(what: str, counts: dict[str, int]) -> str:
    """The detail of the "no box's section more than once" criterion.

    THE DETAIL MUST NOT CLAIM MORE THAN THE CHECK DOES. The criterion was
    named "publishes each box's section exactly once" while the check said
    "at most once" -- and the detail then printed "0 of the 3 required box
    section(s) are published exactly once; 3 ARE ABSENT" UNDER A PASS, a
    sentence that contradicts the verdict on its own line. The criterion
    now says what it tests -- no box's section is published MORE THAN ONCE
    -- and the detail says the same. The PRESENCE of a box's section is
    carried by the sibling per-box criteria, which already FAIL on an
    absent box.
    """
    dupes = _duplicate_box_sections(counts)
    if not dupes:
        present = sum(1 for b in REQUIRED_BOXES if counts.get(b, 0) == 1)
        absent = sorted(b for b in REQUIRED_BOXES if not counts.get(b, 0))
        return (f"no box's section is published more than once -- {present} of "
                f"the {len(REQUIRED_BOXES)} required box section(s) are "
                "published"
                + (f"; {len(absent)} {'IS' if len(absent) == 1 else 'ARE'} ABSENT "
                   f"(e.g. {absent[0]})" if absent else ""))
    first = sorted(dupes)[0]
    return (f"{len(dupes)} box section(s) are published more than once in "
            f"{what} (e.g. {first}: {dupes[first]} publications) -- the rule "
            "cannot tell which publication is this cycle's, and an ambiguous "
            "measurement is a FAILURE, never a skip")


def _first_int(text: str, key: str):
    """The integer the FIRST publication of `key` carries, or None.

    THE WHOLE VALUE, OR NOTHING. The old pattern took a PREFIX of the
    digits -- "guard_log_breaches=0zzz" was read as 0 -- so the rule
    PUBLISHED a figure the guard did not write and PASSED a criterion on
    it. The value is now read WHOLE and handed to the pair, so a value
    that carries anything the guard did not write is UNREADABLE, and an
    unreadable fact is a DAMAGED fact that the caller NAMES.

    THE FIRST PUBLICATION IS THE ONE THAT COUNTS. A damaged publication
    is never skipped in favour of a later good one: dropping a fact can
    only flatter the evidence.
    """
    m = re.search(rf"\b{re.escape(key)}=(\S*)", text)
    if not m:
        return None
    return _safe_int(m.group(1))


def _fact_published(text: str, key: str) -> bool:
    """Is the key PUBLISHED in this text, whatever its value?

    THE COUNTS ARE READ WHOLE, so a published value the rule cannot read
    is UNREADABLE rather than invisible. A sentence that says the count
    is ABSENT -- or that the log was not read -- would then be false of a
    log that published one, so the two cases are told apart here.
    """
    return re.search(rf"\b{re.escape(key.rstrip('='))}=", text) is not None


def _is_number(value) -> bool:
    """A usable number.

    NOT a bool: in Python `True` IS an int, so without this an
    `acceptance_position0: true` would read as 1.0 and PASS a threshold it
    never reached. A bool in a measurement is a missing measurement.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    """A usable number that is also finite. NaN and the infinities are not
    measurements: NaN compares false against everything, so a NaN
    deviation would silently satisfy a `<=` check.

    AN INTEGER TOO LARGE FOR A FLOAT IS NOT A MEASUREMENT EITHER. Python's
    ints are unbounded, so `10**400` parses out of a measurement field
    perfectly -- and then `math.isfinite` RAISES OverflowError on it, and
    `float()` raises too. A rule that raises on hostile evidence cannot be
    used to reject it: a rule that dies has no verdict, and a reader cannot
    tell whether the run meant anything. So the check is a FALSE, not a
    traceback, and the measurement is reported as missing.
    """
    if not _is_number(value):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _num(value, default: float = 0.0) -> float:
    """A usable number, or the default. Used in DETAIL strings, where a
    missing value must still render as something a reader can see.

    NEVER IN A VERDICT. The default is a NUMBER, so a verdict that read
    it would compare a figure the rule never measured against a floor --
    and a CEILING is satisfied by 0.0 out of the box. A criterion is
    judged EITHER on `_safe_float`/`_safe_int`, which answer "the figure
    is missing" with None, OR on a RAW JSON value the rule has already
    admitted with `_is_finite`: the 4-way aggregate and per-stream
    minima, the per-level aggregate, the acceptance rate at position 0
    and the bench's published cross-check percentages are compared on
    the admitted values themselves, which are finite by construction.
    Either way a MISSING figure FAILS the criterion by name.
    """
    if not _is_finite(value):
        return default
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return default


def _safe_int(text):
    """An integer the rule can actually hold, or None.

    `int()` REFUSES a string with more digits than the interpreter's
    integer-conversion limit (4300 by default), and it RAISES rather than
    returning None -- so a hostile token longer than that limit took the
    rule down with a traceback instead of failing a criterion. A number
    the rule cannot hold is a number the rule does not have.
    """
    try:
        return int(text)
    except (TypeError, ValueError, OverflowError):
        return None


# A THOUSANDS-SEPARATED INTEGER IS A NUMBER ONLY IF THE GROUPING IS WELL
# FORMED. The engine prints its request length grouped ("196,608") and the
# capture gate strips the commas before publishing the fact ("196608"), so
# the rule has to read BOTH shapes. What it must NOT do is strip the commas
# ITSELF and call the result a number: "196,60,8" is not 196,608 -- it is a
# figure the evidence never carried, and a rule that publishes one is
# publishing a measurement it made up.
_GROUPED_INT_RE = re.compile(r"^[0-9]{1,3}(?:,[0-9]{3})+$")
_PLAIN_INT_RE = re.compile(r"^[0-9]+$")


def _strict_int(text):
    """An integer the evidence carries, with or without thousands separators.

    A comma grouping that is not well formed ("196,60,8", "1,9", "196,6080")
    is NOT a number: it is refused, so the reader sees what the engine
    published instead of a figure the rule made up by stripping the commas.

    THE RULE IS EXACT:
      * no comma in the token -> the WHOLE token must be digits;
      * a comma in the token -> the WHOLE token must be a well formed
        grouping ("196,608"), and the commas are then dropped;
      * anything else -- a non-digit, a malformed grouping, a trailing or
        leading character, an empty token -- is REFUSED.

    AND THE PARSE CANNOT RAISE: the conversion goes through `_safe_int`
    (which refuses a run of digits longer than the interpreter's own limit
    instead of raising), and a result the rule cannot arithmetic with is
    refused as well. A refused figure is a MISSING figure, and every call
    site treats it as one.
    """
    if text is None:
        return None
    token = str(text).strip()
    if not token:
        return None
    if "," in token:
        if not _GROUPED_INT_RE.match(token):
            return None
        token = token.replace(",", "")
    elif not _PLAIN_INT_RE.match(token):
        return None
    v = _safe_int(token)
    if v is None:
        return None
    if not _is_finite(v):
        return None
    return v


def _safe_float(text):
    """A FINITE float, or None.

    A number that arrives as TEXT is parsed here. A number that arrives
    inside a JSON object was parsed by the JSON parser, so it is admitted
    by `_is_finite` first and only then converted (see `_measured_float`).

    `float()` of a very long run of digits does not raise -- it returns
    INFINITY -- and an infinity is just as useless to the rule as an
    exception: `int(inf)` raises OverflowError, and `inf` compares false
    against every floor. So the finiteness check belongs HERE, at the
    point the number is read, rather than at each of the places that
    later use it.

    **EVERY NUMBER THE RULE READS OUT OF THE EVIDENCE IS ADMITTED BY A
    PARSER THAT CANNOT RAISE.** A number that arrives as TEXT is parsed by
    THIS PAIR of helpers. A number that arrives inside a JSON object was
    parsed by the JSON parser, so the rule admits it with `_is_finite` --
    which rejects a string, a bool, a NaN, an infinity and an integer too
    large for a float, and cannot raise -- and converts it with
    `_safe_float`/`_safe_int` wherever it needs a number to work with. A
    fix applied to one site is a fix to one site; a class of crash is
    closed only when the whole class is routed through a parser that
    cannot raise.
    """
    try:
        v = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if not math.isfinite(v):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return v


def _measured_float(value):
    """A number the rule can measure with, or None.

    THE ADMISSION AND THE PARSE ARE ONE STEP, AND BOTH OF THEM ARE SAFE.
    A number that arrives inside a JSON object has already been parsed by
    the JSON parser, so the rule ADMITS it with `_is_finite` -- which
    rejects a string, a bool, a NaN, an infinity and an integer too large
    for a float -- and then PARSES it with `_safe_float`, which cannot
    raise and cannot hand back a non-finite figure. A number the rule
    cannot read is None, and None is a MISSING measurement at every call
    site: it fails the criterion by name, and it never becomes a default
    that could let a criterion PASS.
    """
    if not _is_finite(value):
        return None
    return _safe_float(value)


def _measured_int(text):
    """An integer the rule can use as a figure, or None.

    THE PARSE AND THE ADMISSION ARE ONE STEP, AND BOTH OF THEM ARE SAFE.
    The parse is `_safe_int`, which cannot raise; the admission is
    `_is_finite`, which rejects an integer too large for a float -- an
    integer Python can hold but the rule cannot arithmetic with. A figure
    the rule cannot arithmetic is a MISSING figure, and it is reported as
    one.
    """
    v = _safe_int(text)
    if v is None:
        return None
    if not _is_finite(v):
        return None
    return v


def _shown(value) -> str:
    """A measurement rendered for a detail string.

    A number that is NOT usable is never rendered as a figure: a
    non-finite NUMBER is rendered as the words that say it is not a
    measurement, and anything else is rendered as its repr -- so a hostile
    string is quoted and a detail never prints a bare `nan%` or `inf%`
    that reads like a measurement. A detail line is the operator's only
    view of WHY a verdict came out the way it did, and a line that says
    `nan%` says nothing.
    """
    if _is_finite(value):
        return f"{value}"
    if _is_number(value):
        return "'not a finite number'"
    return repr(value)


# A FIGURE IS A PLAIN DECIMAL NUMBER, AND THE NUMBER HAS TO END WHERE THE
# NUMBER ENDS. The token is digits with at most one decimal point, and
# the character after it may not be a digit, a letter, an underscore, a
# dot or a comma -- so "1.35" is a figure and "1.2.3" is not, and "1e400"
# is not either.
_FIGURE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?![0-9A-Za-z_.,])")


def _fig(value, places: int = 2, ok=None) -> str:
    """An evidence figure, printed so the PRINTED FORM cannot disagree with
    the verdict the figure is printed under.

    A FIXED-DECIMAL rendering of a measured figure is a LOSSY rendering. A
    figure of 42.004 printed as "42.00" reads as 42.00, and a reader who
    compares it with a 42.0 ceiling concludes that the rule PASSed a figure
    that is over the ceiling -- the printed figure contradicts the verdict it
    sits under. That is the same class as publishing a figure the evidence
    does not carry: the reader is shown a number that was never measured.

    SO THE ROUNDED FORM IS USED ONLY WHILE IT AGREES WITH THE RAW FIGURE
    ABOUT THE VERDICT. `ok` is the criterion's own predicate on the raw
    figure -- `None` when the figure is published as context and judged by
    no criterion of its own -- and when the rounded form would put the
    figure on the OTHER SIDE of the criterion's limit, the shortest
    round-tripping repr is published instead.

    A FIGURE PUBLISHED UNDER MORE THAN ONE CRITERION carries more than one
    verdict, so `ok` may be a TUPLE of predicates: the rounded form is kept
    only while it agrees with the raw figure about EVERY one of them. The
    inter-token figure is published under the ceiling AND under the
    regression line, and a figure that read as clear of the ceiling while
    the ceiling criterion FAILED it would contradict one of the two -- and
    the same figure published two ways in two adjacent lines is not a
    figure the reader can check at all.

    A rounding that does not move the figure across the limit tells the
    reader nothing the verdict does not already say, so the archived
    evidence sets keep their published wording byte-for-byte.
    """
    s = f"{value:.{places}f}"
    if float(s) == value:
        return s
    if ok is None:
        return repr(value)
    preds = ok if isinstance(ok, (list, tuple)) else (ok,)
    for p in preds:
        if p(float(s)) != p(value):
            return repr(value)
    return s


def _fig_or_none(value, places: int = 2, ok=None):
    """`_fig` for a figure that may be MISSING.

    A missing measurement has no rounded form and no lossless form: it is
    published as the word the caller wrote (`None`), never as a number --
    so a figure the rule could not measure is never published as one.
    """
    if value is None:
        return None
    return _fig(value, places, ok)


def _fig_item(value, places: int = 2, ok=None) -> str:
    """The same, in the form the published LISTS use (`round(value, n)`).

    The lists are published as `[round(v, 2), ...]`, so each element is the
    plain `str()` of the rounded float. The rule is the same: the rounded
    form is used only while it agrees with the raw figure about the verdict.
    """
    s = f"{value:.{places}f}"
    if float(s) == value:
        return str(round(value, places))
    if ok is None:
        return repr(value)
    preds = ok if isinstance(ok, (list, tuple)) else (ok,)
    for p in preds:
        if p(float(s)) != p(value):
            return repr(value)
    return str(round(value, places))


def _fig_pair(a, b, places: int = 3, same=None) -> tuple:
    """Two figures that CARRY A COMPARISON between them.

    A note that says "high-draft X vs prose Y" makes a claim about the PAIR,
    and a rounding that moves one figure across the other would make the
    note contradict itself. `same(x, y)` is the claim: True when the pair
    reads as the claim says. The rounded pair is kept only while it
    satisfies the SAME claim; otherwise BOTH figures are published
    losslessly, because a pair that is half-rounded and half-lossless is
    not a pair the reader can compare.
    """
    sa = f"{a:.{places}f}"
    sb = f"{b:.{places}f}"
    if same is None:
        return sa, sb
    if same(float(sa), float(sb)) == same(a, b):
        return sa, sb
    return repr(a), repr(b)


def _figure(text) -> tuple:
    """The number a measurement line publishes, and the token it published.

    THE VALUE IS None when the text is not a BARE number; the token is
    what the bench actually wrote, so a failure can NAME it.

    WHY THE TOKEN IS READ WHOLE. A pattern that takes a PREFIX of the
    text after the label reads the LEADING DIGITS of a figure the bench
    never wrote: "1e400" read as "1" is a measurement of one millisecond
    per token that the evidence does not carry, and a rule that judges it
    and PUBLISHES it is accepting a figure it did not measure. So the
    number has to be the WHOLE number, and a number that carries a sign,
    an exponent, a digit group or a suffix is not a number this rule
    reads at all.
    """
    m = _FIGURE_RE.match(text.lstrip())
    if m:
        return _safe_float(m.group(0)), m.group(0)
    written = text.strip().split(None, 1)
    return None, (written[0] if written else "")


def _split_endpoint(url: str) -> tuple[str, str]:
    """(path, url) for a pinned endpoint. The PATH is the URL without its
    scheme -- that is the form the measurement row labels carry."""
    clean = url.rstrip("/")
    path = clean
    for scheme in ("http://", "https://"):
        if path.startswith(scheme):
            path = path[len(scheme):]
            break
    return path, clean


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """The command line. Every option only overrides a PINNED value; none
    of them can relax a criterion."""
    ap = argparse.ArgumentParser(
        prog="acceptance-check.py",
        description="The acceptance rule for this serving lane: read an "
                    "evidence directory and exit 0 only when every criterion "
                    "is met.")
    ap.add_argument("evidence", nargs="?", default=".",
                    help="the evidence directory")
    ap.add_argument("expected_md5", nargs="?", default=None,
                    help="the digest of the rule file itself")
    ap.add_argument("--evidence-dir", "--logs", dest="evidence_opt",
                    default=None,
                    help="the evidence directory (same as the positional form)")
    ap.add_argument("--lane-url", dest="lane_url", default=None,
                    help="the pinned lane endpoint")
    ap.add_argument("--relay-url", dest="relay_url", default=None,
                    help="the pinned relay endpoint")
    ap.add_argument("--head", dest="head", default=None,
                    help="the head box's label")
    ap.add_argument("--r1", dest="r1", default=None,
                    help="the rank-1 box's label")
    ap.add_argument("--r2", dest="r2", default=None,
                    help="the rank-2 box's label")
    ap.add_argument("--expect-rule-md5", dest="expect_md5", default=None,
                    help="the digest of the rule file itself")
    return ap.parse_args(argv)


def main() -> int:
    global BASELINE_PATH, BASELINE_URL, BASELINE_RELAY_PATH, BASELINE_RELAY_URL
    global REQUIRED_BOXES, GUARD_FLOORS, HEAD_BOX, BOX3
    global CAPTURE_POOL_ACTUAL_MAX_GIB, CAPTURE_POOL_ESTIMATE_MAX_GIB

    args = _parse_args(sys.argv[1:])
    ev = args.evidence_opt or args.evidence or "."
    expected_md5 = args.expect_md5 or args.expected_md5 or EXPECTED_RULE_MD5
    p = lambda *a: os.path.join(ev, *a)  # noqa: E731

    # The pinned endpoints and the pinned box labels can be pointed at
    # another site. Every override replaces a PINNED VALUE; none of them
    # can relax a criterion.
    if args.lane_url:
        BASELINE_PATH, BASELINE_URL = _split_endpoint(args.lane_url)
    if args.relay_url:
        BASELINE_RELAY_PATH, BASELINE_RELAY_URL = _split_endpoint(args.relay_url)
    if args.head or args.r1 or args.r2:
        labels = [args.head or REQUIRED_BOXES[0],
                  args.r1 or REQUIRED_BOXES[1],
                  args.r2 or REQUIRED_BOXES[2]]
        floors = [GUARD_FLOORS[b] for b in REQUIRED_BOXES]
        actuals = [CAPTURE_POOL_ACTUAL_MAX_GIB[b] for b in REQUIRED_BOXES]
        estimates = [CAPTURE_POOL_ESTIMATE_MAX_GIB[b] for b in REQUIRED_BOXES]
        REQUIRED_BOXES = tuple(labels)
        GUARD_FLOORS = dict(zip(labels, floors))
        CAPTURE_POOL_ACTUAL_MAX_GIB = dict(zip(labels, actuals))
        CAPTURE_POOL_ESTIMATE_MAX_GIB = dict(zip(labels, estimates))
        HEAD_BOX = labels[0]
        BOX3 = labels[2]

    # ==================================================================
    # 1. the bench: the speed criteria
    # ==================================================================
    bench_path = p("bench.txt")
    bench_text = _read(bench_path)
    rows: list[tuple[str, dict]] = []
    malformed_rows: list[str] = []
    if not bench_text.strip():
        check("bench evidence exists and is not empty", False,
              f"{bench_path} is missing or empty")
    else:
        check("bench evidence exists and is not empty", True,
              f"{bench_path} ({len(bench_text.splitlines())} lines)")
        rows, malformed_rows = _bench_rows(bench_text)
        # EVERY MEASUREMENT ROW MUST BE READABLE. A row the rule cannot
        # parse is a row it cannot judge, and a row it cannot judge is a
        # missing measurement -- a FAIL. Silently dropping it let a run
        # ACCEPT on the survivors alone.
        check("every measurement row in bench.txt is readable",
              not malformed_rows,
              f"{len(malformed_rows)} unreadable row(s): {malformed_rows[:3]}"
              if malformed_rows
              else f"{len(rows)} measurement row(s), all readable")

    lane_stream: list[tuple[str, dict]] = []
    relay_stream: list[tuple[str, dict]] = []
    high_draft_lane: list[float] = []
    high_draft_relay: list[float] = []
    # THE DECODE ROWS WHOSE CLASS THIS RULE DOES NOT KNOW -- the stream rows,
    # the concurrency rows and the non-stream rows alike. A row whose label
    # is not one of the two published classes is neither of them: it is not
    # counted as prose, it is not counted as high-draft, it is not counted by
    # the concurrency floors either, and it is NOT DROPPED -- it is NAMED,
    # and the criterion below fails by name.
    # (label, the row's own class label)
    unknown_class: list[tuple[str, str]] = []
    conc: dict[int, list[tuple[str, dict]]] = {}
    prefill_rows: list[dict] = []
    acceptance: dict[str, dict] = {}
    stream_rows: list[dict] = []
    # ROWS THAT CANNOT SAY WHICH CONCURRENCY LEVEL THEY MEASURED, ROWS
    # WHOSE TWO SAYINGS DISAGREE, AND ROWS AT A LEVEL THE RULE DOES NOT
    # JUDGE. All three are NAMED below; none is dropped.
    unreadable_conc: list[tuple[str, str, object]] = []
    mismatched_conc: list[tuple[str, str, object, object]] = []
    unjudged_conc: list[tuple[str, str, int]] = []
    # THE ROWS THAT ARE FILED AT A LEVEL AND STILL CARRIED A SOURCE THE
    # RULE COULD NOT READ. (label, mode, the row's own level field, the
    # sources that were unreadable.) A source that yielded NOTHING is a
    # DAMAGED source, and a damaged fact is NAMED -- not dropped, and not
    # quietly counted as "read".
    unreadable_src_conc: list[tuple[str, str, object, list[str]]] = []
    # EVERY ROW THAT MATCHES NO BRANCH OF THE DISPATCH ABOVE. None of them
    # is dropped: each is classified and named (see the `else` at the end of
    # the loop).
    cross_check_records: list[tuple[str, dict]] = []
    bench_records: list[tuple[str, dict]] = []
    unjudged_measurements: list[tuple[str, dict]] = []
    unreadable_measurements: list[tuple[str, dict]] = []
    unattributable: list[tuple[str, dict]] = []

    for label, row in rows:
        kind = row.get("kind")
        mode = row.get("mode") or ""
        # THE MODE IS NORMALIZED BEFORE IT DECIDES WHAT A ROW IS. See
        # _row_is_conc: a case-sensitive, anchored dispatch dropped rows
        # that stated their level in all three places, and a dropped
        # measurement is a skip.
        mode_key = str(mode).strip().lower()
        workload = _workload(row)
        path = _path(label, row)
        kind_key = str(kind or "").strip().lower()
        # A DECODE ROW IS A ROW THE RULE JUDGES AS A DECODE MEASUREMENT: a
        # stream row, a concurrency row, or a non-stream row.  THE PREFILL ROW
        # AND THE ACCEPTANCE WINDOWS ARE NOT DECODE ROWS: they are judged by
        # their KIND -- a measured length, a histogram -- not by a class, so a
        # label on them is not a decode class.
        #
        # THE CLASS IS CHECKED HERE, AT ONE SITE, FOR EVERY DECODE ROW.  The
        # round-15 review found the check living ONLY inside the stream
        # branch: a CONCURRENCY row whose class was not one of the two
        # published classes was still counted by the 4-way floors, and the
        # criterion that names the known-class requirement PASSED.
        #
        # `workload` IS "" EXACTLY WHEN THE ROW CARRIES A LABEL THAT IS NOT
        # ONE OF THE TWO PUBLISHED CLASSES: a missing label means the
        # producer's default (prose), and a label that is present but is not
        # one of the two is a class this rule does not know.
        #
        # A ROW THE RULE CANNOT CLASSIFY IS NAMED, AND IS COUNTED BY NOTHING:
        # not as prose, not as high-draft, and not by the concurrency floors
        # either -- a measurement the rule cannot attribute is not evidence
        # for anything.
        is_decode_row = (kind_key not in ("prefill", "acceptance")
                         and (mode_key in ("stream", "non-stream")
                              or _row_is_conc(label, row)))
        if is_decode_row and workload == "":
            raw_label = row.get("workload")
            unknown_class.append(
                (label, raw_label if isinstance(raw_label, str) else repr(raw_label)))
            continue
        if mode_key == "stream":
            stream_rows.append(row)
            # THE ROW IS ROUTED BY ITS CLASS, NEVER BY "NOT high-draft".
            # The old dispatch sent every class that was not high-draft into
            # the prose lists, so a row labelled "code" -- a class the
            # producer cannot even emit -- was counted as free prose: the
            # criterion that names a free-prose class PASSED on an evidence
            # set that published none, and the 1-way rate floor (which is
            # calibrated on the free-prose class) was judged over the wrong
            # rows. A ROW THE RULE CANNOT CLASSIFY IS NAMED, NOT COUNTED
            # AND NOT DROPPED.
            if workload == "high-draft":
                # THE HIGH-DRAFT CLASS IS A LANE CLASS, and its criterion is
                # judged over the LANE's rows -- like its free-prose sibling
                # right above it, which is judged over `lane_stream`. A
                # criterion that counted the RELAY's high-draft rows would
                # PASS on a boot where the lane itself published no
                # high-draft class at all, which is the fail-open this
                # closes.
                #
                # The high-draft rows are ALSO excluded from both rate
                # floors on purpose: the floors are calibrated on the
                # free-prose class, and the high-draft class is faster, so
                # counting it would inflate the measured rate and make the
                # floor easier to pass.
                #
                # A NaN is TRUTHY, so the old `if row.get("tok_s")` guard
                # let a NaN into this list, and a NaN then poisons every
                # comparison it is later used in.
                if _is_finite(row.get("tok_s")) and row.get("tok_s"):
                    if path == "lane":
                        high_draft_lane.append(row["tok_s"])
                    else:
                        high_draft_relay.append(row["tok_s"])
            elif workload == "prose":
                # THE FREE-PROSE CLASS. `lane_stream` and `relay_stream` hold
                # FREE-PROSE ROWS AND ONLY FREE-PROSE ROWS, so the criteria
                # that name the free-prose class are judged over the class
                # they name.
                if path == "lane":
                    lane_stream.append((label, row))
                else:
                    # The LABEL is kept, like the lane's: the relay's floor is
                    # calibrated on the relay's own baseline path, so the rows
                    # judged must be the ones measured over it.
                    relay_stream.append((label, row))
            # THERE IS NO UNKNOWN-CLASS CASE HERE, AND THERE MUST NOT BE ONE.
            # The class is checked at ONE SITE, above this dispatch, for every
            # decode row -- see the comment there.  A row whose class this
            # rule does not know was NAMED there and counted by nothing
            # before it ever reached this branch.
        elif _row_is_conc(label, row):
            # A ROW THAT CANNOT SAY WHICH LEVEL IT MEASURED IS A NAMED
            # FAILURE, NOT A DROPPED ROW.
            #
            # DROPPING A MEASUREMENT CAN ONLY FLATTER THE EVIDENCE. The
            # old code did `continue` on a row whose level it could not
            # read, and the row vanished: a 4-way row carrying a 1.0 tok/s
            # aggregate (against a floor of 54.5) was dropped and the rule
            # ACCEPTed the lane. Worse, the rule read the row's own
            # `streams` field FIRST and gave up when that field was
            # unreadable -- even though the row's MODE says which level it
            # measured. And when the sweep made the level parse safe, the
            # crash it had replaced became an UNNAMED skip, which is
            # worse than the crash: a crash is loud, and a false ACCEPT is
            # quiet.
            #
            # SO THE LEVEL IS READ FROM ALL THREE PLACES THE ROW
            # SAYS IT -- its own field, its mode and its label -- and:
            #   * a level from any of them is used;
            #   * the places DISAGREEING is a damaged row, named;
            #   * NO place yielding a readable level is a damaged row,
            #     named -- never dropped.
            field_level = None
            raw_level = row.get("streams")
            if raw_level is not None and str(raw_level).strip() != "":
                field_level = _safe_int(str(raw_level).strip())
            mode_level = None
            m = _LEVEL_TOKEN_RE.search(str(mode))
            if m:
                mode_level = _safe_int(m.group(1))
            # THE LEVEL TOKEN IS THE TOKEN, AND NOTHING ELSE.
            #
            # The bench writes the level as "x4" -- in the mode
            # ("concurrent x4") and in the label ("lane concurrent x4
            # <endpoint> [r1]"). So the level is read with a pattern that
            # REQUIRES the "x": "concurrent", whitespace, "x", the digits,
            # and a boundary that is not a word character and not a dot --
            # the token has to END there.
            #
            # WHY THE "x" IS REQUIRED, AND WHY THAT IS THE WHOLE FIX. A
            # pattern that accepts bare digits after "concurrent" reads the
            # ENDPOINT's digits instead: strip the "x4" token from the
            # label and "lane concurrent 8.8.8.8:8100 [r1]" hands the rule
            # the level 8 -- which IS a level the rule judges, so the row is
            # filed under the 8-way criterion (which has no numeric floor)
            # and escapes BOTH judgement AND the naming. An anchoring alone
            # does not help: the endpoint is ADJACENT to the word, so
            # "concurrent\s+x?(\d+)" reads it just the same. The token has
            # to BE the token.
            #
            # THE BOUNDARY MATTERS TOO: "x4.5" and "x45" are not the level
            # 4, and "x4127.0.0.1" is not the level 4127 -- so the digits
            # must not be followed by a word character or a dot. That also
            # rejects a HOST NAME that happens to start with "x" and
            # digits: "lane concurrent x16host:8100 [r1]" must not read as
            # the level 16 out of the middle of a host name.
            #
            # THE MATCH IS CASE-INSENSITIVE because the lane's own earlier
            # bench wrote "lane CONCURRENT x4 [r1]" -- the same token, in
            # caps -- and a rule that stopped reading that evidence would be
            # a rule that stopped judging the lane it was written for.
            label_level = None
            m = _LEVEL_TOKEN_RE.search(str(label))
            if m:
                label_level = _safe_int(m.group(1))
            # EVERY SOURCE IS CLASSIFIED, NOT MERELY COUNTED. A source is:
            #   * ABSENT     -- the row carries no value there. It
            #     contributes nothing, and it is NOT damage: a row that says
            #     nothing is not a row that says something the rule cannot
            #     read.
            #   * READABLE   -- a level was read from it.
            #   * UNREADABLE -- the row carries a value there and NO level
            #     could be read from it. The row said something, and the
            #     rule could not read it: that is a DAMAGED fact, and a
            #     damaged fact is NAMED, not dropped.
            unreadable_sources = []
            if (raw_level is not None and str(raw_level).strip() != ""
                    and field_level is None):
                unreadable_sources.append("its own level field")
            if str(mode).strip() != "" and mode_level is None:
                unreadable_sources.append("its mode")
            if str(label).strip() != "" and label_level is None:
                unreadable_sources.append("its label")
            said = [v for v in (field_level, mode_level, label_level)
                    if v is not None]
            if said and any(v != said[0] for v in said):
                mismatched_conc.append((label, str(mode), raw_level, mode_level))
                continue
            level = said[0] if said else None
            if level is None:
                unreadable_conc.append((label, str(mode), raw_level))
                continue
            if level not in REQUIRED_CONC_LEVELS:
                # A ROW AT A LEVEL THE RULE DOES NOT JUDGE IS A ROW THAT
                # ESCAPES JUDGEMENT. The rule's floors are calibrated on
                # the levels it declares, and the documented bench
                # invocation publishes exactly those; a row at any other
                # level would carry its aggregate into no criterion at all,
                # so a failing one would be invisible. It is NAMED.
                unjudged_conc.append((label, str(mode), level))
                continue
            # THE ROW IS FILED, AND THE SOURCE THAT WAS NOT READ IS NAMED.
            # The level IS determined -- by the sources that did read -- so
            # the criterion still passes; but the detail must not claim that
            # the damaged source was read, so it is named here.
            if unreadable_sources:
                unreadable_src_conc.append((label, str(mode), raw_level,
                                           unreadable_sources))
            conc.setdefault(level, []).append((label, row))
        elif kind == "prefill" or mode_key == "prefill":
            prefill_rows.append(row)
        elif kind == "acceptance":
            # THE WINDOW KEY MUST BE HASHABLE. A list or a dict as the
            # window name is a malformed row, and using it as a dict key
            # raises TypeError.
            window = row.get("window", "?")
            # TWO PUBLICATIONS OF ONE WINDOW ARE MERGED, NOT OVERWRITTEN.
            # Last write wins discarded a FAILING publication that a
            # healthy one followed, and the rule then reported the healthy
            # figure and ACCEPTED. The merge is field-wise, and the VERDICT
            # is order-independent (no criterion reads a list), so the
            # verdict cannot depend on which publication came last.
            key = str(window)
            prev = acceptance.get(key)
            acceptance[key] = row if prev is None else _merge_acceptance(prev, row)
        else:
            # A ROW THAT MATCHES NO BRANCH ABOVE IS A ROW THE RULE CANNOT
            # JUDGE -- and until now it was DROPPED, silently. A dropped
            # measurement can only flatter the evidence: a row whose MODE
            # names a class the rule does not know carried its rate into no
            # criterion at all, and the rule ACCEPTED.
            #
            # EVERY UNMATCHED ROW IS CLASSIFIED, AND THE CLASS IS DECIDED BY
            # FACTS THE RULE CAN EVALUATE FROM THE ROW ITSELF -- never from
            # free text, except for the one label the rule already keys on.
            # See the block above for the four classes and why a catch-all
            # `else` would have false-FAILED on the lane's own evidence.
            if _XCHECK_LABEL_RE.search(str(label)):
                cross_check_records.append((label, row))
            elif not any(k in row for k in _JUDGED_QUANTITIES):
                bench_records.append((label, row))
            else:
                # THE CLASS IS NAMED BY ITS MODE OR BY ITS KIND -- the two
                # places a row states what class of measurement it is.
                rate_field = (_UNJUDGED_MEASUREMENT_CLASSES.get(mode_key)
                              or _UNJUDGED_MEASUREMENT_CLASSES.get(
                                  str(kind or "").strip().lower()))
                if rate_field:
                    rate = row.get(rate_field)
                    # A RATE MUST BE A POSITIVE NUMBER, on the same grounds
                    # as the concurrency aggregate: the bench publishes null
                    # when it could not compute one, and that is a MISSING
                    # measurement, not a rate of zero.
                    # THE RATE IS READ THROUGH THE PAIR, so a rate the rule
                    # cannot read is a MISSING measurement -- it is never
                    # defaulted to a figure that could read as a rate.
                    rate_f = _measured_float(rate)
                    if rate_f is not None and rate_f > 0:
                        unjudged_measurements.append((label, row))
                    else:
                        unreadable_measurements.append((label, row))
                else:
                    unattributable.append((label, row))

    # --- 0z. EVERY ROW IS EITHER JUDGED OR NAMED ---------------------------
    # THE ROWS THAT MATCHED NO BRANCH OF THE DISPATCH ARE CLASSIFIED AND
    # PUBLISHED HERE, so that a row the rule cannot judge is NAMED instead
    # of vanishing. A row the rule cannot judge is a MISSING measurement,
    # and a missing measurement is a FAIL, never a skip.
    check("every measurement row in bench.txt is attributable to a class the "
          "rule judges",
          not unattributable,
          (f"{len(unattributable)} measurement row(s) carry a measurement the "
           f"rule cannot attribute to any class it judges (e.g. "
           f"{unattributable[0][0]}: mode={unattributable[0][1].get('mode')!r}, "
           f"kind={unattributable[0][1].get('kind')!r}) -- a row the rule "
           "cannot judge is a MISSING measurement, and a missing measurement "
           "is a FAIL, never a skip")
          if unattributable else
          (f"{len(rows)} row(s) in bench.txt: every one of them is either "
           "judged by this rule or named as a record it does not judge"))
    check("every measurement row published over a class this rule does not "
          "judge is readable",
          not unreadable_measurements,
          (f"{len(unreadable_measurements)} measurement row(s) over a class "
           f"this rule does not judge carry no readable rate (e.g. "
           f"{unreadable_measurements[0][0]}) -- a rate the rule cannot read "
           "is a MISSING measurement, and a missing measurement is a FAIL")
          if unreadable_measurements else
          (f"{len(unjudged_measurements)} measurement row(s) over a class this "
           "rule does not judge, all of them readable"))
    if unjudged_measurements:
        note(f"{len(unjudged_measurements)} measurement row(s) are published over "
             f"a class this rule's floors are not calibrated on; they are "
             f"published but NOT judged. Rows: "
             + "; ".join(lb for lb, _ in unjudged_measurements[:3]))
    if bench_records:
        note(f"{len(bench_records)} row(s) in bench.txt carry no measurement this "
             f"rule judges (bench records); they are named, not skipped. Rows: "
             + "; ".join(lb for lb, _ in bench_records[:3]))
    if cross_check_records:
        note(f"{len(cross_check_records)} token-accounting cross-check row(s) are "
             f"read by the rule's cross-check scanner, not by the dispatch")

    # --- 1a. THE TOKEN ACCOUNTING (the criterion that makes the rest mean
    #         what they say) -----------------------------------------------
    bad_source = [lb for lb, r in rows
                  if str(r.get("mode") or "").strip().lower() == "stream"
                  and r.get("tokens_source") != TOKEN_SOURCE_REQUIRED]
    check(f"every stream row counts TOKENS (tokens_source={TOKEN_SOURCE_REQUIRED})",
          bool(stream_rows) and not bad_source,
          f"{len(stream_rows)} stream row(s), {len(bad_source)} without the engine's "
          f"usage block" + (f" ({bad_source[:3]})" if bad_source else ""))

    # THE BENCH PUBLISHES ONE CROSS-CHECK PER ROUND, so the lines are
    # SCANNED IN ORDER and the first USABLE one is taken. Reading only the
    # first line meant that a first round whose counters were absent failed
    # the criterion even when every later round was good.
    #
    # A CORRUPT LINE STOPS THE SCAN: a damaged line is NAMED, not skipped,
    # so the scan breaks on it and the criterion fails by name (see the
    # CORRUPT case below). A line whose counters are ABSENT does NOT stop
    # the scan -- it is skipped, so a later good round can be judged. A
    # line that is not a READABLE JSON OBJECT is a THIRD shape again: it
    # is skipped by this scanner too, and it is named by the separate
    # 'every measurement row in bench.txt is readable' criterion, which
    # counts it.
    xcheck = None
    xcheck_unreadable = False
    xcheck_no_usable = False
    xcheck_corrupt = None
    XCHECK_COUNTERS = ("usage_tokens", "engine_draft_rounds",
                       "engine_tokens_per_step")
    xcheck_lines = re.findall(
        r"token accounting cross-check \(1-way lane\):\s*(\{.*\})", bench_text)
    for raw in xcheck_lines:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            xcheck_unreadable = True
            continue
        if not isinstance(parsed, dict):
            xcheck_unreadable = True
            continue
        # A LINE THAT PARSES IS NOT A LINE THAT CAN BE COMPARED. The bench
        # publishes a cross-check for EVERY round, and a round whose
        # spec-decode counters were ABSENT publishes a dict whose counters
        # are null -- that is exactly what the bench writes. So taking the
        # first PARSABLE line took the null-counter round and ignored every
        # later good round, and a boot whose counters were present in 3 of
        # 4 rounds was REJECTED. "Usable" has to mean "its counters are
        # usable NUMBERS", and the rule compares the first line that
        # qualifies. If NO line qualifies, the criterion fails by name.
        #
        # AND ABSENCE IS NOT THE SAME AS CORRUPTION. They are the same
        # shape -- a counter that is not a number -- and they must NOT get
        # the same verdict:
        #   * ABSENT (the counter is null) is the bench's own "this round
        #     recorded no counters" case. It is skipped, so a later good
        #     round can be judged.
        #   * CORRUPT (the counter is PRESENT and is not a number, or is a
        #     number too large for the arithmetic) is a DAMAGED line. A
        #     damaged line is NAMED, not skipped: silently dropping it is
        #     how a rule ends up accepting evidence it has not seen.
        #   * ZERO is a THIRD case and it belongs with ABSENT, not with
        #     CORRUPT. The bench publishes the cross-check LINE for every
        #     round, and publishes the two PERCENTAGES only when all three
        #     counters are truthy -- so a line carrying a zero counter is a
        #     line the bench itself produced, on a round it declined to
        #     compute a comparison for. The rule mirrors the bench here:
        #     such a round carries no comparable measurement and is skipped,
        #     and if NO round qualifies the criterion FAILS by name. It is
        #     not corruption, and calling it corruption would fail a boot
        #     whose later rounds are all good.
        present = [parsed.get(k) for k in XCHECK_COUNTERS]
        if any(v is not None and not _is_finite(v) for v in present):
            xcheck_corrupt = parsed
            break
        if any(v is None for v in present):
            xcheck_no_usable = True
            continue
        if all(v != 0 for v in present):
            # The bench publishes the two percentages ONLY when all three
            # counters are truthy, so a line whose counters are all present
            # and non-zero is a line the bench MUST have published them on.
            # The rule judges it, and checks that it did.
            xcheck = parsed
            break
        # A zero counter is falsy, so the bench published no percentages for
        # this round: it carries no comparable measurement. Skip it.
        xcheck_no_usable = True
        continue
    if xcheck_corrupt is not None:
        check("the token accounting cross-check is published", False,
              "a 'token accounting cross-check (1-way lane)' line in bench.txt "
              "carries a counter that is NOT a usable number "
              f"(usage_tokens={xcheck_corrupt.get('usage_tokens')!r}, "
              f"engine_draft_rounds={xcheck_corrupt.get('engine_draft_rounds')!r}, "
              f"engine_tokens_per_step="
              f"{xcheck_corrupt.get('engine_tokens_per_step')!r}) -- a damaged "
              "cross-check is named, not skipped")
    elif not xcheck:
        if xcheck_no_usable:
            check("the token accounting cross-check is published", False,
                  "the 'token accounting cross-check (1-way lane)' line(s) in "
                  "bench.txt carry no usable counters -- usage_tokens, "
                  "engine_draft_rounds and engine_tokens_per_step must all be "
                  "usable numbers for the two accounts to be comparable")
        elif xcheck_unreadable:
            check("the token accounting cross-check is published", False,
                  ("the 'token accounting cross-check (1-way lane)' line(s) in "
                   "bench.txt are not a readable JSON object"))
        else:
            # THE LINE MAY BE THERE AND STILL NOT MATCH THE SCANNER'S
            # PATTERN. The scanner reads the record with `\{.*\}`, so a
            # line whose body is truncated, or is not a JSON object at
            # all, is INVISIBLE to it -- and "no ... line in bench.txt"
            # would be FALSE of it. The LABEL is the record's own name,
            # and the label is what says the bench published the record.
            if _XCHECK_LABEL_RE.search(bench_text):
                check("the token accounting cross-check is published", False,
                      "a 'token accounting cross-check (1-way lane)' line in "
                      "bench.txt carries no JSON object the rule can read -- "
                      "the scanner reads the record's body as a JSON object, "
                      "and this one has none (the line itself is named by the "
                      "criterion 'every measurement row in bench.txt is "
                      "readable')")
            else:
                check("the token accounting cross-check is published", False,
                      "no 'token accounting cross-check (1-way lane)' line in bench.txt")
    else:
        usage_tok = xcheck.get("usage_tokens")
        chunk_tot = xcheck.get("chunk_total")
        engine_steps = xcheck.get("engine_draft_rounds")
        engine_tps = xcheck.get("engine_tokens_per_step")
        observed_tps = xcheck.get("observed_tokens_per_step")
        check("the token accounting cross-check is published", True,
              f"usage {usage_tok} tokens over the engine's {engine_steps} "
              f"decode steps = {engine_tps} tokens/step "
              f"(the client saw {_shown(chunk_tot)} SSE chunks)")
        # THE RULE DOES NOT TAKE THE BENCH'S PERCENTAGE ON TRUST. It derives
        # BOTH comparisons from the published counters ITSELF, so each
        # criterion is the rule's own arithmetic rather than a number the
        # bench chose. A cross-check whose published deviation was computed
        # against its own definition -- a quantity compared with itself,
        # which is identically 0.0 and can never fail -- does not survive
        # this: the server-side account has to be published for the
        # comparison to exist at all.
        #
        # THE DENOMINATOR IS THE ENGINE'S OWN DECODE-STEP COUNT, not the
        # client's SSE chunk count. The engine coalesces a decode step's
        # tokens into one chunk, so the two agree -- but the chunk count is
        # a step count only because of that engine behaviour, and a rule
        # whose arithmetic depends on it breaks on an engine that streams
        # one delta per token instead. The chunk count is published beside
        # it so the relationship is visible.
        cross_dev = None
        tps_dev = None
        observed_rate = None
        # THE COUNTERS HAVE TO BE NUMBERS **AND NON-ZERO WHERE THEY ARE A
        # DENOMINATOR** before any arithmetic happens. A counter that is a
        # string, or a step count of zero, is a MISSING MEASUREMENT and is
        # reported as a FAIL -- it is not an arithmetic opportunity, and
        # dividing by it must not take the rule down.
        #
        # THE COUNTERS ARE PARSED BY THE PAIR, NOT BY A RAW float(). A
        # counter the rule cannot parse is a MISSING measurement: the
        # criterion below fails BY NAME, and the rule never defaults a
        # counter to a figure that could let the comparison PASS. The
        # admission and the parse are one step, so the whole class of "a
        # number the rule reads out of the evidence" goes through a parser
        # that cannot raise.
        steps_f = _measured_float(engine_steps)
        tps_f = _measured_float(engine_tps)
        usage_f = _measured_float(usage_tok)
        counters_are_numbers = (steps_f is not None and tps_f is not None
                                and usage_f is not None)
        # EVERY DENOMINATOR IS CHECKED FOR ZERO, INCLUDING engine_tps: it is
        # the divisor of the tokens-per-step deviation, and a zero there is
        # a ZeroDivisionError, not a measurement.
        if (counters_are_numbers and engine_steps and usage_tok
                and engine_tps):
            # THE ARITHMETIC IS DONE IN FLOAT, NOT IN INT. `_is_finite`
            # REJECTS an integer too large for a float (it catches the
            # OverflowError `math.isfinite` raises on one) -- but it
            # ADMITS any integer that IS float-representable, and the
            # product of two such integers is a perfectly good Python int
            # that is NOT representable as a float. So `engine_steps *
            # engine_tps` is `int * int`, and dividing THAT by `usage_tok`
            # raises `OverflowError: integer division result too large for
            # a float`. A rule that raises has no verdict, so the whole
            # arithmetic is done in the parsed floats, where an overflow is
            # an INFINITY rather than an exception -- and a non-finite
            # result is a MISSING MEASUREMENT, reported as one.
            predicted = steps_f * tps_f
            if (math.isfinite(predicted) and math.isfinite(usage_f)
                    and usage_f != 0.0):
                observed_rate = usage_f / steps_f if steps_f else None
                cross_dev = abs(predicted - usage_f) / usage_f * 100.0
                tps_dev = (abs(observed_rate - tps_f) / tps_f * 100.0
                           if observed_rate is not None and tps_f else None)
                if not (math.isfinite(cross_dev)
                        and (tps_dev is None or math.isfinite(tps_dev))):
                    # AN INFINITY IS NOT A DEVIATION. It is the arithmetic
                    # saying "this figure is too large to be a percentage",
                    # which is a missing measurement, not a passing one.
                    cross_dev = None
                    tps_dev = None
            else:
                check("the cross-check's counters are usable numbers", False,
                      "the cross-check's counters overflow the arithmetic: "
                      f"usage_tokens={usage_tok!r}, "
                      f"engine_draft_rounds={engine_steps!r}, "
                      f"engine_tokens_per_step={engine_tps!r} -- the predicted "
                      "token count is not a usable number, so the two accounts "
                      "cannot be compared")
        elif usage_tok or engine_steps or engine_tps:
            # THE COUNTERS ARE NOT USABLE. That is a missing measurement,
            # not an arithmetic opportunity: a non-numeric counter is
            # reported as a FAIL rather than crashing the rule. The
            # criterion has its OWN NAME, because a name that is printed as
            # both PASS and FAIL in one operator log tells the reader
            # nothing about which one happened.
            check("the cross-check's counters are usable numbers", False,
                  "the cross-check's counters are not usable numbers "
                  f"(usage_tokens={usage_tok!r}, engine_draft_rounds={engine_steps!r}, "
                  f"engine_tokens_per_step={engine_tps!r})")
        # A PUBLISHED FIGURE MUST NOT DISAGREE WITH THE VERDICT IT SITS
        # UNDER. Every figure below is printed under a criterion that
        # compares it with a LIMIT, so the rounded form is used only while
        # it agrees with the raw figure about that limit; otherwise the
        # lossless form is published, because "10.0%" printed directly
        # above a "10.0%" ceiling reads as a PASS the raw figure does not
        # earn (10.004% is over a 10.0% ceiling).
        check(f"the token accounting cross-check agrees within {TOKEN_CROSSCHECK_MAX_PCT}%",
              cross_dev is not None and abs(cross_dev) <= TOKEN_CROSSCHECK_MAX_PCT,
              f"{_fig_or_none(cross_dev, 2, lambda v: abs(v) <= TOKEN_CROSSCHECK_MAX_PCT)}% "
              f"(the server's usage vs the engine's steps x tokens-per-step)")
        # THE SERVER'S COUNTED TOKENS PER STEP is the OTHER HALF of the
        # comparison the reader does: the deviation is what the reader
        # computes from this figure and the engine's own. So this figure is
        # published under the SAME predicate, applied to the deviation a
        # reader would recompute from it.
        _observed_rate_ok = None
        if tps_f and observed_rate is not None:
            _observed_rate_ok = (
                lambda v: abs(abs(v - tps_f) / tps_f * 100.0) <= TOKENS_PER_STEP_MAX_PCT)
        check(f"the tokens-per-step accounts agree within {TOKENS_PER_STEP_MAX_PCT}%",
              tps_dev is not None and _is_finite(observed_tps)
              and abs(tps_dev) <= TOKENS_PER_STEP_MAX_PCT,
              f"{_fig_or_none(tps_dev, 2, lambda v: abs(v) <= TOKENS_PER_STEP_MAX_PCT)}% "
              f"(the server's counted tokens per step "
              f"{_fig_or_none(observed_rate, 4, _observed_rate_ok)} "
              f"vs the engine's draft-accounted tokens per step {engine_tps})"
              + ("" if _is_finite(observed_tps) else
                 " -- the evidence publishes no server-side tokens-per-step "
                 "account, so the two accounts cannot be compared"))
        # AND THE BENCH'S OWN PUBLISHED PERCENTAGES MUST MATCH THE RULE'S.
        # Otherwise the bench measured something else from the same
        # counters, and the evidence and the verdict have drifted apart.
        #
        # THE PUBLISHED PERCENTAGES ARE GUARDED TOO: they arrive from the
        # same file as everything else, and a string or a NaN there used to
        # raise -- and a NaN silently SATISFIED the check, because
        # `max(0.0, nan)` is 0.0 in Python.
        #
        # AND A PUBLISHED PERCENTAGE THAT IS NOT A USABLE NUMBER IS A FAIL,
        # NOT A SKIP. Skipping it is the defect F-10 named: the corrupted
        # value was DROPPED from the comparison, so the criterion still
        # PASSed on the other percentage and the whole rule ACCEPTed a
        # cross-check whose published deviation was a NaN. The rule's own
        # doctrine is "a missing measurement is a FAIL, never a skip", and
        # that applies to the published number exactly as it applies to the
        # recomputed one. The rule CAN recompute both, so the bench MUST
        # have published both, and each must be a usable number.
        pub_cross = xcheck.get("deviation_pct")
        pub_tps = xcheck.get("tokens_per_step_deviation_pct")
        drift = None
        drift_problems = []
        if cross_dev is not None and tps_dev is not None:
            if not _is_finite(pub_cross):
                drift_problems.append(
                    f"the bench's published deviation_pct is not a usable "
                    f"number ({_shown(pub_cross)})")
            if not _is_finite(pub_tps):
                drift_problems.append(
                    f"the bench's published tokens_per_step_deviation_pct is "
                    f"not a usable number ({_shown(pub_tps)})")
            if not drift_problems:
                drift = max(abs(pub_cross - cross_dev),
                            abs(pub_tps - tps_dev))
        # THE RECOMPUTED FIGURES ARE PUBLISHED UNDER THE CRITERION'S OWN
        # PREDICATE. The criterion is about the DRIFT between the published
        # figure and the recomputed one, and the reader computes that drift
        # from the two printed percentages -- so a rounded recomputed
        # figure is used only while it still reads as the same drift the
        # verdict is about. When the published figure is not a usable
        # number there is no drift to read, and the recomputed figures are
        # published losslessly.
        _drift_ok_cross = None
        _drift_ok_tps = None
        if _is_finite(pub_cross) and _is_finite(pub_tps):
            _drift_ok_cross = (
                lambda v: abs(pub_cross - v) <= CROSSCHECK_DRIFT_MAX_PCT)
            _drift_ok_tps = (
                lambda v: abs(pub_tps - v) <= CROSSCHECK_DRIFT_MAX_PCT)
        if drift_problems:
            check("the published cross-check matches the rule's own recomputation",
                  False,
                  "; ".join(drift_problems)
                  + " -- the published percentages cannot be compared with the "
                  "rule's own recomputation ("
                  f"{_fig_or_none(cross_dev, 2, _drift_ok_cross)}% / "
                  f"{_fig_or_none(tps_dev, 2, _drift_ok_tps)}%)")
        else:
            check("the published cross-check matches the rule's own recomputation",
                  cross_dev is not None and tps_dev is not None
                  and drift is not None and drift <= CROSSCHECK_DRIFT_MAX_PCT,
                  f"the bench published {_shown(pub_cross)}% / "
                  f"{_shown(pub_tps)}%; the rule "
                  f"recomputed {_fig_or_none(cross_dev, 2, _drift_ok_cross)}% / "
                  f"{_fig_or_none(tps_dev, 2, _drift_ok_tps)}%")

    conc_rows = [r for lb, r in rows if _row_is_conc(lb, r)]
    bad_conc = [r.get("mode") for r in conc_rows if r.get("tokens_source") != TOKEN_SOURCE_REQUIRED]
    check(f"every concurrent level counts TOKENS (tokens_source={TOKEN_SOURCE_REQUIRED})",
          bool(conc_rows) and not bad_conc,
          f"{len(conc_rows)} concurrent row(s), {len(bad_conc)} not counted in tokens"
          + (f" ({bad_conc})" if bad_conc else ""))

    # --- 1b. THE 1-WAY LANE STREAM: ACCEPT / NO-EFFECT / REJECT -----------
    # --- 1b-0. THE ENDPOINT THE DRIVER ACTUALLY USED ---------------------
    # The row LABELS carry the URL, but a label is a substring: the paired
    # diagnostic's box-loopback runs produce the SAME label shape. The
    # machine-written fact is the driver's own evidence manifest, which
    # records the endpoint it was pointed at. If it is missing, the path
    # cannot be proven and the criterion below cannot be judged.
    driver_manifest = _read(p("manifest.txt"))

    # --- the rule's OWN identity, against the driver's own record -----------
    # The driver writes `rule_md5=` into the manifest BEFORE the rule runs
    # and pins it against the digest it expects. The rule now checks ITSELF
    # against that record, so a rule run DIRECTLY -- bypassing the driver
    # -- cannot report a verdict over evidence collected for a DIFFERENT
    # rule. A missing record is a FAIL, never a skip.
    try:
        own_md5 = hashlib.md5(Path(__file__).resolve().read_bytes()).hexdigest()
    except OSError as exc:
        own_md5 = f"unreadable ({exc})"
    m_rule = re.search(r"^rule_md5=(\S+)\s*$", driver_manifest, re.M)
    manifest_rule_md5 = m_rule.group(1).strip() if m_rule else None
    manifest_matches = manifest_rule_md5 is not None and manifest_rule_md5 == own_md5
    pinned_matches = expected_md5 is None or expected_md5 == own_md5
    check("the rule that ran is the one the evidence manifest records",
          manifest_matches and pinned_matches,
          f"the rule's own md5 is {own_md5}; the manifest records "
          f"rule_md5={manifest_rule_md5}"
          + ("" if manifest_matches
             else " -- a stale or foreign rule cannot report this verdict")
          + ("" if pinned_matches
             else f" -- the pinned digest is {expected_md5}, which is not this rule")
          + ("" if expected_md5 is not None
             else " (no pinned digest was supplied; CI pins one)")
          if manifest_rule_md5 is not None else
          "the evidence manifest does not record rule_md5 -- which rule produced "
          "this verdict cannot be proven (a missing measurement is a FAIL)")
    lane_url = None
    m_url = re.search(r"^lane_url=(\S+)\s*$", driver_manifest, re.M)
    if m_url:
        lane_url = m_url.group(1).strip()
    check(f"the driver measured the baseline endpoint ({BASELINE_URL})",
          lane_url == BASELINE_URL,
          f"the evidence manifest records lane_url={lane_url}"
          + ("" if lane_url == BASELINE_URL
             else f" -- the threshold was calibrated on {BASELINE_URL}")
          if lane_url is not None else
          "the evidence manifest does not record lane_url -- the measurement "
          "endpoint cannot be proven (a missing measurement is a FAIL)")

    # --- 1b-0-bis. THE SECOND ENDPOINT, ON THE SAME ARGUMENT --------------
    # A second endpoint in front of the lane (a proxy) has its own floor,
    # calibrated on its own path -- so when a deployment HAS one, it must
    # be measured over that same path.
    #
    # A deployment that serves ONE endpoint has no such floor, and the
    # criteria below would be criteria about something that does not exist.
    # So the question is settled by the EVIDENCE, not by a default: if the
    # driver's manifest records a second endpoint, every criterion here
    # applies and fails closed like the rest; if the manifest records none,
    # the deployment has one endpoint and these criteria are REPORTED as
    # not applicable -- explicitly, never as a silent skip.
    relay_url = None
    m_rurl = re.search(r"^relay_url=(\S+)\s*$", driver_manifest, re.M)
    if m_rurl:
        relay_url = m_rurl.group(1).strip()
    second_endpoint_expected = relay_url is not None
    if second_endpoint_expected:
        check(f"the driver measured the second endpoint over the baseline "
              f"second endpoint ({BASELINE_RELAY_URL})",
              relay_url == BASELINE_RELAY_URL,
              f"the evidence manifest records relay_url={relay_url}"
              + ("" if relay_url == BASELINE_RELAY_URL
                 else f" -- the second endpoint's threshold was calibrated on "
                      f"{BASELINE_RELAY_URL}")
              if relay_url is not None else
              "the evidence manifest does not record relay_url -- the second "
              "endpoint's measurement path cannot be proven (a missing "
              "measurement is a FAIL)")
    else:
        note("this deployment serves ONE endpoint: the evidence manifest "
             "records no second endpoint, so the second-endpoint criteria do "
             "not apply. Point the driver at a proxy as well (and record it) "
             "to have them judged.")

    # The rows the criteria are judged on: the BASELINE PATH.
    # BOTH the 1-way lane rate and the TTFT are thresholds calibrated on
    # the baseline path, so both are judged over it -- the relay already is.
    # Judging one over the baseline path and the other over EVERY lane row
    # would be an asymmetry: a row measured somewhere else could have
    # decided the lane criterion. It could only produce a false REJECT,
    # never a false green -- and in a single-endpoint run it is inert --
    # but a criterion is judged on the path it was calibrated on or not at
    # all.
    #
    # AND ON THE FREE-PROSE CLASS: `lane_stream` holds the FREE-PROSE ROWS
    # AND ONLY THE FREE-PROSE ROWS (the dispatch routes by class), so the
    # 1-way rate floor -- which the rule's own comment above the dispatch
    # calls "calibrated on the free-prose class" -- is judged over the
    # free-prose class. A row of a class the rule cannot classify is named
    # by its own criterion and never enters this list.
    base_rows = [(lb, r) for lb, r in lane_stream if BASELINE_PATH in lb]
    other_rows = [(lb, r) for lb, r in lane_stream if BASELINE_PATH not in lb]
    if other_rows:
        note(f"{len(other_rows)} lane stream row(s) were measured over a path "
             f"other than the baseline path ({BASELINE_PATH}); they "
             "are published but NOT judged, because the thresholds were "
             "calibrated on the baseline path. Rows: "
             + ", ".join(f"{lb} -> {_shown(r.get('tok_s'))} tok/s"
                         for lb, r in other_rows))

    if not base_rows:
        check("1-way lane stream measured over the baseline path", False,
              f"{len(lane_stream)} lane row(s), none over the baseline path "
              f"({BASELINE_PATH}) -- the thresholds were calibrated "
              "on it, so they cannot be judged (a missing measurement is a FAIL)")
    else:
        # ONLY NUMERIC RATES COUNT, AND THE DETAIL IS BUILT ONLY WHEN THERE
        # IS A WORST. The shipped bench writes "tok_s": null for an empty
        # completion -- so the rows this rule is fed CAN carry a null, and
        # formatting a None as a float before the None check raised a
        # TypeError and took the whole rule down, instead of reporting the
        # missing measurement as the FAIL it is. A non-numeric rate is the
        # same kind of missing measurement.
        raw_rates = [r.get("tok_s") for _, r in base_rows]
        # THE LANE PATH FILTERS EXACTLY AS THE RELAY PATH DOES. `isinstance(v,
        # (int, float))` admitted a bool (True IS an int) and a NaN, and it
        # admitted an integer too large for a float -- which then raised
        # OverflowError in the `:.2f` formatting below. `_is_finite` admits
        # none of them, so a hostile rate is a MISSING rate, named.
        #
        # THE RATES ARE PARSED BY THE PAIR, NEVER BY `_num`: this criterion
        # is judged on the worst rate, and `_num`'s default is a NUMBER --
        # a number the rule never measured, standing in for a missing one.
        # A rate the rule cannot read is a MISSING measurement, and the
        # criterion below fails BY NAME.
        rates = [x for x in (_measured_float(v) for v in raw_rates)
                 if x is not None]
        worst = min(rates) if rates else None
        if worst is None:
            check("1-way lane stream measured", False,
                  "lane stream rows carry no numeric tok/s"
                  + (f" ({len(raw_rates)} row(s) on the baseline path, "
                     f"{len(raw_rates) - len(rates)} without a numeric rate)"
                     if raw_rates else ""))
        else:
            # THE PRINTED FIGURE MUST NOT CONTRADICT THE VERDICT IT SITS
            # UNDER. The criterion is a FLOOR, so the floor is the predicate:
            # a rounded form that would read as clearing the floor when the
            # raw figure does not -- or the other way round -- is replaced by
            # the lossless form of the figure the rule actually judged.
            _lane_ok = (lambda v: v >= LANE_STREAM_KEEP)

            def _lane_detail(ok):
                return (f"rounds [{', '.join(_fig_item(r, 2, ok) for r in rates)}]"
                        f" (worst {_fig(worst, 2, ok)})")

            detail = _lane_detail(_lane_ok)
            missing = len(raw_rates) - len(rates)
            if missing:
                # EVERY BASELINE-PATH ROUND MUST CARRY A MEASUREMENT. The
                # floor is a WORST-CASE floor, judged over the rounds the
                # criterion is judged on, and a round with no usable rate is
                # a round whose rate is UNKNOWN. An unknown round cannot be
                # assumed to be above the floor, and a round that is
                # silently DROPPED can only flatter the evidence: a hostile
                # integer in ONE round used to make the rule judge the
                # others and ACCEPT. A missing measurement is a FAIL, never
                # a skip.
                check(f"1-way lane stream >= {LANE_STREAM_KEEP} tok/s", False,
                      detail + f" -- {missing} of {len(raw_rates)} "
                      "baseline-path round(s) carry no usable rate, so the "
                      "worst case is unknown")
            elif worst >= LANE_STREAM_KEEP:
                check(f"1-way lane stream >= {LANE_STREAM_KEEP} tok/s", True, detail)
            elif worst > LANE_STREAM_REJECT:
                # THE NO-EFFECT VERDICT ASSERTS A BAND, NOT A FLOOR, so the
                # band is the predicate here: a printed figure that reads
                # OUTSIDE the open band would contradict the verdict the
                # note states, and a figure that reads INSIDE it when the
                # raw figure is outside it would contradict it the other
                # way. The FLOOR stays in the predicate too, because the
                # criterion this line sits under is the floor. The detail
                # and the note carry the SAME figure, so they are rendered
                # with the SAME predicates.
                _band_ok = (lambda v: LANE_STREAM_REJECT < v < LANE_STREAM_KEEP)
                check(f"1-way lane stream >= {LANE_STREAM_KEEP} tok/s", False,
                      _lane_detail((_lane_ok, _band_ok)), "NO-EFFECT")
                note(f"NO-EFFECT: the 1-way lane stream landed at "
                     f"{_fig(worst, 2, (_lane_ok, _band_ok))} tok/s, inside "
                     f"the open band ({LANE_STREAM_REJECT}, {LANE_STREAM_KEEP}). The profile did "
                     "not beat the baseline by enough to pay for its capacity and memory "
                     "cost: REJECTED, and NOT retried without a new variable.")
            else:
                check(f"1-way lane stream >= {LANE_STREAM_KEEP} tok/s", False, detail)
        # (the ROUND COUNT is asserted below, over the BASELINE-PATH rows --
        # counting every lane row would let rounds measured over another path
        # satisfy the rule)

        # --- the TTFT criterion, judged on the BASELINE PATH --------------
        # The baseline was measured over the pinned baseline path. A
        # threshold calibrated on one path cannot be applied to another; the
        # rows are asserted to come from it. (The "no baseline row at all"
        # case already FAILED above, on the 1-way criterion.)
        #
        # THIS IS A PUBLISHED FACT, NOT A CRITERION. It was a `check` with a
        # literal `True`, which made it a PASS that no input could ever
        # turn into a FAIL -- a decorative gate. A gate that cannot fail
        # proves nothing, so it is published as a note instead, and the
        # criterion that CAN fail (the TTFT ceiling itself, below) carries
        # the weight.
        note(f"the TTFT criterion is measured over the baseline path "
             f"({BASELINE_PATH}): {len(base_rows)} lane row(s) over it")
        # The ROUND COUNT is a count of the rows the criterion is judged
        # on -- not of every lane row. Two rounds measured over the LAN
        # are not two rounds over the baseline path.
        check("1-way lane stream measured on >= 2 rounds over the baseline path",
              len(base_rows) >= MIN_ROUNDS,
              f"{len(base_rows)} baseline-path round(s)"
              f" ({len(lane_stream)} lane round(s) in total)")
        # THE TTFT VALUES ARE FILTERED THE SAME WAY THE RATES ARE. A bool is
        # an int, a NaN compares false against everything, and an integer too
        # large for a float used to raise OverflowError in the `:.3f`
        # formatting below. None of them is a measurement.
        #
        # AND THEY ARE PARSED BY THE PAIR, NEVER BY `_num`. The TTFT is
        # judged against a CEILING, and `_num`'s default (0.0) satisfies
        # that ceiling out of the box -- a figure the rule never measured
        # standing in for a missing one. A TTFT the rule cannot read is a
        # MISSING measurement, and the criterion below fails BY NAME.
        ttfts = [x for x in (_measured_float(r.get("ttft_s"))
                             for _, r in base_rows)
                 if x is not None]
        ttft_missing = len(base_rows) - len(ttfts)
        # THE CEILING IS THE PREDICATE. A rounded TTFT that reads as over
        # the ceiling when the raw figure is under it would contradict the
        # verdict it is printed under.
        _ttft_ok = (lambda v: v <= TTFT_MAX_S)
        if not ttfts:
            check("lane TTFT measured", False, "no numeric lane TTFT on the baseline path")
        elif ttft_missing:
            # THE SAME COMPLETENESS RULE AS THE RATE. The TTFT ceiling is a
            # WORST-CASE ceiling over the baseline-path rounds, and a round
            # with no usable TTFT is a round whose TTFT is UNKNOWN.
            check(f"lane TTFT <= {TTFT_MAX_S} s", False,
                  f"worst {_fig(max(ttfts), 3, _ttft_ok)} s over {BASELINE_PATH} "
                  f"(rounds [{', '.join(_fig_item(t, 3, _ttft_ok) for t in ttfts)}]) -- {ttft_missing} "
                  f"of {len(base_rows)} baseline-path round(s) carry no usable "
                  "TTFT, so the worst case is unknown")
        else:
            check(f"lane TTFT <= {TTFT_MAX_S} s", max(ttfts) <= TTFT_MAX_S,
                  f"worst {_fig(max(ttfts), 3, _ttft_ok)} s over {BASELINE_PATH} "
                  f"(rounds [{', '.join(_fig_item(t, 3, _ttft_ok) for t in ttfts)}])")

    # --- 1c. the relay ------------------------------------------------------
    # THE RELAY PATH FILTERS NUMERIC RATES, EXACTLY AS THE LANE PATH DOES.
    # The lane path was fixed for this; leaving the relay path filtering
    # only for None was the asymmetry that let a non-numeric rate reach a
    # float comparison on one path and not the other.
    #
    # AND THE COMPLETENESS COUNT IS TAKEN OVER THE **UNFILTERED** BASELINE
    # ROWS, exactly as the lane path takes it over `raw_rates` rather than
    # over the filtered list. Filtering FIRST and counting afterwards made
    # the difference identically zero, so the completeness rule was DEAD
    # CODE on this path: a relay round with a hostile rate was silently
    # dropped, the surviving round was judged, and the rule ACCEPTed.
    relay_all = [(lb, r) for lb, r in relay_stream if BASELINE_RELAY_PATH in lb]
    relay_rows = [(lb, r) for lb, r in relay_all if _is_finite(r.get("tok_s"))]
    base_relay = relay_rows
    other_relay = [(lb, r) for lb, r in relay_stream
                   if BASELINE_RELAY_PATH not in lb]
    # WAS THE RELAY CRITERION JUDGED AT ALL? A one-endpoint deployment has
    # no second endpoint, so the relay floor does not apply -- and the
    # ACCEPT banner must not then claim the relay floor was met. Crediting
    # a tool with a proof it did not perform is exactly the claim this
    # recipe refuses to make elsewhere.
    relay_judged = False
    if other_relay:
        note(f"{len(other_relay)} relay stream row(s) were measured over a path "
             f"other than the baseline relay path ({BASELINE_RELAY_PATH}); "
             "they are published but NOT judged, because the relay "
             "floor was calibrated on the baseline path. Rows: "
             + ", ".join(f"{lb} -> {_shown(r.get('tok_s'))} tok/s"
                         for lb, r in other_relay))
    if not relay_rows:
        if second_endpoint_expected:
            # THE SENTENCE HAS TO BE TRUE OF THE EVIDENCE IN EVERY ONE OF
            # THE CASES. `relay_rows` is FILTERED on a usable rate, so an
            # empty list means one of THREE things, and they are not the
            # same thing: the second-endpoint stream row is ABSENT, the
            # rows are on ANOTHER PATH, or the rows carry NO USABLE RATE.
            # The bench writes "tok_s": null for an empty completion, so
            # the third case is a row the SHIPPED bench itself produced --
            # and "no second-endpoint stream row in bench.txt" would be
            # FALSE of it. The criterion NAME and its VERDICT do not move:
            # the second-endpoint stream was NOT measured, either way.
            if not relay_stream:
                check("second-endpoint stream measured", False,
                      "no second-endpoint stream row in bench.txt")
            elif not relay_all:
                check("second-endpoint stream measured", False,
                      f"{len(relay_stream)} second-endpoint stream row(s) in "
                      f"bench.txt, none of them over the baseline relay path "
                      f"({BASELINE_RELAY_PATH}) -- the floor was calibrated on "
                      "that path, so no rate can be judged")
            else:
                check("second-endpoint stream measured", False,
                      f"{len(relay_all)} second-endpoint stream row(s) over "
                      f"the baseline relay path ({BASELINE_RELAY_PATH}) in "
                      "bench.txt carry NO USABLE RATE -- a rate the rule "
                      "cannot read is a MISSING measurement, not a rate of "
                      "zero -- so the second-endpoint rate was not measured")
        else:
            note("no second-endpoint stream row: this deployment has one "
                 "endpoint, so the second-endpoint rate is not judged")
    elif not base_relay:
        check(f"the relay stream is measured over the baseline path "
              f"({BASELINE_RELAY_PATH})", False,
              f"{len(relay_rows)} relay row(s), none over the baseline path -- "
              "the floor was calibrated on it, so the criterion "
              "cannot be judged (a missing measurement is a FAIL)")
    else:
        # THE RELAY RATES ARE PARSED BY THE PAIR, NEVER BY `_num`: this
        # criterion is judged on the worst rate, and `_num`'s default is a
        # NUMBER -- a number the rule never measured, standing in for a
        # missing one. A rate the rule cannot read is a MISSING
        # measurement, and the criterion below fails BY NAME.
        relay_rates = [x for x in (_measured_float(r.get("tok_s"))
                                   for _, r in base_relay)
                       if x is not None]
        relay_judged = True
        relay_missing = len(relay_all) - len(relay_rates)
        # THE FLOOR IS THE PREDICATE, exactly as it is for the lane path.
        _relay_ok = (lambda v: v >= RELAY_STREAM_MIN)
        if relay_missing:
            # THE SAME COMPLETENESS RULE AS THE LANE: the relay floor is a
            # worst-case floor over the baseline relay path, and a round
            # with no usable rate is a round whose rate is UNKNOWN.
            check(f"relay stream >= {RELAY_STREAM_MIN} tok/s", False,
                  f"rounds [{', '.join(_fig_item(r, 2, _relay_ok) for r in relay_rates)}] "
                  f"(worst {_fig(min(relay_rates), 2, _relay_ok)}) -- {relay_missing} of "
                  f"{len(relay_all)} baseline relay round(s) carry no usable "
                  "rate, so the worst case is unknown")
        else:
            check(f"relay stream >= {RELAY_STREAM_MIN} tok/s",
                  min(relay_rates) >= RELAY_STREAM_MIN,
                  f"rounds [{', '.join(_fig_item(r, 2, _relay_ok) for r in relay_rates)}] "
                  f"(worst {_fig(min(relay_rates), 2, _relay_ok)})")

    # --- 1d. the engine's inter-token, PER TOKEN ----------------------------
    m = re.search(r"engine inter-token \(1-WAY window, per TOKEN\):(?P<rest>[^\n]*)",
                  bench_text)
    if not m:
        check("engine inter-token measured per TOKEN", False,
              "no 'engine inter-token (1-WAY window, per TOKEN)' line in bench.txt")
    else:
        # THE INTER-TOKEN FIGURE IS A NUMBER OR IT IS NOT A FIGURE, AND THE
        # TOKEN IS THE TOKEN. The figure is read as a WHOLE bare number:
        # "1.35" is a figure and is read as one, and "." and "1.2.3" are
        # not figures at all -- the token has to END where a number ends,
        # so the pair is never asked about a shape it cannot read. A run
        # of digits too long to be a finite number IS a bare number and IS
        # matched, and is then refused by the pair, which answers None,
        # because float() would have read it as an INFINITY. A missing
        # measurement is a FAIL by NAME, not a crash, and the rule never
        # publishes a figure the evidence does not carry.
        inter_ms, written = _figure(m.group("rest"))
        if inter_ms is None:
            check("engine inter-token measured per TOKEN", False,
                  f"the published inter-token figure ({written!r}) is not a "
                  "number the rule can read")
            check(f"engine inter-token <= {INTER_TOKEN_MAX_MS} ms/token", False,
                  f"the published inter-token figure ({written!r}) is not a number")
            # THE NAME STATES THE PREDICATE. A regression is ">=
            # REJECT_MS"; "not a regression" is "< REJECT_MS". A name that
            # reads the other way round is a trap for the reader of the
            # report, who cannot tell which side of the line it is on.
            check(f"engine inter-token not a regression (< {INTER_TOKEN_REJECT_MS} ms)",
                  False, "no usable inter-token figure")
        else:
            # THE CEILING IS THE PREDICATE, AND SO IS THE REGRESSION LINE.
            # A figure of 42.004 ms printed as "42.00" reads as 42.00 -- and
            # a reader who compares it with the 42.0 ceiling concludes that
            # the rule PASSed a figure that is over the ceiling. The printed
            # figure must not disagree with the verdict it sits under, so
            # the lossless form is published whenever the rounded one would
            # move the figure across the limit.
            #
            # AND THE FIGURE IS PUBLISHED UNDER BOTH CRITERIA, so the
            # predicate is BOTH of them: a figure that reads as clear of
            # the ceiling while the ceiling criterion FAILED it would
            # contradict that criterion, and the same figure published two
            # ways in two adjacent lines is not a figure the reader can
            # check at all.
            _inter_ok = (lambda v: v <= INTER_TOKEN_MAX_MS,
                         lambda v: v < INTER_TOKEN_REJECT_MS)
            check(f"engine inter-token <= {INTER_TOKEN_MAX_MS} ms/token",
                  inter_ms <= INTER_TOKEN_MAX_MS,
                  f"{_fig(inter_ms, 2, _inter_ok)} ms/token (the bench's 1-WAY window delta, per token)")
            check(f"engine inter-token not a regression (< {INTER_TOKEN_REJECT_MS} ms)",
                  inter_ms < INTER_TOKEN_REJECT_MS, f"{_fig(inter_ms, 2, _inter_ok)} ms/token")
    m_step = re.search(r"engine inter-token \(1-WAY window, Prometheus sum/count delta\):(?P<rest>[^\n]*)",
                       bench_text)
    if m_step:
        # THE PER-STEP FIGURE IS READ THROUGH THE SAME TOKEN, for the same
        # reason: a figure the rule cannot read is not published as one,
        # and a figure it can only read by taking its leading digits is
        # not published at all.
        step_ms, step_written = _figure(m_step.group("rest"))
        if step_ms is not None:
            note(f"the per-STEP inter-token was {_fig(step_ms, 2)} ms/step; the "
                 "per-token figure above divides it by the measured tokens-per-step ratio")
        else:
            note(f"the published per-STEP inter-token figure "
                 f"({step_written!r}) is not a usable number, so the "
                 "per-step figure is not published")

    # --- 1e. the 4-way CLIFF (ASSERTED, not "a 4-way row exists") ---------
    four_lane = [(lb, r) for lb, r in conc.get(4, []) if _path(lb, r) == "lane"]
    if not four_lane:
        # THE ROW MAY BE THERE AND STILL NOT FILED. A 4-way lane row whose
        # level the rule cannot read -- or whose own places disagree about
        # the level -- is NOT filed under the 4-way level, and "no 4-way
        # concurrent row for the lane in bench.txt" would be FALSE of it.
        # Such a row is NAMED by the criterion that judges the levels, so
        # this criterion says the measurement is missing and says where
        # the row is named.
        check("the 4-way cliff measurement is present", False,
              "no 4-way concurrent row for the lane in bench.txt THAT THE "
              "RULE COULD FILE under the 4-way level -- a row the rule "
              "cannot file is named by the criterion 'every concurrency row "
              "says which level it measured'")
    else:
        # ONLY NUMERIC AGGREGATES AND MINIMA ARE COMPARED. A string, a list
        # or a dict in either field used to raise TypeError on the
        # comparison instead of being reported as a missing measurement.
        aggs = [r.get("aggregate_tok_s") for _, r in four_lane
                if _is_finite(r.get("aggregate_tok_s"))]
        mins = [r.get("min_stream_tok_s") for _, r in four_lane
                if _is_finite(r.get("min_stream_tok_s"))]
        # THE SAME COMPLETENESS RULE. Both are worst-case floors over the
        # 4-way lane rows, so a row that carries no usable figure is a row
        # whose figure is UNKNOWN -- and an unknown row cannot be assumed
        # to be above the floor.
        agg_missing = len(four_lane) - len(aggs)
        min_missing = len(four_lane) - len(mins)
        # THE FLOORS ARE THE PREDICATES. A rounded aggregate that reads as
        # under the floor when the raw figure is over it would contradict
        # the verdict it is printed under.
        _agg_ok = (lambda v: v >= CONC4_AGG_MIN)
        _min_ok = (lambda v: v >= CONC4_PER_STREAM_MIN)
        check(f"4-way lane aggregate >= {CONC4_AGG_MIN} tok/s",
              bool(aggs) and not agg_missing and min(aggs) >= CONC4_AGG_MIN,
              (f"aggregates [{', '.join(_fig_item(a, 2, _agg_ok) for a in aggs)}] "
               f"(worst {_fig(min(aggs), 2, _agg_ok)})"
               + (f" -- {agg_missing} of {len(four_lane)} 4-way lane row(s) "
                  "carry no usable aggregate, so the worst case is unknown"
                  if agg_missing else "")) if aggs
              else "no numeric aggregate published")
        check(f"4-way lane per-stream >= {CONC4_PER_STREAM_MIN} tok/s",
              bool(mins) and not min_missing and min(mins) >= CONC4_PER_STREAM_MIN,
              (f"per-stream minima [{', '.join(_fig_item(m, 2, _min_ok) for m in mins)}] "
               f"(worst {_fig(min(mins), 2, _min_ok)})"
               + (f" -- {min_missing} of {len(four_lane)} 4-way lane row(s) "
                  "carry no usable per-stream minimum, so the worst case is "
                  "unknown" if min_missing else "")) if mins
              else "no numeric per-stream minimum published")
        # THE DEAD STREAMS ARE NAMED. The bench publishes `dead_streams` --
        # the count of the streams that produced NO usable rate -- and a row
        # that carries it is a row that says part of its measurement did not
        # happen. "per-stream minima [0.0]" alone leaves the reader to infer
        # the cause, and the cause matters: a stream that produced nothing is
        # not a slow stream, and the two call for different fixes.
        #
        # THE FIELD IS OPTIONAL. The archived evidence sets predate it, so a
        # row that does NOT carry it is judged exactly as before, on its
        # minimum -- making the field mandatory would reject evidence that
        # is otherwise perfectly readable.
        dead_rows = [(lb, r) for lb, r in four_lane if "dead_streams" in r]
        dead_named = []
        dead_unreadable = []
        for lb, r in dead_rows:
            v = r.get("dead_streams")
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                # A COUNT THE RULE CANNOT READ IS NOT A COUNT OF ZERO.
                dead_unreadable.append((lb, v))
            elif v > 0:
                dead_named.append((lb, v))
        # THE CRITERION IS PRINTED ONLY WHEN THE EVIDENCE CARRIES THE FIELD.
        # The archived evidence sets predate it, so for them the rule says
        # exactly what it said before -- a criterion that appears only for
        # the evidence that can answer it, and never for the evidence that
        # cannot.
        if dead_rows:
            if dead_named:
                dead_ok = False
                dead_detail = (
                    f"{len(dead_named)} 4-way lane row(s) carry dead streams "
                    f"(e.g. {dead_named[0][0]}: "
                    f"dead_streams={dead_named[0][1]}) -- a stream that "
                    "produced no tokens is not a slow stream")
            elif dead_unreadable:
                dead_ok = False
                dead_detail = (
                    f"{len(dead_unreadable)} 4-way lane row(s) carry a "
                    f"dead_streams count the rule cannot read (e.g. "
                    f"{dead_unreadable[0][0]}: "
                    f"dead_streams={dead_unreadable[0][1]!r}) -- a damaged "
                    "fact is NAMED, not dropped, so the dead-stream state of "
                    "the 4-way lane is UNKNOWN")
            else:
                dead_ok = True
                dead_detail = ("every 4-way lane row's streams produced a "
                               f"rate ({len(four_lane)} row(s) judged)")
            check("every 4-way lane stream produced a rate", dead_ok,
                  dead_detail)

    # --- 1f. the WORKLOAD SPLIT must be PUBLISHED -------------------------
    #
    # THE DETAIL IS BUILT BRANCH BY BRANCH, because the sentence the rule
    # PRINTS about the evidence has to be TRUE of it. The first three
    # branches are the ones that were already here, word for word. The
    # fourth is the new one: a row that IS filed at a level -- so its level
    # WAS determined -- but that carried a source the rule could not read.
    # The old sentence claimed that every row's level field, mode AND label
    # were read; when one of them yielded nothing, that claim was false, so
    # the new sentence says what was read and NAMES the source that was not.
    if unreadable_conc:
        conc_level_detail = (
            f"{len(unreadable_conc)} concurrency row(s) name no readable level "
            f"(e.g. {unreadable_conc[0][0]}: mode={unreadable_conc[0][1]!r}, "
            f"streams={unreadable_conc[0][2]!r}) -- a row the rule cannot "
            "attribute to a level is a DAMAGED row, and it is named rather "
            "than dropped")
    elif mismatched_conc:
        conc_level_detail = (
            f"{len(mismatched_conc)} concurrency row(s) disagree about their "
            f"own level (e.g. {mismatched_conc[0][0]}: mode="
            f"{mismatched_conc[0][1]!r}, streams={mismatched_conc[0][2]!r})")
    elif unjudged_conc:
        conc_level_detail = (
            f"{len(unjudged_conc)} concurrency row(s) sit at a level the rule "
            f"does not judge (e.g. {unjudged_conc[0][0]}: level "
            f"{unjudged_conc[0][2]}, which is outside "
            f"{', '.join(str(x) for x in REQUIRED_CONC_LEVELS)}) -- a row at "
            "an unjudged level would carry its aggregate into NO criterion, "
            "so a failing one would be invisible")
    elif unreadable_src_conc:
        conc_level_detail = (
            f"{len(conc)} concurrency level(s) read from the sources each row "
            "carries, and every row's readable sources agreed -- but "
            f"{len(unreadable_src_conc)} row(s) carried a source the rule "
            f"could NOT read (e.g. {unreadable_src_conc[0][0]}: "
            f"{', '.join(unreadable_src_conc[0][3])}) -- a source the rule "
            "cannot read is a DAMAGED source, and it is named rather than "
            "dropped")
    else:
        conc_level_detail = (
            f"{len(conc)} concurrency level(s) read from each row's own level "
            "field, its mode and its label, and every row agreed")
    check("every concurrency row says which level it measured",
          not unreadable_conc and not mismatched_conc and not unjudged_conc,
          conc_level_detail)
    # THE FREE-PROSE CLASS, AND ONLY THE FREE-PROSE CLASS. `lane_stream`
    # now holds the rows whose class IS "prose" -- the dispatch routes by
    # class, not by "not high-draft" -- so the criterion's name and its
    # check say the same thing: the class it names is the class it counts.
    check("the workload split: a free-prose class is published", bool(lane_stream),
          f"{len(lane_stream)} prose stream row(s)")
    # EVERY DECODE ROW NAMES ITS CLASS, AND THE CLASS IS ONE THIS RULE
    # KNOWS. A row labelled with a class the producer cannot emit ("code") is
    # not prose and not high-draft: it is counted as neither, it is counted by
    # the concurrency floors neither, and it is NAMED here rather than quietly
    # folded into the prose class.  THE PREFILL ROW AND THE ACCEPTANCE
    # WINDOWS ARE NOT DECODE ROWS -- they are judged by their kind, not by a
    # class -- so a label on one of them is not a decode class and is not
    # named here.
    if unknown_class:
        unknown_detail = (
            f"{len(unknown_class)} row(s) carry a class this rule does not "
            f"know (e.g. {unknown_class[0][1]!r} on row {unknown_class[0][0]!r}) "
            f"-- the known classes are {', '.join(KNOWN_WORKLOADS)}, and a row "
            "the rule cannot classify is counted by NOTHING: not as prose, not "
            "as high-draft, and not by the concurrency floors either")
    else:
        unknown_detail = (f"every decode row carries one of the known "
                          f"classes ({', '.join(KNOWN_WORKLOADS)})")
    check("the workload split: every decode row carries a known class",
          not unknown_class,
          unknown_detail)
    check("the workload split: a high-draftability class is published",
          bool(high_draft_lane),
          f"{len(high_draft_lane)} high-draft LANE stream row(s)"
          + (f" ({len(high_draft_relay)} on the second endpoint, which is "
             "published but not judged)" if high_draft_relay else ""))
    for level in REQUIRED_CONC_LEVELS:
        rows_l = [(lb, r) for lb, r in conc.get(level, []) if _path(lb, r) == "lane"]
        # AN AGGREGATE MUST BE A POSITIVE NUMBER. The bench publishes null
        # when it could not compute one -- that is a MISSING measurement,
        # not a rate of zero, and the detail says "n/a" rather than 0.00.
        ok = bool(rows_l) and all(
            _is_finite(r.get("aggregate_tok_s")) and r.get("aggregate_tok_s") > 0
            for _, r in rows_l)
        if rows_l:
            # THE PUBLICATION CRITERION IS THE PREDICATE HERE: it requires
            # an aggregate that is a number AND POSITIVE, so a rounded form
            # that reads as zero would publish a zero aggregate under a
            # criterion that says the aggregate was published.
            _published_ok = (lambda v: v > 0)
            aggs = ", ".join(_fig(r.get("aggregate_tok_s"), 2, _published_ok)
                             if _is_finite(r.get("aggregate_tok_s")) else "n/a"
                             for _, r in rows_l)
            # THE PER-STREAM RANGE IS PRINTED ONLY WHEN IT IS A MEASUREMENT.
            # The criterion above judges the AGGREGATE; the per-stream
            # figures are context beside it. A null per-stream figure
            # becomes 0.00 in a formatter, and "per-stream 0.00" is a number
            # the rule did NOT measure -- so it is not printed.
            mins = [_num(r.get("min_stream_tok_s")) for _, r in rows_l]
            maxs = [_num(r.get("max_stream_tok_s")) for _, r in rows_l]
            if rows_l and all(v > 0 for v in mins + maxs):
                per_stream = (f"per-stream {_fig(min(mins), 2, _published_ok)}-"
                              f"{_fig(max(maxs), 2, _published_ok)}")
            else:
                per_stream = "per-stream not published"
            detail = f"aggregates [{aggs}], {per_stream}"
        else:
            detail = f"no usable {level}-way lane row"
        check(f"the workload split: the {level}-way aggregate is published", ok, detail)
    # A PREFILL IS A MEASUREMENT, AND A MEASUREMENT THE RULE CANNOT READ IS
    # A MISSING MEASUREMENT -- NOT A PASS.
    #
    # The condition used to be a PUBLICATION check (`bool(prefill_rows)`): a
    # row whose rate was the string "1200.0zz" PASSED, and the detail
    # published that unusable token in a measurement slot. A criterion whose
    # NAME claims that a prefill was MEASURED has to read the measurement.
    #
    # AND THE PROMPT LENGTH IS PART OF THE MEASUREMENT. The criterion says a
    # LONG-CONTEXT prefill, and a prefill whose prompt length the rule
    # cannot read is not a long-context prefill -- it is a prefill of
    # unknown length.
    #
    # THE PROMPT LENGTH IS `prompt_tokens`, OR IT IS MISSING. There is NO
    # SECOND PLACE TO READ IT FROM. The shipped bench reads the length out
    # of the server's own usage block and writes it as `prompt_tokens`
    # (bench-decode.py:443-448); it never writes a prefill row with a bare
    # `tokens` and no `prompt_tokens`. So the row's generic `tokens` field
    # is NOT the prompt length, and reading it as one would let a row
    # CHOOSE ITS OWN "PROMPT LENGTH" -- a hostile row that omits
    # `prompt_tokens` and carries `tokens: 1` would satisfy a criterion
    # about a LONG-CONTEXT prefill with a one-token prompt. A row that
    # carries no readable `prompt_tokens` has NO readable prompt length.
    #
    # A `prompt_tokens` THE ROW CARRIES BUT THE RULE CANNOT READ IS DAMAGE,
    # and it is never rescued from the other field: the row said something
    # about its prompt length, and the rule could not read it.
    prefill_len = None
    prefill_problem = None
    if prefill_rows:
        prefill_rate = _measured_float(prefill_rows[0].get("prefill_tok_s"))
        prefill_len = _measured_float(prefill_rows[0].get("prompt_tokens"))
        if prefill_len is None:
            if "prompt_tokens" in prefill_rows[0]:
                prefill_problem = (
                    "the published prefill prompt length is not a usable "
                    f"number ({_shown(prefill_rows[0].get('prompt_tokens'))}) "
                    "-- a prefill with no readable prompt length is not a "
                    "long-context prefill, so the measurement is MISSING")
            else:
                prefill_problem = (
                    "the prefill row publishes no usable prompt length -- a "
                    "prefill with no readable prompt length is not a "
                    "long-context prefill, so the measurement is MISSING")
        # THE DECLARED BUDGET IS PART OF THE MEASUREMENT, AND THE MEASURED
        # LENGTH HAS TO REACH IT.
        #
        # THE BENCH DECLARES THE BUDGET IT MEASURED AGAINST and publishes
        # it as `target_tokens`; the engine's own `usage.prompt_tokens` is
        # the MEASURED length. A prefill that published a length but no
        # budget proves nothing about how long the prompt was, and a
        # prefill whose measured length is BELOW the budget it declares did
        # not measure the budget it claims to have measured. So the rule
        # checks the two published numbers AGAINST EACH OTHER -- a
        # consistency check between two numbers the evidence carries, with
        # no new constant of its own -- and checks the declared budget
        # against the recipe's own budget (PREFILL_BUDGET_TOKENS).
        #
        # A READABLE LENGTH IS NOT ENOUGH. A row carrying `prompt_tokens:
        # 1` is readable, and it used to PASS a criterion whose NAME says
        # "a long-context prefill is measured". Now it FAILS BY NAME.
        #
        # THE ORDER OF THE NAMES. The rate outranks the length (the rate is
        # the measurement, the length is its attribution), so the rate's
        # problem is assigned LAST below and wins. The length outranks the
        # budget (the budget is a DECLARATION; the length is what was
        # measured), so the budget's problems are only reached when the
        # length is readable.
        prefill_budget = _measured_float(prefill_rows[0].get("target_tokens"))
        if prefill_len is not None:
            if prefill_budget is None:
                if "target_tokens" in prefill_rows[0]:
                    prefill_problem = (
                        "the declared prefill budget is not a usable number "
                        f"({_shown(prefill_rows[0].get('target_tokens'))}) -- a "
                        "prefill with no readable declared budget does not "
                        "prove a long-context prefill, so the measurement is "
                        "MISSING")
                else:
                    prefill_problem = (
                        "the prefill row publishes no declared budget -- the "
                        "length it publishes is measured against nothing, so "
                        "the measurement is MISSING")
            elif prefill_budget < PREFILL_BUDGET_TOKENS:
                prefill_problem = (
                    "the declared prefill budget "
                    f"({_shown(prefill_rows[0].get('target_tokens'))}) is below "
                    f"the recipe's own long-context budget "
                    f"({PREFILL_BUDGET_TOKENS} tokens) -- a prefill of a "
                    "shorter budget is not the long-context prefill this "
                    "criterion names")
            elif prefill_len < prefill_budget:
                prefill_problem = (
                    "the published prompt length "
                    f"({_shown(prefill_rows[0].get('prompt_tokens'))}) is below "
                    "the declared budget "
                    f"({_shown(prefill_rows[0].get('target_tokens'))}) -- the "
                    "measurement does not reach the budget it declares")
        # THE RATE IS THE MEASUREMENT; THE LENGTH IS ITS ATTRIBUTION. So
        # when BOTH are unusable, the detail names the RATE: that is the
        # figure the criterion is about, and the length only says what the
        # measurement was of.
        if prefill_rate is None:
            prefill_problem = (
                "the published prefill rate is not a usable number "
                f"({_shown(prefill_rows[0].get('prefill_tok_s'))}) -- a rate "
                "the rule cannot read is a MISSING measurement, not a rate of "
                "zero")
    if not prefill_rows:
        check(PREFILL_CRIT, False,
              "no prefill row in bench.txt")
    elif prefill_problem:
        check(PREFILL_CRIT, False,
              prefill_problem)
    else:
        check(PREFILL_CRIT, True,
              (f"prompt_tokens {_shown(prefill_rows[0].get('prompt_tokens'))}, "
               f"{_shown(prefill_rows[0].get('prefill_tok_s'))} tok/s"))
    # THE HISTOGRAM, NOT THE ROW. The old check asked only "is there at
    # least one acceptance ROW?", so deleting the per-position histogram
    # from every row left the criterion PASSing while the very next lines
    # printed "(per-position: None)". THE CRITERION NAMES THE HISTOGRAM, SO
    # THE CHECK READS THE HISTOGRAM.
    #
    # EVERY CORE WINDOW MUST PUBLISH A READABLE HISTOGRAM, AND THE CORE
    # CHECK IS INDEPENDENT OF THE DECLARATION LIST: adding a core window
    # to NO_PER_POSITION_WINDOWS cannot excuse a window that lost its
    # histogram. The declaration list is a SHORT PUBLISHED LIST of the
    # windows that publish no per-position numbers by nature (the prefill
    # window measures a prefill, not a draft acceptance), and it is
    # published in the detail so a reader can see what it excuses.
    per_pos_readable: list[str] = []
    per_pos_declared: list[str] = []
    per_pos_unreadable: list[str] = []
    for window, arow in acceptance.items():
        wname = str(window)
        hist = arow.get("acceptance_per_pos") if isinstance(arow, dict) else None
        if _per_position_readable(hist):
            per_pos_readable.append(wname)
        elif wname in CORE_PER_POSITION_WINDOWS:
            # A CORE WINDOW IS NOT EXCLUDABLE. It is named as unreadable
            # whatever the declaration list says.
            per_pos_unreadable.append(wname)
        elif wname in NO_PER_POSITION_WINDOWS:
            per_pos_declared.append(wname)
        else:
            per_pos_unreadable.append(wname)
    declared_list = sorted(per_pos_declared)
    per_pos_detail = (
        f"{len(per_pos_readable)} acceptance window(s) publish a readable "
        f"per-position histogram; {len(per_pos_declared)} window(s) publish "
        f"none BY DECLARATION: {declared_list}")
    if not acceptance:
        check("the workload split: the per-position acceptance histogram is published",
              False,
              "no acceptance histogram row in bench.txt -- there is no "
              "per-position decay to read")
    elif per_pos_unreadable:
        first = sorted(per_pos_unreadable)[0]
        check("the workload split: the per-position acceptance histogram is published",
              False,
              per_pos_detail + f"; {len(per_pos_unreadable)} acceptance "
              f"window(s) publish NO readable per-position histogram "
              f"(e.g. {first}) -- the per-position decay is neither present "
              "nor readable")
    elif not per_pos_readable:
        check("the workload split: the per-position acceptance histogram is published",
              False,
              per_pos_detail + "; no acceptance window publishes a readable "
              "per-position histogram, so the per-position decay is neither "
              "present nor readable")
    else:
        check("the workload split: the per-position acceptance histogram is published",
              True,
              per_pos_detail)

    # --- 1g. the acceptance histogram --------------------------------------
    hist = acceptance.get("overall")
    if not hist:
        check("acceptance at position 0 >= 0.5", False,
              "no overall acceptance histogram in bench.txt")
    elif hist.get("counters_present") is not True:
        # NOT NAMED `note`: `note` is the FUNCTION this module uses to print
        # an explanatory line, and assigning to that name anywhere in main()
        # makes Python treat it as a local for the WHOLE function -- which
        # crashes the rule at the first `note(...)` call, long before this
        # branch is reached. A prose fix must not take the rule down.
        #
        # THE FLAG IS COMPARED WITH `is not True`, NOT FOR TRUTHINESS. A
        # window whose counters were ABSENT published the flag as the
        # STRING "false" once, and a truthiness test read that as PRESENT --
        # so an absent-counter window was judged as a healthy one. Only the
        # boolean True counts as "the counters were there".
        hist_note = hist.get("note", "")
        check("acceptance at position 0 >= 0.5", False,
              hist_note if hist_note else
              "the spec-decode counters were ABSENT from the exposition")
    else:
        pos0 = hist.get("acceptance_position0")
        if pos0 is None:
            hist_note = hist.get("note", "")
            check("acceptance at position 0 >= 0.5", False,
                  "the histogram carries no acceptance rate (no drafts were "
                  "recorded)" + (f" ({hist_note})" if hist_note else ""))
        elif not _is_finite(pos0):
            # A BOOL IS NOT A RATE: in Python `True` is an int, so
            # `acceptance_position0: true` would otherwise read as 1.0 and
            # PASS a threshold of 0.5 it never reached.
            check(f"acceptance at position 0 >= {ACCEPT_POS0_MIN}", False,
                  f"the histogram's acceptance rate is not a number ({_shown(pos0)})")
        else:
            # THE FLOOR IS THE PREDICATE. A rounded acceptance rate that
            # reads as under the floor when the raw figure is over it would
            # contradict the verdict it is printed under.
            _accept_ok = (lambda v: v >= ACCEPT_POS0_MIN)
            check(f"acceptance at position 0 >= {ACCEPT_POS0_MIN}",
                  pos0 >= ACCEPT_POS0_MIN,
                  f"position 0 = {_fig(pos0, 3, _accept_ok)} (per-position: {hist.get('acceptance_per_pos')})")
            # THE NOTE IS PUBLISHED ON THE PASS BRANCH TOO. A window whose
            # counters were ABSENT IN SOME ROUNDS merges over the rounds
            # that had them, and the note says so -- a reader must be able
            # to see that the green verdict covers a subset.
            hist_note = hist.get("note", "")
            if hist_note:
                note(f"the acceptance histogram carries a note: {hist_note}")
        hd = acceptance.get("high-draft-1way")
        prose = acceptance.get("prose-1way")
        if hd and prose:
            hd0 = hd.get("acceptance_position0")
            pr0 = prose.get("acceptance_position0")
            if (_is_finite(hd0) and _is_finite(pr0)
                    and abs(hd0 - pr0) > 0.05):
                # THE DIRECTION MATTERS, AND "EXPECTED" IS A CLAIM. The
                # profile's thesis is that the draft serves the
                # high-draftability class WELL and decays on free prose. So
                # a high-draft figure BELOW the prose figure is the
                # OPPOSITE of what the profile predicts, and calling that
                # "as expected" tells a reader of the public output that a
                # collapsed class is normal.
                #
                # THE NOTE DESCRIBES THE RULE'S REAL SCOPE, WHICH IS NOT
                # MERGED-ONLY: the floor is applied to the MERGED histogram
                # AND to each class's own histogram, so a collapsed class
                # DOES fail the rule. Saying otherwise here would put a
                # false statement about the rule's own scope into its
                # published output -- and the two class criteria printed
                # immediately below this note would contradict it.
                if hd0 >= pr0:
                    # THE NOTE ASSERTS A COMPARISON, so the comparison is the
                    # predicate: a rounded figure that would put the class
                    # on the OTHER side of the other class would contradict
                    # the note that says which side it is on. The pair is
                    # rendered together, so the two figures a reader
                    # compares are always the SAME two figures.
                    _hd_s, _pr_s = _fig_pair(hd0, pr0, 3,
                                             lambda x, y: x >= y)
                    note(f"the classes differ as expected: high-draft position-0 "
                         f"acceptance {_hd_s} vs prose {_pr_s}. The criterion "
                         f"is applied to the OVERALL histogram AND to each "
                         f"class's own histogram.")
                else:
                    _hd_s, _pr_s = _fig_pair(hd0, pr0, 3,
                                             lambda x, y: x < y)
                    note(f"NOTE -- the classes differ in the OPPOSITE direction to "
                         f"the profile's prediction: high-draft position-0 "
                         f"acceptance {_hd_s} is BELOW prose {_pr_s}. The "
                         f"merged criterion is applied to the OVERALL histogram, "
                         f"and each class is judged on its OWN histogram "
                         f"separately -- so a class that collapses FAILS the "
                         f"rule on its own figure, not only through the merge. "
                         f"The profile predicts the high-draft class to be the "
                         f"STRONGER one.")

    # THE SPLIT IS JUDGED, CLASS BY CLASS. The recipe's contract is that
    # the bench measures BOTH workload classes in ONE run and that the rule
    # judges the SPLIT -- "a free-prose class and a high-draftability
    # class, each with its own acceptance histogram" (README, VALIDATION
    # and the bench's own --help). The code judged the MERGED histogram and
    # only PUBLISHED the per-class figures, so dropping a class histogram
    # removed the only evidence that class was measured for acceptance --
    # and nothing named it. "The per-position acceptance histogram is
    # published" was satisfied by ANY window, so a bench that measured one
    # class per run produced an evidence set the rule ACCEPTED.
    #
    # EACH CLASS IS NOW JUDGED ON ITS OWN: the window must be there, its
    # counters must have been present, its rate must be a number, and the
    # SAME position-0 floor applies to it. The floor is not a new policy:
    # the floors table lists ONE acceptance floor (position 0 >= 0.5) and
    # the lane's own measured class figures clear it with room -- the
    # free-prose class sits near 0.6 and the high-draftability class near
    # 0.99 -- so this adds no new way for a healthy lane to fail.
    for cname in ("prose-1way", "high-draft-1way"):
        crow = acceptance.get(cname)
        criterion = (f"the {cname} class's acceptance at position 0 "
                     f">= {ACCEPT_POS0_MIN}")
        if not crow:
            check(criterion, False,
                  f"no '{cname}' acceptance row in bench.txt -- the bench "
                  "measures BOTH workload classes in one run and the rule "
                  "judges the SPLIT, so a class that is not published is a "
                  "class that was not measured")
        elif crow.get("counters_present") is not True:
            check(criterion, False,
                  "the spec-decode counters were ABSENT from this class's "
                  "exposition" + (f" ({crow.get('note', '')})"
                                  if crow.get("note") else ""))
        else:
            cpos0 = crow.get("acceptance_position0")
            if cpos0 is None or not _is_finite(cpos0):
                check(criterion, False,
                      f"the class's histogram carries no acceptance rate "
                      f"({_shown(cpos0)})")
            else:
                _accept_ok = (lambda v: v >= ACCEPT_POS0_MIN)
                check(criterion, cpos0 >= ACCEPT_POS0_MIN,
                      f"position 0 = {_fig(cpos0, 3, _accept_ok)} (per-position: "
                      f"{crow.get('acceptance_per_pos')})")

    # ==================================================================
    # 2. the capture/config gate: the capacity floor and the proof
    # ==================================================================
    cap_path = p("capture-gate.txt")
    cap_text = _read(cap_path)
    if not cap_text.strip():
        check("capture gate evidence exists and is not empty", False,
              f"{cap_path} is missing or empty")
        cap_sections: dict[str, str] = {}
        cap_counts: dict[str, int] = {}
    else:
        check("capture gate evidence exists and is not empty", True, f"{cap_path}")
        cap_sections, cap_counts = _sections(cap_text)

    # A BOX PUBLISHED TWICE IS AN AMBIGUOUS MEASUREMENT, AND AMBIGUOUS IS
    # WORSE THAN MISSING. The section parser keeps the LAST publication of a
    # name, so a FAILING head capture-gate section published BEFORE the
    # clean one was invisible -- the rule read the clean one, published no
    # FAIL line, and ACCEPTED a failing capture gate. The check is SCOPED TO
    # THE BOXES THE RULE JUDGES: the capture gate's own sub-sections
    # ("=== log source: ... ===", "=== role: ... ===") are not box sections,
    # and an unscoped check would fail a sound evidence set on them.
    check("the capture gate publishes no box's section more than once",
          not _duplicate_box_sections(cap_counts),
          _duplicate_detail("capture-gate.txt", cap_counts))

    for box in REQUIRED_BOXES:
        body = cap_sections.get(box, "")
        if not body:
            check(f"the capture gate ran on {box}", False,
                  f"no '=== {box} ===' section in capture-gate.txt")
            continue
        ok_exit = "EXIT=0" in body
        no_fail = "[FAIL]" not in body
        extracted = "EXIT=extracted" in body
        check(f"the capture gate is clean on {box}", ok_exit and no_fail,
              "EXIT=0 and no [FAIL]" if ok_exit and no_fail
              else ("a [FAIL] line is present" if not no_fail
                    else ("the section is EXTRACT_ONLY -- facts, no verdict"
                          if extracted else "no EXIT=0 line")))

    facts: dict[str, dict[str, str]] = {}
    for box, body in cap_sections.items():
        f = {}
        for mm in re.finditer(r"^\[GATE\] ([A-Za-z0-9_]+)=(.*)$", body, re.M):
            f[mm.group(1)] = mm.group(2).strip()
        facts[box] = f

    # --- 2a. the spec config proven ON EVERY RANK, from the MOUNTED FILE ---
    for box in REQUIRED_BOXES:
        f = facts.get(box)
        if not f:
            check(f"the spec config is proven on {box}", False, "no [GATE] facts for this box")
            continue
        ok = (f.get("spec_cmd") == "yes"
              and f.get("spec_cmd_reads_file") == "yes"
              and f.get("spec_file") == "yes"
              and f.get("spec_file_method") == "yes"
              and f.get("spec_file_k") == "yes")
        check(f"the spec config is proven on {box}", ok,
              f"cmd={f.get('spec_cmd')} cmd_reads_file={f.get('spec_cmd_reads_file')} "
              f"file={f.get('spec_file')} file_method={f.get('spec_file_method')} "
              f"file_k={f.get('spec_file_k')}")

    # --- 2a-bis. THE BUDGET, ON EVERY RANK --------------------------------
    # The engine's OWN resolved budget, read from its startup log by the
    # capture gate. A rank with a different budget would have a different
    # KV pool size inside the TP ring -- a rank asymmetry that looks like
    # the change itself.
    for box in REQUIRED_BOXES:
        f = facts.get(box)
        if not f:
            check(f"the memory budget is {EXPECTED_GMU} on {box}", False,
                  "no [GATE] facts for this box")
            continue
        gmu = f.get("gmu")
        check(f"the memory budget is {EXPECTED_GMU} on {box}", gmu == EXPECTED_GMU,
              f"the engine resolved GMU={gmu} on this rank"
              + ("" if gmu == EXPECTED_GMU
                 else f" -- the profile pins GMU={EXPECTED_GMU}"))

    # --- 2b. the capture COVERAGE -----------------------------------------
    head_facts = None
    for box, f in facts.items():
        if f.get("role") == "head":
            head_facts = f
            break
    if not head_facts:
        check("the capture coverage is proven", False,
              "no head section with [GATE] facts in capture-gate.txt")
    else:
        check("the head's resolved spec config names num_spec_tokens=3",
              head_facts.get("spec_resolved") == "yes"
              and head_facts.get("spec_resolved_num_spec_tokens") == "yes",
              f"resolved={head_facts.get('spec_resolved')} "
              f"num_spec_tokens={head_facts.get('spec_resolved_num_spec_tokens')}")
        # THE CAPTURE SIZE IS READ THROUGH THE PAIR. A size the rule cannot
        # read is a MISSING measurement: the criterion below FAILS BY NAME,
        # and the rule never defaults it to a number that could satisfy the
        # floor on its own.
        max_size = _measured_int(head_facts.get("capture_max_size", "0"))
        check(f"max_cudagraph_capture_size >= {CAPTURE_COVERAGE_MIN}",
              max_size is not None and max_size >= CAPTURE_COVERAGE_MIN,
              f"max_cudagraph_capture_size = {max_size}"
              if max_size is not None else
              f"the engine published capture_max_size="
              f"{head_facts.get('capture_max_size')!r}, which is not a usable "
              "number, so the capture coverage cannot be judged (a missing "
              "measurement is a FAIL)")
        check(f"the capture list CONTAINS {CAPTURE_COVERAGE_MIN}",
              head_facts.get("capture_list_has_128") == "yes",
              f"capture_list_has_128 = {head_facts.get('capture_list_has_128')}")

    # --- 2c. the CAPACITY FLOOR --------------------------------------------
    # THE POOL AND THE CONCURRENCY ARE **HEAD FACTS**. The engine's
    # resolved config -- and with it the KV pool's TOKEN COUNT and the
    # engine's own concurrency figure -- is printed by the HEAD's
    # EngineCore only. The shipped capture gate says so in as many
    # words ("the token count comes from the head's EngineCore"), and it
    # publishes `[GATE] kv_concurrency=` ONLY on the head. The rule
    # already treats the GRAPH COUNTS this way -- they are printed under
    # `is_global_first_rank()`, so only the head has them, and a
    # criterion that expected them on the workers would have failed both
    # workers for a reason that has nothing to do with the boot.
    #
    # SO A RANK'S EMPTY CAPACITY FACT IS THE CONTRACT, NOT A MISSING
    # MEASUREMENT, and the rule must not read it as one.
    #
    # WHAT WAS WRONG. The facts were collected over EVERY box and
    # reduced with min() over the SURVIVORS. A rank whose fact was
    # absent or garbled was dropped, the min ran over the boxes that
    # happened to answer, and the criterion printed an unqualified
    # green: a FALSE ACCEPT on the rule's primary capacity criterion,
    # and a flat contradiction of the rule's own doctrine ("a missing
    # measurement is a FAIL, never a skip"). It passed on real evidence
    # by ACCIDENT -- the ranks publish no token count, so the survivor
    # set happened to be the head alone -- and a hostile or damaged
    # evidence file could hide a rank's disagreement entirely.
    #
    # THE RULE NOW SAYS:
    #   * the HEAD's pool and concurrency are the figures that are
    #     judged, and if the head published no usable one the criterion
    #     FAILS BY NAME;
    #   * a RANK's empty figure is expected, and the report says whose
    #     figure it is;
    #   * a RANK's figure that cannot be read is a DAMAGED FACT and is
    #     named -- never dropped;
    #   * a RANK's figure that DISAGREES with the head's is a ring that
    #     disagrees with itself, and it FAILS.
    def _capacity_fact(box, key, caster):
        """(value, problem) for one box's capacity fact.

        An EMPTY fact is (None, None): whether that is expected or a
        failure depends on the box, and only the caller knows that.

        THE FACT IS READ THROUGH THE PAIR (`caster` is `_safe_int` or
        `_safe_float`), so a fact the rule cannot parse is a MISSING fact
        that is NAMED -- never a figure the rule defaulted into the
        comparison.
        """
        raw = (facts.get(box) or {}).get(key)
        if raw is None or str(raw).strip() == "":
            return None, None
        v = caster(str(raw).strip())
        if v is None:
            return None, f"{box}: {key}={raw!r} is not a usable number"
        # AN INTEGER TOO LARGE FOR A FLOAT IS NOT A POOL FIGURE. It
        # divides fine until it does not: `pool / COMPARISON_CONTEXT`
        # raises OverflowError on it, and an OverflowError is a crash,
        # not a verdict. A figure the rule cannot arithmetic is a
        # MISSING figure, and it is reported as one.
        if not _is_finite(v):
            return None, f"{box}: {key}={raw!r} is not a usable number"
        return v, None

    head_box = HEAD_BOX
    pool, pool_problem = _capacity_fact(head_box, "kv_pool_tokens", _safe_int)
    conc_x, conc_problem = _capacity_fact(head_box, "kv_concurrency", _safe_float)

    # EVERY RANK IS CHECKED AGAINST THE HEAD. The figure a rank
    # publishes must be the head's; a rank that publishes nothing is
    # expected, and one that publishes a DIFFERENT number has a ring
    # that disagrees with itself.
    rank_problems = []
    for box in REQUIRED_BOXES:
        if box == head_box:
            continue
        for key, head_val, caster in (("kv_pool_tokens", pool, _safe_int),
                                       ("kv_concurrency", conc_x, _safe_float)):
            v, problem = _capacity_fact(box, key, caster)
            if problem:
                rank_problems.append(problem)
            elif v is not None and head_val is not None and v != head_val:
                rank_problems.append(
                    f"{box}: {key}={v!r} disagrees with the head's {head_val!r}")
    check("every rank's capacity facts are readable and consistent with the head's",
          not rank_problems,
          "every rank that publishes a capacity figure publishes the head's, "
          "and no rank contradicts the head"
          if not rank_problems else "; ".join(rank_problems[:4]))
    # THE DENOMINATOR COMES FROM THE CAPTURE GATE'S OWN FACT. The engine's
    # line carries the denominator, and the gate EXTRACTS it -- but the gate
    # publishes FACTS, not engine logs, and it used to publish only the
    # FIGURE. The rule parsed a raw engine line that the evidence never
    # contained, so the denominator was always unset: the criterion failed
    # and the rule then CRASHED formatting it. So the fact is read FIRST,
    # and the raw line is only the fallback for an evidence set that
    # carries it.
    #
    # IT IS THE HEAD'S FACT, LIKE THE FIGURE ITSELF: the head's EngineCore
    # is what prints the request length the concurrency was measured at.
    conc_ctx = None
    conc_ctx_raw = None
    # THE DENOMINATOR HAS TWO WITNESSES IN THE EVIDENCE, AND BOTH ARE READ.
    #
    # THE WITNESSES ARE:
    #   * the capture gate's machine-readable FACT -- the `[GATE]
    #     kv_concurrency_ctx=<N>` line. The gate strips the commas itself
    #     (tools/capture-config.sh), so in real evidence it reads "196608".
    #   * the ENGINE'S OWN boot line inside the gate's capture -- "Maximum
    #     concurrency for 196,608 tokens per request: 1.35x". That is the
    #     raw engine output, and it is GROUPED.
    #
    # BOTH ARE READ, ALWAYS. The prose used to be a FALLBACK -- read only
    # when the fact was missing -- and the fact used to be read by STRIPPING
    # ITS COMMAS. So a malformed fact ("196,60,8") was published as
    # "196,608" and the rule ACCEPTed a denominator the evidence never
    # carried, and a malformed prose line was never read at all whenever the
    # fact was there.
    #
    # A WITNESS IS:
    #   * ABSENT   -- the evidence carries no such figure. Absence is NOT
    #     damage: a witness that is not there costs nothing.
    #   * READABLE -- the figure is there and the rule can read it.
    #   * DAMAGED  -- the figure is there and the rule CANNOT read it. A
    #     damaged fact is NAMED, never dropped and never edited into a
    #     figure the rule made up.
    #
    # AND TWO WITNESSES THAT DISAGREE ARE A DENOMINATOR THAT IS UNKNOWN: the
    # rule does not get to pick the one it likes.
    ctx_witnesses = []
    raw_ctx = (facts.get(head_box) or {}).get("kv_concurrency_ctx")
    if raw_ctx is not None and str(raw_ctx).strip():
        ctx_witnesses.append(("head's kv_concurrency_ctx fact",
                              str(raw_ctx).strip(),
                              _strict_int(raw_ctx)))
    conc_m = re.findall(
        r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
        cap_text)
    if conc_m:
        ctx_witnesses.append(
            ("engine's own 'Maximum concurrency for N tokens per request' "
             "line", conc_m[-1][0], _strict_int(conc_m[-1][0])))
    ctx_damaged = [(w, raw) for (w, raw, v) in ctx_witnesses if v is None]
    ctx_readable = [(w, raw, v) for (w, raw, v) in ctx_witnesses if v is not None]
    ctx_problem = None
    ctx_summary = None
    if ctx_damaged:
        ctx_summary = "; ".join(
            f"the {w} published {raw!r}, which is not a number the rule can read"
            for w, raw in ctx_damaged)
        ctx_problem = (ctx_summary + " -- a damaged fact is NAMED, not "
                       "dropped, so the concurrency figure is about an unknown "
                       "request length and cannot be judged (a missing "
                       "measurement is a FAIL)")
    elif len(ctx_readable) > 1 and len({v for _, _, v in ctx_readable}) > 1:
        ctx_summary = " and ".join(
            f"the {w} published {raw!r}" for w, raw, _ in ctx_readable)
        ctx_problem = (ctx_summary + " -- the two witnesses DISAGREE, so the "
                       "denominator is UNKNOWN and the concurrency figure "
                       "cannot be judged (a missing measurement is a FAIL)")
    elif ctx_readable:
        conc_ctx = ctx_readable[0][2]
    # THE CONCURRENCY FACT ITSELF IS NOT RESCUED FROM THE PROSE. A missing
    # or unreadable head fact is a FAILURE BY NAME, not a gap the rule
    # fills in from a line of prose it cannot attribute to a box.

    if pool is None:
        check(f"KV pool >= {KV_FLOOR_TOKENS:,} tokens", False,
              (pool_problem or
               f"the head ({head_box}) published no kv_pool_tokens fact -- the "
               "engine's resolved config lives on the head only, so without "
               "it the pool is UNKNOWN, not absent")
              + (f" -- and {len(rank_problems)} rank fact(s) were unusable: "
                 + "; ".join(rank_problems[:3]) if rank_problems else ""))
    else:
        check(f"KV pool >= {KV_FLOOR_TOKENS:,} tokens", pool >= KV_FLOOR_TOKENS,
              f"{pool:,} tokens (the head's figure -- the engine prints the "
              "pool's token count on the head only)")
        if KV_FLOOR_TOKENS <= pool < KV_NOTE_TOKENS:
            note(f"CAPACITY NOTE: the pool is {pool:,} tokens -- above the floor of "
                 f"{KV_FLOOR_TOKENS:,} but BELOW {KV_NOTE_TOKENS:,}. The price of "
                 "speculative decoding (the draft's weights, the draft's own KV "
                 "group and the capture-pool growth) is paid in CAPACITY and is "
                 "NAMED here, not hidden. The memory budget is not raised to buy "
                 "the pool back; the price is paid in the budget instead.")
    if conc_x is None:
        check(f"the engine's own concurrency >= {CONCURRENCY_MIN}x", False,
              (conc_problem or
               f"the head ({head_box}) published no kv_concurrency fact -- the "
               "engine prints its own concurrency figure on the head only, so "
               "without it the factor is UNKNOWN, not absent"))
    else:
        # THE DENOMINATOR IS ASSERTED, NOT ASSUMED: a figure printed for a
        # different request length is a different measurement. A figure with
        # NO denominator is a figure about an unknown length, so it cannot
        # be judged -- AND A CRITERION THAT WAS NOT JUDGED IS NOT A PASS.
        #
        # AND A DENOMINATOR THE RULE COULD NOT READ IS NAMED, NOT REWRITTEN.
        # The detail quotes what the engine published -- "196,60,8" -- and
        # never a comma-stripped derivative of it, because a figure the rule
        # made up is a measurement the evidence never carried.
        if conc_ctx is not None:
            ctx_detail = (f"the engine reported its concurrency for a "
                          f"{conc_ctx:,}-token request")
            conc_detail = (f"{conc_x}x for a {conc_ctx:,}-token request (the "
                           "lane can still hold one full-length request)")
        elif ctx_problem:
            ctx_detail = ctx_problem
            conc_detail = (f"{conc_x}x, and the denominator is UNKNOWN -- so "
                           "the floor is not applied to a figure about an "
                           f"unknown request length ({ctx_summary})")
        else:
            ctx_detail = ("the concurrency figure carried no denominator, so "
                          "it is about an unknown request length and cannot "
                          "be judged")
            conc_detail = (f"{conc_x}x, with NO denominator in the evidence "
                           "-- so the floor is not applied to a figure about "
                           "an unknown request length")
        check(f"the concurrency figure is about the served context "
              f"({SERVED_CONTEXT:,}-token requests)",
              conc_ctx == SERVED_CONTEXT, ctx_detail)
        check(f"the engine's own concurrency >= {CONCURRENCY_MIN}x",
              conc_x >= CONCURRENCY_MIN and conc_ctx == SERVED_CONTEXT,
              conc_detail)

    # ==================================================================
    # 2d. THE CAPTURE GRID: the cap, the decode coverage, the mechanism,
    #      and the capacity it bought.
    # ==================================================================
    # The facts come from the capture gate's machine-readable [GATE] lines --
    # the MEASUREMENTS, not the gate's verdicts -- so this rule applies its
    # own thresholds and never has to trust the gate's opinion.

    # --- 2d-1. THE CAP IS IN FORCE (the head's resolved grid) ------------
    head_facts = facts.get(HEAD_BOX, {})
    resolved_list = head_facts.get("cg_resolved_list", "")
    # A SIZE THAT CANNOT BE READ IS NOT A SIZE. `str.isdigit()` is true for
    # a four-thousand-digit run of digits, and int() on one of those
    # raises ValueError -- so a hostile or damaged list would take the
    # rule down with a traceback instead of failing a criterion. A size
    # the rule cannot read is simply not in the set.
    resolved_set = tuple(_safe_int(x) for x in resolved_list.split()
                         if x.isdigit() and _safe_int(x) is not None)
    check(f"the resolved capture grid IS the pinned one ({len(EXPECTED_CG_LIST)} sizes)",
          resolved_set == EXPECTED_CG_LIST,
          "the engine resolved ["
          + " ".join(str(x) for x in resolved_set) + "]"
          if resolved_set else
          "the capture gate reported no resolved capture grid (a missing measurement is a FAIL)")
    # THE RESOLVED CAP SIZE IS READ THROUGH THE PAIR. A size the rule
    # cannot read is a MISSING measurement: the criterion below FAILS BY
    # NAME, and the rule never defaults it to the pinned number.
    resolved_max = _measured_int(head_facts.get("capture_max_size", "0"))
    check(f"max_cudagraph_capture_size == {EXPECTED_CG_MAX}",
          resolved_max is not None and resolved_max == EXPECTED_CG_MAX,
          f"the engine resolved {resolved_max}"
          if resolved_max is not None else
          f"the engine published capture_max_size="
          f"{head_facts.get('capture_max_size')!r}, which is not a usable "
          "number, so the capture grid cannot be judged (a missing "
          "measurement is a FAIL)")
    missing_ud = [s for s in UNIFORM_DECODE_SIZES if s not in resolved_set]
    check("all 7 uniform-decode sizes survived the cap (the decode hot path)",
          not missing_ud,
          f"MISSING: {missing_ud} -- the decode hot path lost a graph"
          if missing_ud else
          f"all of {list(UNIFORM_DECODE_SIZES)} are in the resolved grid")

    # --- 2d-2. THE GRAPH COUNTS (the HEAD's facts) -------------------------
    # HEAD ONLY, and this is a MEASURED FACT about the engine, not a
    # convenience: the per-mode capture bars are printed under
    # `is_global_first_rank()`, so only the head has them. On the three-rank
    # lane, rank 1 and rank 2 have ZERO "Capturing CUDA graphs" lines. A
    # criterion that expected them on the workers would have failed both
    # workers for a reason that has nothing to do with the boot -- the
    # per-rank symmetry proof comes from the capture pool's per-rank cost
    # instead (criterion 2d-3).
    # THE GRAPH COUNTS ARE READ THROUGH THE PAIR. A count the rule cannot
    # read is a MISSING measurement, and it is NAMED: the criterion below
    # compares the WHOLE table, so an unreadable count cannot be mistaken
    # for the pinned one.
    head_graphs: dict[str, int | None] = {}
    for k in ("piecewise", "full", "dspark"):
        head_graphs[k] = _measured_int(head_facts.get(f"cg_graph_{k}", "0"))

    def _graph_detail(k, v):
        if v is not None:
            return str(v)
        return repr(head_facts.get("cg_graph_" + k)) + " (not a usable number)"

    check("the captured graphs are the pinned ones",
          head_graphs == EXPECTED_CG_GRAPHS,
          f"piecewise={_graph_detail('piecewise', head_graphs['piecewise'])} "
          f"full={_graph_detail('full', head_graphs['full'])} "
          f"dspark={_graph_detail('dspark', head_graphs['dspark'])}"
          f" (pinned {EXPECTED_CG_GRAPHS['piecewise']}/"
          f"{EXPECTED_CG_GRAPHS['full']}/{EXPECTED_CG_GRAPHS['dspark']})")

    # --- 2d-3. THE MECHANISM MUST HAVE MOVED (every rank) -----------------
    # If the capture pool neither shrank into its expected band nor stayed
    # under the engine's own estimate, the profile did not move the thing it
    # claims to move -- whatever the pool says.
    for box in REQUIRED_BOXES:
        f = facts.get(box, {})
        # THE POOL FIGURES ARE READ THROUGH THE PAIR. A figure the rule
        # cannot read is a MISSING measurement: the criteria below FAIL BY
        # NAME, and the rule never defaults an UNREADABLE value to a
        # number. `float("inf")` used to reach the output as "the engine
        # estimated inf GiB" -- a non-finite figure published as a
        # measurement, which is exactly what this rule refuses to do.
        #
        # A FACT THAT WAS NEVER PUBLISHED is a different case: the pair
        # reads the "0" default and the criterion FAILS on it, because 0
        # is not inside the band. The sibling criterion "the capture gate
        # ran on {box}" is what NAMES the absent box.
        est_raw = f.get("capture_pool_estimated_gib", "0")
        act_raw = f.get("capture_pool_actual_gib", "0")
        est_gib = _safe_float(est_raw)
        act_gib = _safe_float(act_raw)
        max_est = CAPTURE_POOL_ESTIMATE_MAX_GIB.get(box, 0.0)
        max_act = CAPTURE_POOL_ACTUAL_MAX_GIB.get(box, 0.0)
        check(f"the capture-pool ESTIMATE is within the expected band on {box} "
              f"(under {max_est} GiB there)",
              est_gib is not None and 0 < est_gib < max_est,
              f"the engine estimated {est_gib} GiB (the expected band on this rank is "
              f"under {max_est} GiB)"
              if est_gib is not None else
              f"the engine published capture_pool_estimated_gib={est_raw!r}, "
              "which is not a usable number, so the capture-pool estimate "
              "cannot be judged (a missing measurement is a FAIL)")
        check(f"the capture pool's ACTUAL size is within the expected band on {box} "
              f"(under {max_act} GiB there)",
              act_gib is not None and 0 < act_gib < max_act,
              f"the engine's actual pool is {act_gib} GiB (the expected band on this "
              f"rank is under {max_act} GiB)"
              if act_gib is not None else
              f"the engine published capture_pool_actual_gib={act_raw!r}, which "
              "is not a usable number, so the capture pool's actual size "
              "cannot be judged (a missing measurement is a FAIL)")
        # THE PER-RANK PROOF: the actual pool stays under the engine's own
        # estimate for this rank. The estimate is subtracted from the memory
        # budget BEFORE the KV cache is sized, so the gap between the two is
        # capacity the lane pays for and never uses.
        check(f"the capture pool's ACTUAL stays under the engine's own ESTIMATE on {box}",
              act_gib is not None and est_gib is not None and 0 < act_gib < est_gib,
              f"the engine's actual pool is {act_gib} GiB against its own estimate of "
              f"{est_gib} GiB"
              if act_gib is not None and est_gib is not None else
              "the capture pool's actual size and the engine's own estimate "
              "are not both usable numbers, so the two cannot be compared (a "
              "missing measurement is a FAIL)")

    # --- 2d-4. NO CAPACITY LOSS, AND THE CAPACITY MUST HAVE MOVED ---------
    if pool is not None:
        check(f"the pool did not SHRINK below the {POOL_NO_LOSS_TOKENS:,}-token expectation",
              pool >= POOL_NO_LOSS_TOKENS,
              f"{pool:,} tokens (the expectation is {POOL_NO_LOSS_TOKENS:,})"
              + (f" -- a GAIN of {pool - POOL_NO_LOSS_TOKENS:,} tokens"
                 if pool > POOL_NO_LOSS_TOKENS else ""))
        if pool == POOL_NO_LOSS_TOKENS:
            check("the capacity MOVED (the profile bought something)", False,
                  f"the pool is exactly {pool:,} tokens: the profile applied and the "
                  "capacity did not move AT ALL -- it bought nothing",
                  verdict="NO-EFFECT")
        else:
            check("the capacity MOVED (the profile bought something)", True,
                  f"{pool:,} tokens against the {POOL_NO_LOSS_TOKENS:,}-token "
                  f"expectation (+{pool - POOL_NO_LOSS_TOKENS:,} tokens)")
    else:
        check("the capacity moved (the profile bought something)", False,
              "no pool figure in the evidence -- the capacity cannot be judged")
    if conc_x is not None and pool is not None:
        # THE LIKE-FOR-LIKE READING. The engine's own concurrency figure is
        # a property of (pool / request length), so comparing the figure
        # printed for a 196,608-token request against a threshold written
        # for a 131,072-token request compares two different things and
        # would ALWAYS fail. The expectation is applied to the pool AT THE
        # COMPARISON CONTEXT, and the engine's own figure is published
        # beside it, so the change of denominator is visible and not hidden.
        # THE COMPARISON CONTEXT IS A DENOMINATOR AND MUST BE POSITIVE **AND
        # USABLE**. It comes from the environment, and an environment that
        # sets it to 0 used to raise ZeroDivisionError instead of reporting
        # the misconfiguration it is. An environment that sets it to an
        # integer too large for a float raises OverflowError on the same
        # division. Both are misconfigurations, and both are reported as
        # the misconfigurations they are.
        if not (COMPARISON_CONTEXT > 0 and _is_finite(COMPARISON_CONTEXT)):
            check(f"the capacity did not fall below the {CONCURRENCY_NO_LOSS_X}x "
                  f"expectation at a {COMPARISON_CONTEXT:,}-token request",
                  False,
                  f"the comparison context ({COMPARISON_CONTEXT}) is not a "
                  "positive, usable number of tokens, so the ratio cannot be "
                  "computed (a missing measurement is a FAIL)")
            like_for_like = None
        else:
            like_for_like = pool / COMPARISON_CONTEXT
        # THE FIGURE IS PUBLISHED WITH THE DENOMINATOR IT WAS PRINTED FOR,
        # OR WITH AN HONEST STATEMENT THAT IT HAS NONE. Formatting an unset
        # denominator crashed the rule; a number about an unknown length is
        # still worth publishing, as long as it says so.
        conc_about = (f"a {conc_ctx:,}-token request is {conc_x}x"
                      if conc_ctx is not None else
                      f"an unstated request length is {conc_x}x")
        check(f"the capacity did not fall below the {CONCURRENCY_NO_LOSS_X}x "
              f"expectation at a {COMPARISON_CONTEXT:,}-token request",
              like_for_like is not None and like_for_like >= CONCURRENCY_NO_LOSS_X,
              (f"the pool of {pool:,} tokens is "
               f"{_fig_or_none(like_for_like, 2, lambda v: v >= CONCURRENCY_NO_LOSS_X)}x a "
               f"{COMPARISON_CONTEXT:,}-token request (the expectation is "
               f"{CONCURRENCY_NO_LOSS_X}x); the engine's own figure for "
               f"{conc_about}") if like_for_like is not None else
              (f"the pool of {pool:,} tokens cannot be divided by the "
               f"comparison context ({COMPARISON_CONTEXT}), so its ratio is "
               f"unavailable; the engine's own figure for {conc_about}"))

    # ==================================================================
    # 3. the kit: asserted INSIDE each box's own section
    # ==================================================================
    kit_path = p("kit-check.txt")
    kit_text = _read(kit_path)
    kit_sections, kit_counts = _sections(kit_text) if kit_text.strip() else ({}, {})
    # THE SAME AMBIGUITY, IN THE SAME PARSER: a box's kit-check section
    # published twice is a box whose kit identity the rule cannot pin down
    # to this cycle. Scoped to the boxes the rule judges, for the reason
    # given above.
    check("the kit check publishes no box's section more than once",
          not _duplicate_box_sections(kit_counts),
          _duplicate_detail("kit-check.txt", kit_counts))
    for box in REQUIRED_BOXES:
        body = kit_sections.get(box, "")
        check(f"the kit check is OK on {box}",
              "kit-md5: OK" in body,
              "kit-md5: OK in this box's own section" if "kit-md5: OK" in body
              else ("no 'kit-md5: OK' inside this box's section"
                    if body else f"no section for {box} in kit-check.txt"))

    # ==================================================================
    # 4. the correctness battery
    # ==================================================================
    probes = _read(p("probes.txt"))
    check("the 7 thinking probes pass", "all 7 probes PASS" in probes,
          "all 7 probes PASS" if "all 7 probes PASS" in probes
          else ("probes.txt is missing or does not report 7/7 PASS"
                if probes else "probes.txt is missing or empty"))

    # THE ANSWER MUST NAME THE COLOUR. "Red" appearing SOMEWHERE in the
    # file is not an answer: the model's reasoning may name it while the
    # answer says something else, a stray note may name it, and "Reddish"
    # is not "Red". The answer is the content: field when the capture
    # writes one, and the file's text when it does not; and the word must
    # stand alone.
    #
    # AND IT MUST NOT DENY THE COLOUR. NAMING THE COLOUR IS NECESSARY BUT
    # NOT SUFFICIENT: "the image is not Red at all, it is blue" names the
    # colour and is still a broken vision path. The old check PASSed that
    # answer and printed it on the very same line.
    #
    # THE GUARD IS NARROW ON PURPOSE. A WIDE GUARD -- "any negation anywhere
    # in the answer" -- is a FALSE-REJECT risk, and this rule is judged on
    # real answers: "Red, not blue" IS a red answer, and so is "It is not
    # blue, it is Red". So the guard reads ONLY the at most 3 tokens
    # IMMEDIATELY IN FRONT of the whole word "red": a negation there denies
    # the colour, and a negation further away is about something else.
    #
    # AND THE NEGATION MUST BE IN THE SAME CLAUSE AS THE COLOUR. A negation
    # NEARBY is not a denial of THIS colour: "Not blue. Red." denies BLUE
    # and NAMES red. The old guard read the three tokens before the colour
    # word and fired on that negation, so it FALSE-REJECTED the answer and
    # published a denial it had not established -- and the identical
    # assertion in the other order ("Red, not blue") PASSED, so the verdict
    # depended on the ORDER of two sentences. A sentence terminator (".", "!",
    # "?") between the negation and the colour word means the answer moved on
    # to a NEW sentence: the negation belongs to the PREVIOUS one and denies
    # whatever THAT sentence named, not the colour the next sentence names.
    vision = _read(p("vision-smoke.txt"))
    vision_answer = ""
    if vision:
        mv = re.search(r"^content:\s*(.*)$", vision, re.M)
        raw = mv.group(1).strip() if mv else vision
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1]
        vision_answer = raw.strip()
    names_red = bool(vision_answer) and re.search(r"\bred\b", vision_answer,
                                                  re.I) is not None
    denies_negation = ""
    denies_red = False
    if names_red:
        for _m in re.finditer(r"\bred\b", vision_answer, re.I):
            _before = list(re.finditer(r"[\w']+", vision_answer[:_m.start()]))
            _window = [(t.group(0).lower(), t.end()) for t in _before[-3:]]
            for _tok, _tok_end in reversed(_window):
                if not (_tok in NEGATION_TOKENS or _tok.endswith("n't")):
                    continue
                # THE CLAUSE RULE. A sentence terminator between the negation
                # and the colour word means the answer moved on to a NEW
                # sentence -- so the negation denies whatever THAT sentence
                # named, and not the colour this one names.
                if re.search(r"[.!?]", vision_answer[_tok_end:_m.start()]):
                    continue
                denies_red = True
                denies_negation = _tok
                break
            if denies_red:
                break
    if denies_red:
        check("the vision path answers 'Red'", False,
              f"the answer DENIES the colour Red ({vision_answer[:120]!r}) -- "
              f"the negation {denies_negation!r} stands in the SAME clause as "
              "the colour it names, so the answer is not a red answer")
    else:
        check("the vision path answers 'Red'", names_red,
              f"the answer is {vision_answer[:120]!r}" if vision_answer
              else ("vision-smoke.txt carries no answer to read" if vision
                    else "vision-smoke.txt is missing or empty"))

    c1 = _read(p("c1-proof.txt"))
    check("the C-1 byte proof passes", "C-1 PROOF PASS" in c1,
          "C-1 PROOF PASS" if "C-1 PROOF PASS" in c1
          else ("c1-proof.txt is missing or does not report the proof"
                if c1 else "c1-proof.txt is missing or empty"))

    ring = _read(p("ring-proof.txt"))
    check("the ring proof holds", "every rank issued the same collective sequence" in ring,
          "every rank issued the same collective sequence"
          if "every rank issued the same collective sequence" in ring
          else ("ring-proof.txt is missing or reports a divergence"
                if ring else "ring-proof.txt is missing or empty"))

    # ==================================================================
    # 5. the guard: no breaches, and the floor ARMED is the floor PINNED
    #    for this profile
    # ==================================================================
    guard_path = p("guard.txt")
    guard_text = _read(guard_path)
    guard_sections, guard_counts = (_sections(guard_text)
                                    if guard_text.strip() else ({}, {}))
    # AND AGAIN: a box's guard section published twice is a guard window the
    # rule cannot tell apart from another one. Scoped to the boxes the rule
    # judges, for the reason given above.
    check("the guard log publishes no box's section more than once",
          not _duplicate_box_sections(guard_counts),
          _duplicate_detail("guard.txt", guard_counts))
    for box in REQUIRED_BOXES:
        body = guard_sections.get(box, "")
        if box not in guard_sections:
            check(f"the guard evidence exists for {box}", False,
                  f"no '=== {box} ===' section in guard.txt")
        # EVERY NUMBER OUT OF THE GUARD LOG GOES THROUGH THE SAFE
        # PARSERS, AND A NUMBER THAT CANNOT BE READ IS A DAMAGED FACT --
        # NAMED, never dropped. Dropping it would let a hostile log hide a
        # breach behind an unreadable counter and leave the rule judging
        # only the survivors. That is the fail-open this rule has been
        # audited for twice, and it is closed here at the SOURCE.
        #
        # THE DAMAGE IS SPLIT BY COUNTER, because the three counters answer
        # three DIFFERENT questions, and lumping them together made the
        # rule say things that were FALSE of the evidence:
        #   * breaches= -> the BREACH state. Nothing else decides it.
        #   * low_mb=   -> the RUNNING MINIMUM. Nothing else decides it.
        #   * avail_mb= -> the AVAILABLE margin. In this rule it is
        #     DETAIL-ONLY: it feeds no verdict, so a damaged avail series
        #     fails NOTHING -- it is NAMED in the minimum's detail.
        lows = []
        avail = []
        breaches = []
        damaged_low = []
        damaged_avail = []
        damaged_breaches = []
        # THE TWO MARKER BUCKETS. `_GUARD_NO_READING` is the guard saying
        # it COULD NOT READ the counter; `_GUARD_NO_SAMPLE` is the guard
        # saying it HAD NOT SAMPLED YET. Both are written BY THE GUARD and
        # both are excluded from the series and counted, so the detail can
        # say how many ticks carried each of them. Neither is damage, and
        # neither is a figure.
        no_reading = {"low_mb=": 0, "avail_mb=": 0, "breaches=": 0}
        no_sample = {"low_mb=": 0, "avail_mb=": 0, "breaches=": 0}
        for key, target, caster, dmg in (("low_mb=", lows, _safe_float,
                                          damaged_low),
                                         ("avail_mb=", avail, _safe_float,
                                          damaged_avail),
                                         ("breaches=", breaches, _safe_int,
                                          damaged_breaches)):
            # THE WHOLE VALUE, OR NOTHING -- the same standard
            # `_first_int` uses. The old pattern took a PREFIX of the
            # digits, so a sample like "low_mb=7596zzz" was read as
            # 7596: a GOOD sample the guard never wrote, on a counter
            # the rule had promised to name when it could not read it.
            # A sample the rule cannot read is a DAMAGED sample, and it
            # is NAMED below.
            for x in re.findall(rf"\b{re.escape(key)}(\S*)", body):
                if x == _GUARD_NO_READING:
                    # THE GUARD'S OWN MARKER: it is excluded from the
                    # series, and it is counted, so the detail can say how
                    # many ticks the guard could not read.
                    no_reading[key] += 1
                    continue
                if x in _GUARD_NO_SAMPLE:
                    # THE GUARD'S OWN "NO SAMPLE YET" MARKER. The guard
                    # writes the DASH on its sample line while the container
                    # is not up yet, and the bare 999,999 on its exit line,
                    # and it does so on purpose: a sample that said
                    # low_mb=999999 would read as "an enormous free-memory
                    # minimum" and the rule would PASS a box on which
                    # nothing was ever measured.
                    #
                    # IT IS NOT A FIGURE AND IT IS NOT DAMAGE. It is the
                    # guard SAYING that it had not sampled yet, and it hides
                    # no measurement: the guard moves its ratchet only when
                    # it has read an avail, so the tick carries the
                    # ratchet's value as it stood BEFORE the first reading.
                    # The ticks that DID carry a reading are what the
                    # running minimum is judged on; these ticks are named.
                    no_sample[key] += 1
                    continue
                v = caster(x)
                if v is None:
                    dmg.append(x)
                elif key == "low_mb=" and v >= 999_999:
                    # THE LINE IS PUBLISHED. The sentinel is not a reading,
                    # but the LINE that carries it is there -- so the fact
                    # is not ABSENT, and a sentence that says the guard log
                    # has no low_mb= line would be false of it.
                    #
                    # THE 999,999 SENTINEL IS NOT A SAMPLE. The guard's
                    # running minimum starts there, and it means "no sample
                    # yet", not "a very large sample". It is the exit-line
                    # form of the same fact the dash carries, so it is
                    # counted with the dash.
                    no_sample[key] += 1
                    continue
                else:
                    target.append(v)
        # THE STATE OF EACH COUNTER, SEPARATELY:
        #   READABLE   -- there is a reading, and the counter is judged on it.
        #   DAMAGED    -- a sample the rule cannot read. The counter's state
        #                 is UNKNOWN: the rule cannot know that the sample
        #                 it could not read was not the one that mattered.
        #   NO-READING -- the guard's own '?' marker on a counter the guard
        #                 never marks that way: the guard said it could not
        #                 read it, and the rule does not know what it said.
        #   NO-SAMPLE  -- the guard's own "no sample yet" marker. On the
        #                 RATCHET (`low_mb=`) it hides no measurement, so
        #                 the readings judge it and the marker ticks are
        #                 named; on the counters the guard reads on every
        #                 tick it leaves their state UNKNOWN.
        #   ABSENT     -- the counter published no line at all.
        if damaged_breaches:
            breach_state = "damaged"
        elif no_reading["breaches="]:
            breach_state = "no-reading"
        elif breaches:
            breach_state = "readable"
        elif no_sample["breaches="]:
            breach_state = "no-sample"
        else:
            breach_state = "absent"
        if damaged_low:
            low_state = "damaged"
        elif no_reading["low_mb="]:
            low_state = "no-reading"
        elif lows:
            low_state = "readable"
        elif no_sample["low_mb="]:
            low_state = "no-sample"
        else:
            low_state = "absent"
        # DID THE GUARD WRITE ABOUT THIS BOX AT ALL? A box the guard never
        # mentioned has no counters to read, and the rule says so -- twice --
        # in the criteria above.
        section_present = box in guard_sections
        # THE BREACH STATE IS JUDGED ON THE breaches= COUNTER, AND ON
        # NOTHING ELSE. A damaged low_mb= or avail_mb= sample does not make
        # the breach state unknown, and saying so would be FALSE of the
        # evidence: the breaches= counter was perfectly readable.
        if breach_state == "readable":
            breach_lines = sum(1 for b in breaches if b > 0)
            breach_peak = max(breaches)
            check(f"the guard never fired on {box}",
                  breach_lines == 0 and breach_peak == 0,
                  f"{breach_lines} breach line(s), peak {breach_peak}"
                  if (breach_lines or breach_peak) else "0 breaches")
        elif breach_state == "damaged":
            check(f"the guard never fired on {box}", False,
                  f"the guard log for {box} carries {len(damaged_breaches)} "
                  f"breaches= sample(s) the rule cannot read "
                  f"({', '.join(repr(x) for x in damaged_breaches)}) -- a "
                  "damaged fact is NAMED, not dropped, so the breach state "
                  "of this box is UNKNOWN")
        elif breach_state == "no-reading":
            check(f"the guard never fired on {box}", False,
                  f"the guard log for {box} carries {no_reading['breaches=']} "
                  "breaches= sample(s) the guard itself marked '?' -- the "
                  "guard could not read that counter, so the breach state of "
                  "this box is UNKNOWN")
        elif breach_state == "no-sample":
            # THE GUARD'S OWN "NO SAMPLE YET" MARKER ON A COUNTER IT READS
            # ON EVERY TICK. `breaches=` is the guard's own breach COUNT, so
            # a tick that carries no reading of it leaves the breach state
            # UNKNOWN -- the count that tick would have carried could have
            # been the count of a breach.
            check(f"the guard never fired on {box}", False,
                  f"the guard log for {box} carries {no_sample['breaches=']} "
                  "breaches= tick(s) the guard itself marked as no sample "
                  "yet -- the guard carried no reading of its breach "
                  "counter on that tick, so the breach state of this box is "
                  "UNKNOWN")
        elif not section_present:
            # THE GUARD PUBLISHED NO SECTION FOR THIS BOX. There is no
            # guard evidence for the box at all, and the rule says so.
            check(f"the guard never fired on {box}", False,
                  f"no guard evidence for {box}: the guard log has no "
                  "low_mb= and no breaches= line")
        else:
            # THE SECTION IS THERE, AND IT CARRIES NO breaches= LINE.
            check(f"the guard never fired on {box}", False,
                  f"no guard evidence for {box}: the guard log has no "
                  "breaches= line")
        log_breaches = _first_int(body, "guard_log_breaches")
        if log_breaches is None:
            check(f"the guard.log records no breach event on {box}", False,
                  (f"the guard_log_breaches count for {box} is PUBLISHED but "
                   "the rule cannot read its value"
                   if _fact_published(body, "guard_log_breaches") else
                   f"no guard_log_breaches count for {box} -- the guard's own "
                   "log was not read")
                  + " (a missing measurement is a FAIL)")
        else:
            check(f"the guard.log records no breach event on {box}", log_breaches == 0,
                  f"guard_log_breaches={log_breaches}")
        starts = _first_int(body, "guard_start_lines")
        check(f"the guard evidence on {box} is windowed to this cycle",
              starts == 1,
              f"{starts} arming line(s) in guard.log -- arming ROTATES the log, "
              "so exactly 1 means the log covers this cycle alone; more than 1 "
              "means the log was not rotated and the evidence spans cycles"
              if starts is not None else
              ("the guard_start_lines count is PUBLISHED but the rule cannot "
               "read its value"
               if _fact_published(body, "guard_start_lines") else
               "no guard_start_lines count -- the guard's own log was not read"))
        win_breaches = _first_int(body, "guard_window_breaches")
        check(f"the guard never fired inside this cycle's window on {box}",
              win_breaches == 0,
              f"guard_window_breaches={win_breaches} (the highest consecutive "
              "breach count the guard's own counter reached inside the window)"
              if win_breaches is not None else
              (("the guard_window_breaches count is PUBLISHED but the rule "
                "cannot read its value"
                if _fact_published(body, "guard_window_breaches") else
                "no guard_window_breaches count -- the window could not be read")
               + " (a missing measurement is a FAIL)"))
        floor = GUARD_FLOORS.get(box)
        armed = _first_int(body, "armed_threshold_mb")
        if armed is None:
            check(f"{box} ran with the pinned guard floor of {floor} MB", False,
                  ("the armed_threshold_mb in "
                   f"{box}'s guard section is PUBLISHED but the rule cannot "
                   "read its value"
                   if _fact_published(body, "armed_threshold_mb") else
                   f"no armed_threshold_mb in {box}'s guard section")
                  + " -- the floor the GUARD ITSELF was armed at is unknown, so "
                  "the cycle did not run the pinned rule")
        else:
            check(f"{box} ran with the pinned guard floor of {floor} MB",
                  armed == floor,
                  f"the guard's own start line says min={armed} MB"
                  + ("" if armed == floor
                     else f" -- the pinned floor is {floor} MB"))
        # THE RUNNING MINIMUM IS JUDGED ON THE low_mb= COUNTER, AND ON
        # NOTHING ELSE. A criterion that was NOT JUDGED is not a pass, so
        # the running minimum -- and the margin this profile's price is
        # paid in -- are judged BY NAME whenever that counter carries no
        # reading. They are never silently skipped.
        #
        # AND A BOX THE GUARD NEVER WROTE ABOUT IS NOT THIS CRITERION'S
        # BUSINESS: the rule says, twice, above, that there is no guard
        # evidence for that box at all, and a box the guard never mentioned
        # has no running minimum to judge.
        if low_state == "readable":
            min_mb = min(lows)
            # THE FIGURES ARE PRINTED THROUGH THE PAIR, like every other
            # number the rule reads out of the evidence. The running
            # minimum is a finite float the guard's own parsers produced,
            # so the integer rendering cannot fail.
            min_mb_i = _safe_int(min_mb)
            avail_i = _safe_int(min(avail)) if avail else None
            # THE avail_mb= SERIES FEEDS NO VERDICT IN THIS RULE, so a
            # damaged or incomplete avail series fails NOTHING -- but it is
            # NAMED here, because a reader who sees "available low 7632 MB"
            # is entitled to know that some ticks carried no reading at all.
            avail_note = ""
            if no_sample["low_mb="]:
                # THE GUARD'S OWN "NO SAMPLE YET" TICKS. They carry no
                # reading and they hide none, so the running minimum above
                # is the minimum over the ticks that DID read -- and the
                # ticks that did not are named here.
                avail_note = (f" -- the guard had not sampled yet at "
                              f"{no_sample['low_mb=']} tick(s), so that tick "
                              "carries no reading")
            if no_reading["avail_mb="]:
                avail_note += (f" -- the guard could not read meminfo at "
                               f"{no_reading['avail_mb=']} tick(s), so that "
                               "tick carries no available reading")
            if no_sample["avail_mb="]:
                avail_note += (f" -- and the guard had not sampled yet at "
                               f"{no_sample['avail_mb=']} tick(s) of that "
                               "counter")
            if damaged_avail:
                avail_note += (f" -- and {len(damaged_avail)} avail_mb= "
                               "sample(s) the rule cannot read are named, not "
                               "dropped")
            check(f"{box}'s guard running minimum >= {floor} MB",
                  min_mb >= floor,
                  f"{min_mb_i} MB (the running minimum over the whole guard log)"
                  + (f", available low {avail_i} MB" if avail else "")
                  + avail_note)
            # THE PRICE, named. This profile's price is paid in the last
            # rank's margin over its pinned floor, so the margin has to be
            # there: at least 1,000 MB above the floor. The mechanism
            # predicts the price is near ZERO, and this criterion is what
            # says so out loud.
            if box == BOX3:
                need = BOX3_MIN_AVAILABLE_MB - floor
                check(f"{box} kept at least {need} MB of margin over its {floor} MB floor",
                      min_mb >= BOX3_MIN_AVAILABLE_MB,
                      f"the running minimum was {min_mb_i} MB -- the margin was "
                      f"{min_mb_i - floor} MB (this profile requires at least {need} MB)")
        elif section_present:
            # THE GUARD PUBLISHED A SECTION FOR THIS BOX, AND THE low_mb=
            # COUNTER IN IT CARRIES NO READING. The running minimum is
            # UNKNOWN, and a criterion that was not judged is not a pass:
            # it is judged, and it FAILS by name.
            if low_state == "damaged":
                low_detail = (
                    f"the guard log for {box} carries {len(damaged_low)} "
                    f"low_mb= sample(s) the rule cannot read "
                    f"({', '.join(repr(x) for x in damaged_low)}) -- a damaged "
                    "fact is NAMED, not dropped, so the running minimum of "
                    "this box is UNKNOWN")
            elif low_state == "no-reading":
                low_detail = (
                    f"the guard log for {box} carries {no_reading['low_mb=']} "
                    "low_mb= sample(s) the guard itself marked '?' -- the "
                    "guard could not read that counter, so the running "
                    "minimum of this box is UNKNOWN")
            elif low_state == "no-sample":
                # EVERY SAMPLE ON THIS BOX CARRIED THE GUARD'S OWN "NO
                # SAMPLE YET" MARKER: the container never came up, so the
                # box has no reading at all. The running minimum is
                # UNKNOWN -- and the detail says the guard had no sample
                # yet, NOT that a sample was damaged.
                low_detail = (
                    "the low_mb= line is PUBLISHED and carries the guard's "
                    "own 'no sample yet' sentinel, not a reading -- the guard "
                    f"had not sampled yet at {no_sample['low_mb=']} tick(s), "
                    "so the running minimum of this box is UNKNOWN")
            else:
                low_detail = (
                    f"the guard log for {box} carries no low_mb= line -- so "
                    "the running minimum of this box is UNKNOWN")
            check(f"{box}'s guard running minimum >= {floor} MB", False,
                  low_detail)
            if box == BOX3:
                need = BOX3_MIN_AVAILABLE_MB - floor
                check(f"{box} kept at least {need} MB of margin over its {floor} MB floor",
                      False,
                      "the running minimum is UNKNOWN, so the margin over the "
                      f"{floor} MB floor is UNKNOWN too (this profile requires "
                      f"at least {need} MB) -- a missing measurement is a FAIL")

    # ==================================================================
    # the verdict
    # ==================================================================
    print()
    if no_effect:
        print(f"acceptance-check: NO-EFFECT -- {no_effect}. The profile did nothing "
              "worth its price and is not retried without a new variable.")
    if fails:
        print(f"acceptance-check: {len(fails)} criterion/criteria FAILED: {fails}")
        return 1
    if no_effect:
        return 1
    # THE BANNER LISTS ONLY THE CRITERIA THAT WERE JUDGED. A one-endpoint
    # deployment has no relay floor, and a banner that claims it was met
    # credits the rule with a proof it did not perform.
    relay_clause = (f", relay >= {RELAY_STREAM_MIN} tok/s" if relay_judged else "")
    print(f"acceptance-check: every acceptance criterion is met -- ACCEPT "
          f"(1-way lane >= {LANE_STREAM_KEEP} tok/s{relay_clause}, "
          f"per-token inter-token <= "
          f"{INTER_TOKEN_MAX_MS} ms, TTFT <= {TTFT_MAX_S} s over the baseline path, "
          f"pool >= {KV_FLOOR_TOKENS:,} tokens, concurrency >= {CONCURRENCY_MIN}x, "
          f"acceptance at position 0 >= {ACCEPT_POS0_MIN}, budget {EXPECTED_GMU} on "
          f"every rank, token accounting verified, AND the capture grid capped at "
          f"{EXPECTED_CG_MAX} with all {len(UNIFORM_DECODE_SIZES)} uniform-decode "
          f"sizes and {EXPECTED_CG_GRAPHS['full']} decode graphs intact, the capture "
          f"pool within its expected band and under the engine's own estimate on "
          f"every rank, and no capacity lost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
