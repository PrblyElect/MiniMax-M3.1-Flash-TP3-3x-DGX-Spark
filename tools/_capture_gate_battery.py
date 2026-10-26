#!/usr/bin/env python3
"""_capture_gate_battery.py -- the battery for this kit's CAPTURE GATES.

WHY THIS EXISTS
===============
The capture gates are two of this kit's producers: `capture-config.sh` and
`capture-base-config.sh` publish the engine's own capacity figures and its
capture bars, and the acceptance rule judges what they publish. They are
PRODUCERS, and a producer that publishes a figure the source did not carry
is the worst kind of defect in this kit -- because the rule CANNOT SEE IT.
The rule reads the FIGURE, and a figure that is wrong at the source is
judged as though it were right.

Two such defects lived here, and both were invisible to a reader of the
evidence:

  * the capture pool's thousands grouping was STRIPPED before the figure
    was checked, so a malformed grouping ("20,8,128") became the clean
    integer (200000) -- passed the gate's own floor check and published as
    a figure the engine never wrote;
  * the capture bar's DENOMINATOR was dropped, so an interrupted bar whose
    last line read "19/35" published "19" -- the very figure a COMPLETE
    "19/19" publishes, and a figure the source never carried.

So the gates shipped with NO regression test inside the artifact. This one
is that test.

WHAT IT DRIVES
--------------
The REAL blocks of the REAL scripts: each block is extracted from the
script's text BY ANCHOR and run in a bash subshell over a synthetic engine
log. NOTHING here re-types an expression, so the battery tracks the shipped
scripts byte for byte -- if a script changes, the battery follows it.

THE TARGETS are this kit's own two gates. Nothing else: the battery is
about the artifact, and it does not reach outside it.

The synthetic engine log carries no name, no address and no date, and
every figure in it is synthetic.

Usage:
    python3 -B tools/_capture_gate_battery.py
    python3 -B tools/_capture_gate_battery.py --target capture-config
    python3 -B tools/_capture_gate_battery.py --script PATH

Exit: 0 when every case of every target is GREEN, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile

# NEVER leave bytecode next to the kit: this folder is shipped as it is.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))


class BatteryError(Exception):
    """The script under test does not have the shape the battery needs."""


# --------------------------------------------------------------------------
# THE BLOCKS ARE EXTRACTED FROM EACH SCRIPT, BY ANCHOR.
#
# Each block is (start_anchor, end_anchor, end_mode). The block runs from the
# line that STARTS WITH the start anchor up to (but excluding) the line that
# matches the end anchor. "exact" mode compares the stripped line, "prefix"
# mode uses startswith.
# --------------------------------------------------------------------------
def _find(lines, pred, start=0):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    return -1


def extract_blocks(script_text: str, spec) -> str:
    """Return the extracted blocks of one script, as one bash fragment."""
    lines = script_text.splitlines()
    out = []
    for start_anchor, end_anchor, end_mode in spec:
        i = _find(lines, lambda l: l.startswith(start_anchor))
        if i < 0:
            raise BatteryError("the start anchor %r is not in the script"
                               % start_anchor)
        if end_mode == "exact":
            j = _find(lines, lambda l: l.strip() == end_anchor, i + 1)
        else:
            j = _find(lines, lambda l: l.startswith(end_anchor), i + 1)
        if j < 0:
            raise BatteryError("the end anchor %r is not in the script"
                               % end_anchor)
        out.append("\n".join(lines[i:j]))
    return "\n".join(out)


def run_blocks(script_text: str, spec, preamble: str, log_text: str):
    """Run the extracted blocks of one script over a synthetic engine log."""
    body = extract_blocks(script_text, spec)
    fd, log_path = tempfile.mkstemp(prefix="capture-gate-", suffix=".log")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(log_text)
        runner = (preamble
                  + "\nTMP=" + shlex.quote(log_path) + "\n"
                  + body + "\n")
        proc = subprocess.run(["bash", "-c", runner],
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# THE SYNTHETIC ENGINE LOG.
#
# The engine's own capacity line, in the shape it prints it. Every figure is
# synthetic, and there is no timestamp in it: a date in a shipped file is a
# correlation bridge between identities.
# --------------------------------------------------------------------------
POOL = "200,000"
CTX = "196608"
CONCURRENCY = "1.05"


def engine_log(pool: str = POOL, ctx: str = CTX, mult: str = CONCURRENCY) -> str:
    """A synthetic head log carrying the engine's own capacity line."""
    return (
        "(EngineCore pid=541) INFO [core.py:133] Initializing a V1 LLM engine "
        "(v0.1.dev23341) with config: model='/model', max_seq_len=196608, "
        "enforce_eager=False\n"
        "(Worker_TP0 pid=613) INFO [gpu_worker.py:767] Available KV cache "
        "memory: 8.10 GiB\n"
        "(EngineCore pid=541) INFO [kv_cache_utils.py:1104] GPU KV cache size: "
        "%s tokens, Maximum concurrency for %s tokens per request: %sx\n"
        "(APIServer pid=1) INFO [api_utils.py:357] Application startup "
        "complete.\n"
    ) % (pool, ctx, mult)


def capture_bar_log(pw="19/19", full="17/17", ds="14/14") -> str:
    """A synthetic head log carrying the engine's per-mode capture bars.

    The bars are tqdm lines "N/M" -- the graphs captured so far, over the
    graphs the mode has to capture.
    """
    return (
        "(EngineCore pid=541) INFO [core.py:133] Capturing CUDA graphs "
        "(PIECEWISE): %s [00:00<00:00, 19.00it/s]\n"
        "(EngineCore pid=541) INFO [core.py:133] Capturing CUDA graphs (FULL): "
        "%s [00:00<00:00, 17.00it/s]\n"
        "(EngineCore pid=541) INFO [core.py:133] Capturing dspark CUDA graphs "
        "(FULL): %s [00:00<00:00, 14.00it/s]\n"
        "(EngineCore pid=541) INFO [kv_cache_utils.py:1104] GPU KV cache size: "
        "200,000 tokens, Maximum concurrency for 196608 tokens per request: "
        "1.05x\n"
    ) % (pw, full, ds)


# --------------------------------------------------------------------------
# THE CASE VOCABULARY.
#
#   facts_exact : exact lines that MUST be published
#   notes_exact : exact lines that MUST be printed
#   notes_regex : regexes that MUST match some printed line
#   forbid      : substrings that MUST NOT appear anywhere in the output
#   forbid_lines: exact lines that MUST NOT be printed at all
# --------------------------------------------------------------------------
MALFORMED_NOTE = (r"^\[FAIL\](?: extract-only:)? .*%s.*MALFORMED.*cannot be "
                  r"judged.*$")
INCOMPLETE_NOTE = (r"^\[FAIL\](?: extract-only:)? the capture bar is INCOMPLETE:"
                   r".*%s.*an interrupted capture is not a count of captured "
                   r"graphs.*$")


# --------------------------------------------------------------------------
# THE TARGETS: this kit's own two gates.
# --------------------------------------------------------------------------
TARGETS = []

# ---- 1. the kit's own gate -------------------------------------------------
TARGETS.append({
    "name": "capture-config",
    "script": os.path.join(HERE, "capture-config.sh"),
    "spec": [
        ("# --- 6. the CAPACITY FLOOR",
         "# --- 7. the capture pool's ACTUAL memory", "prefix"),
        ("# --- the machine-readable facts", 'echo "--- end GATE facts ---"',
         "exact"),
    ],
    "preamble": r"""
set -u
HOST="extract-only"
ROLE="head"
KV_FLOOR=196608
KV_FLOOR_GIB=7.51
EXPECTED_CTX=196608
CONCURRENCY_MIN=1.00
KIT_FACT=skipped
SPEC_CMD=no
SPEC_CMD_READS_FILE=no
SPEC_FILE=no
SPEC_FILE_METHOD=no
SPEC_FILE_K=no
SPEC_RESOLVED=no
SPEC_RESOLVED_NUM_SPEC_TOKENS=no
CAPTURE_MAX=0
CAPTURE_COVERAGE_MIN=128
CAPTURE_HAS_128=no
GRAPH_ACTUAL=""
GRAPH_EST=""
RESOLVED_LIST=""
PW_N=0
FULL_N=0
DS_N=0
RESOLVED_GMU=""
KV_TOKENS=""
KV_TOKENS_FACT=""
CONC_CTX=""
CONC_CTX_FACT=""
KV_GIB=""
CONC_FACT=""
fail=0
note_fail() { echo "[FAIL] $HOST: $1"; fail=1; }
note_pass() { echo "[PASS] $HOST: $1"; }
""",
    "cases": [
        # ---- the controls: the normal case must not move ------------------
        {
            "name": "control: well-formed pool grouping (200,000)",
            "log": engine_log(),
            "facts_exact": [
                "[GATE] kv_pool_tokens=200000",
                "[GATE] kv_concurrency_ctx=196608",
                "[GATE] kv_concurrency=1.05",
            ],
            "notes_exact": [
                "[PASS] extract-only: KV pool 200000 tokens (the floor is "
                "196608)",
                "[PASS] extract-only: the engine's own concurrency for a "
                "196608-token request is 1.05x (>= 1.00x)",
            ],
            "notes_regex": [],
            "forbid": ["[FAIL]"],
        },
        {
            "name": "control: the pool figure without a separator (200000)",
            "log": engine_log(pool="200000"),
            "facts_exact": [
                "[GATE] kv_pool_tokens=200000",
            ],
            "notes_exact": [
                "[PASS] extract-only: KV pool 200000 tokens (the floor is "
                "196608)",
            ],
            "notes_regex": [],
            "forbid": ["[FAIL]"],
        },
        # ---- the malformed cases: the defect class ------------------------
        {
            "name": "malformed: pool grouping (20,0,000)",
            "log": engine_log(pool="20,0,000"),
            "facts_exact": [
                "[GATE] kv_pool_tokens=20,0,000",
            ],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % "20,0,000"],
            "forbid": [
                "[GATE] kv_pool_tokens=20000",
                "[GATE] kv_pool_tokens=200000",
                "[PASS] extract-only: KV pool 20000 tokens",
                "[PASS] extract-only: KV pool 200000 tokens",
            ],
        },
        {
            "name": "malformed: denominator (196,60,8)",
            "log": engine_log(ctx="196,60,8"),
            "facts_exact": [
                "[GATE] kv_pool_tokens=200000",
                # the FIGURE is well formed and readable, so it is still
                # published: a malformed denominator refuses the JUDGMENT,
                # not the fact.
                "[GATE] kv_concurrency=1.05",
                "[GATE] kv_concurrency_ctx=196,60,8",
            ],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % "196,60,8"],
            "forbid": [
                "[GATE] kv_concurrency_ctx=196608",
                "[PASS] extract-only: the engine's own concurrency for a "
                "196608-token request",
            ],
        },
        {
            "name": "malformed: denominator (1,9)",
            "log": engine_log(ctx="1,9"),
            "facts_exact": [
                "[GATE] kv_concurrency_ctx=1,9",
            ],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % "1,9"],
            "forbid": [
                "[GATE] kv_concurrency_ctx=196608",
                "[GATE] kv_concurrency_ctx=19",
            ],
        },
        {
            "name": "malformed: denominator (196,6080)",
            "log": engine_log(ctx="196,6080"),
            "facts_exact": [
                "[GATE] kv_concurrency_ctx=196,6080",
            ],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % "196,6080"],
            "forbid": [
                "[GATE] kv_concurrency_ctx=196608",
                "[GATE] kv_concurrency_ctx=1966080",
            ],
        },
        {
            "name": "malformed: the multiplier is not a decimal (1.2.3x)",
            "log": engine_log(mult="1.2.3"),
            "facts_exact": [
                "[GATE] kv_concurrency=",
            ],
            "notes_exact": [],
            "notes_regex": [r"^\[FAIL\] extract-only: .*concurrency figure.*"
                            r"not a number.*$"],
            "forbid": ["[GATE] kv_concurrency=1.2.3"],
        },
    ],
})

# ---- 2. the kit's own base gate -------------------------------------------
TARGETS.append({
    "name": "base",
    "script": os.path.join(HERE, "capture-base-config.sh"),
    "spec": [
        ("# --- 5b. the KV pool", "# --- 5c. THE MEMORY BUDGET", "prefix"),
    ],
    "preamble": r"""
set -u
ROLE="head"
KV_FLOOR=196608
KV_FLOOR_GIB=7.51
fail=0
note_fail() { echo "[FAIL] $1"; fail=1; }
note_pass() { echo "[PASS] $1"; }
""",
    "cases": [
        {
            "name": "control: well-formed pool grouping (200,000)",
            "log": engine_log(),
            "facts_exact": [],
            "notes_exact": [
                "[PASS] KV pool 200000 tokens (floor 196608)",
                "[PASS] this rank's KV budget 8.10 GiB >= the 7.51 GiB the "
                "196608-token floor needs",
            ],
            "notes_regex": [],
            "forbid": ["[FAIL]"],
        },
        {
            "name": "malformed: pool grouping (20,0,000)",
            "log": engine_log(pool="20,0,000"),
            "facts_exact": [],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % "20,0,000"],
            "forbid": [
                "[PASS] KV pool 20000 tokens",
                "[PASS] KV pool 200000 tokens",
                "[PASS] KV pool 20,0,000 tokens",
            ],
        },
        {
            "name": "malformed: pool grouping with a trailing comma (200,000,)",
            "log": engine_log(pool="200,000,"),
            "facts_exact": [],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % "200,000,"],
            "forbid": ["[PASS] KV pool 200000 tokens"],
        },
        {
            "name": "malformed: pool grouping with a leading comma (,200,000)",
            "log": engine_log(pool=",200,000"),
            "facts_exact": [],
            "notes_exact": [],
            "notes_regex": [MALFORMED_NOTE % ",200,000"],
            "forbid": ["[PASS] KV pool 200000 tokens"],
        },
    ],
})

# ---- 3. the kit's own gate: the CAPTURE BARS -----------------------------
TARGETS.append({
    "name": "capture-config-bars",
    "script": os.path.join(HERE, "capture-config.sh"),
    "spec": [
        ("# --- 5b. THE CAPTURE-GRID FACTS", "# --- 6. the CAPACITY FLOOR",
         "prefix"),
        ("# --- the machine-readable facts", 'echo "--- end GATE facts ---"',
         "exact"),
    ],
    "preamble": r"""
set -u
HOST="extract-only"
ROLE="head"
CAPTURE_MAX=128
CAPTURE_LIST="cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]"
CAPTURE_COVERAGE_MIN=128
CAPTURE_HAS_128=yes
EXPECTED_CG_MAX=128
EXPECTED_CG_LIST="1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128"
EXPECTED_CG_GRAPHS="19,17,14"
KIT_FACT=skipped
SPEC_CMD=no
SPEC_CMD_READS_FILE=no
SPEC_FILE=no
SPEC_FILE_METHOD=no
SPEC_FILE_K=no
SPEC_RESOLVED=no
SPEC_RESOLVED_NUM_SPEC_TOKENS=no
GRAPH_ACTUAL=""
GRAPH_EST=""
RESOLVED_LIST=""
PW_N=0
FULL_N=0
DS_N=0
RESOLVED_GMU=""
KV_TOKENS=""
KV_TOKENS_FACT=""
CONC_CTX=""
CONC_CTX_FACT=""
KV_GIB=""
CONC_FACT=""
fail=0
note_fail() { echo "[FAIL] $HOST: $1"; fail=1; }
note_pass() { echo "[PASS] $HOST: $1"; }
""",
    "cases": [
        {
            "name": "control: the COMPLETE bar (19/19, 17/17, 14/14)",
            "log": capture_bar_log(),
            "facts_exact": [
                "[GATE] cg_graph_piecewise=19",
                "[GATE] cg_graph_full=17",
                "[GATE] cg_graph_dspark=14",
            ],
            "notes_exact": [
                "[PASS] extract-only: the captured graphs are the pinned ones: "
                "piecewise=19 full=17 dspark=14",
            ],
            "notes_regex": [],
            "forbid": [],
            "forbid_lines": [],
        },
        {
            "name": "malformed: the INTERRUPTED bar (19/35) is not a count",
            "log": capture_bar_log(pw="19/35"),
            "facts_exact": [
                "[GATE] cg_graph_piecewise=19/35",
                "[GATE] cg_graph_full=17",
                "[GATE] cg_graph_dspark=14",
            ],
            "notes_exact": [],
            "notes_regex": [INCOMPLETE_NOTE % "19/35"],
            "forbid": [],
            # the laundered figure and the laundered verdict
            "forbid_lines": [
                "[GATE] cg_graph_piecewise=19",
                "[PASS] extract-only: the captured graphs are the pinned ones: "
                "piecewise=19 full=17 dspark=14",
            ],
        },
        {
            "name": "malformed: an INTERRUPTED dspark bar (14/35)",
            "log": capture_bar_log(ds="14/35"),
            "facts_exact": [
                "[GATE] cg_graph_piecewise=19",
                "[GATE] cg_graph_dspark=14/35",
            ],
            "notes_exact": [],
            "notes_regex": [INCOMPLETE_NOTE % "14/35"],
            "forbid": [],
            "forbid_lines": ["[GATE] cg_graph_dspark=14"],
        },
    ],
})


# --------------------------------------------------------------------------
# THE RUN.
# --------------------------------------------------------------------------
def check_case(target, script_text, case) -> list:
    """Return a list of failure reasons (empty = the case is GREEN)."""
    rc, out, err = run_blocks(script_text, target["spec"],
                              target["preamble"], case["log"])
    problems: list = []

    lines = out.splitlines()
    if err.strip():
        problems.append("the run wrote to stderr (a bash error): %r"
                        % err.strip()[:400])
    for exact in case.get("facts_exact", []):
        if exact not in lines:
            problems.append("the fact %r is NOT published" % exact)
    for exact in case.get("notes_exact", []):
        if exact not in lines:
            problems.append("the note %r is NOT printed" % exact)
    for pattern in case.get("notes_regex", []):
        rx = re.compile(pattern)
        if not any(rx.search(line) for line in lines):
            problems.append("no printed note matches %r" % pattern)
    for bad in case.get("forbid", []):
        if bad in out:
            problems.append("the output contains the forbidden %r" % bad)
    for bad in case.get("forbid_lines", []):
        if bad in lines:
            problems.append("the output contains the forbidden LINE %r" % bad)
    return problems


def main(argv: list) -> int:
    only = None
    script_override = None
    quiet = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--quiet":
            quiet = True
        elif a == "--target":
            i += 1
            only = argv[i]
        elif a == "--script":
            i += 1
            script_override = argv[i]
        else:
            print("usage: _capture_gate_battery.py [--target NAME] "
                  "[--script PATH] [--quiet]")
            return 2
        i += 1

    targets = [t for t in TARGETS if only is None or t["name"] == only]
    if not targets:
        print("_capture_gate_battery: no target named %r" % only)
        return 2

    red = 0
    total = 0
    for target in targets:
        path = script_override or target["script"]
        if not os.path.exists(path):
            print("_capture_gate_battery: %s -- MISSING %s"
                  % (target["name"], path))
            red += 1
            total += 1
            continue
        with open(path) as fh:
            script_text = fh.read()
        try:
            md5 = subprocess.run(["md5", "-q", path], capture_output=True,
                                 text=True).stdout.strip()
        except OSError:
            md5 = ""
        print("=" * 72)
        print("_capture_gate_battery: target %s" % target["name"])
        print("_capture_gate_battery: script %s" % path)
        if md5:
            print("_capture_gate_battery: md5 %s" % md5)

        for case in target["cases"]:
            total += 1
            try:
                problems = check_case(target, script_text, case)
            except BatteryError as exc:
                problems = ["the script does not have the shape the battery "
                            "needs: %s" % exc]
            if problems:
                red += 1
                print("RED   [%s] %s" % (target["name"], case["name"]))
                for p in problems:
                    print("        - %s" % p)
                if not quiet:
                    _, out, _err = run_blocks(script_text, target["spec"],
                                              target["preamble"], case["log"])
                    for line in out.splitlines():
                        print("        | %s" % line)
            else:
                print("GREEN [%s] %s" % (target["name"], case["name"]))

    print("=" * 72)
    print("_capture_gate_battery: %d case(s) over %d target(s), %d RED"
          % (total, len(targets), red))
    print("_capture_gate_battery: %s" % ("RED" if red else "GREEN"))
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
