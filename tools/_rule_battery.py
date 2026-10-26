#!/usr/bin/env python3
"""_rule_battery.py -- the battery for the RULE this kit ships.

WHY THIS EXISTS
===============
The RULE (tools/acceptance-check.py) is this kit's main product: it is the
thing that turns an evidence set into a verdict, and it is the only thing
here that can say ACCEPT or REJECT. It shipped with no regression test
inside the artifact at all: the kit carried a battery for the bench and
the assembler, and none for the rule that judges their output.

A RULE THAT IS NOT TESTED FAILS QUIETLY. The classes this battery pins
are all the same class: the rule reaching the RIGHT verdict by a WRONG
route, or reaching it and then publishing a figure that DISAGREES with
the verdict it sits under. A reader of a verdict line cannot see either
one, so a rule that does them is a rule that cannot be trusted to have
measured what it says.

THE CLASSES
-----------
  R1  The prefill. A prefill whose MEASURED length does not reach the
      budget it DECLARES is not a long-context prefill, and a one-token
      prefill is not a long-context prefill either. The criterion says
      so by name, and the failure names the shortfall.
  R2  The published figure must not contradict the verdict it sits
      under. A deviation of 10.004% printed as "10.0%" directly under a
      "10.0%" ceiling reads as a PASS the raw figure does not earn; a
      ratio of 1.3966x printed as "1.40x" under "the expectation is 1.4x"
      reads the same way.
  R3  The comparison must be made on the RAW figure. A rule that
      compares the ROUNDED value accepts a value that must fail: 10.004
      rounds to 10.0 and passes a 10.0% ceiling, and 1.3966 rounds to
      1.40 and passes a 1.4x expectation.
  R4  The inter-token figure. Its ceiling and its regression line are
      compared on the raw figure, and it is published in the rounded
      form only while the rounded form agrees with the verdict.
  R5  A counter the rule cannot read is a MISSING measurement, not a
      zero: a counter read as 0 would pass a criterion that says "the
      guard never fired".
  R6  A dead stream fails the criterion named for it.

WHAT IT RUNS
------------
A SYNTHETIC evidence set, built here, with every figure synthetic and no
name, address or date in it. The rule is run over it as a subprocess with
-B, so no bytecode is ever written next to the kit.

THE MUTANTS are the other half: each one puts a known defect back into a
copy of the rule under test, and the battery asserts that a case which is
GREEN on the rule turns RED on the mutant. A case that cannot tell the
fixed rule from the broken one is not a test.

Usage:
    python3 -B tools/_rule_battery.py
    python3 -B tools/_rule_battery.py --rule PATH
    python3 -B tools/_rule_battery.py --rule PATH --no-mutants

Exit: 0 when every case passes and every mutant is caught, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# NEVER leave bytecode next to the kit: this folder is shipped as it is.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULE = os.path.join(HERE, "acceptance-check.py")

# THE BOXES. The recipe's own placeholders -- the rule's defaults, and the
# labels the capture gate's sections carry.
BOXES = ("spark-head.local", "spark-a.local", "spark-b.local")
# THE PINNED GUARD FLOOR PER BOX (the guard is ARMED at its own box's floor).
GUARD_FLOORS = {"spark-head.local": 4096, "spark-a.local": 3584,
                "spark-b.local": 2432}
# THE RESOLVED CAPTURE GRID, exactly the pinned one, space-separated.
GRID = "1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128"
# THE LONG-CONTEXT PREFILL FLOOR this kit declares.
PREFILL_FLOOR = 8192
# THE CAPACITY FLOOR, and the context the like-for-like ratio is taken at.
POOL_FLOOR = 196608
COMPARISON_CONTEXT = 131072


# ══ THE SYNTHETIC EVIDENCE SET ═══════════════════════════════════════════
#
# EVERY figure here is synthetic. The timestamps are built from plain
# integers at RUN TIME, because a literal date in a shipped file is a
# correlation bridge between identities -- and a date in a FIXTURE is a
# date in a shipped file.
_FIXTURE_YEAR = 2000
_FIXTURE_MONTH = 1
_FIXTURE_DAY = 1


def _stamp(offset_s: int) -> str:
    """A full timestamp, `offset_s` seconds after the fixture's own epoch."""
    return (f"{_FIXTURE_YEAR:04d}-{_FIXTURE_MONTH:02d}-{_FIXTURE_DAY:02d} "
            f"{offset_s // 3600 % 24:02d}:{offset_s // 60 % 60:02d}:"
            f"{offset_s % 60:02d}")


# THE ROWS ARE IN THE SHIPPED BENCH'S EXACT SHAPE. The bench publishes
# these keys for a stream round and for a concurrent round, and the rule
# files a row by its LABEL and its MODE -- so a fixture that publishes a
# shape the bench never emits is blind to the classes the bench can reach.
STREAM_ROW = {
    "mode": "stream", "path": "lane", "workload": "prose", "round": 1,
    "tok_s": 42.0, "ttft_s": 0.35, "per_token_ms": 23.81,
    "tokens": 800, "tokens_source": "usage", "streamed_chunks": 401,
}
CONC_ROW = {
    "mode": "concurrent", "path": "lane", "streams": "4", "tok_s": 100.0,
    "ttft_s": 0.40, "tokens": 3200, "tokens_source": "usage",
    "aggregate_tok_s": 100.0, "min_stream_tok_s": 100.0,
    "max_stream_tok_s": 100.0, "workload": "prose",
}
CONC_X4 = dict(CONC_ROW, mode="concurrent x4", streams=4)
CONC_X8 = dict(CONC_ROW, mode="concurrent x8", streams=8, tok_s=180.0,
               aggregate_tok_s=180.0)
CONC_X16 = dict(CONC_ROW, mode="concurrent x16", streams=16, tok_s=300.0,
                aggregate_tok_s=300.0)
CONC_R4 = dict(CONC_ROW, mode="concurrent x4", streams=4, path="relay",
               tok_s=90.0, aggregate_tok_s=90.0)
# THE PREFILL ROW: the MEASURED length and the DECLARED budget it reached.
PREFILL_ROW = {"kind": "prefill", "path": "lane", "prefill_tok_s": 1200.0,
               "tokens": 8192, "prompt_tokens": 8192, "target_tokens": 8192,
               "tokens_source": "usage"}
ACC_ROW = {"kind": "acceptance", "path": "lane", "window": "overall",
           "counters_present": True, "acceptance_position0": 0.62,
           "acceptance_per_pos": {"0": 0.62}}
ACC_HD = dict(ACC_ROW, window="high-draft-1way", acceptance_position0=0.66,
              acceptance_per_pos={"0": 0.66})
ACC_PR = dict(ACC_ROW, window="prose-1way", acceptance_position0=0.60,
              acceptance_per_pos={"0": 0.60})
XCHECK = {
    "usage_tokens": 1600, "chunk_total": 773, "engine_draft_rounds": 805,
    "engine_tokens_per_step": 1.9863, "observed_tokens_per_step": 1.9876,
    "deviation_pct": 0.06, "predicted_tokens": 1599.0,
    "tokens_per_step_deviation_pct": 0.06,
}


def bench_text() -> str:
    """The bench's published evidence, in the shipped labels and shapes."""
    lane = "http://127.0.0.1:8100"
    relay = "http://127.0.0.1:8001"
    lines = [
        f"lane {lane.split('://')[1]} [r1]: " + json.dumps(STREAM_ROW),
        f"lane {lane.split('://')[1]} [r2]: " + json.dumps(
            dict(STREAM_ROW, workload="prose", round=2, tok_s=40.0, ttft_s=0.38,
                 per_token_ms=25.0)),
        f"lane {lane.split('://')[1]} [r3]: " + json.dumps(
            dict(STREAM_ROW, workload="high-draft", round=3, tok_s=44.0,
                 per_token_ms=22.73)),
        f"relay {relay.split('://')[1]} [r1]: " + json.dumps(
            dict(STREAM_ROW, path="relay", round=1, tok_s=40.0,
                 per_token_ms=25.0)),
        f"lane concurrent x4 {lane.split('://')[1]} [r1]: " + json.dumps(CONC_X4),
        f"lane concurrent x8 {lane.split('://')[1]} [r1]: " + json.dumps(CONC_X8),
        f"lane concurrent x16 {lane.split('://')[1]} [r1]: " + json.dumps(CONC_X16),
        f"relay concurrent x4 {relay.split('://')[1]} [r1]: " + json.dumps(CONC_R4),
        lane + " prefill: " + json.dumps(PREFILL_ROW),
        lane + " acceptance overall: " + json.dumps(ACC_ROW),
        lane + " acceptance high-draft-1way: " + json.dumps(ACC_HD),
        lane + " acceptance prose-1way: " + json.dumps(ACC_PR),
        "token accounting cross-check (1-way lane): " + json.dumps(XCHECK),
        "engine inter-token (1-WAY window, per TOKEN): 12.50",
        "engine inter-token (1-WAY window, Prometheus sum/count delta): 25.00",
    ]
    return "\n".join(lines) + "\n"


def capture_gate_text() -> str:
    """The capture gate's aggregate, in the shipped shape."""
    out = []
    for box in BOXES:
        is_head = box == BOXES[0]
        out.append(f"=== {box} ===")
        out.append(f"[GATE] role={'head' if is_head else 'worker'}")
        out.append("[GATE] spec_cmd=yes")
        out.append("[GATE] spec_cmd_reads_file=yes")
        out.append("[GATE] spec_file=yes")
        out.append("[GATE] spec_file_method=yes")
        out.append("[GATE] spec_file_k=yes")
        out.append("[GATE] gmu=0.78")
        # THE CAPACITY FACTS ARE PUBLISHED BY THE HEAD ONLY. A worker has no
        # engine config dump, so the shipped gate publishes an EMPTY value
        # for the pool and the concurrency -- and a fixture that published a
        # pool on every box would hide the whole class of bug that is about
        # a rank whose fact is ABSENT.
        if is_head:
            out.append("[GATE] kv_concurrency=1.52")
            out.append(f"[GATE] kv_concurrency_ctx={POOL_FLOOR}")
            out.append("[GATE] kv_pool_tokens=200000")
        else:
            out.append("[GATE] kv_concurrency=")
            out.append("[GATE] kv_pool_tokens=")
        out.append("[GATE] spec_resolved=yes")
        out.append("[GATE] spec_resolved_num_spec_tokens=yes")
        out.append("[GATE] capture_max_size=128")
        out.append("[GATE] capture_list_has_128=yes")
        out.append("[GATE] cg_resolved_list=" + GRID)
        out.append("[GATE] cg_graph_piecewise=19")
        out.append("[GATE] cg_graph_full=17")
        out.append("[GATE] cg_graph_dspark=14")
        out.append("[GATE] capture_pool_estimated_gib=2.0")
        out.append("[GATE] capture_pool_actual_gib=0.5")
        out.append("[GATE] EXIT=0")
    return "\n".join(out) + "\n"


def kit_text() -> str:
    out = []
    for box in BOXES:
        out.append(f"=== {box} ===")
        out.append("kit-md5: OK -- every kit file is byte-identical to "
                   "the manifest")
    return "\n".join(out) + "\n"


def guard_text() -> str:
    out = []
    for box in BOXES:
        out.append(f"=== {box} ===")
        out.append(f"armed_threshold_mb={GUARD_FLOORS[box]}")
        out.append("guard_start_lines=1")
        out.append("guard_log_breaches=0")
        out.append("guard_window_breaches=0")
        out.append(f"{_stamp(0)} avail_mb=7600 low_mb=7500 up=1 breaches=0")
    return "\n".join(out) + "\n"


def manifest_text(rule_path: str) -> str:
    """The driver's own record of what the evidence was collected for."""
    with open(rule_path, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    return (f"rule_md5={md5}\n"
            "lane_url=http://127.0.0.1:8100\n"
            "relay_url=http://127.0.0.1:8001\n")


def base_files(rule_path: str) -> dict:
    """The whole synthetic evidence set."""
    return {
        "bench.txt": bench_text(),
        "capture-gate.txt": capture_gate_text(),
        "kit-check.txt": kit_text(),
        "guard.txt": guard_text(),
        "manifest.txt": manifest_text(rule_path),
        "probes.txt": "all 7 probes PASS\n",
        "vision-smoke.txt": "the image is Red\n",
        "c1-proof.txt": "C-1 PROOF PASS\n",
        "ring-proof.txt": "every rank issued the same collective sequence\n",
    }


# ══ EDITING THE FIXTURE ══════════════════════════════════════════════════

def _edit_json_rows(text: str, edit) -> str:
    """Rewrite the JSON rows of a bench.txt, one line at a time."""
    out = []
    for line in text.splitlines():
        if ": {" not in line:
            out.append(line)
            continue
        label, _, js = line.partition(": {")
        try:
            row = json.loads("{" + js)
        except Exception:
            out.append(line)
            continue
        if not isinstance(row, dict):
            out.append(line)
            continue
        new = edit(label.strip(), row)
        if new is None:
            out.append(line)
            continue
        out.append(label + ": " + json.dumps(new))
    return "\n".join(out) + "\n"


def set_prefill(files: dict, **fields) -> dict:
    """The prefill row, with these fields set (a None value DELETES one)."""
    out = dict(files)

    def edit(label, row):
        if row.get("kind") != "prefill":
            return None
        for key, value in fields.items():
            if value is None:
                row.pop(key, None)
            else:
                row[key] = value
        return row

    out["bench.txt"] = _edit_json_rows(out["bench.txt"], edit)
    return out


def set_inter_token(files: dict, value: str) -> dict:
    """The bench's own per-TOKEN inter-token figure."""
    out = dict(files)
    out["bench.txt"] = out["bench.txt"].replace(
        "engine inter-token (1-WAY window, per TOKEN): 12.50",
        f"engine inter-token (1-WAY window, per TOKEN): {value}")
    return out


def set_dead_streams(files: dict, value: int) -> dict:
    """The count of streams that produced no usable rate, on the 4-way row."""
    out = dict(files)

    def edit(label, row):
        if row.get("mode") != "concurrent x4" or row.get("path") != "lane":
            return None
        row["dead_streams"] = value
        return row

    out["bench.txt"] = _edit_json_rows(out["bench.txt"], edit)
    return out


def set_cross_check(files: dict, **fields) -> dict:
    """The token-accounting cross-check row, with these counters."""
    out = dict(files)
    good = "token accounting cross-check (1-way lane): " + json.dumps(XCHECK)
    bad = "token accounting cross-check (1-way lane): " + json.dumps(
        dict(XCHECK, **fields))
    out["bench.txt"] = out["bench.txt"].replace(good, bad)
    return out


def set_pool(files: dict, value: int) -> dict:
    """The head box's KV pool figure, in the capture-gate aggregate."""
    out = dict(files)
    out["capture-gate.txt"] = out["capture-gate.txt"].replace(
        "[GATE] kv_pool_tokens=200000",
        f"[GATE] kv_pool_tokens={value}")
    return out


def set_guard_counter(files: dict, value: str) -> dict:
    """A box's windowed breach count -- readable, or not."""
    out = dict(files)
    out["guard.txt"] = out["guard.txt"].replace(
        "guard_window_breaches=0", f"guard_window_breaches={value}", 1)
    return out


# ══ EDITING THE FIXTURE: THE ROUND-16 CLASSES ═════════════════════════════
#
# A CLASS LABEL THE PRODUCER CANNOT EMIT. The bench's vocabulary is closed
# (tools/bench-decode.py: WORKLOADS = {"prose": ..., "high-draft": ...}), so
# "code" is a label the producer can never publish -- and a rule that counts
# it as free prose is counting a class it does not know.


def relabel_stream_workload(files: dict, value: str, only_label: str = None) -> dict:
    """Relabel the LANE's FREE-PROSE stream rows with this class label.

    Only the rows whose class IS "prose" are relabelled: the high-draft
    row keeps its class, so the case asks about the PROSE class and the
    high-draft class separately.

    `only_label` picks ONE row, so a case can ask about a single unknown
    label in a set that still publishes a free-prose class.
    """
    out = dict(files)

    def edit(label, row):
        if row.get("mode") != "stream" or row.get("path") != "lane":
            return None
        if row.get("workload") != "prose":
            return None
        if only_label is not None and only_label not in label:
            return None
        row["workload"] = value
        return row

    out["bench.txt"] = _edit_json_rows(out["bench.txt"], edit)
    return out


# ══ THE ROUND-17 CLASSES: THE CLASS CHECK AT ONE SITE ═════════════════════
#
# THE SHIPPED BATTERY PINNED THE STREAM BRANCH ONLY. But a CONCURRENCY row
# and a NON-STREAM row carry a class too -- the producer publishes one on
# both -- and the rule judges them as DECODE MEASUREMENTS. So a class the
# producer cannot emit on one of them has to be NAMED there as well, and the
# row has to be counted by nothing: not by the 4-way floors either.

# THE LANE'S NON-STREAM ROW, IN THE LANE'S OWN SHAPE. The published bench
# returns the raw response, which carries no class at all; the lane's bench
# adds the mode, the kind and the class. The row is a decode measurement the
# rule publishes but does not put a floor on.
NON_STREAM_ROW = {
    "mode": "non-stream", "kind": "non-stream", "path": "lane",
    "workload": "prose", "total_s": 1.50, "tokens": 100,
    "tokens_source": "usage", "tok_s": 66.67,
    "reasoning_chars": 240, "content_chars": 96,
}
NON_STREAM_LABEL = "lane non-stream [r1]"


def relabel_concurrency_workload(files: dict, value: str,
                                 only_mode: str = None) -> dict:
    """Relabel the LANE's CONCURRENCY rows with this class label.

    `only_mode` picks ONE level ("concurrent x4"), so a case can ask about a
    single row in a set that still publishes the other levels.
    """
    out = dict(files)

    def edit(label, row):
        mode = str(row.get("mode") or "")
        if not mode.startswith("concurrent"):
            return None
        if row.get("path") != "lane":
            return None
        if only_mode is not None and mode != only_mode:
            return None
        row["workload"] = value
        return row

    out["bench.txt"] = _edit_json_rows(out["bench.txt"], edit)
    return out


def add_non_stream_row(files: dict, workload: str = "prose") -> dict:
    """ADD the lane's non-stream row -- a decode row that carries a class."""
    out = dict(files)
    row = dict(NON_STREAM_ROW, workload=workload)
    out["bench.txt"] = out["bench.txt"].rstrip("\n") + "\n" + (
        NON_STREAM_LABEL + ": " + json.dumps(row)) + "\n"
    return out


def drop_per_position(files: dict, only_window: str = None) -> dict:
    """Remove the per-position histogram from the acceptance rows.

    `only_window` removes it from ONE window; without it, from EVERY row --
    which is the shape that used to PASS the criterion that names the
    histogram.
    """
    out = dict(files)

    def edit(label, row):
        if row.get("kind") != "acceptance":
            return None
        if only_window is not None and row.get("window") != only_window:
            return None
        row.pop("acceptance_per_pos", None)
        return row

    out["bench.txt"] = _edit_json_rows(out["bench.txt"], edit)
    return out


def set_prefill_histogram(files: dict) -> dict:
    """Add the prefill window, with an all-null per-position histogram.

    The prefill window measures a prefill, not a draft acceptance, so it
    carries no per-position numbers -- and the rule declares that by name.
    """
    out = dict(files)
    row = {"kind": "acceptance", "path": "lane", "window": "prefill",
           "counters_present": True, "acceptance_position0": None,
           "acceptance_per_pos": {"0": None, "1": None, "2": None},
           "note": ""}
    out["bench.txt"] = out["bench.txt"].rstrip("\n") + "\n" + (
        "lane http://127.0.0.1:8100 acceptance prefill: " + json.dumps(row)) + "\n"
    return out


def duplicate_box_section(files: dict, which: str = "capture-gate.txt") -> dict:
    """Publish ONE box's section TWICE in the aggregate.

    A box published twice is an ambiguous measurement, and the rule must
    say so -- the criterion that names the duplicate must FAIL.
    """
    out = dict(files)
    text = out[which]
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"=== {BOXES[0]} ==="):
            start = i
            break
    if start is None:
        raise AssertionError(f"no section for {BOXES[0]} in {which}")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("=== "):
            end = i
            break
    block = lines[start:end]
    out[which] = "\n".join(lines[:end] + block + lines[end:]) + "\n"
    return out


def set_vision(files: dict, text: str) -> dict:
    """The vision smoke's answer -- what the model actually said."""
    out = dict(files)
    out["vision-smoke.txt"] = text + "\n"
    return out


def drop_box_section(files: dict, which: str, box: str) -> dict:
    """Publish only SOME of the required box sections -- ONE IS ABSENT.

    The box-section criterion says "no box's section is published MORE THAN
    ONCE", so an ABSENT box does not fail it -- and it must not: a missing
    measurement is not an ambiguous one.  But the DETAIL it publishes must
    not claim that the absent box is published, and THAT is what this
    fixture exists to pin.  Without it, a detail that says "3 of the 3
    required box section(s) are published" while one box is ABSENT is GREEN
    against the shipped battery -- a FALSE published string nobody notices.
    """
    out = dict(files)
    lines = out[which].splitlines()
    keep: list[str] = []
    dropping = False
    seen = False
    for line in lines:
        if line.strip() == f"=== {box} ===":
            dropping = True
            seen = True
            continue
        if dropping:
            if line.startswith("=== "):
                dropping = False
            else:
                continue
        keep.append(line)
    if not seen:
        raise AssertionError(f"no section for {box} in {which}")
    out[which] = "\n".join(keep) + "\n"
    return out


# ══ RUNNING THE RULE ═════════════════════════════════════════════════════

def run(files: dict, rule: str) -> tuple[int, str]:
    """Run the rule over an evidence set. Returns (rc, the whole output)."""
    td = tempfile.mkdtemp(prefix="rule-battery-")
    try:
        for name, body in files.items():
            with open(os.path.join(td, name), "w", encoding="utf-8") as fh:
                fh.write(body)
        proc = subprocess.run(
            [sys.executable, "-B", rule, "--evidence-dir", td],
            capture_output=True, text=True)
        return proc.returncode, proc.stdout + proc.stderr
    finally:
        shutil.rmtree(td, ignore_errors=True)


def criterion_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines()
            if ln.startswith("[PASS]") or ln.startswith("[FAIL]")]


def verdict_of(out: str, name_prefix: str):
    """The verdict of the criterion whose name starts with `name_prefix`."""
    for line in criterion_lines(out):
        name = line.split("]", 1)[1].split(" -- ", 1)[0].strip()
        if name.startswith(name_prefix):
            return "PASS" if line.startswith("[PASS]") else "FAIL"
    return None


def detail_of(out: str, name_prefix: str) -> str:
    """The detail of the criterion whose name starts with `name_prefix`."""
    for line in criterion_lines(out):
        name = line.split("]", 1)[1].split(" -- ", 1)[0].strip()
        if name.startswith(name_prefix):
            return line.split(" -- ", 1)[1] if " -- " in line else ""
    return ""


def known_class_verdict(out: str):
    """The verdict of the criterion that names the known-class requirement.

    THE LOOKUP IS BY SHAPE, NOT BY NAME. The criterion was renamed, and a
    case that asks a BEHAVIOURAL question must be able to ask it against
    either name -- so that a RED against the pre-fix rule is RED because of
    the BEHAVIOUR and not because of the rename. The NAME is pinned by a
    case of its own.
    """
    for line in criterion_lines(out):
        name = line.split("]", 1)[1].split(" -- ", 1)[0].strip()
        if (name.startswith("the workload split: every ")
                and name.endswith("carries a known class")):
            return "PASS" if line.startswith("[PASS]") else "FAIL"
    return None


def known_class_detail(out: str) -> str:
    """The detail of the criterion that names the known-class requirement."""
    for line in criterion_lines(out):
        name = line.split("]", 1)[1].split(" -- ", 1)[0].strip()
        if (name.startswith("the workload split: every ")
                and name.endswith("carries a known class")):
            return line.split(" -- ", 1)[1] if " -- " in line else ""
    return ""


def figure_text(out: str) -> str:
    """The DETAILS and the NOTES -- the places a FIGURE can be published."""
    parts = []
    for line in out.splitlines():
        if line.startswith("[NOTE]"):
            parts.append(line)
        elif line.startswith("[PASS]") or line.startswith("[FAIL]"):
            parts.append(line.split(" -- ", 1)[1] if " -- " in line else "")
    return "\n".join(parts)


# THE CRITERION NAMES THE CASES LOOK FOR.
PREFILL_CRIT = "the workload split: a long-context prefill"
CROSS_CRIT = "the token accounting cross-check agrees within"
TPS_CRIT = "the tokens-per-step accounts agree within"
CAPACITY_CRIT = "the capacity did not fall below the"
CEILING_CRIT = "engine inter-token <= 42.0 ms/token"
REGRESSION_CRIT = "engine inter-token not a regression (< 52.0 ms)"
DEAD_CRIT = "every 4-way lane stream produced a rate"
GUARD_CRIT = "the guard never fired inside this cycle's window on"
# THE CONCURRENCY FLOORS. The 4-way aggregate is judged over the LANE's
# 4-way rows -- so a row the rule cannot classify must not be judged there.
CONC4_AGG_CRIT = "4-way lane aggregate"
CONC4_CLIFF_CRIT = "the 4-way cliff measurement is present"
# THE ROUND-16 CLASSES. Every one of them is a criterion whose NAME asserts
# a property its CHECK must test.
FREE_PROSE_CRIT = "the workload split: a free-prose class is published"
# THE RENAMED CRITERION. The rule judges a prefill row and an acceptance
# window by their KIND, not by a class, so "every workload row" over-claimed
# the criterion's scope: the name now says "every decode row". The rename is
# a RENAME -- the same verdict on both sides -- and a case below pins the
# name itself.
KNOWN_CLASS_CRIT = "the workload split: every decode row carries a known class"
PER_POSITION_CRIT = ("the workload split: the per-position acceptance "
                     "histogram is published")
DUPLICATE_CRIT = "the capture gate publishes no box's section more than once"
# THE SAME SHAPED CRITERION, IN THE OTHER TWO AGGREGATES. Each one publishes
# its own detail, so each one needs its own pin.
KIT_DUPLICATE_CRIT = "the kit check publishes no box's section more than once"
GUARD_DUPLICATE_CRIT = "the guard log publishes no box's section more than once"
VISION_CRIT = "the vision path answers 'Red'"
# THE BOX THAT THE ABSENT-BOX FIXTURE DROPS. The LAST one, so the fixture
# exercises a head-present, one-worker-absent shape.
ABSENT_BOX = BOXES[2]


# ══ THE CASES ═════════════════════════════════════════════════════════════
#
# Each case returns (ok, detail). A case that raises is a failure, and the
# battery says so.

def _healthy(rc: int, out: str) -> tuple[bool, str]:
    """The synthetic evidence set is ACCEPTed, so the cases mean something."""
    if rc != 0 or "ACCEPT" not in out:
        return False, (f"the synthetic evidence set is not ACCEPTed (rc={rc}) "
                       f"-- a rule that fails early never reaches the classes "
                       f"this battery tests")
    return True, "the synthetic evidence set is ACCEPTed"


def _case_r0_healthy(ctx: dict) -> tuple[bool, str]:
    """The synthetic evidence set reaches ACCEPT."""
    return _healthy(ctx["rc"], ctx["out"])


def _case_r1_one_token_prefill(ctx: dict) -> tuple[bool, str]:
    """A ONE-TOKEN prefill fails BY NAME, and the failure names it.

    The criterion says "a long-context prefill of at least 8192 tokens is
    measured". A row whose MEASURED length is 1 is not that, and a rule
    that accepted it would be accepting a measurement it never made.
    """
    out = ctx["one_token_prefill"]
    verdict = verdict_of(out, PREFILL_CRIT)
    if verdict != "FAIL":
        return False, (f"a prefill row with prompt_tokens 1 is {verdict!r} -- "
                       f"the criterion that names a long-context prefill "
                       f"accepted a one-token prompt")
    detail = detail_of(out, PREFILL_CRIT)
    if "below the declared budget" not in detail:
        return False, (f"the failure does not name the shortfall: {detail!r}")
    if "1" not in detail:
        return False, f"the failure does not name the length it read: {detail!r}"
    return True, ("a one-token prefill FAILS by name and the failure names "
                  "the declared budget it did not reach")


def _case_r1_declared_budget_missing(ctx: dict) -> tuple[bool, str]:
    """A prefill row that declares NO budget fails BY NAME.

    The measured length alone proves nothing about HOW LONG the prefill
    was: a row that publishes a length and no budget has published a
    measurement of nothing in particular.
    """
    out = ctx["no_budget_prefill"]
    verdict = verdict_of(out, PREFILL_CRIT)
    if verdict != "FAIL":
        return False, (f"a prefill row with no declared budget is "
                       f"{verdict!r} -- a length with no budget proves no "
                       f"long-context prefill")
    detail = detail_of(out, PREFILL_CRIT)
    if "no declared budget" not in detail:
        return False, f"the failure does not name the missing budget: {detail!r}"
    return True, "a prefill row with no declared budget FAILS by name"


def _case_r1_budget_below_the_floor(ctx: dict) -> tuple[bool, str]:
    """A prefill row that declares a budget BELOW the floor fails BY NAME."""
    out = ctx["small_budget_prefill"]
    verdict = verdict_of(out, PREFILL_CRIT)
    if verdict != "FAIL":
        return False, (f"a prefill row whose declared budget is 1024 is "
                       f"{verdict!r} -- a short budget is not the long-context "
                       f"prefill the criterion names")
    detail = detail_of(out, PREFILL_CRIT)
    if "below the recipe's own long-context budget" not in detail:
        return False, f"the failure does not name the floor: {detail!r}"
    return True, "a prefill row below the floor FAILS by name"


def _case_r1_healthy_prefill_publishes_the_measured_length(ctx: dict) -> tuple[bool, str]:
    """The healthy prefill row publishes the MEASURED length, not the budget.

    The two numbers are the same in the healthy fixture, so the case pins
    WHICH one is published by way of the hostile variant below -- where
    they differ.
    """
    out = ctx["out"]
    verdict = verdict_of(out, PREFILL_CRIT)
    if verdict != "PASS":
        return False, (f"the healthy prefill row is {verdict!r} -- the "
                       f"fixture does not clear the criterion, so the hostile "
                       f"variants would prove nothing")
    detail = detail_of(out, PREFILL_CRIT)
    if f"prompt_tokens {PREFILL_FLOOR}" not in detail:
        return False, f"the detail does not publish the measured length: {detail!r}"
    return True, "the healthy prefill row publishes the measured length"


def _case_r1_unreadable_prefill_length_is_named(ctx: dict) -> tuple[bool, str]:
    """An UNREADABLE prefill length is named, never defaulted to a number.

    A length the rule cannot read is a MISSING measurement. Defaulting it
    to a number -- to 1.0, to anything -- is a measurement the rule never
    made, and it is published as one.
    """
    out = ctx["unreadable_prefill_length"]
    verdict = verdict_of(out, PREFILL_CRIT)
    if verdict != "FAIL":
        return False, (f"a prefill row whose prompt length the rule cannot "
                       f"read is {verdict!r} -- the unreadable length was "
                       f"read as a number")
    detail = detail_of(out, PREFILL_CRIT)
    if "is not a usable number" not in detail:
        return False, (f"the failure does not name the unreadable length: "
                       f"{detail!r}")
    if re.search(r"prompt length \(abc\) is below the declared budget",
                 figure_text(out)):
        return False, ("the failure reports an unreadable length as a number "
                       "that is below a budget -- a measurement it never made")
    return True, "the unreadable length is NAMED, not defaulted to a number"


def _case_r2_inter_token_published_losslessly(ctx: dict) -> tuple[bool, str]:
    """An inter-token figure just above the ceiling is published in FULL.

    42.004 printed as "42.00" reads as 42.0, which is ON the ceiling --
    so a reader concludes the rule PASSED a figure that is over the line.
    """
    out = ctx["inter_token_above_ceiling"]
    detail = detail_of(out, CEILING_CRIT)
    if "42.004 ms/token" not in detail:
        return False, (f"the inter-token figure is not published as the "
                       f"measured 42.004: {detail!r}")
    if re.search(r"42\.00 ms/token", figure_text(out)):
        return False, ("the output publishes the rounded 42.00 ms/token -- a "
                       "figure that reads as ON the ceiling")
    verdict = verdict_of(out, CEILING_CRIT)
    if verdict != "FAIL":
        return False, f"the ceiling criterion is {verdict!r}, expected FAIL"
    return True, "the figure is published as 42.004 and the ceiling FAILS"


def _case_r2_inter_token_rounded_when_it_agrees(ctx: dict) -> tuple[bool, str]:
    """A figure whose ROUNDED form agrees with the verdict is rounded.

    41.996 under a 42.0 ceiling is published as "42.00": the rounded form
    reads as 42.00, which is under the ceiling, exactly as the raw figure
    is. The verdict-aware helper publishes the rounded form only while it
    agrees with the raw figure about the verdict -- and this case pins that
    the helper is USED.
    """
    out = ctx["inter_token_below_ceiling"]
    detail = detail_of(out, CEILING_CRIT)
    if "42.00 ms/token" not in detail:
        return False, (f"the inter-token figure is not published in the "
                       f"rounded form: {detail!r}")
    if "41.996 ms/token" in figure_text(out):
        return False, ("the output publishes the lossless 41.996 ms/token -- "
                       "the verdict-aware helper is not in use at this site")
    verdict = verdict_of(out, CEILING_CRIT)
    if verdict != "PASS":
        return False, f"the ceiling criterion is {verdict!r}, expected PASS"
    return True, "the figure is published as 42.00 and the ceiling PASSES"


def _case_r2_cross_check_published_losslessly(ctx: dict) -> tuple[bool, str]:
    """A cross-check deviation just over the ceiling is published in FULL.

    10.004% printed as "10.0%" directly under a "10.0%" ceiling reads as
    a PASS the raw figure does not earn. The published figure is the rule's
    OWN recomputation, in full -- so the case checks the published string
    against the figure the rule computed, not against a rounded form of it.
    """
    out = ctx["cross_check_above_ceiling"]
    detail = detail_of(out, CROSS_CRIT)
    predicted = 1000 * 1.10004
    honest = repr(abs(predicted - 1000) / 1000 * 100.0)
    if honest not in detail:
        return False, (f"the cross-check's deviation is not published as the "
                       f"measured {honest}: {detail!r}")
    if re.search(r"10\.0% \(the server's usage", figure_text(out)):
        return False, ("the output publishes the rounded 10.0% -- a figure "
                       "that reads as ON the ceiling")
    return True, f"the deviation is published as {honest}"


def _case_r2_like_for_like_published_losslessly(ctx: dict) -> tuple[bool, str]:
    """A like-for-like ratio just under the expectation is published in FULL.

    1.3966x printed as "1.40x" under "the expectation is 1.4x" reads as a
    PASS the raw ratio does not earn.
    """
    out = ctx["capacity_below_expectation"]
    detail = detail_of(out, CAPACITY_CRIT)
    if "1.3966445922851562" not in detail:
        return False, (f"the like-for-like ratio is not published as the "
                       f"measured 1.3966445922851562: {detail!r}")
    if "1.40x" in figure_text(out):
        return False, ("the output publishes the rounded 1.40x -- a figure "
                       "that reads as ON the expectation")
    return True, "the ratio is published as 1.3966445922851562"


def _case_r3_cross_check_comparison_is_on_the_raw_figure(ctx: dict) -> tuple[bool, str]:
    """The cross-check is compared on the RAW figure, so 10.004% FAILS."""
    out = ctx["cross_check_above_ceiling"]
    verdict = verdict_of(out, CROSS_CRIT)
    if verdict != "FAIL":
        return False, (f"a cross-check deviation of 10.004% under a 10.0% "
                       f"ceiling is {verdict!r} -- the comparison was made on "
                       f"the rounded figure, which passes")
    return True, "the cross-check's 10.004% FAILS a 10.0% ceiling"


def _case_r3_like_for_like_comparison_is_on_the_raw_figure(ctx: dict) -> tuple[bool, str]:
    """The like-for-like ratio is compared on the RAW figure."""
    out = ctx["capacity_below_expectation"]
    verdict = verdict_of(out, CAPACITY_CRIT)
    if verdict != "FAIL":
        return False, (f"a like-for-like ratio of 1.3966x under a 1.4x "
                       f"expectation is {verdict!r} -- the comparison was made "
                       f"on the rounded figure, which passes")
    return True, "the like-for-like ratio of 1.3966x FAILS a 1.4x expectation"


def _case_r4_inter_token_ceiling_is_on_the_raw_figure(ctx: dict) -> tuple[bool, str]:
    """The inter-token CEILING is compared on the raw figure."""
    out = ctx["inter_token_above_ceiling"]
    verdict = verdict_of(out, CEILING_CRIT)
    if verdict != "FAIL":
        return False, (f"an inter-token figure of 42.004 under a 42.0 ceiling "
                       f"is {verdict!r} -- the ceiling was compared on the "
                       f"rounded figure, which passes")
    return True, "the inter-token ceiling FAILS on 42.004"


def _case_r4_inter_token_regression_is_on_the_raw_figure(ctx: dict) -> tuple[bool, str]:
    """The inter-token REGRESSION line is compared on the raw figure."""
    out = ctx["inter_token_above_reject"]
    verdict = verdict_of(out, REGRESSION_CRIT)
    if verdict != "FAIL":
        return False, (f"an inter-token figure of 52.004 under a 52.0 "
                       f"regression line is {verdict!r} -- the comparison was "
                       f"made on the rounded figure, which passes")
    return True, "the inter-token regression FAILS on 52.004"


def _case_r4_inter_token_regression_is_on_the_raw_figure(ctx: dict) -> tuple[bool, str]:
    """The inter-token REGRESSION line is compared on the raw figure.

    A figure of 51.996 is under the 52.0 regression line, and it has to be
    judged as such: the ROUNDED figure is 52.00, which is not under it, so
    a rule that compares the rounded value rejects a figure that is
    inside the line.
    """
    out = ctx["inter_token_below_reject"]
    verdict = verdict_of(out, REGRESSION_CRIT)
    if verdict != "PASS":
        return False, (f"an inter-token figure of 51.996 under a 52.0 "
                       f"regression line is {verdict!r} -- the comparison was "
                       f"made on the rounded figure, which rejects it")
    return True, "the inter-token regression PASSES on 51.996"


def _case_r5_unreadable_counter_is_not_zero(ctx: dict) -> tuple[bool, str]:
    """A counter the rule cannot read is a MISSING measurement, not a zero.

    A counter read as 0 would pass a criterion that says "the guard never
    fired" -- on a box whose counter the rule could not read at all.
    """
    out = ctx["unreadable_counter"]
    verdict = verdict_of(out, GUARD_CRIT)
    if verdict != "FAIL":
        return False, (f"a guard section whose breach count the rule cannot "
                       f"read is {verdict!r} -- the counter was laundered "
                       f"into a number")
    detail = detail_of(out, GUARD_CRIT)
    if "cannot read its value" not in detail:
        return False, f"the failure does not name the unreadable counter: {detail!r}"
    return True, "the unreadable counter FAILS its criterion by name"


def _case_r6_dead_stream_fails_its_criterion(ctx: dict) -> tuple[bool, str]:
    """A dead stream fails the criterion named for it."""
    out = ctx["dead_stream"]
    verdict = verdict_of(out, DEAD_CRIT)
    if verdict != "FAIL":
        return False, (f"a 4-way row carrying a dead stream is {verdict!r} -- "
                       f"the criterion named for the dead streams passed")
    return True, "the dead stream FAILS its criterion by name"


# ══ THE ROUND-16 CLASSES ═════════════════════════════════════════════════
#
# EVERY CRITERION WHOSE NAME ASSERTS A PROPERTY ITS CHECK MUST TEST.


def _case_r7_unknown_class_is_named(ctx: dict) -> tuple[bool, str]:
    """A CLASS THE PRODUCER CANNOT EMIT IS NAMED, AND IT IS NOT PROSE.

    The bench's vocabulary is closed -- "prose" and "high-draft" are the
    only two classes it publishes -- so a row labelled "code" is a class
    this rule does not know. The old dispatch counted it as free prose, so
    an evidence set with NO free-prose class at all PASSED the criterion
    that names one. Now the row is named, the free-prose criterion says
    there is no free-prose class, and the unknown class has a criterion of
    its own.
    """
    out = ctx["all_prose_relabelled"]
    verdict = verdict_of(out, FREE_PROSE_CRIT)
    if verdict != "FAIL":
        return False, (f"an evidence set whose only stream rows carry the "
                       f"unknown class is {verdict!r} for the free-prose "
                       f"criterion -- a class the rule cannot classify was "
                       f"counted as the free-prose class")
    known = verdict_of(out, KNOWN_CLASS_CRIT)
    if known != "FAIL":
        return False, (f"an evidence set carrying an unknown class is "
                       f"{known!r} for the known-class criterion -- the "
                       f"unknown class was not named")
    detail = detail_of(out, KNOWN_CLASS_CRIT)
    if "code" not in detail:
        return False, (f"the failure does not name the class it does not "
                       f"know: {detail!r}")
    return True, ("the unknown class FAILS the free-prose criterion by name "
                  "and the known-class criterion names the label")


def _case_r7_one_unknown_class_keeps_the_prose_class(ctx: dict) -> tuple[bool, str]:
    """ONE unknown label does not erase the free-prose class.

    The other half of the class above: the free-prose criterion is about
    the PROSE rows, so a set that still publishes one prose row still
    PASSES it -- while the unknown class is still named. A rule that
    failed the free-prose criterion for any unknown label would be a rule
    that rejects an evidence set for a reason its own name does not state.
    """
    out = ctx["one_prose_relabelled"]
    verdict = verdict_of(out, FREE_PROSE_CRIT)
    if verdict != "PASS":
        return False, (f"an evidence set that still publishes ONE free-prose "
                       f"row is {verdict!r} for the free-prose criterion")
    detail = detail_of(out, FREE_PROSE_CRIT)
    if "1 prose stream row" not in detail:
        return False, (f"the free-prose criterion does not count the prose "
                       f"rows it names: {detail!r}")
    known = verdict_of(out, KNOWN_CLASS_CRIT)
    if known != "FAIL":
        return False, (f"an evidence set carrying one unknown label is "
                       f"{known!r} for the known-class criterion")
    detail = detail_of(out, KNOWN_CLASS_CRIT)
    if "code" not in detail:
        return False, (f"the failure does not name the class it does not "
                       f"know: {detail!r}")
    return True, ("the free-prose class still PASSES with one prose row, and "
                  "the one unknown class is named")


def _case_r8_the_histogram_is_judged(ctx: dict) -> tuple[bool, str]:
    """THE HISTOGRAM IS MISSING FROM EVERY WINDOW -- THE CRITERION FAILS.

    The old check asked only "is there an acceptance ROW?", so the
    per-position histogram could be deleted from every row and the
    criterion still PASSed while the rule printed "(per-position: None)".
    """
    out = ctx["no_histogram_anywhere"]
    verdict = verdict_of(out, PER_POSITION_CRIT)
    if verdict != "FAIL":
        return False, (f"an evidence set with NO per-position histogram is "
                       f"{verdict!r} for the criterion that names the "
                       f"histogram")
    detail = detail_of(out, PER_POSITION_CRIT)
    if "readable per-position histogram" not in detail:
        return False, (f"the failure does not name the histogram it is "
                       f"about: {detail!r}")
    return True, ("no per-position histogram FAILS the criterion that names "
                  "the histogram")


def _case_r8_one_core_window_loses_its_histogram(ctx: dict) -> tuple[bool, str]:
    """ONE CORE WINDOW LOSES ITS HISTOGRAM -- THE CRITERION NAMES IT.

    The core windows are NOT excludable: the declaration list cannot be
    used to excuse a window that lost its histogram. This case is the one
    that pins that -- the failure has to NAME the window, or a rule that
    failed for an unnamed reason would pass it.
    """
    out = ctx["core_window_loses_histogram"]
    verdict = verdict_of(out, PER_POSITION_CRIT)
    if verdict != "FAIL":
        return False, (f"an evidence set whose 'prose-1way' window lost its "
                       f"histogram is {verdict!r} for the criterion that "
                       f"names the histogram")
    detail = detail_of(out, PER_POSITION_CRIT)
    if "prose-1way" not in detail:
        return False, (f"the failure does not name the window that lost its "
                       f"histogram: {detail!r}")
    return True, ("a core window that lost its histogram FAILS the criterion "
                  "and the failure names the window")


def _case_r8_the_declared_window_is_published(ctx: dict) -> tuple[bool, str]:
    """THE PREFILL WINDOW PUBLISHES NO HISTOGRAM, AND THAT IS DECLARED.

    The prefill window measures a prefill, not a draft acceptance, so it
    carries no per-position numbers. The exclusion is a SHORT PUBLISHED
    LIST, and the criterion publishes it -- so a reader can see what the
    rule excuses, and not only that it excused something.
    """
    out = ctx["prefill_window_no_histogram"]
    verdict = verdict_of(out, PER_POSITION_CRIT)
    if verdict != "PASS":
        return False, (f"the prefill window's all-null histogram is "
                       f"{verdict!r} -- it is excluded by declaration, and "
                       f"the rule says so by name")
    detail = detail_of(out, PER_POSITION_CRIT)
    if "prefill" not in detail:
        return False, (f"the criterion does not publish the declaration "
                       f"list: {detail!r}")
    return True, ("the declared window is excused and the declaration list "
                  "is published")


def _case_r9_a_duplicated_box_section_still_fails(ctx: dict) -> tuple[bool, str]:
    """A BOX PUBLISHED TWICE STILL FAILS THE RENAMED CRITERION.

    The criterion was renamed from "publishes each box's section exactly
    once" to "publishes no box's section more than once" because the check
    says "at most once", not "exactly once". A RENAME MUST NOT GUT THE
    CHECK: an ambiguous measurement is still a failure, and the renamed
    criterion has to say so.
    """
    out = ctx["duplicate_box_section"]
    verdict = verdict_of(out, DUPLICATE_CRIT)
    if verdict != "FAIL":
        return False, (f"a box section published twice is {verdict!r} for "
                       f"the criterion that names the duplicate")
    detail = detail_of(out, DUPLICATE_CRIT)
    if "more than once" not in detail:
        return False, (f"the failure does not name the duplicate: {detail!r}")
    return True, ("a duplicated box section FAILS the renamed criterion by "
                  "name")


def _case_r10_the_vision_answer_must_not_deny_red(ctx: dict) -> tuple[bool, str]:
    """AN ANSWER THAT DENIES THE COLOUR FAILS BY NAME.

    "the image is not Red at all, it is blue" NAMES the colour and is
    still a broken vision path. The criterion named the colour, so the
    check has to test it.
    """
    out = ctx["vision_denies_red"]
    verdict = verdict_of(out, VISION_CRIT)
    if verdict != "FAIL":
        return False, (f"an answer that denies the colour is {verdict!r} for "
                       f"the criterion that names the colour")
    detail = detail_of(out, VISION_CRIT)
    if "DENIES" not in detail:
        return False, (f"the failure does not name the denial: {detail!r}")
    return True, ("an answer that denies the colour FAILS the criterion by "
                  "name")


def _case_r10_a_mention_of_another_colour_still_passes(ctx: dict) -> tuple[bool, str]:
    """AN ANSWER THAT NAMES ANOTHER COLOUR IS STILL A RED ANSWER.

    "Red, not blue" IS a red answer. The guard is narrow on purpose: a
    wide guard would reject a sound answer, and a battery that cannot PASS
    a valid case is not a proof either.
    """
    out = ctx["vision_names_another_colour"]
    verdict = verdict_of(out, VISION_CRIT)
    if verdict != "PASS":
        return False, (f"the answer 'Red, not blue' is {verdict!r} -- the "
                       f"guard rejects a sound answer")
    return True, "'Red, not blue' PASSES -- the guard is narrow"


def _case_r10_a_red_answer_still_passes(ctx: dict) -> tuple[bool, str]:
    """A PLAIN RED ANSWER STILL PASSES.

    The guard must not touch the healthy case: the synthetic evidence set
    answers "the image is Red" and the criterion must still PASS on it.
    """
    out = ctx["out"]
    verdict = verdict_of(out, VISION_CRIT)
    if verdict != "PASS":
        return False, (f"the healthy answer is {verdict!r} for the criterion "
                       f"that names the colour")
    return True, "a plain red answer PASSES"


# ══ THE ROUND-17 CLASSES: THE CLASS CHECK AT ONE SITE ═════════════════════
#
# EVERY CASE HERE ASKS THE SAME QUESTION OF A DIFFERENT ROW KIND: a row the
# rule judges as a DECODE MEASUREMENT, carrying a class the producer cannot
# emit, must be NAMED -- and a row the rule judges by its KIND must not be.


def _case_r11_a_concurrency_row_with_an_unknown_class_is_named(ctx: dict) -> tuple[bool, str]:
    """A CONCURRENCY ROW WITH A CLASS THE PRODUCER CANNOT EMIT IS NAMED.

    The producer publishes a class on its concurrency rows, and the rule
    judges them as decode measurements -- so the class check covers them. A
    concurrency row labelled "code" must be NAMED, and it must be counted
    by NOTHING: not by the 4-way floors either.
    """
    out = ctx["concurrency_class_relabelled"]
    verdict = known_class_verdict(out)
    if verdict != "FAIL":
        return False, (f"a concurrency row carrying the unknown class is "
                       f"{verdict!r} for the known-class criterion -- the class "
                       f"was not named")
    detail = known_class_detail(out)
    # THE LABEL IS CHECKED AS IT IS PUBLISHED -- quoted, by repr(). A bare
    # "code" would match "decode", which is in the PASS branch's own wording.
    if "'code'" not in detail:
        return False, (f"the failure does not name the class it does not "
                       f"know: {detail!r}")
    if "concurrent x4" not in detail:
        return False, (f"the failure does not name the row it is about: {detail!r}")
    # THE ROW IS COUNTED BY NOTHING: the 4-way floors must not judge a
    # measurement the rule cannot attribute.
    if verdict_of(out, CONC4_AGG_CRIT) == "PASS":
        return False, ("the 4-way floors still judge the concurrency row whose "
                       "class the rule cannot name -- a measurement the rule "
                       "cannot attribute is not evidence for anything")
    return True, ("the concurrency row's unknown class FAILS the known-class "
                  "criterion by name, the failure names the row and its label, "
                  "and the 4-way floors no longer count it")


def _case_r11_a_non_stream_row_with_an_unknown_class_is_named(ctx: dict) -> tuple[bool, str]:
    """A NON-STREAM ROW WITH A CLASS THE PRODUCER CANNOT EMIT IS NAMED.

    The same hole, in the other branch: the non-stream row is a decode
    measurement the rule publishes, and the lane's producer publishes a
    class on it -- so the class check covers it too.
    """
    out = ctx["non_stream_class_relabelled"]
    verdict = known_class_verdict(out)
    if verdict != "FAIL":
        return False, (f"a non-stream row carrying the unknown class is "
                       f"{verdict!r} for the known-class criterion -- the class "
                       f"was not named")
    detail = known_class_detail(out)
    if "'code'" not in detail:
        return False, (f"the failure does not name the class it does not "
                       f"know: {detail!r}")
    if "non-stream" not in detail:
        return False, (f"the failure does not name the row it is about: {detail!r}")
    return True, ("the non-stream row's unknown class FAILS the known-class "
                  "criterion by name, and the failure names the row")


def _case_r12_a_prefill_row_with_a_class_label_is_not_named(ctx: dict) -> tuple[bool, str]:
    """A PREFILL ROW WITH A CLASS LABEL ON IT IS NOT NAMED.

    The prefill is judged by its KIND -- a measured length -- not by a class,
    so a label on it is not a decode class. THIS IS THE CASE THAT KEEPS THE
    RULE FROM REFUSING THE LANE'S OWN GOOD EVIDENCE: the lane's producer
    annotates its prefill row with its own label, and a rule that named it
    would reject a sound evidence set over an annotation on a row it does
    not judge by a class.
    """
    for key, label in (("prefill_class_labelled", "code"),
                       ("prefill_lane_class_labelled", "long-context")):
        out = ctx[key]
        verdict = known_class_verdict(out)
        if verdict != "PASS":
            return False, (f"a prefill row labelled {label!r} is {verdict!r} "
                           f"for the known-class criterion -- the prefill row "
                           f"is judged by its kind, not by a class")
        detail = known_class_detail(out)
        # THE ROW ITSELF MUST NOT BE NAMED. A bare "code" would match
        # "decode", which is in the PASS branch's own wording, so the ROW is
        # what is looked for: a named row is published by repr() of its
        # label, and the label says "prefill".
        if "prefill" in detail:
            return False, (f"the known-class criterion names the prefill "
                           f"row (labelled {label!r}): {detail!r}")
    return True, ("a prefill row labelled 'code' is NOT named, and the lane's "
                  "own 'long-context' annotation is not named either")


def _case_r12_the_known_class_criterion_is_named_for_the_decode_rows(ctx: dict) -> tuple[bool, str]:
    """THE KNOWN-CLASS CRITERION IS NAMED FOR THE DECODE ROWS.

    The old name over-claimed: the rule judges a prefill row and an
    acceptance window by their kind, not by a class, so "every workload row"
    was not what the check tested. The rename is a RENAME -- the same verdict
    on both sides -- and this case pins the NAME.
    """
    out = ctx["out"]
    if verdict_of(out, KNOWN_CLASS_CRIT) is None:
        return False, (f"the rule publishes no criterion named "
                       f"{KNOWN_CLASS_CRIT!r} -- the criterion still carries "
                       f"the old name")
    verdict = known_class_verdict(out)
    if verdict != "PASS":
        return False, (f"the healthy evidence set is {verdict!r} for the "
                       f"known-class criterion")
    return True, ("the known-class criterion is named for the decode rows, and "
                  "the healthy evidence set still PASSes it")


def _case_r13_a_concurrency_row_with_the_prose_class_is_still_counted(ctx: dict) -> tuple[bool, str]:
    """A CONCURRENCY ROW WHOSE CLASS IS "prose" IS STILL COUNTED.

    The check must NOT over-reject: a row the rule CAN classify is counted
    exactly as before, and the 4-way floors still judge it.
    """
    out = ctx["out"]
    verdict = known_class_verdict(out)
    if verdict != "PASS":
        return False, (f"the healthy evidence set is {verdict!r} for the "
                       f"known-class criterion -- the check over-rejects")
    agg = verdict_of(out, CONC4_AGG_CRIT)
    if agg != "PASS":
        return False, (f"the 4-way aggregate criterion is {agg!r} -- the "
                       f"prose-classed concurrency row is no longer counted")
    detail = detail_of(out, CONC4_AGG_CRIT)
    if "100.00" not in detail:
        return False, (f"the 4-way criterion does not publish the prose "
                       f"row's aggregate: {detail!r}")
    return True, ("the prose-classed concurrency row is still counted by the "
                  "4-way floors")


# ══ THE ROUND-18 CLASSES ══════════════════════════════════════════════════
#
# TWO CLASSES, AND BOTH OF THEM ARE ABOUT A PUBLISHED STRING.
#
# r14  THE BOX-SECTION DETAIL. The criterion says "no box's section is
#      published MORE THAN ONCE", so an ABSENT box does not fail it -- and
#      the DETAIL it publishes beside that PASS must not claim the absent
#      box is published.  A detail that says "3 of the 3 required box
#      section(s) are published" while one box is ABSENT is a FALSE string,
#      and the shipped battery did not notice it.
#
# r15  THE CLAUSE. A negation NEARBY the colour word is not a denial of
#      THAT colour: "Not blue. Red." denies BLUE and NAMES red.  The guard
#      must require the negation to be in the SAME CLAUSE, and the detail
#      must NAME the negation it matched so a reader can see the denial
#      instead of taking the rule's word for it.


def _absent_box_detail_is_truthful(out: str, crit: str, absent_box: str
                                    ) -> tuple[bool, str]:
    """THE DETAIL MUST NOT CLAIM A BOX THAT IS ABSENT.

    The criterion PASSes -- "no box's section is published more than once"
    is TRUE when a box is absent, because a MISSING measurement is not an
    AMBIGUOUS one -- so the DETAIL is the only thing that can be wrong
    here, and a wrong detail is a FALSE published string on a PASS line.
    """
    verdict = verdict_of(out, crit)
    if verdict != "PASS":
        return False, (f"the criterion that names the duplicate is {verdict!r} -- "
                       f"an ABSENT box is not a duplicate")
    detail = detail_of(out, crit)
    all_published = (f"{len(BOXES)} of the {len(BOXES)} required box "
                     f"section(s) are published")
    if all_published in detail:
        return False, (f"the detail claims EVERY box section is published while "
                       f"{absent_box} is absent: {detail!r}")
    published = (f"{len(BOXES) - 1} of the {len(BOXES)} required box "
                 f"section(s) are published")
    if published not in detail:
        return False, (f"the detail does not say how many box sections are "
                       f"published: {detail!r}")
    if "ABSENT" not in detail or absent_box not in detail:
        return False, (f"the detail does not name the absent box: {detail!r}")
    return True, ("the detail names the absent box and does not claim it is "
                  "published")


def _case_r14_the_capture_gate_detail_names_the_absent_box(ctx: dict) -> tuple[bool, str]:
    """THE CAPTURE GATE'S DETAIL MUST NOT CLAIM AN ABSENT BOX.

    One of the three required box sections is not published at all.  The
    criterion still PASSes -- nothing is published twice -- but the detail
    beside it must say which box is missing, and must not say all three are
    published.
    """
    out = ctx["capture_gate_one_box_absent"]
    ok, detail = _absent_box_detail_is_truthful(
        out, DUPLICATE_CRIT, ABSENT_BOX)
    return ok, detail


def _case_r14_the_kit_check_detail_names_the_absent_box(ctx: dict) -> tuple[bool, str]:
    """THE KIT CHECK'S DETAIL MUST NOT CLAIM AN ABSENT BOX.

    The kit check has its own box-section criterion with its own detail, so
    it needs its own pin -- the same false string in a second place would
    be just as false.
    """
    out = ctx["kit_check_one_box_absent"]
    ok, detail = _absent_box_detail_is_truthful(
        out, KIT_DUPLICATE_CRIT, ABSENT_BOX)
    return ok, detail


def _case_r14_the_guard_log_detail_names_the_absent_box(ctx: dict) -> tuple[bool, str]:
    """THE GUARD LOG'S DETAIL MUST NOT CLAIM AN ABSENT BOX.

    The guard log has its own box-section criterion with its own detail, so
    it needs its own pin -- the same false string in a third place would
    be just as false.
    """
    out = ctx["guard_one_box_absent"]
    ok, detail = _absent_box_detail_is_truthful(
        out, GUARD_DUPLICATE_CRIT, ABSENT_BOX)
    return ok, detail


def _case_r15_a_negation_in_another_sentence_does_not_deny_the_colour(ctx: dict) -> tuple[bool, str]:
    """"NOT BLUE. RED." IS A RED ANSWER.

    The negation denies BLUE; the answer NAMES red.  The old guard fired on
    the negation of the OTHER colour and published a denial it had not
    established -- and the identical assertion in the other order ("Red,
    not blue") PASSED, so the verdict depended on the ORDER of two
    sentences.  A sentence terminator between the negation and the colour
    word means the answer moved on to a NEW sentence.
    """
    out = ctx["vision_negation_in_another_sentence"]
    verdict = verdict_of(out, VISION_CRIT)
    if verdict != "PASS":
        return False, (f"the answer 'Not blue. Red.' is {verdict!r} -- the "
                       f"negation denies blue, not red")
    return True, ("a negation in another sentence does not deny the colour")


def _case_r15_a_negation_outside_the_window_does_not_deny_the_colour(ctx: dict) -> tuple[bool, str]:
    """"IT IS NOT BLUE, IT IS RED" IS A RED ANSWER.

    The negation is four tokens back -- outside the three-token window -- so
    it is about something else.  This is the case that catches a guard
    widened to the WHOLE answer.
    """
    out = ctx["vision_negation_outside_the_window"]
    verdict = verdict_of(out, VISION_CRIT)
    if verdict != "PASS":
        return False, (f"the answer 'It is not blue, it is Red' is {verdict!r} "
                       f"-- the negation is outside the window")
    return True, ("a negation outside the window does not deny the colour")


def _case_r15_the_failing_detail_names_the_negation(ctx: dict) -> tuple[bool, str]:
    """"THE IMAGE IS NOT REALLY RED AT ALL" FAILS, AND THE DETAIL NAMES THE
    NEGATION IT MATCHED.

    The negation is two tokens back -- inside the window -- and in the SAME
    clause, so it denies the colour.  The detail must not merely SAY that
    the answer denies the colour: it must NAME the negation it matched, so
    a reader can see the denial instead of taking the rule's word for it.
    """
    out = ctx["vision_negation_two_tokens_back"]
    verdict = verdict_of(out, VISION_CRIT)
    if verdict != "FAIL":
        return False, (f"an answer that denies the colour is {verdict!r} -- "
                       f"the negation and the colour are in the same clause")
    detail = detail_of(out, VISION_CRIT)
    if "DENIES" not in detail:
        return False, (f"the failure does not name the denial: {detail!r}")
    if "'not'" not in detail:
        return False, (f"the failure does not NAME the negation it matched: "
                       f"{detail!r}")
    return True, ("the denial FAILS by name and the detail names the negation "
                  "it matched")


CASES = [
    ("r0_the_synthetic_evidence_set_is_accepted", _case_r0_healthy),
    ("r1_a_one_token_prefill_fails_by_name", _case_r1_one_token_prefill),
    ("r1_a_prefill_with_no_declared_budget_fails_by_name",
     _case_r1_declared_budget_missing),
    ("r1_a_prefill_below_the_floor_fails_by_name",
     _case_r1_budget_below_the_floor),
    ("r1_an_unreadable_prefill_length_is_named",
     _case_r1_unreadable_prefill_length_is_named),
    ("r1_the_healthy_prefill_publishes_the_measured_length",
     _case_r1_healthy_prefill_publishes_the_measured_length),
    ("r2_the_inter_token_figure_is_published_in_full",
     _case_r2_inter_token_published_losslessly),
    ("r2_the_inter_token_figure_is_rounded_when_it_agrees",
     _case_r2_inter_token_rounded_when_it_agrees),
    ("r2_the_cross_check_deviation_is_published_in_full",
     _case_r2_cross_check_published_losslessly),
    ("r2_the_like_for_like_ratio_is_published_in_full",
     _case_r2_like_for_like_published_losslessly),
    ("r3_the_cross_check_is_compared_on_the_raw_figure",
     _case_r3_cross_check_comparison_is_on_the_raw_figure),
    ("r3_the_like_for_like_ratio_is_compared_on_the_raw_figure",
     _case_r3_like_for_like_comparison_is_on_the_raw_figure),
    ("r4_the_inter_token_ceiling_is_compared_on_the_raw_figure",
     _case_r4_inter_token_ceiling_is_on_the_raw_figure),
    ("r4_the_inter_token_regression_is_compared_on_the_raw_figure",
     _case_r4_inter_token_regression_is_on_the_raw_figure),
    ("r4_the_inter_token_regression_line_is_compared_on_the_raw_figure",
     _case_r4_inter_token_regression_is_on_the_raw_figure),
    ("r5_an_unreadable_counter_is_not_read_as_zero",
     _case_r5_unreadable_counter_is_not_zero),
    ("r6_a_dead_stream_fails_its_criterion", _case_r6_dead_stream_fails_its_criterion),
    ("r7_an_unknown_workload_class_is_named_and_is_not_prose",
     _case_r7_unknown_class_is_named),
    ("r7_one_unknown_label_keeps_the_free_prose_class",
     _case_r7_one_unknown_class_keeps_the_prose_class),
    ("r8_the_per_position_histogram_is_judged",
     _case_r8_the_histogram_is_judged),
    ("r8_a_core_window_that_loses_its_histogram_is_named",
     _case_r8_one_core_window_loses_its_histogram),
    ("r8_the_declared_window_is_published",
     _case_r8_the_declared_window_is_published),
    ("r9_a_duplicated_box_section_still_fails",
     _case_r9_a_duplicated_box_section_still_fails),
    ("r10_an_answer_that_denies_the_colour_fails_by_name",
     _case_r10_the_vision_answer_must_not_deny_red),
    ("r10_a_mention_of_another_colour_still_passes",
     _case_r10_a_mention_of_another_colour_still_passes),
    ("r10_a_plain_red_answer_still_passes",
     _case_r10_a_red_answer_still_passes),
    ("r11_a_concurrency_row_with_an_unknown_class_is_named",
     _case_r11_a_concurrency_row_with_an_unknown_class_is_named),
    ("r11_a_non_stream_row_with_an_unknown_class_is_named",
     _case_r11_a_non_stream_row_with_an_unknown_class_is_named),
    ("r12_a_prefill_row_with_a_class_label_is_not_named",
     _case_r12_a_prefill_row_with_a_class_label_is_not_named),
    ("r12_the_known_class_criterion_is_named_for_the_decode_rows",
     _case_r12_the_known_class_criterion_is_named_for_the_decode_rows),
    ("r13_a_concurrency_row_with_the_prose_class_is_still_counted",
     _case_r13_a_concurrency_row_with_the_prose_class_is_still_counted),
    ("r14_the_capture_gate_detail_names_the_absent_box",
     _case_r14_the_capture_gate_detail_names_the_absent_box),
    ("r14_the_kit_check_detail_names_the_absent_box",
     _case_r14_the_kit_check_detail_names_the_absent_box),
    ("r14_the_guard_log_detail_names_the_absent_box",
     _case_r14_the_guard_log_detail_names_the_absent_box),
    ("r15_a_negation_in_another_sentence_does_not_deny_the_colour",
     _case_r15_a_negation_in_another_sentence_does_not_deny_the_colour),
    ("r15_a_negation_outside_the_window_does_not_deny_the_colour",
     _case_r15_a_negation_outside_the_window_does_not_deny_the_colour),
    ("r15_the_failing_detail_names_the_negation",
     _case_r15_the_failing_detail_names_the_negation),
]


# ══ THE MUTANTS ═══════════════════════════════════════════════════════════
#
# Each mutant puts a KNOWN DEFECT back into a copy of the rule under test.
# A mutant that no case catches is a case that cannot tell the fixed rule
# from the broken one -- and a test that cannot tell the two apart is not
# a test.
MUTANTS = [
    ("J1 the prefill length check is removed",
     'prefill_len = _measured_float(prefill_rows[0].get("prompt_tokens"))',
     'prefill_len = _measured_float(prefill_rows[0].get("prompt_tokens")) or 1.0',
     "r1_an_unreadable_prefill_length_is_named"),
    ("J1b the consistency check between the measured length and the declared "
     "budget is removed",
     "            elif prefill_len < prefill_budget:\n",
     "            elif False:\n",
     "r1_a_one_token_prefill_fails_by_name"),
    ("J2 the inter-token detail drops the verdict-aware helper",
     'f"{_fig(inter_ms, 2, _inter_ok)} ms/token (the bench\'s 1-WAY window '
     'delta, per token)"',
     'f"{_fig(inter_ms, 2)} ms/token (the bench\'s 1-WAY window delta, per '
     'token)"',
     "r2_the_inter_token_figure_is_rounded_when_it_agrees"),
    ("J3 an unreadable counter is laundered into a zero",
     "    try:\n        return int(text)\n    except (TypeError, ValueError, "
     "OverflowError):\n        return None",
     "    try:\n        return int(text)\n    except (TypeError, ValueError, "
     "OverflowError):\n        return 0",
     "r5_an_unreadable_counter_is_not_read_as_zero"),
    ("J4 the dead-stream criterion always passes",
     "            if dead_named:\n                dead_ok = False\n",
     "            if dead_named:\n                dead_ok = True\n",
     "r6_a_dead_stream_fails_its_criterion"),
    ("J5 the like-for-like comparison is made on the ROUNDED value",
     "like_for_like is not None and like_for_like >= CONCURRENCY_NO_LOSS_X,",
     "like_for_like is not None and round(like_for_like, 2) >= "
     "CONCURRENCY_NO_LOSS_X,",
     "r3_the_like_for_like_ratio_is_compared_on_the_raw_figure"),
    ("J6 the cross-check comparison is made on the ROUNDED value",
     "cross_dev is not None and abs(cross_dev) <= TOKEN_CROSSCHECK_MAX_PCT,",
     "cross_dev is not None and abs(round(cross_dev, 2)) <= "
     "TOKEN_CROSSCHECK_MAX_PCT,",
     "r3_the_cross_check_is_compared_on_the_raw_figure"),
    ("J7 the lossless fallback is removed from the figure helper",
     "    s = f\"{value:.{places}f}\"\n    if float(s) == value:\n        return s\n"
     "    if ok is None:\n        return repr(value)\n    preds = ok if "
     "isinstance(ok, (list, tuple)) else (ok,)\n    for p in preds:\n"
     "        if p(float(s)) != p(value):\n            return repr(value)\n"
     "    return s",
     "    s = f\"{value:.{places}f}\"\n    return s",
     "r2_the_inter_token_figure_is_published_in_full"),
    ("J8 the inter-token ceiling is compared on the ROUNDED value",
     "                  inter_ms <= INTER_TOKEN_MAX_MS,",
     "                  round(inter_ms, 2) <= INTER_TOKEN_MAX_MS,",
     "r4_the_inter_token_ceiling_is_compared_on_the_raw_figure"),
    ("J9 the inter-token regression is compared on the ROUNDED value",
     "inter_ms < INTER_TOKEN_REJECT_MS, f\"{_fig(inter_ms, 2, _inter_ok)} "
     "ms/token\")",
     "round(inter_ms, 2) < INTER_TOKEN_REJECT_MS, f\"{_fig(inter_ms, 2, "
     "_inter_ok)} ms/token\")",
     "r4_the_inter_token_regression_line_is_compared_on_the_raw_figure"),
    ("F22a the cross-check's deviation goes back to the rounded form",
     "f\"{_fig_or_none(cross_dev, 2, lambda v: abs(v) <= "
     "TOKEN_CROSSCHECK_MAX_PCT)}% \"",
     "f\"{round(cross_dev, 2) if cross_dev is not None else None}% \"",
     "r2_the_cross_check_deviation_is_published_in_full"),
    ("F22b the like-for-like ratio goes back to the rounded form",
     "f\"{_fig_or_none(like_for_like, 2, lambda v: v >= "
     "CONCURRENCY_NO_LOSS_X)}x a \"",
     "f\"{like_for_like:.2f}x a \"",
     "r2_the_like_for_like_ratio_is_published_in_full"),
    # ── THE ROUND-16 MUTANTS ─────────────────────────────────────────────
    # Each one puts a KNOWN DEFECT back into the fix it pins, so the case
    # that pins it goes RED on the mutant.
    ("F24a a class the rule does not know is counted as prose again",
     '    return w if w in KNOWN_WORKLOADS else ""',
     '    return w if w in KNOWN_WORKLOADS else "prose"',
     "r7_an_unknown_workload_class_is_named_and_is_not_prose"),
    ("F24b a row whose class the rule does not know is dropped, not named",
     "            unknown_class.append(\n"
     "                (label, raw_label if isinstance(raw_label, str) else "
     "repr(raw_label)))",
     "            pass",
     "r7_one_unknown_label_keeps_the_free_prose_class"),
    ("F25a the histogram is deemed readable whenever it is present at all",
     "    if not isinstance(hist, dict) or not hist:\n        return False\n",
     "    if not isinstance(hist, dict) or not hist:\n        return True\n",
     "r8_the_per_position_histogram_is_judged"),
    ("F25b the histogram is deemed unreadable even when it is published",
     "        if isinstance(v, (int, float)) and _is_finite(v):\n"
     "            return True\n    return False",
     "        if isinstance(v, (int, float)) and _is_finite(v):\n"
     "            return True\n    return True",
     "r8_the_declared_window_is_published"),
    ("F25c the core window moves into the declaration list (the loophole)",
     'CORE_PER_POSITION_WINDOWS = ("overall", "prose-1way", "high-draft-1way")\n'
     "\n"
     "# THE WINDOWS THAT PUBLISH NO PER-POSITION HISTOGRAM, BY NAME. The "
     "prefill\n"
     "# window measures a prefill, not a draft acceptance: it carries no\n"
     "# per-position acceptance numbers. The exclusion is a SHORT PUBLISHED "
     "LIST,\n"
     "# and it can only cover windows that are NOT core windows.\n"
     'NO_PER_POSITION_WINDOWS = ("prefill",)',
     'CORE_PER_POSITION_WINDOWS = ("overall", "high-draft-1way")\n'
     "\n"
     "# THE WINDOWS THAT PUBLISH NO PER-POSITION HISTOGRAM, BY NAME. The "
     "prefill\n"
     "# window measures a prefill, not a draft acceptance: it carries no\n"
     "# per-position acceptance numbers. The exclusion is a SHORT PUBLISHED "
     "LIST,\n"
     "# and it can only cover windows that are NOT core windows.\n"
     'NO_PER_POSITION_WINDOWS = ("prefill", "prose-1way")',
     "r8_a_core_window_that_loses_its_histogram_is_named"),
    ("F26 the duplicate-section check is removed",
     "    check(\"the capture gate publishes no box's section more than once\",\n"
     "          not _duplicate_box_sections(cap_counts),",
     "    check(\"the capture gate publishes no box's section more than once\",\n"
     "          True,",
     "r9_a_duplicated_box_section_still_fails"),
    ("F27a the denial guard is removed",
     "    denies_red = False\n    if names_red:",
     "    denies_red = False\n    if False and names_red:",
     "r10_an_answer_that_denies_the_colour_fails_by_name"),
    ("F27b the denial guard is widened to the whole answer",
     "            _before = list(re.finditer(r\"[\\w']+\", vision_answer[:_m.start()]))\n"
     "            _window = [(t.group(0).lower(), t.end()) for t in _before[-3:]]",
     "            _before = list(re.finditer(r\"[\\w']+\", vision_answer))\n"
     "            _window = [(t.group(0).lower(), t.end()) for t in _before]",
     "r10_a_mention_of_another_colour_still_passes"),
    ("F27c the denial guard fires on every answer",
     "    denies_red = False\n    if names_red:",
     "    denies_red = True\n    if names_red:",
     "r10_a_plain_red_answer_still_passes"),
    # ── THE ROUND-17 MUTANTS ─────────────────────────────────────────────
    # Each one puts a KNOWN DEFECT back into the fix it pins, so the case
    # that pins it goes RED on the mutant.
    ("F31a the class check is narrowed back to the stream branch",
     "        is_decode_row = (kind_key not in (\"prefill\", \"acceptance\")\n"
     "                         and (mode_key in (\"stream\", \"non-stream\")\n"
     "                              or _row_is_conc(label, row)))",
     "        is_decode_row = (kind_key not in (\"prefill\", \"acceptance\")\n"
     "                         and mode_key == \"stream\")",
     "r11_a_concurrency_row_with_an_unknown_class_is_named"),
    ("F31b the prefill row is judged as a decode row",
     "        is_decode_row = (kind_key not in (\"prefill\", \"acceptance\")\n"
     "                         and (mode_key in (\"stream\", \"non-stream\")\n"
     "                              or _row_is_conc(label, row)))",
     "        is_decode_row = (kind_key not in (\"acceptance\")\n"
     "                         and (mode_key in (\"stream\", \"non-stream\")\n"
     "                              or _row_is_conc(label, row)\n"
     "                              or kind_key == \"prefill\"))",
     "r12_a_prefill_row_with_a_class_label_is_not_named"),
    ("F31c a row the rule cannot classify is still counted by the floors",
     "            unknown_class.append(\n"
     "                (label, raw_label if isinstance(raw_label, str) else repr(raw_label)))\n"
     "            continue\n",
     "            unknown_class.append(\n"
     "                (label, raw_label if isinstance(raw_label, str) else repr(raw_label)))\n",
     "r11_a_concurrency_row_with_an_unknown_class_is_named"),
    ("F31d the class check fires on every decode row",
     "        if is_decode_row and workload == \"\":\n",
     "        if is_decode_row:\n",
     "r13_a_concurrency_row_with_the_prose_class_is_still_counted"),
    ("F31e the known-class criterion keeps its old name",
     "    check(\"the workload split: every decode row carries a known class\",\n",
     "    check(\"the workload split: every workload row carries a known class\",\n",
     "r12_the_known_class_criterion_is_named_for_the_decode_rows"),
    ("F31f the non-stream row is not a decode row",
     "        is_decode_row = (kind_key not in (\"prefill\", \"acceptance\")\n"
     "                         and (mode_key in (\"stream\", \"non-stream\")\n"
     "                              or _row_is_conc(label, row)))",
     "        is_decode_row = (kind_key not in (\"prefill\", \"acceptance\")\n"
     "                         and (mode_key == \"stream\"\n"
     "                              or _row_is_conc(label, row)))",
     "r11_a_non_stream_row_with_an_unknown_class_is_named"),
    # ── THE ROUND-18 MUTANTS ─────────────────────────────────────────────
    # Each one puts a KNOWN DEFECT back into the fix it pins, so the case
    # that pins it goes RED on the mutant.
    ("F32 the box-section detail claims an ABSENT box is published",
     "        present = sum(1 for b in REQUIRED_BOXES if counts.get(b, 0) == 1)\n"
     "        absent = sorted(b for b in REQUIRED_BOXES if not counts.get(b, 0))\n"
     "        return (f\"no box's section is published more than once -- {present} of \"\n"
     "                f\"the {len(REQUIRED_BOXES)} required box section(s) are \"\n"
     "                \"published\"\n"
     "                + (f\"; {len(absent)} {'IS' if len(absent) == 1 else 'ARE'} ABSENT \"\n"
     "                   f\"(e.g. {absent[0]})\" if absent else \"\"))",
     "        return (f\"no box's section is published more than once -- \"\n"
     "                f\"{len(REQUIRED_BOXES)} of the {len(REQUIRED_BOXES)} required \"\n"
     "                \"box section(s) are published\")",
     "r14_the_capture_gate_detail_names_the_absent_box"),
    ("F33c the clause rule is dropped (a negation in another sentence denies)",
     "                if re.search(r\"[.!?]\", vision_answer[_tok_end:_m.start()]):\n"
     "                    continue",
     "                if False:\n"
     "                    continue",
     "r15_a_negation_in_another_sentence_does_not_deny_the_colour"),
]


def _mutant_rule(fixed: str, old: str, new: str, workdir: str):
    """A copy of the rule under test with one defect put back."""
    with open(fixed, encoding="utf-8") as fh:
        text = fh.read()
    if text.count(old) != 1:
        return None
    path = os.path.join(workdir, "mutant-rule.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(old, new))
    return path


# ══ MAIN ══════════════════════════════════════════════════════════════════

def _build_context(rule: str, work: str) -> dict:
    """Every run of the rule the cases read."""
    ctx: dict = {}
    base = base_files(rule)
    rc, out = run(base, rule)
    ctx["rc"] = rc
    ctx["out"] = out

    def variant(files):
        r, o = run(files, rule)
        return o

    ctx["one_token_prefill"] = variant(set_prefill(base, prompt_tokens=1))
    ctx["unreadable_prefill_length"] = variant(
        set_prefill(base, prompt_tokens="abc"))
    ctx["no_budget_prefill"] = variant(set_prefill(base, target_tokens=None))
    ctx["small_budget_prefill"] = variant(set_prefill(base, target_tokens=1024))
    ctx["inter_token_above_ceiling"] = variant(
        set_inter_token(base, "42.004"))
    ctx["inter_token_below_ceiling"] = variant(
        set_inter_token(base, "41.996"))
    ctx["inter_token_above_reject"] = variant(
        set_inter_token(base, "52.004"))
    ctx["inter_token_below_reject"] = variant(
        set_inter_token(base, "51.996"))
    ctx["cross_check_above_ceiling"] = variant(set_cross_check(
        base, usage_tokens=1000, engine_draft_rounds=1000,
        engine_tokens_per_step=1.10004, deviation_pct=10.004,
        tokens_per_step_deviation_pct=9.0945))
    ctx["capacity_below_expectation"] = variant(set_pool(base, 183061))
    ctx["unreadable_counter"] = variant(set_guard_counter(base, "abc"))
    ctx["dead_stream"] = variant(set_dead_streams(base, 1))
    # THE ROUND-16 CLASSES.
    # EVERY prose lane row relabelled with a class the producer cannot emit.
    ctx["all_prose_relabelled"] = variant(
        relabel_stream_workload(base, "code"))
    # EXACTLY ONE prose lane row relabelled: the free-prose class survives.
    ctx["one_prose_relabelled"] = variant(
        relabel_stream_workload(base, "code", only_label="[r2]"))
    # THE HISTOGRAM GONE FROM EVERY ACCEPTANCE ROW.
    ctx["no_histogram_anywhere"] = variant(drop_per_position(base))
    # THE HISTOGRAM GONE FROM ONE CORE WINDOW.
    ctx["core_window_loses_histogram"] = variant(
        drop_per_position(base, only_window="prose-1way"))
    # THE PREFILL WINDOW, WITH NO PER-POSITION NUMBERS.
    ctx["prefill_window_no_histogram"] = variant(set_prefill_histogram(base))
    # A BOX PUBLISHED TWICE.
    ctx["duplicate_box_section"] = variant(duplicate_box_section(base))
    # THE VISION ANSWER: one that denies the colour, and one that names
    # another colour.
    ctx["vision_denies_red"] = variant(set_vision(
        base, "the image is not Red at all, it is blue"))
    ctx["vision_names_another_colour"] = variant(set_vision(base, "Red, not blue"))
    # THE ROUND-18 CLASSES.
    # ONE BOX'S SECTION ABSENT, in each of the three aggregates.
    ctx["capture_gate_one_box_absent"] = variant(
        drop_box_section(base, "capture-gate.txt", ABSENT_BOX))
    ctx["kit_check_one_box_absent"] = variant(
        drop_box_section(base, "kit-check.txt", ABSENT_BOX))
    ctx["guard_one_box_absent"] = variant(
        drop_box_section(base, "guard.txt", ABSENT_BOX))
    # THE VISION ANSWER, AGAIN: a negation in ANOTHER sentence, a negation
    # OUTSIDE the window, and a denial whose negation is two tokens back.
    ctx["vision_negation_in_another_sentence"] = variant(
        set_vision(base, "Not blue. Red."))
    ctx["vision_negation_outside_the_window"] = variant(
        set_vision(base, "It is not blue, it is Red"))
    ctx["vision_negation_two_tokens_back"] = variant(
        set_vision(base, "the image is not really Red at all"))
    # THE ROUND-17 CLASSES.
    # A CONCURRENCY ROW WITH A CLASS THE PRODUCER CANNOT EMIT.
    ctx["concurrency_class_relabelled"] = variant(
        relabel_concurrency_workload(base, "code", only_mode="concurrent x4"))
    # A NON-STREAM ROW WITH A CLASS THE PRODUCER CANNOT EMIT.
    ctx["non_stream_class_relabelled"] = variant(
        add_non_stream_row(base, workload="code"))
    # A PREFILL ROW WITH A CLASS LABEL ON IT -- judged by its KIND.
    ctx["prefill_class_labelled"] = variant(set_prefill(base, workload="code"))
    # THE LANE'S OWN PREFILL ANNOTATION -- also judged by its KIND.
    ctx["prefill_lane_class_labelled"] = variant(
        set_prefill(base, workload="long-context"))
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default=DEFAULT_RULE,
                    help="the acceptance rule to exercise")
    ap.add_argument("--no-mutants", action="store_true",
                    help="run the cases only (the mutant proof is skipped)")
    args = ap.parse_args()

    if not os.path.isfile(args.rule):
        print(f"_rule_battery: no such rule: {args.rule}")
        return 2

    work = tempfile.mkdtemp(prefix="rule-battery-")
    failures = []
    try:
        ctx = _build_context(args.rule, work)
        print("_rule_battery: the cases")
        for name, func in CASES:
            try:
                ok, detail = func(ctx)
            except Exception as exc:  # noqa: BLE001 - a case that raises is a failure
                ok, detail = False, f"the case raised {exc!r}"
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
            if not ok:
                failures.append(name)

        if not args.no_mutants:
            print("\n_rule_battery: the mutants")
            for label, old, new, case_name in MUTANTS:
                mutant = _mutant_rule(args.rule, old, new, work)
                if mutant is None:
                    print(f"  [FAIL] {label} -- the rule under test does not "
                          f"carry the text the mutant replaces, so the mutant "
                          f"could not be built")
                    failures.append(f"mutant:{label}")
                    continue
                sub = _build_context(mutant, work)
                func = dict(CASES)[case_name]
                try:
                    ok, detail = func(sub)
                except Exception as exc:  # noqa: BLE001
                    ok, detail = False, f"the case raised {exc!r}"
                if ok:
                    print(f"  [FAIL] {label} -- {case_name} is still GREEN on "
                          f"the mutant, so it cannot see this defect")
                    failures.append(f"mutant:{label}")
                else:
                    print(f"  [PASS] {label} -- caught by {case_name}: {detail}")

        print()
        if failures:
            print(f"_rule_battery: FAIL -- {len(failures)} case(s) failed: "
                  f"{', '.join(failures)}")
            return 1
        print("_rule_battery: PASS -- every case is green and every mutant is "
              "caught.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
