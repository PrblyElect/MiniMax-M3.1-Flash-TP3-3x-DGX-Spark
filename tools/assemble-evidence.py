#!/usr/bin/env python3
"""assemble-evidence.py -- turn the PER-BOX evidence into the three aggregate
files the acceptance rule reads.

WHY THIS EXISTS
===============
The acceptance rule (tools/acceptance-check.py) reads its evidence from a
directory of NINE files, and it is fail-closed about every one of them: a
missing measurement is a FAIL, never a skip. Three of those nine are
AGGREGATES over the three boxes, and the per-box evidence that goes into
them is produced by the shipped tools:

  capture-gate.txt   the three boxes' capture-gate sections, merged
  kit-check.txt      the kit verdict, asserted inside each box's own section
  guard.txt          each box's guard evidence, in the rule's key contract

Without this tool, a reader could produce the per-box facts but not the
aggregates, and the rule would reject its own evidence set. A documented
procedure that cannot be executed is a documented lie, so this tool ships.

WHAT IT READS (in the evidence directory)
=========================================
  capture-gate-<label>.txt   one per box: capture-config.sh --out DIR, run
                             ON THAT BOX. The section header carries the box's
                             LANE LABEL (the one passed to --head/--r1/--r2),
                             because the rule keys its sections by that label.
  guard-<label>.log         one per box: the guard's guard.log (the arming
                             line and any breach line).
  guard-<label>-mem.log     one per box: the guard's guard_mem.log (the
                             free-memory samples).

WHAT IT WRITES (into the evidence directory)
============================================
  capture-gate.txt, kit-check.txt, guard.txt

WHAT IT DOES NOT WRITE
======================
The rule's other four evidence files -- probes.txt, vision-smoke.txt,
c1-proof.txt and ring-proof.txt -- are MEASUREMENTS the operator has to
make. This tool prints their exact contract and writes NOTHING for them, on
purpose: a rule that can be satisfied by writing a template is not a rule.
The four of them are measurements the reader performs with their own
harness; the three this tool assembles are bookkeeping over evidence the
shipped tools already produced.

Usage:
    python3 tools/assemble-evidence.py --evidence-dir logs/evidence \
        --head spark-head.local --r1 spark-a.local --r2 spark-b.local
Exit: 0 when every aggregate was written, 1 otherwise (fail-closed: a box
with no evidence is an error, not an empty section).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

DEFAULT_BOXES = ("spark-head.local", "spark-a.local", "spark-b.local")

# The guard's own start line: "<date> <time> guard start min=4096MB ...".
START_RE = re.compile(r"guard start min=(\d+)MB")
# A FULL timestamp: "<date> <time> ...". The arming line AND every
# memory sample carry one, so the cycle's window is compared in DATE AND
# TIME. A time of day alone cannot order the two across midnight: a sample
# at 00:04:11 sorts BEFORE an arming at 23:58:02, so a cycle that crosses
# midnight would lose every sample and the rule would fail a healthy box.
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
# A memory sample: "<date> <time> avail_mb=7632 low_mb=7393 up=1 breaches=0".
#
# THE WHOLE VALUE, OR NOTHING. The old pattern took a PREFIX of the digits
# ("breaches=3x" was read as 3), so a counter the guard did not write was
# laundered into a clean integer and published as though the guard had
# written it. The value is now taken whole, and a value that is not a whole
# plain integer is REFUSED further down -- a damaged counter is NAMED,
# never rounded into a figure.
SAMPLE_RE = re.compile(r"\b(low_mb|avail_mb|breaches)=(\S*)")
# A WHOLE, PLAIN, ASCII INTEGER -- and nothing else.
INT_RE = re.compile(r"[0-9]+")
# THE LONGEST COUNTER THIS TOOL WILL READ AS A NUMBER. A run of digits
# longer than this is DAMAGE, not a measurement.
#
# CPython refuses to convert more than `sys.get_int_max_str_digits()`
# digits (4,300 by default) and raises ValueError, so an unbounded read
# CRASHES this tool on a damaged log -- and a crash here left the previous
# cycle's aggregates in place for the rule to judge as this cycle's
# evidence. The bound is applied BEFORE the parse, so the read never
# depends on the interpreter's limit, and the `try`/`except` below is the
# second line of the same defence: an over-long run is UNREADABLE, never a
# truncated integer and never a crash.
MAX_COUNTER_DIGITS = 20
# THE GUARD'S OWN MARKERS. The guard says out loud when it could not read a
# counter ('?') and when it had not sampled yet ('-'), and the rule treats
# those as the guard saying so -- not as corruption. They are not figures
# either, so they contribute no number; a counter that carries one is
# UNKNOWN, and an unknown count cannot be assumed to be zero.
GUARD_MARKERS = ("?", "-")
# The token published in place of a counter that cannot be read. It is NOT
# a number, so a reader that reads the counter as a number reads NOTHING --
# and a fact nobody can read fails its criterion BY NAME instead of
# quietly passing on a laundered value.
UNREADABLE = "unreadable"


def read_counter(value: str):
    """The counter's value, or None when this tool will not read it.

    A counter is read only when it is a WHOLE, PLAIN and SHORT run of
    digits. A run that is too long is not a number this tool will accept --
    and a number this tool will not accept is UNREADABLE, never a
    truncated one (the old `int(v[:3])` shape laundered a damaged counter
    into a clean small integer) and never a crash.
    """
    if not INT_RE.fullmatch(value):
        return None
    if len(value) > MAX_COUNTER_DIGITS:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sections(text: str) -> dict[str, str]:
    """Split '=== <label> ===' sections, exactly as the rule splits them."""
    out: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^=== (\S+) ===$", line.strip())
        if m:
            if current:
                out[current] = "\n".join(buf)
            current = m.group(1)
            buf = []
        elif current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf)
    return out


def read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def parse_guard(log_text: str, mem_text: str) -> tuple[dict, list[str], list[str], list[str]]:
    """Return (the rule's keys, the memory-sample lines, the problems, the
    counters that could not be read).

    THE WINDOW IS THIS CYCLE'S: the samples at or after the LAST arming
    line. The guard ROTATES its log when it arms, so the normal case is
    one arming line and the whole log IS the window -- and the rule's
    "windowed to this cycle" criterion (exactly one arming line) is what
    says so. A log carrying more than one arming line is a log that was
    not rotated, and the rule fails it rather than guessing.
    """
    keys: dict = {}
    problems: list[str] = []

    starts = [ln for ln in log_text.splitlines() if START_RE.search(ln)]
    breaches = [ln for ln in log_text.splitlines() if "BREACH " in ln]
    keys["guard_start_lines"] = len(starts)
    keys["guard_log_breaches"] = len(breaches)

    if starts:
        m = START_RE.search(starts[-1])
        # THE ARMED THRESHOLD IS READ AS A WHOLE, BOUNDED COUNTER. The last
        # arming line is the one this cycle's window is computed against, so
        # its threshold is the floor the cycle ran with. A threshold this
        # tool cannot read is published as UNREADABLE -- the rule then fails
        # it BY NAME -- and it is never a truncated integer.
        armed = read_counter(m.group(1)) if m else None
        if armed is None:
            keys["armed_threshold_mb"] = UNREADABLE
            problems.append("the last 'guard start' line's armed threshold "
                            f"({(m.group(1)[:40] + '...') if m else '?'}) is "
                            "not a counter this tool can read -- the floor the "
                            "guard ran with is UNKNOWN")
        else:
            keys["armed_threshold_mb"] = armed
    else:
        problems.append("no 'guard start' line in the guard log -- the guard "
                        "was never armed, so the floor it ran with is unknown")

    samples = [ln for ln in mem_text.splitlines() if SAMPLE_RE.search(ln)]
    if not samples:
        problems.append("the guard's memory log has no free-memory samples")

    # The window: the samples at or after the LAST arming. The arming line
    # and the samples both carry a FULL timestamp, so the comparison is on
    # date AND time -- exact, including across midnight.
    window = samples
    if starts:
        m = TS_RE.search(starts[-1])
        if m:
            armed_at = m.group(1)
            window = []
            unplaced = 0
            for ln in samples:
                sm = TS_RE.search(ln)
                if sm is None:
                    # A sample whose timestamp cannot be read cannot be
                    # ordered against the arming. It is KEPT -- dropping a
                    # measurement can only make the evidence look better,
                    # and evidence must never be flattered -- and it is
                    # counted, so the reader can see that it happened.
                    unplaced += 1
                    window.append(ln)
                elif sm.group(1) >= armed_at:
                    window.append(ln)
            if unplaced:
                problems.append(f"{unplaced} memory sample(s) carry no parsable "
                                "timestamp and were kept unplaced -- the window "
                                "may include a sample from another cycle")
        else:
            problems.append("the last arming line has no parsable timestamp, "
                            "so the cycle's window could not be bounded")
    if not window and samples:
        problems.append("no memory sample falls inside this cycle's window "
                        "(the log predates the last arming) -- the window "
                        "cannot be measured")
    # The window's breach count is the HIGHEST the guard's own consecutive
    # breach counter reached inside the window -- not the number of sample
    # lines that happen to carry a non-zero counter. The guard's counter is
    # the number of CONSECUTIVE low samples it has seen, so the highest
    # value it reached inside the window is what "the guard fired here"
    # means, and it is 0 exactly when the guard never fired.
    #
    # A COUNTER THAT CANNOT BE READ IS NOT COUNTED, AND IT IS NEVER ROUNDED
    # INTO ONE. The old pattern took a PREFIX of the digits, so a sample
    # reading "breaches=3x" was read as the clean integer 3 -- a figure the
    # guard never wrote, published as though it had. The whole value is
    # taken here, and a value that is neither a plain integer nor one of
    # the guard's own markers is REFUSED: the reader cannot know that the
    # sample it could not read was not the one that mattered, and an
    # unknown count cannot be assumed to be zero. So the window's count is
    # published as UNREADABLE instead -- a fact that is NAMED and that the
    # rule fails by name, rather than a clean integer that flatters the
    # evidence.
    win_breaches = 0
    unreadable: list[str] = []
    for ln in window:
        for k, v in SAMPLE_RE.findall(ln):
            if k != "breaches":
                continue
            # THE WHOLE VALUE, BOUNDED. A counter that is not a plain, short
            # run of digits is DAMAGE: it is named, and the window's count is
            # published as UNREADABLE. It is never rounded into a number and
            # never a crash (a five-thousand-digit run raised ValueError in
            # `int()`, and the crash left the previous cycle's aggregates in
            # place for the rule to judge).
            value = read_counter(v)
            if value is None:
                unreadable.append(v)
                continue
            win_breaches = max(win_breaches, value)
    if unreadable:
        keys["guard_window_breaches"] = UNREADABLE
    else:
        keys["guard_window_breaches"] = win_breaches

    return keys, samples, problems, unreadable


# THE AGGREGATES THIS TOOL OWNS. They are the previous cycle's answer until
# this cycle's answer replaces them, and the rule reads them as THIS cycle's
# evidence.
AGGREGATE_FILES = ("capture-gate.txt", "kit-check.txt", "guard.txt")


def remove_stale_aggregates(ev: str) -> list[str]:
    """The previous cycle's aggregates are removed. Returns their names.

    THE REMOVAL IS NOT A REFUSAL PATH. It happens at the START of every
    run, before anything else, so that EVERY exit -- a refusal, a crash, a
    usage error -- leaves no previous cycle's answer in place for the rule
    to judge as this cycle's evidence. The removal used to happen only on
    the refusal path, and a CRASH therefore left the old aggregates sitting
    in the directory: the assembler died, and the rule judged the STALE
    files and reported PASS for a box whose evidence was not there. A
    refusal that leaves the old answer in place is worse than a crash, and
    so is a crash that leaves it in place.
    """
    removed = []
    for name in AGGREGATE_FILES:
        stale = os.path.join(ev, name)
        try:
            if os.path.exists(stale):
                os.remove(stale)
                removed.append(name)
        except OSError as exc:
            print(f"assemble-evidence: WARNING -- the stale aggregate {stale} "
                  f"could not be removed ({exc}) -- it would be judged as this "
                  f"cycle's evidence", file=sys.stderr)
    return removed


def _evidence_dir_hint(argv: list[str]) -> str:
    """The evidence directory the command line names, read WITHOUT argparse.

    The stale aggregates are removed at the START of every run, so the
    directory has to be known even when the rest of the command line is
    wrong: argparse exits on a usage error, and a usage error must not
    leave the previous cycle's answer in place either. argparse accepts any
    unambiguous PREFIX of `--evidence-dir`, so this scan accepts the same
    prefixes, and it falls back to argparse's own default.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--evidence"):
            if "=" in arg:
                return arg.split("=", 1)[1]
            if i + 1 < len(argv):
                return argv[i + 1]
    return "."


def main() -> int:
    ev_hint = _evidence_dir_hint(sys.argv[1:])
    # THE STALE AGGREGATES GO FIRST, BEFORE ANYTHING ELSE.
    removed = remove_stale_aggregates(ev_hint)
    if removed:
        print("assemble-evidence: removed the previous cycle's stale "
              f"aggregates ({', '.join(removed)}) so the rule cannot judge "
              "them as this cycle's evidence.", file=sys.stderr)
    wrote = False
    try:
        rc = _assemble()
        wrote = (rc == 0)
        return rc
    finally:
        # AND ON EVERY EXIT THAT IS NOT A SUCCESS. A crash halfway through
        # the writes must not leave a half-written aggregate behind for the
        # rule to judge, and a usage error must not leave the previous
        # cycle's answer in place either.
        if not wrote:
            remove_stale_aggregates(ev_hint)


def _assemble() -> int:
    ap = argparse.ArgumentParser(
        description="assemble the acceptance rule's aggregate evidence files")
    ap.add_argument("--evidence-dir", default=".",
                    help="the evidence directory (read and written)")
    ap.add_argument("--head", default=DEFAULT_BOXES[0], help="the head's lane label")
    ap.add_argument("--r1", default=DEFAULT_BOXES[1], help="the first worker's lane label")
    ap.add_argument("--r2", default=DEFAULT_BOXES[2], help="the second worker's lane label")
    args = ap.parse_args()

    ev = args.evidence_dir
    labels = [args.head, args.r1, args.r2]
    problems: list[str] = []

    gate_sections: list[tuple[str, str]] = []
    kit_sections: list[str] = []
    guard_sections: list[str] = []

    for label in labels:
        # --- the capture gate's section for this box -----------------------
        cap_path = os.path.join(ev, f"capture-gate-{label}.txt")
        cap_text = read(cap_path)
        if not cap_text.strip():
            problems.append(f"{label}: no capture-gate file ({cap_path}) -- run "
                            f"capture-config.sh --out on that box and copy the "
                            f"file here")
            continue
        secs = sections(cap_text)
        if label not in secs:
            found = ", ".join(sorted(secs)) or "none"
            problems.append(f"{label}: the capture-gate file has no '=== {label} ===' "
                            f"section (it has: {found}) -- the rule keys its "
                            f"sections by the LANE LABEL")
            continue
        body = secs[label]
        gate_sections.append((label, body))

        # --- the kit verdict, asserted inside this box's own section ------
        kit_lines = [ln for ln in body.splitlines() if "kit-md5" in ln]
        if kit_lines:
            kit_sections.append(f"=== {label} ===\n" + "\n".join(kit_lines))
        else:
            kit_sections.append(f"=== {label} ===\n"
                                "kit-md5: NOT REPORTED -- this box's capture-gate "
                                "section carries no kit-md5 line")

        # --- the guard's evidence -----------------------------------------
        gl_path = os.path.join(ev, f"guard-{label}.log")
        gm_path = os.path.join(ev, f"guard-{label}-mem.log")
        gl_text = read(gl_path)
        gm_text = read(gm_path)
        if not gl_text.strip():
            problems.append(f"{label}: no guard log ({gl_path}) -- copy the "
                            f"guard's guard.log here")
            continue
        if not gm_text.strip():
            problems.append(f"{label}: no guard memory log ({gm_path}) -- copy "
                            f"the guard's guard_mem.log here")
            continue
        keys, samples, guard_problems, unreadable = parse_guard(gl_text, gm_text)
        for problem in guard_problems:
            problems.append(f"{label}: {problem}")
        block = [f"=== {label} ==="]
        for key in ("armed_threshold_mb", "guard_start_lines",
                    "guard_log_breaches", "guard_window_breaches"):
            if key in keys:
                block.append(f"{key}={keys[key]}")
        block.append("--- the guard's own arming line ---")
        starts = [ln for ln in gl_text.splitlines() if START_RE.search(ln)]
        if starts:
            block.append(starts[-1])
        block.append("--- the guard's memory samples ---")
        block.extend(samples)
        if unreadable:
            # THE COUNTER THAT COULD NOT BE READ IS NAMED IN THE EVIDENCE,
            # so a reader can see WHY the window's count is unreadable
            # without reading the rule's source. The raw sample line above
            # is carried unchanged as well: it is the SECOND witness, and it
            # is what the rule reads the damage from.
            parts = []
            for v in unreadable:
                if v in GUARD_MARKERS:
                    kind = "the guard's own marker"
                elif INT_RE.fullmatch(v):
                    kind = (f"a run of {len(v)} digits -- longer than the "
                            f"{MAX_COUNTER_DIGITS} this tool reads")
                else:
                    kind = "not a number"
                # The value is shown SHORTENED when it is long: the raw
                # sample line is carried unchanged above, so the reader can
                # always see the whole thing, and a five-thousand-digit run
                # does not have to be reprinted to be identified.
                shown = v if len(v) <= 40 else v[:40] + "..."
                parts.append(f"{shown!r} ({kind})")
            block.append("--- a counter the rule cannot read ---")
            block.append("the guard's own breach counter was published as "
                         + ", ".join(parts)
                         + " -- the window's breach count is UNREADABLE and was "
                           "NOT reduced to a number")
            print(f"assemble-evidence: WARNING -- {label}: the guard's memory "
                  f"log carries {len(unreadable)} breach counter(s) the rule "
                  f"cannot read ({', '.join(repr(v if len(v) <= 40 else v[:40] + '...') for v in unreadable)}) -- the "
                  f"window's breach count is published as UNREADABLE, not as a "
                  f"number", file=sys.stderr)
        guard_sections.append("\n".join(block))

    if problems:
        print("assemble-evidence: FAILED -- the evidence set is incomplete:",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        # THE STALE AGGREGATES ARE ALREADY GONE. They are removed at the
        # START of every run (see `main`), not on this path: a refusal that
        # leaves the old answer in place is worse than a crash, and so is a
        # CRASH that leaves it in place -- the removal used to live here, so
        # a crash (a damaged counter with five thousand digits raised
        # ValueError in `int()`) left the previous cycle's aggregates
        # sitting in the directory for the rule to judge as this cycle's
        # evidence, and the rule reported PASS for a box whose evidence was
        # not there.
        print("assemble-evidence: nothing was written. A box with no evidence is "
              "an error, not an empty section.", file=sys.stderr)
        return 1

    with open(os.path.join(ev, "capture-gate.txt"), "w", encoding="utf-8") as fh:
        for label, body in gate_sections:
            fh.write(f"=== {label} ===\n{body}\n\n")
    with open(os.path.join(ev, "kit-check.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(kit_sections) + "\n")
    with open(os.path.join(ev, "guard.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(guard_sections) + "\n")

    print(f"assemble-evidence: wrote capture-gate.txt, kit-check.txt and "
          f"guard.txt into {ev} for {', '.join(labels)}")
    print()
    print("THE FOUR FILES YOU STILL HAVE TO MEASURE. This tool deliberately "
          "does not write them: a rule that can be satisfied by writing a "
          "template is not a rule. Each is a real measurement, and the rule "
          "fails closed while it is missing.")
    print("  probes.txt       must contain the literal: all 7 probes PASS")
    print("  vision-smoke.txt must carry an answer that names the colour Red")
    print("  c1-proof.txt     must contain the literal: C-1 PROOF PASS")
    print("  ring-proof.txt   must contain the literal: every rank issued the "
          "same collective sequence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
