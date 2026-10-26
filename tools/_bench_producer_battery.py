#!/usr/bin/env python3
"""_bench_producer_battery.py -- the producer-side battery for this kit.

WHY THIS EXISTS
===============
TWO TOOLS PUBLISH THE NUMBERS THE ACCEPTANCE RULE JUDGES, and both of them
can publish a figure the source did not carry:

  bench-decode.py       the measurement rows -- the per-stream minimum and
                        the engine's per-token inter-token figure;
  assemble-evidence.py  the guard aggregates -- the window's breach count.

A producer that publishes a figure the source did not carry is the worst
kind of defect in this kit, because the rule CANNOT SEE IT. The rule reads
the FIGURE, and a figure that is wrong at the source is judged as though it
were right. A published minimum that quietly drops a stream which produced
no tokens, and a published counter that quietly rounds a damaged sample
into a clean integer, are both invisible to a reader of the evidence -- and
the rule is built to be the reader that cannot be fooled.

SO THIS BATTERY DRIVES THE REAL PRODUCERS and pins the behaviours that must
not regress:

  F1  A stream that produced no tokens STAYS in the per-stream minimum, the
      count of such streams is published beside it, and the bench calls the
      level broken.
  F2  The per-token inter-token figure reaches the rule with the precision
      the source measured, so a value just above the ceiling cannot round
      ONTO the ceiling and pass a limit it does not clear.
  F3  A guard counter the rule cannot read is NAMED, never laundered into a
      clean integer -- and the assembler does not CRASH on a damaged one,
      because a crash leaves the previous cycle's aggregates in place for
      the rule to judge as this cycle's evidence.
  F4  The prefill's published length is the ENGINE'S OWN count of the prompt
      it was given, not the budget the bench declared -- and a budget the
      engine cannot reach publishes NO prefill row at all.
  F5  No previous cycle's aggregate survives ANY exit of the assembler: not a
      refusal, and not a crash.
  F6  The armed threshold the aggregate publishes is the one this cycle's
      window is computed against, not the one from an earlier arming line.

WHAT IT RUNS
------------
A miniature OpenAI-compatible engine (standard library only) serves the
bench, and the bench runs against it on a DETERMINISTIC CLOCK: two runs of
the same code publish the same bytes, so a byte-for-byte comparison of two
producer versions compares the PRODUCERS and nothing else.

The MUTANTS are the other half of the battery: each one puts a known
defect back into a copy of the fixed producer, and the battery asserts that
a case which is GREEN on the fixed bytes turns RED on the mutant. A case
that cannot tell the fixed code from the broken code is not a test.

Usage:
    python3 tools/_bench_producer_battery.py
    python3 tools/_bench_producer_battery.py --bench PATH --assemble PATH
    python3 tools/_bench_producer_battery.py --compare-bench PATH
    python3 tools/_bench_producer_battery.py --with-judge

Exit: 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# NEVER leave bytecode next to the kit: this folder is shipped as it is.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCH = os.path.join(HERE, "bench-decode.py")
DEFAULT_ASSEMBLE = os.path.join(HERE, "assemble-evidence.py")
DEFAULT_JUDGE = os.path.join(HERE, "acceptance-check.py")

# THE BOXES. The assembler's own defaults, which are the kit's placeholders.
BOXES = ("spark-head.local", "spark-a.local", "spark-b.local")

# THE PREFILL BUDGET THE FIXTURE DECLARES. It is deliberately NOT a multiple
# of the fixture engine's word count (36), so the engine's own count can
# never EQUAL the declared budget -- and the case can therefore tell a
# MEASURED length from the DECLARED one.
_PREFILL_BUDGET = 700

# ══ THE FIXTURE'S CLOCK ═══════════════════════════════════════════════════
#
# The bench's published figures are ratios of wall-clock intervals, so two
# runs of the SAME code do not publish the SAME bytes -- and a byte-for-byte
# comparison of two producer versions would then be a comparison of noise.
#
# The fixture's clock serves a fixed three-step schedule: the request's
# start, the first token, and the end of the decode phase. Every call after
# that repeats the schedule, so a measurement that makes a different number
# of clock calls (the concurrent level's wall time) cannot shift the phase
# of the next measurement.
#
# THE WINDOW IS THE POINT OF F2. The rule compares the per-token figure
# with a CEILING, and a figure that is rounded on the way out can round
# ONTO that ceiling: a true 42.004 published as "42.00" is read as 42.0 and
# passes a limit it does not clear. So the fixture measures a value INSIDE
# that window -- above the ceiling, and low enough that a two-decimal
# format rounds it down onto it.
_CEILING = 42.0
_WINDOW = (42.0, 42.005)
_TARGET = 42.004
_TOKENS = 16            # tokens per measured call
_CHUNKS = 8             # stream chunks per measured call
_TOKENS_PER_CHUNK = 2
_TTFT = 0.1             # the time to the first token
_TAIL = 0.73006         # the decode phase's length


def _simulated_per_token(decode_s: float) -> float:
    """The per-token figure the bench will measure, computed its way.

    The bench computes (total_s - ttft_s) * 1000 / (tokens - 1), and the
    fixture's clock hands it the three offsets that produce those two
    intervals. So this is the value the bench measures, not an estimate of
    it -- and the battery compares the PUBLISHED token against the value
    the bench itself published in its measurement row.
    """
    t0 = 0.0
    ttft = _TTFT
    t1 = _TTFT + decode_s
    total_s = t1 - t0
    ttft_s = ttft - t0
    return (total_s - ttft_s) * 1000.0 / (_TOKENS - 1)


def _decode_duration() -> float:
    """The decode phase that lands the per-token figure in its window."""
    d = _TARGET * (_TOKENS - 1) / 1000.0
    for _ in range(1000):
        if _WINDOW[0] < _simulated_per_token(d) <= _WINDOW[1]:
            return d
        d = math.nextafter(d, 0.0)
    raise SystemExit("_bench_producer_battery: the fixture clock could not "
                     "place the per-token figure in its rounding window")


_DECODE_S = _decode_duration()

# THE CLOCK ITSELF.
_OFFSETS = (0.0, _TTFT, _TTFT + _DECODE_S)


class _FixtureClock:
    """A deterministic replacement for time.perf_counter.

    PER THREAD, so the concurrent streams each see the same schedule
    instead of racing for a shared counter -- a measurement whose threads
    interleave differently would publish different bytes on every run.
    """

    def __init__(self, offsets=_OFFSETS):
        self._offsets = offsets
        self._local = threading.local()

    def __call__(self):
        n = getattr(self._local, "n", 0)
        self._local.n = n + 1
        return self._offsets[n % len(self._offsets)]


class _MonotonicClock:
    """A clock that only moves FORWARD.

    The fixture's own clock hands the bench three fixed offsets and cycles
    through them, so a measurement that reads it twice can see its second
    reading EARLIER than its first. The prefill measurement reads the clock
    twice per attempt and the growth loop makes SEVERAL attempts, so its
    runs use this clock instead: a duration has to be a duration.
    """

    STEP = 0.05

    def __init__(self, step: float = STEP):
        self._step = step
        self._local = threading.local()

    def __call__(self):
        n = getattr(self._local, "n", 0)
        self._local.n = n + 1
        return n * self._step


# ══ THE FIXTURE'S ENGINE ══════════════════════════════════════════════════
#
# A miniature OpenAI-compatible engine. Two modes:
#
#   healthy   every stream answers with a usage block;
#   one-dead  exactly ONE stream in the whole run answers WITHOUT one --
#             the first request that arrives while another one is already
#             in flight. A stream that returns no usage block produced no
#             tokens, and the measurement has to say so.
#
# The response is held open long enough for a concurrent batch to form, so
# the batch is a batch and not four requests that happened to be issued
# together.
_STUB_SOURCE = r'''
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = sys.argv[1]
CHUNKS = int(sys.argv[2])
TOKENS_PER_CHUNK = int(sys.argv[3])
HOLD_S = float(sys.argv[4])
RECORD = sys.argv[5] if len(sys.argv) > 5 else ""

_lock = threading.Lock()
_inflight = 0
_dead_claimed = False
_poll = [0]


def _record(value):
    """The engine's OWN count of the prompt it was given, written down.

    The battery compares the length the bench PUBLISHED against the length
    the engine MEASURED, so the measurement has to be reachable from outside
    the engine.
    """
    if not RECORD:
        return
    with open(RECORD, "a") as fh:
        fh.write("%d\n" % value)

METRICS = """# HELP vllm:spec_decode_num_accepted_tokens_total x
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{position="0"} %d
vllm:spec_decode_num_accepted_tokens_total{position="1"} %d
vllm:spec_decode_num_accepted_tokens_total{position="2"} %d
vllm:spec_decode_num_drafts_total %d
vllm:spec_decode_num_draft_tokens_total %d
vllm:time_to_first_token_seconds_sum %f
vllm:time_to_first_token_seconds_count %d
vllm:num_requests_running 0
vllm:num_requests_waiting 0
"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/metrics"):
            with _lock:
                _poll[0] += 1
                n = _poll[0]
            body = (METRICS % (100 + n, 100 + n, 100 + n, 100 + n, 100 + n,
                               2.0 + n, 10 + n)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/v1/models"):
            body = json.dumps({"data": [{"id": "minimax31-flash"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        global _inflight, _dead_claimed
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        if not req.get("stream"):
            # THE ENGINE REFUSES THE PROMPT. A context length, a payload too
            # large, an authentication failure -- the bench has to survive
            # any of them, because a rejected prefill must not cost the lane
            # its decode rows, its histogram and its manifest.
            if MODE == "reject-prefill":
                self.send_error(400, "maximum context length exceeded")
                return
            # THE ENGINE'S OWN COUNT OF THE PROMPT. In the "count-tokens"
            # mode the count is the number of words in the prompt the engine
            # was actually given -- so it grows with the prompt, and the
            # bench's growth loop can reach a declared token budget. In
            # every other mode the count is a constant, so a declared
            # budget is UNREACHABLE and the bench has to fail closed.
            prompt = ""
            for message in (req.get("messages") or []):
                prompt += str(message.get("content") or "")
            count = len(prompt.split())
            n_prompt = count if MODE == "count-tokens" else 12
            _record(n_prompt)
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "tok "},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": 8,
                          "total_tokens": n_prompt + 8}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        dead = False
        with _lock:
            _inflight += 1
            if (MODE == "one-dead" and _inflight >= 2 and not _dead_claimed):
                _dead_claimed = True
                dead = True
        try:
            time.sleep(HOLD_S)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(payload):
                self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
                self.wfile.flush()

            for _ in range(CHUNKS):
                emit(("data: " + json.dumps(
                    {"choices": [{"delta": {"content": "tok "},
                                  "finish_reason": None}]}) + "\n\n").encode())
            if not dead:
                emit(("data: " + json.dumps(
                    {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 12,
                               "completion_tokens": CHUNKS * TOKENS_PER_CHUNK,
                               "total_tokens": 12 + CHUNKS * TOKENS_PER_CHUNK}})
                    + "\n\n").encode())
            emit(b"data: [DONE]\n\n")
            emit(b"")
        finally:
            with _lock:
                _inflight -= 1


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
'''


class _Engine:
    """The fixture's engine, on its own process and its own port."""

    def __init__(self, mode: str):
        self._dir = tempfile.mkdtemp(prefix="bench-battery-engine-")
        path = os.path.join(self._dir, "engine.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_STUB_SOURCE)
        # THE ENGINE'S OWN COUNTS, written down as it measures them.
        self.record = os.path.join(self._dir, "counts.txt")
        self._proc = subprocess.Popen(
            [sys.executable, "-B", path, mode, str(_CHUNKS),
             str(_TOKENS_PER_CHUNK), "0.25", self.record],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.port = int(self._proc.stdout.readline().strip())

    def counts(self) -> list[int]:
        """The prompt-token counts the engine measured, in order."""
        out = []
        try:
            with open(self.record, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out.append(int(line))
        except OSError:
            pass
        return out

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        shutil.rmtree(self._dir, ignore_errors=True)


# ══ THE FIXTURE'S GUARD LOGS ══════════════════════════════════════════════
#
# The assembler bounds a cycle's window by comparing FULL timestamps, so the
# fixture's samples have to carry one. A literal date in a shipped file is
# a correlation bridge, so the fixture builds its clock from plain integers
# at RUN TIME instead.
_FIXTURE_YEAR = 2000
_FIXTURE_MONTH = 1
_FIXTURE_DAY = 1


def _stamp(offset_s: int) -> str:
    """A full timestamp, `offset_s` seconds after the fixture's own epoch."""
    total = offset_s
    return (f"{_FIXTURE_YEAR:04d}-{_FIXTURE_MONTH:02d}-{_FIXTURE_DAY:02d} "
            f"{total // 3600 % 24:02d}:{total // 60 % 60:02d}:{total % 60:02d}")


def _write_guard_inputs(evidence_dir: str, damaged: bool,
                        breach_value: str | None = None,
                        armings: tuple[str, ...] | None = None) -> None:
    """The per-box guard evidence the assembler reads.

    `breach_value` replaces the breach counter the LAST box's LAST sample
    carries -- so a counter of any shape can be driven through the
    assembler. `armings` replaces the arming lines, so a log with more than
    one arming line (and different floors) can be driven through it too.
    """
    for index, label in enumerate(BOXES):
        with open(os.path.join(evidence_dir, f"capture-gate-{label}.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write(f"=== {label} ===\n"
                     f"kit-md5: 00000000000000000000000000000000  "
                     f"tools/bench-decode.py\n")
        with open(os.path.join(evidence_dir, f"guard-{label}.log"),
                  "w", encoding="utf-8") as fh:
            if armings:
                for offset, text in enumerate(armings):
                    fh.write(f"{_stamp(offset * 5)} {text}\n")
            else:
                fh.write(f"{_stamp(0)} guard start min=4096MB debounce=3 "
                         f"container=minimax31-flash-tp3 pid=1\n")
        lines = []
        for tick in range(3):
            # The LAST box's LAST sample is the damaged one, and the damage
            # is a counter that is not a number at all.
            breach = "3x" if (damaged and index == len(BOXES) - 1
                              and tick == 2) else "0"
            if breach_value is not None and tick == 2:
                # The damage is on EVERY box's last sample: it is a shape of
                # counter, not one box's accident.
                breach = breach_value
            lines.append(f"{_stamp(10 + tick)} avail_mb=7632 low_mb=7393 "
                         f"up=1 breaches={breach}\n")
        with open(os.path.join(evidence_dir, f"guard-{label}-mem.log"),
                  "w", encoding="utf-8") as fh:
            fh.writelines(lines)


# ══ RUNNING THE REAL PRODUCERS ═══════════════════════════════════════════

_ENV_KEYS = ("LANE_URL", "RELAY_URL", "RELAY_KEY", "MODEL")


def _run_bench(script: str, evidence_dir: str, mode: str,
               prefill_tokens: int = 0, clock=None) -> dict:
    """Run the REAL bench against the fixture's engine, on the fixture clock."""
    engine = _Engine(mode)
    try:
        argv = [
            script,
            "--lane-url", f"http://127.0.0.1:{engine.port}",
            "--model", "minimax31-flash",
            "--tokens", str(_TOKENS),
            "--rounds", "1",
            "--workload", "prose",
            "--streams", "1,4",
            "--prefill-tokens", str(prefill_tokens),
            "--evidence-dir", evidence_dir,
        ]
        saved_argv = list(sys.argv)
        saved_env = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
        saved_clock = time.perf_counter
        out = io.StringIO()
        sys.argv = argv
        time.perf_counter = clock or _FixtureClock()
        code = 0
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    runpy.run_path(script, run_name="__main__")
                except SystemExit as exc:
                    code = 0 if exc.code is None else (
                        exc.code if isinstance(exc.code, int) else 1)
                except BaseException as exc:  # noqa: BLE001
                    # THE PRODUCER CRASHED. The battery RECORDS the crash
                    # instead of dying with it: a producer that raises has
                    # to be a RED case, not a dead battery.
                    code = 1
                    out.write(f"Traceback (most recent call last):\n{exc!r}\n")
        finally:
            sys.argv = saved_argv
            time.perf_counter = saved_clock
            for key, value in saved_env.items():
                if value is not None:
                    os.environ[key] = value
    finally:
        # THE ENGINE'S OWN COUNTS, read BEFORE its directory goes away.
        counts = engine.counts()
        engine.close()
    bench_path = os.path.join(evidence_dir, "bench.txt")
    text = ""
    if os.path.isfile(bench_path):
        with open(bench_path, encoding="utf-8") as fh:
            text = fh.read()
    return {"code": code, "output": out.getvalue(), "bench": text,
            "engine_counts": counts}


def _run_assembler(script: str, evidence_dir: str) -> dict:
    """Run the REAL assembler over the fixture's guard evidence."""
    proc = subprocess.run(
        [sys.executable, "-B", script, "--evidence-dir", evidence_dir,
         "--head", BOXES[0], "--r1", BOXES[1], "--r2", BOXES[2]],
        capture_output=True, text=True)
    guard_path = os.path.join(evidence_dir, "guard.txt")
    text = ""
    if os.path.isfile(guard_path):
        with open(guard_path, encoding="utf-8") as fh:
            text = fh.read()
    return {"code": proc.returncode, "output": proc.stdout + proc.stderr,
            "guard": text, "evidence_dir": evidence_dir}


# ══ READING THE PUBLISHED EVIDENCE ═══════════════════════════════════════

def _rows(bench_text: str) -> list[tuple[str, dict]]:
    rows = []
    for line in bench_text.splitlines():
        if ": {" not in line:
            continue
        label, _, js = line.partition(": {")
        try:
            row = json.loads("{" + js)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append((label.strip(), row))
    return rows


def _concurrency_row(bench_text: str) -> dict:
    for label, row in _rows(bench_text):
        if row.get("mode") == "concurrent x4" and row.get("path") == "lane":
            return row
    return {}


def _inter_token(bench_text: str) -> str:
    m = re.search(r"engine inter-token \(1-WAY window, per TOKEN\):([^\n]*)",
                  bench_text)
    return m.group(1).strip() if m else ""


def _honest_per_token(bench_text: str) -> float | None:
    """The per-token figure the bench's own measurement row carries."""
    for label, row in _rows(bench_text):
        if row.get("mode") == "stream" and row.get("path") == "lane":
            value = row.get("per_token_ms")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _guard_section(guard_text: str, box: str) -> str:
    """One box's section of the published guard aggregate."""
    m = re.search(rf"^=== {re.escape(box)} ===$(.*?)(?=^=== |\Z)",
                  guard_text, re.MULTILINE | re.DOTALL)
    return m.group(1) if m else ""


def _guard_key(section: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}=(.*)$", section, re.MULTILINE)
    return m.group(1).strip() if m else None


def _prefill_row(bench_text: str) -> dict:
    for label, row in _rows(bench_text):
        if row.get("kind") == "prefill":
            return row
    return {}


# THE MARKER IN THE PREVIOUS CYCLE'S AGGREGATES. Content that carries it is
# the PREVIOUS cycle's answer, and the rule -- which reads those three files
# as THIS cycle's evidence -- would judge it as this cycle's.
_STALE_MARKER = "PREVIOUS-CYCLE-AGGREGATE"


def _write_stale_aggregates(evidence_dir: str) -> None:
    """The previous cycle's answer, sitting where this cycle's goes."""
    for name in ("capture-gate.txt", "kit-check.txt", "guard.txt"):
        with open(os.path.join(evidence_dir, name), "w", encoding="utf-8") as fh:
            fh.write(f"{_STALE_MARKER}\n")


def _aggregates_left(evidence_dir: str) -> list[str]:
    """The aggregate files that are still in the evidence directory.

    A directory is not an aggregate: the rule cannot read one, and one was
    never this cycle's answer.
    """
    left = []
    for name in ("capture-gate.txt", "kit-check.txt", "guard.txt"):
        path = os.path.join(evidence_dir, name)
        if os.path.isdir(path):
            continue
        if os.path.exists(path):
            left.append(name)
    return left


def _stale_aggregates_left(evidence_dir: str) -> list[str]:
    """The previous cycle's answer that survived an exit."""
    left = []
    for name in ("capture-gate.txt", "kit-check.txt", "guard.txt"):
        path = os.path.join(evidence_dir, name)
        if os.path.isdir(path):
            # A directory was never a valid aggregate.
            continue
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            if _STALE_MARKER in fh.read():
                left.append(name)
    return left


# ══ THE CASES ═════════════════════════════════════════════════════════════

def _case_f1_minimum(ctx: dict) -> tuple[bool, str]:
    """A dead stream stays in the published per-stream minimum."""
    run = ctx["dead_run"]
    row = _concurrency_row(run["bench"])
    if not row:
        return False, "the bench published no 4-way lane concurrency row"
    minimum = row.get("min_stream_tok_s")
    if minimum != 0.0:
        return False, (f"min_stream_tok_s is {minimum!r} -- the stream that "
                       f"produced no tokens was dropped from the minimum "
                       f"instead of being reported as the rate it measured")
    return True, "min_stream_tok_s is 0.0 -- the dead stream is in the figure"


def _case_f1_dead_count(ctx: dict) -> tuple[bool, str]:
    """The count of streams that produced no rate is published."""
    row = _concurrency_row(ctx["dead_run"]["bench"])
    if "dead_streams" not in row:
        return False, ("the concurrency row publishes no count of the "
                       "streams that produced no usable rate")
    if row.get("dead_streams") != 1:
        return False, f"dead_streams is {row.get('dead_streams')!r}, expected 1"
    return True, "dead_streams is 1"


def _case_f1_bench_fails_closed(ctx: dict) -> tuple[bool, str]:
    """The bench itself calls a level with a dead stream broken."""
    run = ctx["dead_run"]
    if run["code"] == 0:
        return False, ("the bench exited 0 over a concurrent level with a "
                       "dead stream -- the broken measurement passed silently")
    if "no usable rate" not in run["output"]:
        return False, ("the bench failed, but not about the dead stream: "
                       + run["output"].strip().splitlines()[-1])
    return True, "the bench exited non-zero and named the dead stream"


def _case_f1_healthy_level(ctx: dict) -> tuple[bool, str]:
    """A healthy level publishes no dead streams and a real minimum."""
    row = _concurrency_row(ctx["healthy_run"]["bench"])
    if not row:
        return False, "the bench published no 4-way lane concurrency row"
    if row.get("dead_streams") != 0:
        return False, f"dead_streams is {row.get('dead_streams')!r}, expected 0"
    minimum = row.get("min_stream_tok_s")
    if not isinstance(minimum, (int, float)) or minimum <= 0:
        return False, f"min_stream_tok_s is {minimum!r} on a healthy level"
    return True, f"dead_streams is 0 and the minimum is {minimum!r}"


def _case_f2_exact_figure(ctx: dict) -> tuple[bool, str]:
    """The published inter-token figure is the figure the bench measured."""
    bench_text = ctx["healthy_run"]["bench"]
    written = _inter_token(bench_text)
    if not written:
        return False, "the bench published no per-token inter-token figure"
    honest = _honest_per_token(bench_text)
    if honest is None:
        return False, "the bench's own measurement row carries no per_token_ms"
    if not _WINDOW[0] < honest <= _WINDOW[1]:
        return False, (f"the fixture measured {honest!r}, which is not inside "
                       f"the rounding window {_WINDOW} -- the case would not "
                       f"be testing the rounding at all")
    try:
        read = float(written)
    except ValueError:
        return False, f"the published figure {written!r} is not a number"
    if read != honest:
        return False, (f"the published figure {written!r} reads as {read!r} but "
                       f"the measurement row carries {honest!r} -- the figure "
                       f"was rounded on the way out")
    return True, f"the published figure {written!r} IS the measured {honest!r}"


def _case_f2_ceiling_verdict(ctx: dict) -> tuple[bool, str]:
    """The honest figure does not clear the ceiling, and the published one
    does not either -- so the rule's verdict is the honest one."""
    bench_text = ctx["healthy_run"]["bench"]
    honest = _honest_per_token(bench_text)
    written = _inter_token(bench_text)
    if honest is None or not written:
        return False, "the bench published no inter-token figure to judge"
    if honest <= _CEILING:
        return False, (f"the fixture measured {honest!r}, which clears the "
                       f"ceiling {f'{_CEILING}'} -- the case would not be "
                       f"testing a figure that must fail")
    return True, (f"the measured figure {honest!r} is above the ceiling "
                  f"{_CEILING} and the published figure says so")


def _case_f3_not_laundered(ctx: dict) -> tuple[bool, str]:
    """A counter the rule cannot read is not published as a clean integer."""
    guard = ctx["damaged_run"]["guard"]
    if not guard:
        return False, "the assembler published no guard.txt"
    section = _guard_section(guard, BOXES[-1])
    if not section:
        return False, f"the published evidence has no {BOXES[-1]} section"
    written = _guard_key(section, "guard_window_breaches")
    if written is None:
        return False, "the assembler published no guard_window_breaches"
    if written.isdigit():
        return False, (f"guard_window_breaches is the clean integer {written!r} "
                       f"-- a counter the rule cannot read was laundered into a "
                       f"number the guard never wrote")
    return True, f"guard_window_breaches is {written!r} -- not a number"


def _case_f3_raw_sample_carried(ctx: dict) -> tuple[bool, str]:
    """The raw sample line is still carried: it is the second witness."""
    guard = ctx["damaged_run"]["guard"]
    if "breaches=3x" not in guard:
        return False, ("the raw sample line is not in the published evidence "
                       "-- the second witness is gone")
    return True, "the raw sample line is carried unchanged"


def _case_f3_damage_named(ctx: dict) -> tuple[bool, str]:
    """The damage is named in the evidence, not only implied by it."""
    guard = ctx["damaged_run"]["guard"]
    if "a counter the rule cannot read" not in guard:
        return False, "the published evidence does not name the counter it could not read"
    if "3x" not in guard:
        return False, "the published evidence does not name the value it could not read"
    return True, "the evidence names the counter and the value"


def _case_f3_wellformed_is_an_integer(ctx: dict) -> tuple[bool, str]:
    """A well-formed counter is still published as a plain integer."""
    guard = ctx["healthy_guard_run"]["guard"]
    for box in BOXES:
        section = _guard_section(guard, box)
        if not section:
            return False, f"the published evidence has no {box} section"
        written = _guard_key(section, "guard_window_breaches")
        if written is None:
            return False, f"the assembler published no guard_window_breaches for {box}"
        if written != "0":
            return False, f"guard_window_breaches is {written!r} for {box}, expected '0'"
        if "a counter the rule cannot read" in section:
            return False, f"a well-formed log produced a damage section for {box}"
    return True, "every box's guard_window_breaches is the integer 0"


def _case_f3_long_counter_is_unreadable(ctx: dict) -> tuple[bool, str]:
    """A five-thousand-digit counter is UNREADABLE, not a truncated integer.

    The run of digits is longer than CPython will convert (`int()` refuses
    more than 4,300 digits by default), so an unbounded read CRASHES the
    assembler -- and a crash leaves the previous cycle's aggregates in place
    for the rule to judge as this cycle's evidence. And a read that took the
    FIRST THREE DIGITS would publish 999, a figure the guard never wrote.
    """
    run = ctx.get("long_counter_run")
    if run is None:
        return False, "the five-thousand-digit counter run was not made"
    if "Traceback" in run["output"]:
        return False, ("the assembler RAISED on a five-thousand-digit counter "
                       "-- the counter read is not bounded")
    for box in BOXES:
        section = _guard_section(run["guard"], box)
        if not section:
            return False, f"the published evidence has no {box} section"
        written = _guard_key(section, "guard_window_breaches")
        if written is None:
            return False, f"the assembler published no guard_window_breaches for {box}"
        if written.isdigit():
            return False, (f"guard_window_breaches is the clean integer "
                           f"{written!r} for {box} -- the five-thousand-digit "
                           f"counter was laundered into a number")
    if run["code"] != 0:
        return False, (f"the assembler exited {run['code']} over a damaged "
                       f"counter: {run['output'].strip().splitlines()[-1:]}")
    return True, ("every box's counter is published as unreadable, with no "
                  "traceback and no truncated integer")


def _case_f4_prefill_is_the_measured_length(ctx: dict) -> tuple[bool, str]:
    """The published prefill length is the ENGINE'S count, not the budget.

    The bench DECLARES a budget and the engine MEASURES the prompt it was
    given. The published length has to be the engine's own count -- and the
    rule checks the two against each other, so publishing the declared
    budget as if it were the measured length would let a one-token prefill
    pass a criterion that says "a long-context prefill is measured".
    """
    run = ctx.get("prefill_run")
    if run is None:
        return False, "the prefill run was not made"
    row = _prefill_row(run["bench"])
    if not row:
        return False, "the bench published no prefill row"
    published = row.get("prompt_tokens")
    target = row.get("target_tokens")
    if not isinstance(published, int) or isinstance(published, bool):
        return False, f"the published prompt_tokens is {published!r}, not a number"
    if not isinstance(target, int) or isinstance(target, bool):
        return False, f"the published target_tokens is {target!r}, not a number"
    if published < target:
        return False, (f"the measured prompt length {published} is BELOW the "
                       f"declared budget {target} -- the measurement does not "
                       f"reach the budget it declares")
    if published == target:
        return False, (f"the published prompt length is exactly the declared "
                       f"budget {target} -- the case cannot tell a MEASURED "
                       f"length from the DECLARED one")
    counts = run.get("engine_counts") or []
    if not counts:
        return False, "the fixture engine recorded no prompt-token count"
    if published != counts[-1]:
        return False, (f"the published prompt length {published} is not the "
                       f"engine's own count {counts[-1]} -- the bench "
                       f"published a figure the engine did not measure")
    return True, (f"the published length {published} IS the engine's own "
                  f"count, and it reaches the declared budget {target}")


def _case_f4_prefill_fails_closed(ctx: dict) -> tuple[bool, str]:
    """A budget the engine cannot reach publishes NO prefill row.

    The engine's count is a constant in this run, so no amount of growth
    reaches the declared budget. The bench must then publish NO prefill row
    -- a short row would be a row the rule has to reject, and publishing it
    would leave the bench claiming a long-context prefill it did not
    measure.
    """
    run = ctx.get("prefill_unreachable_run")
    if run is None:
        return False, "the unreachable-budget run was not made"
    if _prefill_row(run["bench"]):
        return False, ("the bench published a prefill row even though the "
                       "engine's count never reached the declared budget")
    if run["code"] == 0:
        return False, ("the bench exited 0 although it could not measure the "
                       "long-context prefill it declared")
    if "never reached the declared floor" not in run["output"]:
        return False, ("the bench failed, but not about the unreachable "
                       "prefill budget: "
                       + str(run["output"].strip().splitlines()[-1:]))
    return True, ("the bench published no prefill row, exited non-zero and "
                  "named the unreachable budget")


def _case_f4_rejected_prompt(ctx: dict) -> tuple[bool, str]:
    """A REJECTED prefill prompt fails closed and costs the run nothing.

    The engine refusing the request used to raise out of the bench's prefill
    step and kill the whole run, so no `bench.txt` was written at all and
    the lane lost its decode rows, its histogram and its manifest to one
    rejected prefill. Now the refusal is caught, named, and reported as a
    failure of the prefill step -- and the rest of the run is still written.
    """
    run = ctx.get("rejected_prefill_run")
    if run is None:
        return False, "the rejected-prompt run was not made"
    problems = []
    if "Traceback" in run["output"]:
        problems.append("the bench RAISED on a rejected prefill prompt")
    if _prefill_row(run["bench"]):
        problems.append("the bench published a prefill row for a prompt the "
                        "engine refused")
    if not run["bench"].strip():
        problems.append("the bench wrote NO bench.txt -- the rejected prefill "
                        "took the whole run down with it")
    elif '"mode": "stream"' not in run["bench"]:
        problems.append("the bench wrote bench.txt without the decode rows -- "
                        "the rejected prefill cost the lane its measurement")
    if run["code"] == 0:
        problems.append("the bench exited 0 although the prefill was refused")
    if "REFUSED the prefill prompt" not in run["output"]:
        problems.append("the bench did not name the refusal: "
                        + str(run["output"].strip().splitlines()[-1:]))
    if problems:
        return False, "; ".join(problems)
    return True, ("the refusal is named, no prefill row is published, and "
                  "the decode rows are still published")


def _case_f5_stale_aggregates(ctx: dict) -> tuple[bool, str]:
    """No previous cycle's aggregate survives ANY exit of the assembler.

    The refusal is the exit the tool used to clean up on. The CRASH is the
    exit it did not: an unhandled exception left the previous cycle's
    aggregates in the evidence directory, and the rule -- which reads those
    three files as this cycle's evidence -- judged the STALE ones and
    reported PASS for a box whose evidence was not there.
    """
    problems = []
    for label, key in (("the refusal", "stale_refusal_run"),
                       ("the crash", "stale_crash_run")):
        run = ctx.get(key)
        if run is None:
            problems.append(f"the {label} run was not made")
            continue
        left = _aggregates_left(run["evidence_dir"])
        if left:
            stale = _stale_aggregates_left(run["evidence_dir"])
            problems.append(f"after {label}, the aggregate file(s) {left} are "
                            f"still there"
                            + (f" -- and {stale} still carry the previous "
                               f"cycle's answer" if stale else
                               " -- a half-written aggregate set is not this "
                               "cycle's evidence either")
                            + " -- the rule would judge them as this cycle's "
                              "evidence")
    if problems:
        return False, "; ".join(problems)
    return True, ("after the refusal and after the crash, no aggregate file "
                  "is left in the evidence directory")


def _case_f6_arming_line(ctx: dict) -> tuple[bool, str]:
    """The armed threshold is the one THIS cycle's window is computed against.

    A guard log with two arming lines is a log that was not rotated, and the
    rule fails it rather than guessing -- but the aggregate still has to
    publish the threshold the window was computed against, which is the LAST
    arming line's.
    """
    run = ctx.get("two_arming_run")
    if run is None:
        return False, "the two-arming-lines run was not made"
    for box in BOXES:
        section = _guard_section(run["guard"], box)
        if not section:
            return False, f"the published evidence has no {box} section"
        written = _guard_key(section, "armed_threshold_mb")
        if written != "4096":
            return False, (f"armed_threshold_mb is {written!r} for {box}, "
                           f"expected '4096' -- the threshold was taken from "
                           f"the wrong arming line")
    return True, ("every box's armed threshold is the one this cycle's "
                  "window is computed against (4096)")


def _case_normal_bench_unchanged(ctx: dict) -> tuple[bool, str]:
    """The well-formed measurement is unchanged, except for the two
    deliberate changes: the inter-token figure's precision, and the new
    count of the streams that produced no rate."""
    other = ctx.get("compare_bench_run")
    if other is None:
        return True, "skipped (no comparison producer given)"
    mine = _normalize(ctx["healthy_run"]["bench"]).splitlines()
    theirs = _normalize(other["bench"]).splitlines()
    if len(mine) != len(theirs):
        return False, (f"the published evidence has {len(mine)} line(s) and the "
                       f"comparison producer's has {len(theirs)}")
    deliberate = []
    problems = []
    for a, b in zip(mine, theirs):
        if a == b:
            continue
        if "engine inter-token (1-WAY window, per TOKEN):" in a:
            deliberate.append("the inter-token figure, now the exact value "
                              f"{_inter_token(ctx['healthy_run']['bench'])!r} "
                              f"instead of {b.rsplit(':', 1)[-1].strip()!r}")
            continue
        if _differ_only_by(a, b, "dead_streams"):
            deliberate.append("the concurrency row's new dead_streams count")
            continue
        problems.append((a, b))
    if problems:
        return False, (f"{len(problems)} line(s) changed that should not have: "
                       f"{problems[:2]}")
    if not deliberate:
        return True, "the published evidence is byte-identical"
    return True, ("the published evidence is byte-identical apart from: "
                  + "; ".join(deliberate))


def _case_normal_assembler_unchanged(ctx: dict) -> tuple[bool, str]:
    """The well-formed guard aggregate is byte-identical to the older
    producer's."""
    other = ctx.get("compare_assemble_run")
    if other is None:
        return True, "skipped (no comparison producer given)"
    if other["guard"] != ctx["healthy_guard_run"]["guard"]:
        mine = ctx["healthy_guard_run"]["guard"].splitlines()
        theirs = other["guard"].splitlines()
        differing = [(a, b) for a, b in zip(mine, theirs) if a != b]
        return False, (f"the well-formed guard aggregate changed "
                       f"({len(differing)} line(s) differ: {differing[:2]})")
    return True, "the published guard aggregate is byte-identical"


def _case_judge(ctx: dict) -> tuple[bool, str]:
    """The rule reads the published evidence and judges it the honest way.

    The evidence it reads is the one with a DEAD STREAM and with a
    per-token figure just above the ceiling -- so the honest verdicts are
    two FAILURES, and the rule has to reach them.
    """
    out = ctx.get("judge_output")
    if not out:
        return True, "skipped (the rule was not run)"
    problems = []
    readable = _judge_line(out, "every measurement row in bench.txt is readable")
    if "[FAIL]" in readable:
        problems.append("the rule rejected the bench row as unreadable")
    per_stream = _judge_line(out, "per-stream >=")
    if not per_stream:
        problems.append("the rule published no per-stream criterion")
    elif "[FAIL]" not in per_stream:
        problems.append("the rule PASSED the per-stream criterion")
    inter = _judge_line(out, "inter-token <=")
    if not inter:
        problems.append("the rule published no inter-token criterion")
    elif "[FAIL]" not in inter:
        problems.append("the rule PASSED the inter-token criterion")
    if problems:
        return False, "; ".join(problems)
    return True, ("the rule read the row and failed both: "
                  f"{per_stream.strip()} | {inter.strip()}")


def _judge_line(output: str, needle: str) -> str:
    for line in output.splitlines():
        if needle in line:
            return line
    return ""


# THE PUBLISHED ENDPOINT'S PORT is the fixture's own, so it differs between
# two runs of the same code. It is not part of any figure, so it is
# normalized out before two published evidence sets are compared.
_PORT_RE = re.compile(r"127\.0\.0\.1:\d+")


def _normalize(text: str) -> str:
    return _PORT_RE.sub("127.0.0.1:<port>", text)


def _differ_only_by(a: str, b: str, key: str) -> bool:
    """Two published lines that differ ONLY by one added field."""
    try:
        ja = json.loads("{" + a.partition(": {")[2])
        jb = json.loads("{" + b.partition(": {")[2])
    except Exception:
        return False
    if not isinstance(ja, dict) or not isinstance(jb, dict):
        return False
    if (key in ja) == (key in jb):
        return False
    ja = {k: v for k, v in ja.items() if k != key}
    jb = {k: v for k, v in jb.items() if k != key}
    return ja == jb


CASES = [
    ("f1_dead_stream_stays_in_the_minimum", _case_f1_minimum),
    ("f1_dead_streams_are_counted", _case_f1_dead_count),
    ("f1_bench_fails_closed_on_a_dead_stream", _case_f1_bench_fails_closed),
    ("f1_a_healthy_level_stays_healthy", _case_f1_healthy_level),
    ("f2_inter_token_is_published_exactly", _case_f2_exact_figure),
    ("f2_the_honest_figure_does_not_clear_the_ceiling", _case_f2_ceiling_verdict),
    ("f3_damaged_counter_is_not_laundered", _case_f3_not_laundered),
    ("f3_the_raw_sample_is_still_carried", _case_f3_raw_sample_carried),
    ("f3_the_damage_is_named", _case_f3_damage_named),
    ("f3_a_wellformed_counter_is_an_integer", _case_f3_wellformed_is_an_integer),
    ("f3_a_five_thousand_digit_counter_is_unreadable",
     _case_f3_long_counter_is_unreadable),
    ("f4_prefill_length_is_the_measured_one", _case_f4_prefill_is_the_measured_length),
    ("f4_an_unreachable_budget_publishes_no_prefill_row",
     _case_f4_prefill_fails_closed),
    ("f4_a_rejected_prompt_fails_closed_and_costs_the_run_nothing",
     _case_f4_rejected_prompt),
    ("f5_stale_aggregates_are_removed_on_every_exit", _case_f5_stale_aggregates),
    ("f6_the_threshold_comes_from_the_arming_line_this_cycle_uses",
     _case_f6_arming_line),
    ("normal_case_bench_output_is_unchanged", _case_normal_bench_unchanged),
    ("normal_case_assembler_output_is_unchanged", _case_normal_assembler_unchanged),
    ("the_rule_reads_it_the_honest_way", _case_judge),
]


# ══ THE MUTANTS ═══════════════════════════════════════════════════════════
#
# Each mutant puts a KNOWN DEFECT back into a copy of the fixed producer.
# A mutant that no case catches is a case that cannot tell the fixed code
# from the broken code -- and a test that cannot tell the two apart is not
# a test.
MUTANTS = [
    ("F1 restores the filter that drops a dead stream", "bench",
     'per_stream = list(conc.get("per_stream_tok_s") or [])',
     'per_stream = [t for t in (conc.get("per_stream_tok_s") or []) if t]',
     "f1_dead_stream_stays_in_the_minimum"),
    ("F2 restores the two-decimal rounding", "bench",
     'f"{min(per_token)!r}"', 'f"{min(per_token):.2f}"',
     "f2_inter_token_is_published_exactly"),
    ("F3 restores the prefix read of a counter", "assemble",
     r'SAMPLE_RE = re.compile(r"\b(low_mb|avail_mb|breaches)=(\S*)")',
     r'SAMPLE_RE = re.compile(r"\b(low_mb|avail_mb|breaches)=(\d+)")',
     "f3_damaged_counter_is_not_laundered"),
    # ---- the round-14 mutants (F20, F21) ---------------------------------
    ("B4 publishes the DECLARED budget as the MEASURED length", "bench",
     'prow = {"kind": "prefill", "path": "lane", **pf}',
     'prow = {"kind": "prefill", "path": "lane", **pf, '
     '"prompt_tokens": pf.get("target_tokens")}',
     "f4_prefill_length_is_the_measured_one"),
    ("A2 leaves the previous cycle's aggregates in place", "assemble",
     "    for name in AGGREGATE_FILES:",
     "    for name in ():",
     "f5_stale_aggregates_are_removed_on_every_exit"),
    ("A3 reads a counter by its first three digits", "assemble",
     "    if len(value) > MAX_COUNTER_DIGITS:\n"
     "        return None\n"
     "    try:\n"
     "        return int(value)\n"
     "    except ValueError:\n"
     "        return None\n",
     "    try:\n"
     "        return int(value[:3])\n"
     "    except ValueError:\n"
     "        return None\n",
     "f3_a_five_thousand_digit_counter_is_unreadable"),
    ("A4 takes the armed threshold from the FIRST arming line", "assemble",
     "        m = START_RE.search(starts[-1])",
     "        m = START_RE.search(starts[0])",
     "f6_the_threshold_comes_from_the_arming_line_this_cycle_uses"),
    # ---- this wave's own mutants -----------------------------------------
    ("B5 stops growing the prompt until the budget is reached", "bench",
     "if measured >= target_tokens:",
     "if measured >= 0:",
     "f4_prefill_length_is_the_measured_one"),
    ("B6 publishes a prefill row for a floor it could not reach", "bench",
     "        pf, pf_problem = bench_prefill(base, args.model, args.prefill_tokens)\n",
     "        pf, pf_problem = bench_prefill(base, args.model, args.prefill_tokens)\n"
     "        pf = pf or {\"prompt_tokens\": args.prefill_tokens,\n"
     "                  \"target_tokens\": args.prefill_tokens, \"total_s\": 0.0,\n"
     "                  \"prefill_tok_s\": 0.0}\n",
     "f4_an_unreachable_budget_publishes_no_prefill_row"),
    ("B7 the rejected prompt takes the whole run down (the refusal is not "
     "caught)", "bench",
     "        except urllib.error.HTTPError as exc:\n",
     "        except urllib.error.HTTPError as exc:\n"
     "            raise\n",
     "f4_a_rejected_prompt_fails_closed_and_costs_the_run_nothing"),
    # THE CRASH MUTANT. The counter read has TWO defences -- a length bound
    # BEFORE the parse, and a try/except around it -- and they are REDUNDANT
    # with each other: removing EITHER one alone still leaves the crash
    # impossible (the length bound returns None first; the try/except
    # catches the ValueError). So only the mutant that removes BOTH brings
    # the crash back, and that is the one this battery carries. (The two
    # single-defence mutants are unobservable, and that is a property of the
    # fix rather than a gap in the case.)
    ("A5 reads a counter with no bound and no guard (the crash comes back)",
     "assemble",
     "    if len(value) > MAX_COUNTER_DIGITS:\n"
     "        return None\n"
     "    try:\n"
     "        return int(value)\n"
     "    except ValueError:\n"
     "        return None\n",
     "    return int(value)\n",
     "f3_a_five_thousand_digit_counter_is_unreadable"),
    ("A6 leaves the aggregates in place when the assembler CRASHES", "assemble",
     "    finally:\n"
     "        # AND ON EVERY EXIT THAT IS NOT A SUCCESS. A crash halfway through\n"
     "        # the writes must not leave a half-written aggregate behind for the\n"
     "        # rule to judge, and a usage error must not leave the previous\n"
     "        # cycle's answer in place either.\n"
     "        if not wrote:\n"
     "            remove_stale_aggregates(ev_hint)\n",
     "    finally:\n        pass\n",
     "f5_stale_aggregates_are_removed_on_every_exit"),
]

# WHERE THE MUTANTS ARE BUILT FROM: the kit's own fixed producers.
MUTANT_SOURCE = {"bench": DEFAULT_BENCH, "assemble": DEFAULT_ASSEMBLE}


def _mutant_script(fixed: str, old: str, new: str, workdir: str) -> str | None:
    """A copy of the fixed producer with one defect put back."""
    with open(fixed, encoding="utf-8") as fh:
        text = fh.read()
    if text.count(old) != 1:
        return None
    path = os.path.join(workdir, os.path.basename(fixed))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(old, new))
    return path


# ══ MAIN ═════════════════════════════════════════════════════════════════

def _bench_context(bench: str, work: str) -> dict:
    """Every run of the BENCH the cases read."""
    ctx: dict = {}
    healthy_dir = os.path.join(work, "healthy")
    dead_dir = os.path.join(work, "dead")
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(dead_dir, exist_ok=True)
    ctx["healthy_run"] = _run_bench(bench, healthy_dir, "healthy")
    ctx["dead_run"] = _run_bench(bench, dead_dir, "one-dead")
    # THE EVIDENCE DIRECTORY of the dead run, so the judge can be run over
    # it: the rule's verdicts are read from what the bench actually wrote.
    ctx["dead_evidence_dir"] = dead_dir

    # THE PREFILL RUNS. The "count-tokens" engine counts the prompt it was
    # given, so a declared budget is REACHABLE; the "healthy" engine's count
    # is a constant, so it is NOT.
    prefill_dir = os.path.join(work, "prefill")
    unreachable_dir = os.path.join(work, "prefill-unreachable")
    os.makedirs(prefill_dir, exist_ok=True)
    os.makedirs(unreachable_dir, exist_ok=True)
    ctx["prefill_run"] = _run_bench(bench, prefill_dir, "count-tokens",
                                    prefill_tokens=_PREFILL_BUDGET,
                                    clock=_MonotonicClock())
    ctx["prefill_unreachable_run"] = _run_bench(
        bench, unreachable_dir, "healthy",
        prefill_tokens=_PREFILL_BUDGET, clock=_MonotonicClock())

    # THE ENGINE REFUSES THE PREFILL PROMPT.
    rejected_dir = os.path.join(work, "prefill-rejected")
    os.makedirs(rejected_dir, exist_ok=True)
    ctx["rejected_prefill_run"] = _run_bench(
        bench, rejected_dir, "reject-prefill",
        prefill_tokens=_PREFILL_BUDGET, clock=_MonotonicClock())
    return ctx


def _assembler_context(assemble: str, work: str) -> dict:
    """Every run of the ASSEMBLER the cases read."""
    ctx: dict = {}
    good_ev = os.path.join(work, "guard-good")
    bad_ev = os.path.join(work, "guard-bad")
    os.makedirs(good_ev, exist_ok=True)
    os.makedirs(bad_ev, exist_ok=True)
    _write_guard_inputs(good_ev, damaged=False)
    _write_guard_inputs(bad_ev, damaged=True)
    ctx["healthy_guard_run"] = _run_assembler(assemble, good_ev)
    ctx["damaged_run"] = _run_assembler(assemble, bad_ev)

    # THE FIVE-THOUSAND-DIGIT COUNTER.
    long_ev = os.path.join(work, "guard-long")
    os.makedirs(long_ev, exist_ok=True)
    _write_guard_inputs(long_ev, damaged=False, breach_value="9" * 5000)
    ctx["long_counter_run"] = _run_assembler(assemble, long_ev)

    # TWO ARMING LINES, WITH DIFFERENT FLOORS.
    two_ev = os.path.join(work, "guard-two-armings")
    os.makedirs(two_ev, exist_ok=True)
    _write_guard_inputs(two_ev, damaged=False, armings=(
        "guard start min=2048MB debounce=3 container=minimax31-flash-tp3 pid=1",
        "guard start min=4096MB debounce=3 container=minimax31-flash-tp3 pid=1"))
    ctx["two_arming_run"] = _run_assembler(assemble, two_ev)

    # THE STALE AGGREGATES: the previous cycle's answer, and the two exits
    # that must not leave it in place.
    refusal_ev = os.path.join(work, "stale-refusal")
    os.makedirs(refusal_ev, exist_ok=True)
    # ... and the per-box evidence of only TWO boxes, so the assembler
    # refuses: it has to refuse, and the refusal must not leave the old
    # answer behind.
    _write_guard_inputs(refusal_ev, damaged=False)
    for label in BOXES[2:]:
        for name in (f"capture-gate-{label}.txt", f"guard-{label}.log",
                     f"guard-{label}-mem.log"):
            os.remove(os.path.join(refusal_ev, name))
    _write_stale_aggregates(refusal_ev)
    ctx["stale_refusal_run"] = _run_assembler(assemble, refusal_ev)

    crash_ev = os.path.join(work, "stale-crash")
    os.makedirs(crash_ev, exist_ok=True)
    _write_guard_inputs(crash_ev, damaged=False)
    _write_stale_aggregates(crash_ev)
    # A DIRECTORY named kit-check.txt: the removal cannot delete it, and the
    # write of that aggregate fails -- so the assembler CRASHES halfway
    # through writing, with the previous cycle's guard.txt still in place.
    os.remove(os.path.join(crash_ev, "kit-check.txt"))
    os.mkdir(os.path.join(crash_ev, "kit-check.txt"))
    ctx["stale_crash_run"] = _run_assembler(assemble, crash_ev)
    return ctx


def _build_context(args) -> dict:
    work = tempfile.mkdtemp(prefix="bench-battery-")
    ctx = {"work": work}
    ctx.update(_bench_context(args.bench, work))
    ctx.update(_assembler_context(args.assemble, work))

    if args.compare_bench:
        other_dir = os.path.join(work, "healthy-compare")
        os.makedirs(other_dir)
        ctx["compare_bench_run"] = _run_bench(args.compare_bench, other_dir, "healthy")
    if args.compare_assemble:
        other_ev = os.path.join(work, "guard-good-compare")
        os.makedirs(other_ev)
        _write_guard_inputs(other_ev, damaged=False)
        ctx["compare_assemble_run"] = _run_assembler(args.compare_assemble, other_ev)

    if args.with_judge and os.path.isfile(args.judge):
        # The rule is run over the evidence with the DEAD STREAM and with a
        # per-token figure just above the ceiling: the two honest verdicts
        # are failures, and the rule has to reach both of them.
        proc = subprocess.run(
            [sys.executable, "-B", args.judge, "--evidence-dir",
             ctx["dead_evidence_dir"]],
            capture_output=True, text=True)
        ctx["judge_output"] = proc.stdout + proc.stderr
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=DEFAULT_BENCH,
                    help="the bench-decode.py to exercise")
    ap.add_argument("--assemble", default=DEFAULT_ASSEMBLE,
                    help="the assemble-evidence.py to exercise")
    ap.add_argument("--compare-bench", default="",
                    help="a second bench to compare the published bytes with "
                         "(the normal-case check)")
    ap.add_argument("--compare-assemble", default="",
                    help="a second assembler to compare the published bytes with")
    ap.add_argument("--judge", default=DEFAULT_JUDGE,
                    help="the acceptance rule to run over the published evidence")
    ap.add_argument("--with-judge", action="store_true",
                    help="also run the acceptance rule over the published "
                         "evidence and check its verdicts")
    ap.add_argument("--keep", action="store_true",
                    help="keep the fixture's working directory")
    args = ap.parse_args()

    ctx = _build_context(args)
    work = ctx["work"]
    failures = []
    try:
        print("_bench_producer_battery: the cases")
        for name, func in CASES:
            try:
                ok, detail = func(ctx)
            except Exception as exc:  # noqa: BLE001 - a case that raises is a failure
                ok, detail = False, f"the case raised {exc!r}"
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
            if not ok:
                failures.append(name)

        print("\n_bench_producer_battery: the mutants")
        for index, (label, which, old, new, case_name) in enumerate(MUTANTS):
            # THE MUTANT IS BUILT FROM THE KIT'S OWN FIXED PRODUCER, never
            # from the producer this run is exercising: a mutant of the
            # pre-fix code is not a mutant at all, it is the pre-fix code.
            # The point of a mutant is that the case is GREEN on the fixed
            # bytes and RED on the mutant, and the fixed bytes are the
            # kit's own.
            fixed = MUTANT_SOURCE[which]
            mutant = _mutant_script(fixed, old, new, work)
            if mutant is None:
                print(f"  [FAIL] {label} -- the fixed producer does not carry "
                      f"the text the mutant replaces, so the mutant could not "
                      f"be built")
                failures.append(f"mutant:{label}")
                continue
            # The mutant's own context: only the producer under mutation is
            # re-run, and it runs in its OWN directory (a directory left
            # behind by an earlier mutant is not this mutant's evidence).
            sub = {"work": work}
            if which == "bench":
                sub.update(_bench_context(
                    mutant, os.path.join(work, f"mut-bench-{index}")))
            else:
                sub.update(_assembler_context(
                    mutant, os.path.join(work, f"mut-assemble-{index}")))
            func = dict(CASES)[case_name]
            try:
                ok, detail = func(sub)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"the case raised {exc!r}"
            if ok:
                print(f"  [FAIL] {label} -- {case_name} is still GREEN on the "
                      f"mutant, so it cannot see this defect")
                failures.append(f"mutant:{label}")
            else:
                print(f"  [PASS] {label} -- caught by {case_name}: {detail}")

        print()
        if failures:
            print(f"_bench_producer_battery: FAIL -- {len(failures)} case(s) "
                  f"failed: {', '.join(failures)}")
            return 1
        print("_bench_producer_battery: PASS -- every case is green and every "
              "mutant is caught.")
        return 0
    finally:
        if args.keep:
            print(f"_bench_producer_battery: the fixture's working directory is "
                  f"{work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
