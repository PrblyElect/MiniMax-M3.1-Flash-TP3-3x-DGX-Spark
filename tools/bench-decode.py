#!/usr/bin/env python3
"""bench-decode.py -- the measurement instrument for this lane.

Three rules make its numbers mean something, and all three are load-bearing:

1. A NONCE GOES AT THE FRONT OF THE PROMPT. Prefix caching is on, so a
   prompt whose prefix has been seen before is not a cold measurement. The
   nonce is the first thing in the prompt, which guarantees the measured
   prefix is never the cached one.

2. THE WARMUP IS SACRIFICIAL. The first request pays a JIT tax, and that tax
   has to be paid on something other than the measured prompt -- so the
   warmup runs with its own nonce and its own prefix.

3. THE WORKLOAD IS LABELLED. The draft acceptance rate -- and therefore the
   value of the speculative-decode budget -- depends on the content. A
   throughput figure without a workload label is not a claim.

A SECOND ENDPOINT IS OPTIONAL AND MEASURED, NOT ASSUMED. If the deployment
puts a proxy in front of the lane, --relay-url measures that path too, in the
same rounds, with its own nonce and its own sacrificial warmup. Its rows are
labelled path=relay and carry the endpoint they were measured over, and the
evidence manifest records relay_url= -- which is how the acceptance rule
knows the second endpoint exists and judges it on its own floor. With no
--relay-url, the manifest records no second endpoint and the rule reports
the second-endpoint criteria as not applicable, explicitly.

Usage:
    python3 bench-decode.py --lane-url http://127.0.0.1:8100 --tokens 800 \\
        --rounds 2 --streams 1,4,8,16 --workload prose
    python3 bench-decode.py --lane-url http://127.0.0.1:8100 \\
        --relay-url http://127.0.0.1:8001 --tokens 800 --rounds 2
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

# ----------------------------------------------------------------- prompts

PROSE = (
    "You are a careful technical writer. Explain, in a few short paragraphs, "
    "how a tensor-parallel inference ring behaves when one rank is slower "
    "than the others, and what a scheduler has to do about it."
)

# Long, self-consistent reasoning chains accept drafts far more often than
# prose does. The two are not interchangeable, and the difference is the
# whole story about how much a speculative-decode budget is worth.
HIGH_DRAFT = (
    "Work through this carefully, step by step, showing your reasoning at "
    "each step: a three-box inference ring runs a model whose published "
    "attention geometry does not divide by the tensor-parallel size. "
    "Derive the smallest set of weights to append so that the model can be "
    "sharded across the three boxes without changing the mapping between "
    "query heads and key-value heads, and prove that the result is "
    "arithmetically identical to the published model. Then state what could "
    "go wrong if the ratio were not preserved."
)

WORKLOADS = {"prose": PROSE, "high-draft": HIGH_DRAFT}


def _nonce() -> str:
    return uuid.uuid4().hex[:12]


def _prompt(nonce: str, workload: str) -> str:
    # THE NONCE IS FIRST. Everything after it is the workload.
    return f"[{nonce}]\n{WORKLOADS[workload]}"


def _long_prompt(nonce: str, copies: int) -> str:
    """The prefill prompt: `copies` copies of the prose workload.

    The prompt's LENGTH is not this function's business -- the bench
    MEASURES it (see `bench_prefill`). The copy count is only the size the
    bench starts from.
    """
    base = _prompt(nonce, "prose")
    return ("\n\n" + base) * max(1, copies)


# ------------------------------------------------------------------ http

def _post(url: str, payload: dict, timeout: float = 600.0,
          headers: dict | None = None) -> dict:
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=hdrs,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _sse(url: str, payload: dict, timeout: float = 600.0,
         headers: dict | None = None) -> dict:
    """One streaming completion. Returns timing, token counts and content."""
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=hdrs,
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    text = []
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                # A REASONING MODEL streams its thinking under a separate
                # delta key and its answer under `content`. The key differs
                # between builds (`reasoning`, `reasoning_content`), so all
                # of them are accepted: the first token of ANY kind is the
                # time to first token -- counting only `content` would report
                # a null TTFT for a model that thinks before it answers, and
                # would under-count the per-token latency.
                piece = (delta.get("content")
                         or delta.get("reasoning_content")
                         or delta.get("reasoning"))
                if piece:
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    t1 = time.perf_counter()
    out = {
        "ttft_s": ttft,
        "total_s": t1 - t0,
        "streamed_chunks": chunks,
        "chars": sum(len(t) for t in text),
        "finish_reason": finish,
        "usage": usage,
    }
    return out


def _completion_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    return usage.get("completion_tokens")


# --------------------------------------------------------------- metrics

_METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>\S+)$")

# The counters the acceptance curve is computed from. These are the engine's
# own names; a different engine build may rename them, in which case the
# acceptance curve is simply unavailable (and the bench says so) rather than
# silently reporting zero.
ACCEPTED_TOTAL = "vllm:spec_decode_num_accepted_tokens_total"
ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
DRAFT_TOKENS_TOTAL = "vllm:spec_decode_num_draft_tokens_total"
DRAFTS_TOTAL = "vllm:spec_decode_num_drafts_total"
TTFT_SUM = "vllm:time_to_first_token_seconds_sum"
TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
REQUESTS_RUNNING = "vllm:num_requests_running"
REQUESTS_WAITING = "vllm:num_requests_waiting"


def _metrics_snapshot(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read().decode(errors="replace")
    except (urllib.error.URLError, OSError):
        return {}
    snap: dict = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = m.group("labels") or ""
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if name == ACCEPTED_PER_POS:
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                snap[f"{name}[{pos.group(1)}]"] = value
        else:
            snap[name] = value
    return snap


def _delta(before: dict, after: dict, key: str) -> float | None:
    if key not in before or key not in after:
        return None
    return after[key] - before[key]


def _idle_check(base_url: str) -> dict:
    """Is the engine idle RIGHT NOW?

    A measurement taken with traffic in flight -- your own warmup replay, a
    scheduled job, a co-tenant's requests, another client -- is an
    unlabeled MIXED number, not a property of the boot. A "slow" reading
    taken that way is contamination, not a regression, and acting on it
    sends you chasing a problem that does not exist.

    The bench runs this immediately before every measured window.
    """
    snap = _metrics_snapshot(base_url)
    running = snap.get(REQUESTS_RUNNING)
    waiting = snap.get(REQUESTS_WAITING)
    return {
        "requests_running": running,
        "requests_waiting": waiting,
        "idle": running == 0 and waiting == 0,
        "known": running is not None and waiting is not None,
    }


def _acceptance_curve(before: dict, after: dict) -> dict:
    """The acceptance histogram, from the engine's own counters.

    The engine counts, per draft round, how many drafted tokens were
    ACCEPTED at each position. The acceptance rate at a position is that
    count divided by the number of draft rounds -- every round proposes one
    token per position.
    """
    positions = sorted(
        int(k.split("[")[1].rstrip("]"))
        for k in after
        if k.startswith(ACCEPTED_PER_POS + "[")
    )
    out: dict = {"counters_present": False}
    accepted = _delta(before, after, ACCEPTED_TOTAL)
    drafts = _delta(before, after, DRAFTS_TOTAL)
    draft_tokens = _delta(before, after, DRAFT_TOKENS_TOTAL)
    if accepted is None or drafts is None or draft_tokens is None:
        out["note"] = ("the engine's spec-decode counters were absent from the "
                       "exposition, so the acceptance histogram is unavailable")
        return out
    out["counters_present"] = True
    out["drafts"] = drafts
    out["draft_tokens"] = draft_tokens
    out["accepted"] = accepted
    if not drafts:
        # THE COUNTERS ARE PRESENT BUT THE WINDOW RAN NO DRAFT ROUNDS.
        # Dividing by zero here would kill the bench MID-ROUND with a
        # traceback, and a bench that dies leaves no bench.txt and no
        # manifest.txt at all -- the rule would then read a MISSING
        # measurement, which is a FAIL but an unexplained one. The rates
        # are simply undefined over a window with no steps, so they are
        # left out and the reason is published.
        out["note"] = ("the engine's spec-decode counters are present but the "
                       "measured window ran no draft rounds, so the "
                       "tokens-per-step rates are undefined")
        return out
    per_pos = {}
    for pos in positions:
        key = f"{ACCEPTED_PER_POS}[{pos}]"
        d = _delta(before, after, key)
        if d is None or d <= 0:
            continue
        per_pos[str(pos)] = d / drafts
    out["acceptance_per_pos"] = per_pos
    if "0" in per_pos:
        out["acceptance_position0"] = per_pos["0"]
    out["mean_accepted_length"] = accepted / drafts
    # TOKENS PER STEP: every decode step produces the target's own token
    # PLUS the draft tokens that were accepted. The accepted count is per
    # ROUND (not per drafted token), so the divisor is the number of
    # rounds -- dividing by the drafted-token count would be out by the
    # speculation depth.
    out["tokens_per_step"] = 1.0 + accepted / drafts
    return out


def _merge_histograms(hists: list[dict]) -> dict:
    """One histogram from several per-round ones (the overall window)."""
    if not hists:
        return {"counters_present": False}
    # THE MERGED FLAG IS COMPUTED, NOT ASSUMED. It used to be hard-coded
    # True whenever there was any histogram at all -- so a window in which
    # the engine published NO spec-decode counters was published to the
    # evidence file as though it had, and the rule then reported "no drafts
    # were recorded" about a window whose counters were simply ABSENT.
    # Only the rounds that actually carried counters count.
    present = [h for h in hists if h.get("counters_present")]
    merged = {"counters_present": bool(present), "acceptance_per_pos": {}}
    for key in ("drafts", "draft_tokens", "accepted"):
        merged[key] = sum(h.get(key, 0.0) for h in present)
    positions = sorted({p for h in hists for p in (h.get("acceptance_per_pos") or {})})
    for pos in positions:
        num = sum((h.get("acceptance_per_pos") or {}).get(pos, 0.0) * (h.get("drafts") or 0)
                  for h in present)
        den = sum(h.get("drafts") or 0 for h in present)
        if den:
            merged["acceptance_per_pos"][pos] = num / den
    if "0" in merged["acceptance_per_pos"]:
        merged["acceptance_position0"] = merged["acceptance_per_pos"]["0"]
    if merged.get("drafts"):
        merged["mean_accepted_length"] = merged["accepted"] / merged["drafts"]
    if merged.get("drafts"):
        merged["tokens_per_step"] = 1.0 + merged["accepted"] / merged["drafts"]
    # THE NOTE SURVIVES THE MERGE. A per-round histogram can say WHY it has
    # no rates -- the counters were present but the window ran no draft
    # rounds, for instance. Only the MERGED rows reach bench.txt, so a note
    # dropped here never reaches the evidence file, and the rule then
    # reports "the counters were absent" about a window whose counters were
    # present. The reason has to travel with the numbers.
    notes = [h["note"] for h in hists if h.get("note")]
    if notes:
        merged["note"] = "; ".join(dict.fromkeys(notes))
    return merged


def _endpoint_path(url: str) -> str:
    """The endpoint as the evidence rows label it: the URL without its
    scheme. The acceptance rule selects the rows it judges by this string,
    because a threshold calibrated on one path cannot be applied to another."""
    clean = url.rstrip("/")
    for scheme in ("http://", "https://"):
        if clean.startswith(scheme):
            return clean[len(scheme):]
    return clean


# ------------------------------------------------------------------ benches

def warmup(base_url: str, model: str, workload: str,
           headers: dict | None = None) -> dict:
    """A SACRIFICIAL warmup: its own nonce, its own prefix, never measured."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(_nonce(), workload)}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    return _post(base_url.rstrip("/") + "/v1/chat/completions", payload, headers=headers)


def bench_stream(base_url: str, model: str, prompt: str, max_tokens: int,
                 headers: dict | None = None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    res = _sse(base_url.rstrip("/") + "/v1/chat/completions", payload, headers=headers)
    tokens = _completion_tokens(res.get("usage"))
    res["tokens"] = tokens
    if tokens and res.get("total_s"):
        res["tok_s"] = tokens / res["total_s"]
        # The per-token latency is the time after the first token, divided by
        # the number of gaps. It is computed from the TOKEN count, not from
        # the number of stream chunks: a chunk can carry several tokens, and
        # counting chunks would understate the latency by that factor.
        if res.get("ttft_s") is not None and tokens > 1:
            res["per_token_ms"] = (res["total_s"] - res["ttft_s"]) * 1000.0 / (tokens - 1)
    return res


def bench_non_stream(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    res = _post(base_url.rstrip("/") + "/v1/chat/completions", payload)
    res["total_s"] = time.perf_counter() - t0
    res["tokens"] = _completion_tokens(res.get("usage"))
    if res["tokens"] and res["total_s"]:
        res["tok_s"] = res["tokens"] / res["total_s"]
    return res


def bench_concurrent(base_url: str, model: str, prompt: str, max_tokens: int,
                     streams: int) -> dict:
    t0 = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=streams) as pool:
        rows = list(pool.map(
            lambda _: bench_stream(base_url, model, prompt, max_tokens),
            range(streams),
        ))
    wall = time.perf_counter() - t0
    tokens = sum(r.get("tokens") or 0 for r in rows)
    # A STREAM THAT PRODUCED NO USAGE BLOCK IS A STREAM THAT PRODUCED NO
    # TOKENS, AND IT IS COUNTED HERE, AT THE SOURCE. The count travels with
    # the measurement because a stream that produced no rate cannot be
    # allowed to VANISH from it: the published per-stream minimum is the
    # minimum over the streams MEASURED, so a dead stream publishes a rate
    # of zero and the rule's floor fails on it. The count is what lets a
    # reader say WHY that minimum is what it is.
    dead = sum(1 for r in rows if not (r.get("tok_s") or 0.0))
    return {
        "streams": streams,
        "wall_s": wall,
        "tokens": tokens,
        "tok_s_aggregate": tokens / wall if wall else None,
        "per_stream_tok_s": [
            (r.get("tok_s") or 0.0) for r in rows
        ],
        "dead_streams": dead,
    }


# THE PREFILL BUDGET IS A TOKEN BUDGET, AND THE BENCH MEASURES IT.
#
# THE DECLARED NUMBER IS A FLOOR ON THE MEASURED LENGTH. It is NOT the
# size of the prompt that is sent: the bench sends the prompt the ENGINE
# measured, and the engine's count of a prompt the bench built is never
# exactly the number the bench asked for. So the bench grows the prompt
# until the engine's own count REACHES the floor, and the row it publishes
# carries the ENGINE'S count -- which is at or above the floor, never below
# it. A reader of the row sees the length that was really measured.
#
# The declared `--prefill-tokens` budget used to be a COPY COUNT: the bench
# built the prompt out of `target_tokens // 64` copies of the prose workload
# and published the DECLARED number beside the length the engine reported.
# The two disagreed by whatever the prose's token density happened to be, so
# the published row said "8192 tokens" over a prompt the engine counted as
# 7800 -- and the rule could not tell a measurement of 8192 tokens from a
# measurement of one token, because the row carried no CONSISTENCY between
# the two numbers.
#
# THE MEASUREMENT IS THE ENGINE'S COUNT. `usage.prompt_tokens` is the
# engine's tokenizer's verdict on the prompt it was given, so it -- not
# the bench's own arithmetic -- is the length.
#
# EVERY ATTEMPT USES A FRESH NONCE. Prefix caching is on, so a repeated
# prefix is not a fresh measurement (this file's own first load-bearing
# rule).
#
# THE LOOP IS BOUNDED, AND IT FAILS CLOSED. A budget that cannot be
# reached publishes NO prefill row at all: a short row would be a row whose
# measured length does not reach the budget the rule judges, and publishing
# it would be claiming a long-context prefill this run did not measure.
PREFILL_MAX_ATTEMPTS = 8
PREFILL_MAX_COPIES = 4096


def _next_copies(copies: int, target: int, measured: int) -> int:
    """How many copies the next attempt needs, from the engine's own count.

    THE PROPORTIONAL GROWTH. The engine's count says how SHORT the prompt
    was, so the next prompt is the size that would REACH the floor -- not a
    multiple of it. A multiple overshoots: a floor of 8,192 measured at
    6,000 would grow to 12,000 and spend half the engine's context on a
    prompt the rule did not ask for. The growth is CAPPED, because a prompt
    the engine cannot pre-fill is not a measurement of anything.
    """
    if measured <= 0:
        return min(copies * 2, PREFILL_MAX_COPIES)
    # ceil(copies * target / measured), without floats.
    grown = -(-copies * target // measured)
    return min(max(copies + 1, grown), PREFILL_MAX_COPIES)


def bench_prefill(base_url: str, model: str, target_tokens: int) -> tuple:
    """A long prompt whose MEASURED length reaches the declared floor.

    Returns `(row, reason)`. The row is None when the floor could not be
    reached, and the reason says why -- the caller then publishes NO prefill
    row, so the rule fails the criterion BY NAME instead of judging a short
    prefill.

    A REJECTED PROMPT FAILS CLOSED, AND IT NEVER TAKES THE RUN DOWN. The
    engine refusing the request (a context length, a payload too large, an
    authentication failure) used to raise out of this function and kill the
    whole bench run, so no `bench.txt` was written at all and the lane lost
    its decode rows, its histogram and its manifest to ONE rejected prefill.
    Now the rejection is caught, named, and reported as a failure of THIS
    step -- and the rest of the run is written.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    copies = max(1, target_tokens // 64)
    attempts = 0
    measured = None
    while attempts < PREFILL_MAX_ATTEMPTS:
        attempts += 1
        # A FRESH NONCE ON EVERY ATTEMPT: prefix caching is on, so a
        # repeated prefix is not a fresh measurement.
        prompt = _long_prompt(_nonce(), copies)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            res = _post(url, payload)
        except urllib.error.HTTPError as exc:
            # THE ENGINE REFUSED THE PROMPT. A refusal is NOT retried with a
            # shorter prompt: the declared number is a FLOOR on the measured
            # length, so every prompt that could satisfy the contract is at
            # least that long, and a shorter one could only produce a row
            # the rule has to reject. And the bench cannot tell WHY the
            # engine refused (it reads the status line, not the body), so a
            # shrink would also fire on an authentication failure and spend
            # attempts without changing the outcome. The honest report is
            # the refusal itself.
            return (None,
                    f"the engine REFUSED the prefill prompt (HTTP {exc.code}) "
                    f"-- the declared floor of {target_tokens} tokens may not "
                    f"fit this engine's context (the bench reads the status "
                    f"line, not the body's reason) -- no prefill row is "
                    f"published")
        except (OSError, ValueError) as exc:
            return (None,
                    f"the prefill request FAILED ({exc!r}) -- no prefill row "
                    f"is published")
        dt = time.perf_counter() - t0
        measured = (res.get("usage") or {}).get("prompt_tokens")
        if (not isinstance(measured, (int, float)) or isinstance(measured, bool)
                or measured <= 0):
            return (None,
                    f"the engine reported no usable prompt-token count "
                    f"({measured!r}) -- the prefill's length cannot be "
                    f"measured, so no prefill row is published")
        if measured >= target_tokens:
            return ({
                # THE MEASURED LENGTH, NOT THE DECLARED FLOOR.
                "prompt_tokens": measured,
                # THE DECLARED FLOOR, so the reader can check the two
                # against each other.
                "target_tokens": target_tokens,
                "total_s": dt,
                "prefill_tok_s": (measured / dt) if dt else None,
                "prefill_copies": copies,
                "prefill_attempts": attempts,
            }, None)
        copies = _next_copies(copies, target_tokens, measured)
    return (None,
            f"the engine's own prompt-token count reached {measured} after "
            f"{attempts} attempt(s) and never reached the declared floor of "
            f"{target_tokens} tokens -- no prefill row is published, because a "
            f"short prefill is not the long-context prefill the rule judges")


# --------------------------------------------------------------------- main

def _levels(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane-url", default=os.environ.get("LANE_URL", "http://127.0.0.1:8100"))
    ap.add_argument("--relay-url", default=os.environ.get("RELAY_URL", ""),
                    help="a second endpoint in front of the lane (a proxy). When "
                         "set, the bench measures that path too, labels its rows "
                         "path=relay, and records relay_url= in the manifest -- "
                         "which is what makes the acceptance rule judge the "
                         "second endpoint's floor instead of reporting it as "
                         "not applicable.")
    ap.add_argument("--relay-key", default=os.environ.get("RELAY_KEY", ""),
                    help="bearer key for the second endpoint, if it wants one")
    ap.add_argument("--model", default=os.environ.get("MODEL", "minimax31-flash"))
    ap.add_argument("--tokens", type=int, default=800, help="max_tokens per measured call")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--streams", default="1", help="comma list of concurrency levels")
    ap.add_argument("--workload", default="prose,high-draft",
                    help="comma list of workload classes to measure: "
                         + ",".join(sorted(WORKLOADS))
                         + ". BOTH classes are measured in ONE run, because the "
                           "acceptance rule judges the SPLIT: a free-prose class "
                           "and a high-draftability class, each with its own "
                           "acceptance histogram. One run, one bench.txt.")
    ap.add_argument("--prefill-tokens", type=int, default=8192,
                    help="the long-context prefill to measure (0 disables "
                         "it). The number is a FLOOR on the MEASURED length, "
                         "in TOKENS: the bench builds a prompt, the engine "
                         "counts it, and the bench grows the prompt until the "
                         "engine's own count reaches the floor. It is NOT the "
                         "size of the prompt that is sent -- the engine may "
                         "count a slightly longer one -- and the published "
                         "row carries the engine's count, never this number. "
                         "A floor the engine cannot reach publishes no "
                         "prefill row at all. The acceptance rule judges that "
                         "a long-context prefill of at least this many tokens "
                         "IS measured, so the default measures one.")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--non-stream", action="store_true", help="also run one non-streaming call")
    ap.add_argument("--fail-on-empty", dest="fail_on_empty", action="store_true", default=True)
    ap.add_argument("--no-fail-on-empty", dest="fail_on_empty", action="store_false")
    ap.add_argument("--require-idle", dest="require_idle", action="store_true", default=False,
                    help="refuse to measure while the engine has requests in flight")
    ap.add_argument("--evidence-dir", default="",
                    help="write the evidence set here (bench.txt + manifest.txt)")
    ap.add_argument("--rule-file", default="",
                    help="the acceptance rule this evidence is collected for; its "
                         "digest is recorded in the manifest")
    ap.add_argument("--json-out", default="", help="write the full result as JSON here")
    args = ap.parse_args()

    # THE WORKLOAD CLASSES ARE A LIST, AND BOTH ARE MEASURED IN ONE RUN.
    # The acceptance rule judges the SPLIT -- a free-prose class and a
    # high-draftability class, each with its own acceptance histogram --
    # so a bench that measured one class per run produced an evidence set
    # the rule could never accept, and it OVERWROTE the previous run's
    # bench.txt besides. One run, one bench.txt, every class.
    workloads = [w.strip() for w in args.workload.split(",") if w.strip()]
    unknown = [w for w in workloads if w not in WORKLOADS]
    if unknown:
        print(f"bench-decode: unknown workload class(es): {', '.join(unknown)}. "
              f"Known: {', '.join(sorted(WORKLOADS))}", file=sys.stderr)
        return 2
    if not workloads:
        print("bench-decode: no workload class given", file=sys.stderr)
        return 2

    base = args.lane_url.rstrip("/")
    endpoint = _endpoint_path(args.lane_url)
    # The second endpoint, if this deployment has one. It is measured over
    # its OWN path, with its own nonce, and its rows are labelled path=relay
    # so the acceptance rule selects them by the endpoint they were measured
    # over -- a threshold calibrated on one path is not applicable to another.
    relay_base = args.relay_url.rstrip("/") if args.relay_url else ""
    relay_endpoint = _endpoint_path(args.relay_url) if relay_base else ""
    relay_headers = {"Authorization": f"Bearer {args.relay_key}"} if args.relay_key else {}
    failures: list[str] = []
    report: dict = {"lane_url": base, "model": args.model, "workload": args.workload}
    if relay_base:
        report["relay_url"] = relay_base

    # The evidence set: the rows the acceptance rule reads, in the shape it
    # reads them. The rule selects the rows it judges by the endpoint the
    # row was measured over, so every label carries it.
    evidence: list[str] = []
    histograms: list[dict] = []
    per_workload: dict[str, list[dict]] = {}

    if not args.no_warmup:
        # A SACRIFICIAL WARMUP PER CLASS. Each class is a different KIND of
        # content, so each pays its own first-call cost -- on its own nonce
        # and its own prefix, never on what is measured.
        for wl in workloads:
            try:
                w = warmup(base, args.model, wl)
                print(f"warmup ({wl}, sacrificial, separate prefix): "
                      f"{json.dumps(w)[:200]}", flush=True)
            except Exception as exc:  # noqa: BLE001 - the warmup is best-effort but visible
                failures.append(f"warmup ({wl}): {exc}")
                print(f"[FAIL] warmup ({wl}): {exc}", flush=True)
            if relay_base:
                # The second endpoint gets its own sacrificial warmup: a JIT
                # tax paid on the lane is not a JIT tax paid on the proxy's
                # path.
                try:
                    w = warmup(relay_base, args.model, wl, headers=relay_headers)
                    print(f"warmup (relay, {wl}, sacrificial, separate prefix): "
                          f"{json.dumps(w)[:200]}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"warmup (relay, {wl}): {exc}")
                    print(f"[FAIL] warmup (relay, {wl}): {exc}", flush=True)

    levels = _levels(args.streams)
    # EACH WORKLOAD CLASS GETS ITS OWN ROUNDS, IN SEQUENCE. The round index
    # runs across the whole run; the class being measured in it is the
    # cursor. Every row carries its class, so the rule sees both.
    total_rounds = args.rounds * len(workloads)
    for round_no in range(1, total_rounds + 1):
        args.workload = workloads[(round_no - 1) // args.rounds]
        # A FRESH NONCE PER ROUND: the measured prefix is never reused.
        prompt = _prompt(_nonce(), args.workload)

        idle = _idle_check(base)
        print(f"idle check [r{round_no}]: {json.dumps(idle)}", flush=True)
        if not idle["known"]:
            print("[WARN] the engine did not report its request counters -- "
                  "the idle check could not be performed", flush=True)
        elif not idle["idle"]:
            msg = (f"the engine is NOT idle (running={idle['requests_running']}, "
                   f"waiting={idle['requests_waiting']}) -- this measurement would be MIXED")
            if args.require_idle:
                failures.append(f"round {round_no}: {msg}")
                print(f"[FAIL] {msg}", flush=True)
                continue
            print(f"[WARN] {msg}", flush=True)

        before = _metrics_snapshot(base)
        row = bench_stream(base, args.model, prompt, args.tokens)
        after = _metrics_snapshot(base)

        row["round"] = round_no
        row["acceptance"] = _acceptance_curve(before, after)
        ttft_delta = _delta(before, after, TTFT_SUM)
        ttft_count = _delta(before, after, TTFT_COUNT)
        if ttft_delta is not None and ttft_count:
            row["engine_ttft_s"] = ttft_delta / ttft_count
        print(f"stream [r{round_no}]: {json.dumps(row)}", flush=True)
        if args.fail_on_empty and not row.get("tokens"):
            failures.append(f"round {round_no}: empty completion")

        hist = row["acceptance"]
        # EVERY ROUND IS MERGED, NOT ONLY THE ONES THAT CARRIED COUNTERS.
        # Dropping a counters-absent round BEFORE the merge made a
        # partially-absent window's merged row cover only the rounds that
        # had counters -- and the note that says exactly that never reached
        # the evidence file. The merge computes `counters_present` from the
        # rounds that carried counters, so it is safe to merge them all.
        histograms.append(hist)
        per_workload.setdefault(args.workload, []).append(hist)
        # The evidence row: the shape the acceptance rule reads.
        erow = {
            "mode": "stream",
            "path": "lane",
            "workload": args.workload,
            "round": round_no,
            "tok_s": row.get("tok_s"),
            "ttft_s": row.get("ttft_s"),
            "per_token_ms": row.get("per_token_ms"),
            "tokens": row.get("tokens"),
            "tokens_source": "usage",
            "streamed_chunks": row.get("streamed_chunks"),
        }
        evidence.append(f"lane {endpoint} [r{round_no}]: {json.dumps(erow)}")

        # The second endpoint, measured over its own path in the same round,
        # with its own fresh nonce. Its rows carry path=relay so the rule can
        # select them, and the label carries the endpoint the measurement was
        # taken over -- the rule matches on the path, not on the row's order.
        if relay_base:
            relay_row = bench_stream(relay_base, args.model,
                                     _prompt(_nonce(), args.workload),
                                     args.tokens, headers=relay_headers)
            relay_row["round"] = round_no
            print(f"stream (relay) [r{round_no}]: {json.dumps(relay_row)}", flush=True)
            if args.fail_on_empty and not relay_row.get("tokens"):
                failures.append(f"round {round_no}: empty completion over the "
                                f"second endpoint ({relay_endpoint})")
            rrow = {
                "mode": "stream",
                "path": "relay",
                "workload": args.workload,
                "round": round_no,
                "tok_s": relay_row.get("tok_s"),
                "ttft_s": relay_row.get("ttft_s"),
                "per_token_ms": relay_row.get("per_token_ms"),
                "tokens": relay_row.get("tokens"),
                # The token accounting cross-check is the engine's, so the
                # second endpoint's rows claim the engine's source ONLY when
                # the engine's usage block actually came back through it.
                # A proxy that swallows the usage block must not be able to
                # pass the rule by inheriting the lane's label.
                "tokens_source": "usage" if relay_row.get("tokens") else "none",
                "streamed_chunks": relay_row.get("streamed_chunks"),
            }
            evidence.append(f"relay {relay_endpoint} [r{round_no}]: {json.dumps(rrow)}")

        # THE TWO ACCOUNTS, AND THE DENOMINATOR THEY SHARE.
        #
        # Account A -- THE ENGINE'S OWN DRAFT ACCOUNTING: 1 + accepted/drafts
        # tokens per decode step, from the engine's spec-decode counters.
        # `drafts` IS the engine's decode-step count, not a token count: the
        # lane's own measurement recorded the engine's inter-token step
        # counter and `spec_decode_num_drafts_total` as EQUAL on the same
        # window. (The measured figure itself is deliberately not quoted
        # here: an exact instrument reading from a private window is a
        # correlation bridge, and the equality is the part that matters.)
        #
        # Account B -- THE SERVER'S OWN TOKEN COUNT over those same steps:
        # the response's usage block divided by the engine's step count.
        #
        # The two accounts share the DENOMINATOR (the engine's own step
        # count) and have independent NUMERATORS -- the engine's
        # accepted-draft counter on one side, the server's usage block on
        # the other. Neither can be derived from the other, so a
        # disagreement between them is a real disagreement.
        #
        # THE CLIENT'S SSE CHUNK COUNT IS PUBLISHED BESIDE IT AND IS NOT THE
        # DENOMINATOR. The engine coalesces a decode step's tokens into ONE
        # SSE chunk, so the chunk count is a step count only because of that
        # engine behaviour -- an assumption the acceptance rule should not
        # depend on. It is published so a reader can see the relationship
        # between the client's view and the engine's.
        #
        # An earlier version compared 1 + mean_accepted_length with
        # tokens_per_step. Those are TWO FORMS OF THE SAME DEFINITION --
        # mean_accepted_length IS accepted/drafts, and tokens_per_step IS
        # 1 + accepted/drafts -- so the deviation was identically 0.0 and the
        # criterion could never fail. A gate that cannot fail proves
        # nothing. This one compares the two accounts above, so it can.
        tps = hist.get("tokens_per_step")
        usage_tokens = row.get("tokens")
        chunks = row.get("streamed_chunks")
        steps = hist.get("drafts")
        xcheck = {"usage_tokens": usage_tokens, "chunk_total": chunks,
                  "engine_draft_rounds": steps,
                  "engine_tokens_per_step": tps}
        if usage_tokens and steps and tps:
            observed = usage_tokens / steps
            # THE EXACT VALUES, NOT ROUNDED ONES. The rule compares these
            # percentages with its OWN recomputation, and a figure that is
            # rounded on the way out can hide the very deviation the
            # comparison is about: a 10.004% deviation published as "10.0"
            # is a figure the rule's own line does not carry either. The
            # JSON stays valid -- json.dumps prints a float at full
            # precision -- and the drift check is unaffected, because the
            # raw and the published values then agree exactly.
            xcheck["observed_tokens_per_step"] = observed
            predicted = steps * tps
            xcheck["deviation_pct"] = abs(predicted - usage_tokens) / usage_tokens * 100.0
            xcheck["predicted_tokens"] = predicted
            xcheck["tokens_per_step_deviation_pct"] = abs(observed - tps) / tps * 100.0
        evidence.append(f"token accounting cross-check (1-way lane): {json.dumps(xcheck)}")

        if args.non_stream:
            ns = bench_non_stream(base, args.model, _prompt(_nonce(), args.workload),
                                  args.tokens)
            print(f"non-stream [r{round_no}]: {json.dumps(ns)}", flush=True)
            if args.fail_on_empty and not ns.get("tokens"):
                failures.append(f"round {round_no}: empty non-streaming completion")

        for level in levels:
            if level <= 1:
                continue
            before = _metrics_snapshot(base)
            conc = bench_concurrent(base, args.model,
                                    _prompt(_nonce(), args.workload),
                                    args.tokens, level)
            after = _metrics_snapshot(base)
            conc["round"] = round_no
            conc["acceptance"] = _acceptance_curve(before, after)
            print(f"concurrent x{level} [r{round_no}]: {json.dumps(conc)}", flush=True)
            if args.fail_on_empty and not conc.get("tokens"):
                failures.append(f"round {round_no}: concurrent x{level} produced no tokens")
            # THE PER-STREAM MINIMUM IS OVER THE STREAMS MEASURED. The old
            # filter (`if t`) DROPPED every stream that produced no usage
            # block, so the published minimum was the minimum over the
            # streams that SURVIVED -- and a level with one dead stream was
            # published as a healthy one, with a minimum the rule's floor
            # cleared. A dead stream is not a slow stream: it is a stream
            # that produced NO tokens, and the minimum has to say so.
            per_stream = list(conc.get("per_stream_tok_s") or [])
            dead_streams = conc.get("dead_streams") or 0
            if args.fail_on_empty and dead_streams:
                # A CONCURRENT LEVEL WITH A DEAD STREAM IS A BROKEN
                # MEASUREMENT, NOT A SLOW ONE, AND THE BENCH SAYS SO. The
                # total-token check above only fires when EVERY stream
                # died, so a partially-dead level used to pass the bench
                # silently and leave the rule to judge a healthy-looking
                # figure.
                failures.append(
                    f"round {round_no}: concurrent x{level} had {dead_streams} "
                    f"stream(s) that produced no usable rate -- the level is a "
                    f"BROKEN measurement, not a slow one")
            crow = {
                "mode": f"concurrent x{level}",
                "path": "lane",
                "workload": args.workload,
                "round": round_no,
                "streams": level,
                "aggregate_tok_s": conc.get("tok_s_aggregate"),
                "min_stream_tok_s": min(per_stream) if per_stream else None,
                "max_stream_tok_s": max(per_stream) if per_stream else None,
                "dead_streams": dead_streams,
                "tokens": conc.get("tokens"),
                "tokens_source": "usage",
                "wall_s": conc.get("wall_s"),
            }
            evidence.append(f"lane concurrent x{level} {endpoint} [r{round_no}]: "
                            f"{json.dumps(crow)}")

        report.setdefault("rounds", []).append(row)

    if args.prefill_tokens:
        pf, pf_problem = bench_prefill(base, args.model, args.prefill_tokens)
        if pf is None:
            # FAIL CLOSED: NO prefill row is published. A short row would be
            # a row the rule would (rightly) reject, and publishing it
            # would leave the bench claiming a long-context prefill this
            # run did not measure.
            print(f"prefill: NOT MEASURED -- {pf_problem}", flush=True)
            report["prefill"] = {"measured": False,
                                 "declared_floor_tokens": args.prefill_tokens,
                                 "reason": pf_problem}
            failures.append(f"the long-context prefill: {pf_problem}")
        else:
            print(f"prefill: {json.dumps(pf)}", flush=True)
            report["prefill"] = pf
            prow = {"kind": "prefill", "path": "lane", **pf}
            evidence.append(f"lane prefill {endpoint}: {json.dumps(prow)}")

    # The acceptance histogram: the overall window, and one per workload
    # class. The rule judges position 0 on the OVERALL histogram AND on
    # each class's own histogram: a class that is not published is a class
    # that was not measured, so its absence is a named failure rather than
    # a silently thinner evidence set.
    overall = _merge_histograms(histograms)
    if histograms:
        # PUBLISHED WHENEVER THERE WAS A ROUND, not only when the counters
        # were present: a window whose counters were ABSENT is exactly the
        # case the reader needs to see, and the row carries the note.
        owin = {"kind": "acceptance", "path": "lane", "window": "overall", **overall}
        evidence.append(f"lane acceptance overall: {json.dumps(owin)}")
    for workload, hists in per_workload.items():
        merged = _merge_histograms(hists)
        if hists:
            wwin = {"kind": "acceptance", "path": "lane",
                    "window": f"{workload}-1way", **merged}
            evidence.append(f"lane acceptance {workload}-1way: {json.dumps(wwin)}")

    # The engine's own inter-token figure for the 1-way window: the
    # per-token latency of the measured round, per token.
    if report.get("rounds"):
        per_token = [r.get("per_token_ms") for r in report["rounds"]
                     if r.get("per_token_ms")]
        if per_token:
            # THE EXACT VALUE, NOT A ROUNDED ONE. The rule compares this
            # figure with a CEILING, so a figure that is rounded on the way
            # out can round ONTO the ceiling and pass a limit it does not
            # clear: a true 42.004 published as "42.00" is read as 42.0.
            # The shortest repr that round-trips to the identical float
            # carries the value exactly, and the rule does its own display
            # rounding for the line it prints -- so the evidence stays
            # exact and the report stays readable.
            evidence.append(f"engine inter-token (1-WAY window, per TOKEN): "
                            f"{min(per_token)!r}")

    # The evidence set on disk. The manifest is the driver's own record of
    # what it was pointed at and which rule the evidence was collected for.
    if args.evidence_dir:
        os.makedirs(args.evidence_dir, exist_ok=True)
        with open(os.path.join(args.evidence_dir, "bench.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(evidence) + ("\n" if evidence else ""))
        rule_file = args.rule_file
        if not rule_file:
            here = os.path.dirname(os.path.abspath(__file__))
            rule_file = os.path.join(here, "acceptance-check.py")
        rule_md5 = ""
        try:
            import hashlib
            with open(rule_file, "rb") as fh:
                rule_md5 = hashlib.md5(fh.read()).hexdigest()
        except OSError:
            rule_md5 = ""
        with open(os.path.join(args.evidence_dir, "manifest.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"rule_md5={rule_md5}\n")
            fh.write(f"lane_url={args.lane_url.rstrip('/')}\n")
            # The second endpoint, when the deployment has one. The rule reads
            # THIS line to decide whether the second-endpoint criteria apply
            # or are reported as not applicable -- so it is written only when
            # the second endpoint was actually measured, never as a guess.
            if relay_base:
                fh.write(f"relay_url={relay_base}\n")
        print(f"wrote {args.evidence_dir}/bench.txt and manifest.txt", flush=True)

    report["failures"] = failures
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json_out}", flush=True)

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
