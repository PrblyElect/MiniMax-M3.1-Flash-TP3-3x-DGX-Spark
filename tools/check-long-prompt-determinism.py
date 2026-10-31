#!/usr/bin/env python3
"""check-long-prompt-determinism.py -- the LONG-PROMPT determinism gate.

WHY THIS GATE EXISTS
--------------------
A prompt that crosses the engine's batched-token budget is served in
chunks, and a chunked prefill that ends up on a different path than a
single-shot one is the classic place for a serving engine to stop being
deterministic. Upstream has shipped defects where a long prompt produced a
wrong answer from the FIRST generated token, and where a speculative-decode
worker died during long-context decode.

A lane that is deterministic for a 200-token turn and NOT deterministic for
an 8,000-token turn is a lane whose long answers cannot be trusted -- and
the difference is invisible until somebody sends a long one.

WHAT IT CHECKS
--------------
The SAME long prompt, sent three times at temperature 0:

  run 1   COLD. Nothing has been computed for this prompt.
  run 2   the same request again: the prefix cache is warm.
  run 3   the same request again: the prefix cache is warm.

The generated TOKEN SEQUENCE is compared, not the text: two runs can agree
on the text and disagree on the tokens (or the other way round), and the
token sequence is the thing that has to be stable. A comparison of two
EMPTY sequences always passes, so the sequences are shown to exist before
they are compared.

THE PROMPT IS EXACTLY `max_num_batched_tokens + last_chunk` TOKENS
-------------------------------------------------------------------
The gate has to land where the bug lives, so the count is exact -- and a
character-level search cannot make it exact: byte-level BPE counts are not
monotonic in the prefix length (a longer prefix can tokenize into FEWER
tokens), so a binary search over characters has no correctness guarantee
and can burn thousands of round trips before it lands or gives up. So the
ids come from the engine's own /tokenize endpoint, exactly that many of
them are taken, and the engine's own `usage.prompt_tokens` is asserted
afterwards: the count is the ENGINE's, not ours.

A FRESH NONCE SITS AT THE HEAD of the prompt, so the first call is
genuinely cold even when the same gate is run again on the same lane: a
cached prefix would otherwise make the "cold" run warm, and the gate would
be measuring nothing.

RUN IT IN A BOOT WINDOW
-----------------------
This gate is LOAD-BEARING: the long-context answers are the ones a serving
lane is judged on, and a defect here is invisible in every short request.
Run it on the head while the lane is up, and treat a FAIL as a reason not
to ship the lane.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLUSTER_ENV = os.path.join(HERE, "..", "config", "cluster.env")

# A deterministic corpus. Numbered lines so the text is not a repeated
# n-gram: a repetitive prompt can tokenize and cache differently from a
# real one, and this gate is about the engine, not about the prose.
CORPUS = [
    "Line {n:05d}: the harbour lights came on before the fog had finished settling.",
    "Line {n:05d}: a cartographer marked the ridge, then crossed out the mark.",
    "Line {n:05d}: two hundred crates of salt went up the coast road each autumn.",
    "Line {n:05d}: the old pump worked until somebody sealed it with mortar.",
]


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def profile_mnbt() -> int:
    """The batched-token budget the engine will actually run with.

    Resolution order, matching the launcher
    (`MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"`):

      1. the environment, when it is set to a number -- the launcher
         resolves the same way, so the environment wins;
      2. the profile of record, in EITHER of the two forms it can take:
         `MAX_NUM_BATCHED_TOKENS=8192` or
         `MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"`;
      3. 0, meaning "not resolvable" -- the caller then uses its fallback.

    WHY THE SECOND FORM MATTERS: a reader that matches only plain digits
    reads the quoted default-form as "no value", returns 0, and the
    caller silently falls back to a hardcoded constant. That constant
    happens to equal the shipped default today, so the gate looks right --
    and the moment an operator overrides the budget, the gate aims its
    prompt at the WRONG chunked-prefill boundary. The prompt then does not
    cross the boundary the engine actually uses, the crossing guard
    compares against the same wrong number, and the gate passes without
    having tested a chunked prefill at all. A gate that cannot fail is
    worse than no gate, so this resolves the profile the way the engine
    will.
    """
    env = os.environ.get("MAX_NUM_BATCHED_TOKENS", "").strip()
    if env.isdigit():
        return int(env)
    try:
        with open(DEFAULT_CLUSTER_ENV, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(
                    r'^\s*MAX_NUM_BATCHED_TOKENS="?\$\{MAX_NUM_BATCHED_TOKENS:-(\d+)\}"?\s*$',
                    line)
                if m:
                    return int(m.group(1))
                m = re.match(r"^\s*MAX_NUM_BATCHED_TOKENS=(\d+)\s*$", line)
                if m:
                    return int(m.group(1))
    except (OSError, ValueError):
        pass
    return 0


def build_prompt_ids(base: str, model: str, target_tokens: int, nonce: str) -> list[int]:
    """Token IDs for a prompt of EXACTLY `target_tokens` tokens.

    The nonce is at the HEAD, so the prefix cache cannot serve this prompt
    from an earlier run: the first call is genuinely cold.

    The IDs come from the engine's own tokenizer and are sent back as the
    prompt, so the engine counts exactly what was asked for.
    """
    lines = [f"nonce {nonce}"]
    i = 0
    while True:
        lines.append(CORPUS[i % len(CORPUS)].format(n=i))
        i += 1
        if i % 64 == 0 and len("\n".join(lines)) > target_tokens * 8:
            break
    text = "\n".join(lines)

    data = post(f"{base}/tokenize", {"model": model, "prompt": text})
    ids = list(data.get("tokens") or [])
    if len(ids) < target_tokens:
        raise SystemExit(f"corpus too short: tokenized to {len(ids)} tokens, "
                         f"need {target_tokens}")
    return ids[:target_tokens]


def complete(base: str, model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    t0 = time.time()
    data = post(f"{base}/v1/completions", {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "logprobs": 1,
    })
    dt = time.time() - t0
    ch = (data.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    return {
        "text": ch.get("text", "") or "",
        "tokens": lp.get("tokens"),
        "token_logprobs": lp.get("token_logprobs"),
        "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
        "seconds": dt,
    }


def token_sequence(base: str, model: str, run: dict) -> list[str] | None:
    """The generated TOKEN sequence, at token granularity.

    The engine's own logprobs carry it when it returns them. When it does
    not, the completion is re-tokenized with the engine's own tokenizer --
    still a token sequence, still the engine's own view of the text. The
    text itself is NOT a token sequence and is never used as one.
    """
    if run.get("tokens"):
        return list(run["tokens"])
    text = (run.get("text") or "").strip()
    if not text:
        return None
    try:
        data = post(f"{base}/tokenize", {"model": model, "prompt": text})
    except Exception:
        return None
    ids = data.get("tokens") or []
    return [str(t) for t in ids] if ids else None


def logprob_drift(a: dict, b: dict) -> float | None:
    """The largest absolute difference between two token_logprobs lists.

    None when the two runs are not comparable (a missing list, or two runs
    that generated a different number of tokens).
    """
    la = a.get("token_logprobs")
    lb = b.get("token_logprobs")
    if not la or not lb or len(la) != len(lb):
        return None
    return max(abs(x - y) for x, y in zip(la, lb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="")
    ap.add_argument("--mnbt", type=int, default=0,
                    help="max_num_batched_tokens; 0 = read it from the profile "
                         "of record (config/cluster.env)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--last-chunk", type=int, default=10,
                    help="tokens in the final prefill chunk")
    ap.add_argument("--nonce", default="",
                    help="prompt nonce; the default is fresh on every run so "
                         "the first call is genuinely cold")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ok = True

    def check(cond: bool, msg: str) -> bool:
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)
        return bool(cond)

    model = args.model
    if not model:
        try:
            data = json.loads(get_text(f"{base}/v1/models"))
            model = (data.get("data") or [{}])[0].get("id", "")
        except Exception:
            model = ""
    if not model:
        print("FAIL could not determine the served model name")
        return 1

    mnbt = args.mnbt or profile_mnbt() or 8192
    target = mnbt + args.last_chunk

    nonce = args.nonce or secrets.token_hex(6)
    print(f"model={model}  max_num_batched_tokens={mnbt}  "
          f"target prompt={target} tokens  nonce={nonce}")

    prompt_ids = build_prompt_ids(base, model, target, nonce)
    check(len(prompt_ids) == target,
          f"prompt is {target} token IDs (got {len(prompt_ids)})")
    if len(prompt_ids) <= mnbt:
        print("FAIL the prompt does not cross the chunked-prefill boundary; "
              "this gate would pass without testing anything")
        return 1

    a = complete(base, model, prompt_ids, args.max_tokens)
    b = complete(base, model, prompt_ids, args.max_tokens)
    c = complete(base, model, prompt_ids, args.max_tokens)

    print(f"  timings: {a['seconds']:.1f}s / {b['seconds']:.1f}s / {c['seconds']:.1f}s")

    check(a["prompt_tokens"] == target,
          f"engine counted the prompt as {target} (got {a['prompt_tokens']})")

    sa = token_sequence(base, model, a)
    sb = token_sequence(base, model, b)
    sc = token_sequence(base, model, c)

    # A comparison of two EMPTY sequences always passes, so the thing being
    # compared has to be shown to exist before it is compared.
    if sa is not None:
        check(len(sa) > 0, "the engine returned the generated token sequence")
        print(f"  generated: {' '.join(str(t) for t in sa[:12])}"
              f"{' ...' if len(sa) > 12 else ''}")
    else:
        check(len(a["text"]) > 0,
              "the engine returned a completion, but no token sequence could "
              "be read from it -- this gate compares TOKEN sequences, so it "
              "cannot pass")
        print(f"  generated: {a['text'][:80]!r}")
        print("\nLONG-PROMPT-DETERMINISM FAIL")
        return 1

    def same(x: list[str] | None, y: list[str] | None, label: str) -> None:
        if x is None or y is None:
            check(False, f"{label}: no token sequence to compare")
            return
        check(x == y, f"{label} produced the same token sequence"
              + ("" if x == y else f"\n         left ={x[:12]}\n         right={y[:12]}"))

    # 1: the same request twice must produce the same tokens.
    same(sa, sb, "the cold run and the repeat run")
    # 2: the third call, with the prefix cache warm from the first.
    same(sa, sc, "the cold run and the prefix-cached run")
    # 3: the two cached runs against each other.
    same(sb, sc, "the two prefix-cached runs")

    # The decode logprobs, cold against cached. A logprob shift need not
    # change the argmax token, so the token comparison above cannot see it.
    # It is REPORTED, not gated: this profile keeps its KV cache in fp8 and
    # reads it through a sparse indexer, so a cold and a cached run are not
    # required to agree to the last decimal, and a gate that cannot explain
    # its own failure is worse than a note that says what it saw.
    drift = logprob_drift(a, c)
    if drift is None:
        print("  note: the cold and the cached run did not return comparable "
              "decode logprobs")
    else:
        print(f"  note: the largest decode-logprob drift between the cold and "
              f"the cached run is {drift:.6g} (reported, not gated)")

    # The engine's own view of the cache, if it exposes one. /metrics is
    # PROMETHEUS TEXT, not JSON -- parsing it as JSON always fails, and a
    # failure swallowed here would silently drop the note.
    try:
        m = get_text(f"{base}/metrics")
        hit = re.search(r"vllm:prefix_cache_queries_total\{[^}]*\}\s+(\S+)", m)
        if hit:
            print(f"  note: prefix cache queries counter reads {hit.group(1)}")
    except Exception:
        pass

    print("\nLONG-PROMPT-DETERMINISM " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
