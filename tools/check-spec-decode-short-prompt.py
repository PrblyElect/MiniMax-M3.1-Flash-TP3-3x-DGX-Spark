#!/usr/bin/env python3
"""check-spec-decode-short-prompt.py -- the spec-decode SHORT-PROMPT gate.

WHY THIS GATE EXISTS
--------------------
With speculative decode on, the upstream engine classifies a batch as a
uniform-decode step by SHAPE alone: a prefill whose scheduled token count
happens to equal `uniform_decode_query_len * num_reqs` is dispatched as a
uniform-decode batch instead of a prefill. Upstream reproduced the
consequence -- deterministic garbage from the first generated token -- on a
3-token prompt with MTP k=2, where the trap window is 1 + k.

This recipe runs DSpark with k=3, so the trap window is FOUR tokens: a
4-token prompt, one request, is the shape the upstream dispatch check
misreads. The exact window is the ENGINE'S OWN `uniform_decode_query_len`,
not a constant of this recipe, so this gate reads it from the engine (the
boot log line the recipe's own draft-pad gate prints) and accepts an
explicit override.

WHAT IT CHECKS
--------------
A prompt of EXACTLY that many tokens, at temperature 0, must produce a sane
answer. The same probe is then sent one token LONGER: one token more is not
a multiple of the trap shape, so the control takes the normal prefill path.
The gate therefore compares the engine against ITSELF at two batch shapes,
and a shape-dependent answer is the bug.

It is also a correctness gate for the SHAPE, not only for the words: a wrong
answer at the trap shape and a right one at the control shape is exactly
what the upstream defect produces.

THE PROMPT IS BUILT FROM TOKEN IDS, NOT FROM CHARACTERS
--------------------------------------------------------
The prompt must be EXACTLY N tokens, and a character-level search cannot
guarantee that: byte-level BPE counts are not monotonic in the prefix length
(a longer prefix can tokenize into FEWER tokens), so a binary search over
characters has no correctness guarantee. So the ids come from the engine's
own /tokenize endpoint, and the engine's own `usage.prompt_tokens` is
asserted afterwards -- the count is the engine's, not ours.

COLDNESS
--------
The upstream defect needs a VIRGIN request (one that has computed nothing
yet), and a prefix-cache hit would turn the prompt into a decode and hide
the bug. A prompt shorter than one KV block cannot be served from the
prefix cache -- the whole prompt is a single partial block -- so every run
of this gate is already cold. A nonce is therefore belt-and-braces on top
of that, and it is ON by default (`--no-nonce` turns it off).

The nonce must cost EXACTLY ONE token: the trap window is only a handful
of tokens, and a longer nonce pushes the question out of the prompt
entirely -- the engine is then asked nothing, and whatever it answers says
nothing about the trap shape. A nonce that costs more than one token is a
HARD FAILURE of this gate, not a note, and so is a probe whose question
cannot fit the window whole.

RUN IT IN A BOOT WINDOW
-----------------------
This gate is LOAD-BEARING: a lane that serves a 4-token prompt wrong is a
lane that serves a short chat turn wrong, and the symptom (garbage) looks
like a model problem rather than a dispatch problem. Run it on the head
while the lane is up, and treat a FAIL as a reason not to ship the lane.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC_CONFIG = os.path.join(HERE, "..", "config", "spec-config.json")

# The probes, in order. Each is a question whose answer is a single
# well-known word, so "sane" is a DECISION and not a judgement call.
#
# The questions are ordered so that the ones which stay determinate when
# their TAIL is cut come first: the prompt is trimmed to an exact token
# count, and "The capital of France" truncated to "The capital of" is not a
# question any more. The arithmetic ones survive it: "2+2" is still a
# question, and still has one answer.
PROBES = [
    ("2+2", "4"),
    ("3+3", "6"),
    ("1+1", "2"),
    ("2+2=", "4"),
    ("The capital of France is", "Paris"),
    ("Rome is the capital of", "Italy"),
    ("The largest planet in the solar system is", "Jupiter"),
]

# The engine's own boot log line, as printed by the recipe's own draft-pad
# gate (patches/verify-draft-pad.py):
#   capture            : uniform_decode_query_len 4, max_cudagraph_capture_size ...
QUERY_LEN_RE = re.compile(r"uniform_decode_query_len[ =:]+(\d+)")


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def served_model(base: str, wanted: str) -> str:
    if wanted:
        return wanted
    try:
        data = json.loads(get_text(f"{base}/v1/models"))
        return (data.get("data") or [{}])[0].get("id", "")
    except Exception:
        return ""


def engine_query_len(base: str, container: str, explicit: int) -> tuple[int, str]:
    """The engine's OWN uniform_decode_query_len, and where it came from.

    Three sources, in order of preference, and the source is always
    reported: a value the reader states, the engine's own boot log, or --
    as a last resort, and clearly labelled as DERIVED rather than measured
    -- the recipe's own speculative config, where the window is 1 + k.
    """
    if explicit > 0:
        return explicit, "stated on the command line"
    if container:
        try:
            r = subprocess.run(["docker", "logs", container],
                               capture_output=True, text=True, timeout=120)
            m = QUERY_LEN_RE.search((r.stdout or "") + (r.stderr or ""))
            if m:
                return int(m.group(1)), (f"the engine's own boot log "
                                         f"(container {container})")
        except Exception:
            pass
    try:
        with open(DEFAULT_SPEC_CONFIG, encoding="utf-8") as fh:
            spec = json.load(fh)
        k = int(spec.get("num_speculative_tokens", 0))
        if k > 0:
            return k + 1, (f"DERIVED from the recipe's own speculative config "
                           f"(1 + k = 1 + {k}), not measured")
    except (OSError, ValueError, TypeError):
        pass
    raise SystemExit(
        "could not determine the engine's uniform_decode_query_len.\n"
        "  The engine's own boot log did not carry it (is the container "
        "running, and is --container right?),\n"
        "  and the recipe's speculative config could not be read.\n"
        "  Pass --query-len <n> to state it.")


def tokenize(base: str, model: str, text: str) -> list[int]:
    data = post(f"{base}/tokenize", {"model": model, "prompt": text})
    return list(data.get("tokens") or [])


def neutral_token(base: str, model: str) -> int:
    """One token that carries no meaning, for padding a prompt to an exact
    length. A whitespace token is the right kind of filler: it changes the
    length without changing the question."""
    for text in (" ", "\n", "  "):
        ids = tokenize(base, model, text)
        if len(ids) == 1:
            return ids[0]
        if ids:
            return ids[-1]
    raise SystemExit("could not build a neutral padding token from the "
                     "engine's own tokenizer")


def build_probe(base: str, model: str, want: int, nonce_ids: list[int],
                question_ids: list[int], pad_id: int) -> tuple[list[int], bool]:
    """Exactly `want` token ids, with the QUESTION INTACT.

    Returns (ids, question_was_trimmed).

    The question must survive whole. The trap window is
    `uniform_decode_query_len` tokens -- FOUR for this recipe's k=3 -- and a
    probe whose question gets cut out of the window does not test the
    dispatch at all: the engine is asked nothing, and whatever it answers
    says nothing about the trap shape. A probe that has to be trimmed is
    therefore a HARD FAILURE of this gate, not a note.

    A question SHORTER than the window is padded with a neutral trailing
    token, which leaves the question intact and makes the prompt length
    exact. Padding at the tail is safe; padding at the head is not.
    """
    room = want - len(nonce_ids)
    trimmed = len(question_ids) > room
    if trimmed:
        return (nonce_ids + question_ids)[:want], True
    ids = list(nonce_ids) + list(question_ids)
    while len(ids) < want:
        ids.append(pad_id)
    return ids[:want], False


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
        "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
        "seconds": dt,
    }


def degenerate(tokens: list[str] | None, text: str) -> str | None:
    """None when the output looks like an answer; a reason when it does not."""
    if not text.strip():
        return "the completion is empty"
    if "\ufffd" in text:
        return "the completion contains a replacement character"
    if any(ord(c) < 32 and c not in "\n\t\r" for c in text):
        return "the completion contains control characters"
    if tokens:
        if len(tokens) >= 8 and len(set(tokens)) <= 2:
            return ("the completion is a degenerate repetition: "
                    f"{len(set(tokens))} distinct token(s) over {len(tokens)}")
    elif len(text.strip()) >= 40 and len(set(text.strip())) <= 2:
        return "the completion is a degenerate repetition"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="")
    ap.add_argument("--query-len", type=int, default=0,
                    help="the engine's uniform_decode_query_len; 0 = read it "
                         "from the engine's own boot log")
    ap.add_argument("--container", default="minimax31-flash-tp3",
                    help="the container whose boot log carries the engine's own "
                         "uniform_decode_query_len")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--no-nonce", action="store_true",
                    help="do not put a fresh token at the head of the prompt "
                         "(the prompt is shorter than one KV block, so it "
                         "cannot be served from the prefix cache anyway)")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ok = True

    def check(cond: bool, msg: str) -> bool:
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)
        return bool(cond)

    model = served_model(base, args.model)
    if not model:
        print("FAIL could not determine the served model name")
        return 1

    want, source = engine_query_len(base, args.container, args.query_len)
    print(f"model={model}  uniform_decode_query_len={want} (from {source})")
    print(f"  the trap window is {want} tokens: a {want}-token prompt, one "
          f"request, is the shape the upstream dispatch check misreads")

    nonce = "" if args.no_nonce else secrets.token_hex(4)
    nonce_ids = tokenize(base, model, nonce) if nonce else []
    # The nonce must cost EXACTLY ONE token of the window. A longer one eats
    # the question out of the prompt and the gate then tests nothing.
    if len(nonce_ids) > 1:
        print(f"FAIL the nonce is {len(nonce_ids)} tokens; it must be exactly "
              f"one, or the question does not fit the {want}-token window")
        print("     run with --no-nonce, or the window is too small for a "
              "nonce and a question together")
        return 1

    # Pick the probe: the first whose question fits the window WHOLE.
    scored = []
    for question, answer in PROBES:
        q_ids = tokenize(base, model, question)
        scored.append((question, answer, q_ids))
    fitting = [s for s in scored if len(s[2]) <= want - len(nonce_ids)]
    if not fitting:
        print(f"FAIL no probe fits the {want}-token trap window with its "
              f"question intact -- this gate would test nothing")
        for question, answer, q_ids in scored:
            print(f"     {question!r} is {len(q_ids)} token(s)")
        return 1
    question, answer, q_ids = fitting[0]
    print(f"  probe: {question!r} -> {answer!r} "
          f"({len(q_ids)} token(s) in the question)")

    pad_id = neutral_token(base, model)
    trap_ids, trap_trimmed = build_probe(base, model, want, nonce_ids, q_ids,
                                         pad_id)
    # The CONTROL is the trap shape plus ONE more neutral token. It is
    # APPENDED, never sliced: slicing a shorter list produced a control that
    # was the same length as the trap, or shorter.
    control_ids = list(trap_ids) + [pad_id]
    if trap_trimmed:
        print(f"FAIL the question was trimmed to fit {want} tokens -- the "
              f"engine would be asked something else, and this gate would "
              f"prove nothing")
        return 1
    if not check(len(trap_ids) == want,
                 f"the trap prompt is exactly {want} token ids "
                 f"(got {len(trap_ids)})"):
        return 1

    trap = complete(base, model, trap_ids, args.max_tokens)
    control = complete(base, model, control_ids, args.max_tokens)
    print(f"  timings: trap {trap['seconds']:.1f}s / control "
          f"{control['seconds']:.1f}s")

    check(trap["prompt_tokens"] == want,
          f"the engine counted the trap prompt as {want} "
          f"(got {trap['prompt_tokens']})")
    check(control["prompt_tokens"] == want + 1,
          f"the engine counted the control prompt as {want + 1} "
          f"(got {control['prompt_tokens']})")

    why = degenerate(trap["tokens"], trap["text"])
    check(why is None, "the trap-shape completion is not degenerate"
          + (f" -- {why}" if why else ""))
    why = degenerate(control["tokens"], control["text"])
    check(why is None, "the control-shape completion is not degenerate"
          + (f" -- {why}" if why else ""))

    trap_text = trap["text"].lower()
    control_text = control["text"].lower()
    # The answer must appear as a WHOLE token. A bare substring match would
    # accept a digit that happens to occur somewhere in free text, which is
    # not evidence that the question was answered.
    answer_re = re.compile(rf"(?<![0-9a-z]){re.escape(answer.lower())}(?![0-9a-z])")
    trap_has = bool(answer_re.search(trap_text))
    control_has = bool(answer_re.search(control_text))
    check(trap_has, f"the trap-shape completion answers the probe ({answer!r}"
          + ("" if trap_has else f" -- got {trap['text'][:120]!r}"))
    check(control_has, "the control-shape completion answers the probe"
          + ("" if control_has else f" -- got {control['text'][:120]!r}"))

    if trap_has and not control_has:
        print("  note: the CONTROL shape failed too, so the probe itself may be "
              "the wrong question for this engine -- read both outputs before "
              "treating this as a dispatch defect")
    if not trap_has and control_has:
        print("  note: the trap shape failed where the control shape passed -- "
              "this is the shape-dependent answer the upstream dispatch "
              "defect produces")

    print("\nSPEC-DECODE-SHORT-PROMPT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
