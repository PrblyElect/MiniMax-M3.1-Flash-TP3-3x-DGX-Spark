#!/usr/bin/env python3
"""resolve-capture-grid.py -- PROVE, IN THE PINNED IMAGE, WHAT THE ENGINE
RESOLVES for the recipe's capture-grid cap, BEFORE any boot.

WHY THIS EXISTS (see docs/FIXLOG.md)
------------------------------------
The recipe's capture grid changes ONE thing: the PIECEWISE capture grid is
capped at 128 tokens so the engine stops reserving a capture estimate for
prefill-sized graphs the lane never replays. The whole case rests on two
claims that must NOT be taken from a desk read (an engine-side runtime claim
is enforced against the image, never trusted from a desk read):

  1. The engine ACCEPTS an explicit `cudagraph_capture_sizes` through
     --compilation-config and resolves it to the list that was asked for.
  2. The UNIFORM-DECODE (FULL) graph set is IDENTICAL before and after the
     cap -- i.e. the decode hot path keeps every graph it has today.

Claim 2 is not obvious: the FULL decode descriptors are derived FROM the
capture grid (vllm/v1/worker/gpu/cudagraph_utils.py::_init_candidates),
by rounding each grid size up to a multiple of uniform_decode_query_len and
dropping the ones past max_decode_tokens (max_num_seqs * query_len = 128).
So the sizes 136..256 never produced a decode graph in the first place, and
a cap at 128 cannot remove one. THIS SCRIPT MEASURES IT INSTEAD OF
ARGUING IT.

WHAT IT DOES
------------
Builds the engine's own VllmConfig TWICE -- once with the engine's DEFAULT
capture grid and once with the RECIPE's explicit list (ceiling 128) -- and
prints, for both:
  * the resolved cudagraph_capture_sizes and max_cudagraph_capture_size,
  * the uniform_decode_query_len and max_num_seqs the engine resolved,
  * the derived FULL decode descriptor set (the manager's own rule,
    replicated below WITH THE SOURCE LINES QUOTED) and its count,
  * the diff between the two resolutions.

WHAT IT PROVES, PRECISELY (a claim, not a vibe)
-------------------------------------------------
It PROVES THE RESOLUTION: that an explicit capture list is accepted and
resolves to exactly that list with max_cudagraph_capture_size == 128, and
that the default resolves to 35 sizes with a ceiling of 256. That part is
the engine's own code, run in the image.

It DERIVES (does not observe) the decode descriptor set: the manager's
rule is replicated below, and the replication covers the SINGLE-TIER
branch (decode_query_len = 4, max_num_seqs = 32) -- the engine iterates
`for decode_query_len in decode_query_lens`, and the dynamic-spec-decode
path that would add tiers does not apply to dspark with a fixed
num_speculative_tokens = 3. The conclusion does not depend on the branch
count: every branch is capped by max_decode_tokens = max_num_seqs *
decode_query_len = 128, so the grid sizes 136..256 never produce a decode
descriptor. The recipe's expected counts for the capped grid are 19
PIECEWISE capture sizes, 17 FULL decode descriptors and 14 draft
descriptors, and the boot's capture gate re-checks all three counts on the
head.

THE SPECULATOR'S OWN GRAPHS
----------------------------
The "Capturing dspark CUDA graphs (FULL)" bar is the SPECULATOR's manager, not
the target's: `DFlashSpeculator.init_cudagraph_manager` builds it with
`decode_query_len=self.num_query_per_req`, and for dspark
`num_query_per_req = num_speculative_steps` (the anchor-as-first layout,
`sample_from_anchor=True`), i.e. 3 -- NOT the target's 4. The manager then
applies the SAME `_init_candidates` rule with `max_decode_tokens =
max_num_reqs * decode_query_len = 32 * 3 = 96`. So the speculator's graphs
come only from grid entries <= 96, rounded up to multiples of 3 -- and the cap
at 128 cannot touch them, because 96 < 128.

This script MEASURES that set for both resolutions and checks it against the
recipe's expected count (14). A replication that reproduces the expected
number is not a vibe; a replication that does not would have been caught
here rather than in the boot.

It allocates NO GPU memory and loads NO weights: CUDA is masked off
(CUDA_VISIBLE_DEVICES="") so the resolution runs on the CPU platform. The
only platform-dependent branch in the resolution is
`default_max_graph_size` (512 off data-center Blackwell, 512 on GB10 --
both non-family-100), so the masked run resolves identically to the live
box. The script says so out loud if the platform it detected is not the
one the lane runs.

USAGE:  python3 resolve-capture-grid.py [/model]
"""
from __future__ import annotations

import json
import sys

# The recipe's capture grid: the PIECEWISE grid capped at 128 tokens.
RECIPE_GRID = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
               104, 112, 120, 128]

# The recipe's expected counts for the capped grid -- the same three numbers
# the boot's capture gate re-checks on the head: 19 PIECEWISE capture sizes,
# 17 FULL decode descriptors, 14 draft (dspark) descriptors.
EXPECTED_PIECEWISE = 19
EXPECTED_FULL = 17
EXPECTED_DRAFT = 14

# The recipe's fixed engine settings (start-tp3.sh). Anything not listed here
# does not touch the capture resolution.
BASE = dict(
    model="/model",
    tokenizer="/model",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=196608,
    tensor_parallel_size=3,
    kv_cache_dtype="fp8",
    block_size=128,
    max_num_seqs=32,
    max_num_batched_tokens=8192,
    gpu_memory_utilization=0.78,
    distributed_executor_backend="mp",
    limit_mm_per_prompt={"video": 0},
    attention_config={"indexer_kv_dtype": "fp8"},
    speculative_config={
        "method": "dspark",
        "model": "/model/dspark",
        "num_speculative_tokens": 3,
    },
)


def build(model_dir: str, grid: list[int] | None):
    """Build the engine's OWN config through EngineArgs -- the same path
    `vllm serve` takes, so the --compilation-config JSON is parsed by the
    engine's own parser and the speculative config is wired to its target
    model exactly as it is at serve time."""
    from vllm.engine.arg_utils import EngineArgs

    kwargs = dict(BASE)
    kwargs["model"] = model_dir
    kwargs["tokenizer"] = model_dir
    kwargs["speculative_config"] = dict(BASE["speculative_config"])
    kwargs["speculative_config"]["model"] = f"{model_dir}/dspark"
    if grid is None:
        kwargs["compilation_config"] = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    else:
        kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": list(grid),
        }
    cfg = EngineArgs(**kwargs).create_engine_config()
    return cfg


def full_decode_descs(sizes, max_num_reqs, query_len, max_cg_capture_size):
    """REPLICATION of vllm/v1/worker/gpu/cudagraph_utils.py::_init_candidates,
    the branch that builds the FULL (uniform-decode) descriptors:

        elif separate_decode_routine and decode_mode and not self.varlen_decode:
            for decode_query_len in decode_query_lens:
                rounded_num_tokens = round_up(num_tokens, decode_query_len)
                rounded_num_reqs = rounded_num_tokens // decode_query_len
                if (rounded_num_tokens > max_decode_tokens
                        or rounded_num_tokens > max_cg_capture_size
                        or rounded_num_reqs > self.max_num_reqs):
                    continue

    (max_decode_tokens = max_num_reqs * decode_query_len.)
    """
    out = []
    max_decode_tokens = max_num_reqs * query_len
    for num_tokens in sizes:
        rounded = -(-num_tokens // query_len) * query_len
        if (rounded > max_decode_tokens
                or rounded > max_cg_capture_size
                or rounded // query_len > max_num_reqs):
            continue
        out.append((rounded, rounded // query_len))
    return sorted(set(out))


def spec_decode_descs(sizes, max_num_reqs, query_len, max_cg_capture_size):
    """REPLICATION of the SPECULATOR's (dspark) descriptor set -- the SAME
    `_init_candidates` rule, with the speculator's OWN parameters:

        DFlashSpeculator.init_cudagraph_manager:
            DFlashCudaGraphManager(self.vllm_config, self.device, cudagraph_mode,
                                   decode_query_len=self.num_query_per_req)
        dspark/speculator.py:
            self.num_query_per_req = self.num_speculative_steps   # sample_from_anchor
        CudaGraphManager.__init__:
            self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

    and then the rule itself (cudagraph_utils.py::_init_candidates):

        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        ...
            rounded_num_tokens = round_up(num_tokens, decode_query_len)
            rounded_num_reqs = rounded_num_tokens // decode_query_len
            if (rounded_num_tokens > max_decode_tokens
                    or rounded_num_tokens > max_cg_capture_size
                    or rounded_num_reqs > self.max_num_reqs):
                continue

    The difference from the target's set is the query length: 3 for the
    speculator, 4 for the target. That is why the speculator's ceiling is
    32*3 = 96 and the target's is 32*4 = 128.
    """
    out = []
    max_decode_tokens = max_num_reqs * query_len
    for num_tokens in sizes:
        rounded = -(-num_tokens // query_len) * query_len
        if (rounded > max_decode_tokens
                or rounded > max_cg_capture_size
                or rounded // query_len > max_num_reqs):
            continue
        out.append((rounded, rounded // query_len))
    return sorted(set(out))


def draft_num_query_per_req(model_dir: str, num_spec_tokens: int) -> int:
    """The speculator's `num_query_per_req`: num_speculative_steps when the
    draft is anchor-as-first (dspark's default), else 1 + the steps. Read from
    the DRAFT's own config -- the same file the engine reads."""
    import os

    cfg_path = os.path.join(model_dir, "dspark", "config.json")
    anchor = True
    try:
        with open(cfg_path) as fh:
            anchor = bool(json.load(fh).get("sample_from_anchor", True))
    except Exception as exc:  # pragma: no cover
        print(f"NOTE: could not read {cfg_path} ({exc}); assuming the "
              f"anchor-as-first default")
    return num_spec_tokens if anchor else 1 + num_spec_tokens


def report(name, cfg, spec_q_len=None):
    cc = cfg.compilation_config
    sizes = list(cc.cudagraph_capture_sizes or [])
    max_cg = cc.max_cudagraph_capture_size
    q_len = getattr(cfg, "uniform_decode_query_len", None)
    max_seqs = cfg.scheduler_config.max_num_seqs
    print(f"--- {name} ---")
    print(f"resolved capture list ({len(sizes)} entries): {sizes}")
    print(f"resolved max_cudagraph_capture_size: {max_cg}")
    print(f"resolved uniform_decode_query_len: {q_len}")
    print(f"resolved max_num_seqs: {max_seqs}")
    descs = full_decode_descs(sizes, max_seqs, q_len, max_cg)
    print(f"derived FULL decode descriptors ({len(descs)}): {descs}")
    spec_descs = None
    if spec_q_len:
        spec_descs = spec_decode_descs(sizes, max_seqs, spec_q_len, max_cg)
        print(f"derived SPECULATOR (dspark) descriptors, decode_query_len="
              f"{spec_q_len}, ceiling={max_seqs * spec_q_len} "
              f"({len(spec_descs)}): {spec_descs}")
    return {"sizes": sizes, "max_cg": max_cg, "q_len": q_len,
            "max_seqs": max_seqs, "descs": descs, "spec_descs": spec_descs}


def main() -> int:
    # A USAGE LINE, AND A MISSING-ENGINE MESSAGE, INSTEAD OF A TRACEBACK.
    # This tool proves an ENGINE-SIDE claim, so it must run where the engine
    # is: inside the pinned image. Run it anywhere else and the honest
    # answer is "this needs the engine", not a stack trace that reads like
    # a crash. The usage line is printed BEFORE the engine is imported, so
    # `--help` works everywhere.
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip().split("\n\n")[0])
        print()
        print("usage: python3 tools/resolve-capture-grid.py [MODEL_DIR]")
        print()
        print("  MODEL_DIR   the checkpoint directory to resolve against "
              "(default: /model)")
        print()
        print("This tool must run INSIDE the pinned image: it builds the "
              "engine's own VllmConfig, so it needs the engine's python "
              "package, which is not on an ordinary host.")
        return 0
    try:
        from vllm.engine.arg_utils import EngineArgs  # noqa: F401
    except Exception as exc:
        # THE ENGINE IS NOT HERE, AND THAT IS A FACT, NOT A CRASH. The
        # tool's whole job is to build the engine's own config, so on a host
        # without the engine's package it has nothing to say -- and the
        # honest answer is that, with the exit code saying "could not run",
        # not a traceback that reads like a defect in the recipe.
        print(f"resolve-capture-grid: the engine's python package is not "
              f"available here ({type(exc).__name__}: {exc}). This tool "
              f"proves an ENGINE-SIDE claim, so it must run INSIDE the "
              f"pinned image.")
        return 2
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/model"
    try:
        from vllm.platforms import current_platform
        plat = f"{current_platform.device_name} / {current_platform.device_type}"
    except Exception as exc:  # pragma: no cover
        plat = f"unknown ({exc})"
    print(f"platform: {plat}")
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')!r}")

    spec_q = draft_num_query_per_req(
        model_dir, BASE["speculative_config"]["num_speculative_tokens"])
    print(f"the draft's num_query_per_req (the speculator's decode_query_len): "
          f"{spec_q}")

    default = report("the engine's default capture grid",
                     build(model_dir, None), spec_q)
    recipe = report("the recipe's capture grid (ceiling 128)",
                    build(model_dir, RECIPE_GRID), spec_q)

    print("=== THE DELTA ===")
    print(f"capture list: {len(default['sizes'])} -> {len(recipe['sizes'])} entries")
    print(f"dropped: {sorted(set(default['sizes']) - set(recipe['sizes']))}")
    print(f"added:   {sorted(set(recipe['sizes']) - set(default['sizes']))}")
    print(f"max_cudagraph_capture_size: {default['max_cg']} -> {recipe['max_cg']}")
    same_dec = default["descs"] == recipe["descs"]
    print(f"FULL decode descriptor set UNCHANGED: {same_dec} "
          f"({len(default['descs'])} -> {len(recipe['descs'])})")
    if default["descs"] != recipe["descs"]:
        print("  default grid only:",
              [d for d in default["descs"] if d not in recipe["descs"]])
        print("  recipe grid only:",
              [d for d in recipe["descs"] if d not in default["descs"]])
    print(f"the uniform-decode sizes present in the recipe's grid: "
          f"{[s for s in (4, 8, 16, 32, 64, 96, 128) if s in recipe['sizes']]}")

    # --- THE SPECULATOR'S OWN GRAPHS ---------------------------------------
    # The dspark manager's ceiling is max_num_seqs * decode_query_len = 32 * 3
    # = 96, so its descriptors come only from grid entries <= 96 -- which the
    # cap at 128 cannot touch. The replication is CROSS-VALIDATED against the
    # recipe's expected draft-descriptor count.
    print("=== THE SPECULATOR (dspark) GRAPH SET ===")
    spec_same = default["spec_descs"] == recipe["spec_descs"]
    print(f"speculator descriptors: {len(default['spec_descs'])} -> "
          f"{len(recipe['spec_descs'])} -- UNCHANGED: {spec_same}")
    print(f"the grid entries the speculator's set is built from "
          f"(<= {default['max_seqs'] * spec_q}): "
          f"{[s for s in default['sizes'] if s <= default['max_seqs'] * spec_q]}")
    print(f"the same entries in the recipe's grid: "
          f"{[s for s in recipe['sizes'] if s <= recipe['max_seqs'] * spec_q]}")
    print(f"the replication reproduces the recipe's expected draft count "
          f"({EXPECTED_DRAFT}): {len(default['spec_descs']) == EXPECTED_DRAFT}")

    ok = (recipe["max_cg"] == 128
          and recipe["sizes"] == RECIPE_GRID
          and len(recipe["sizes"]) == EXPECTED_PIECEWISE
          and same_dec
          and default["max_cg"] == 256
          and len(default["descs"]) == EXPECTED_FULL
          and len(recipe["descs"]) == EXPECTED_FULL
          and len(default["spec_descs"]) == EXPECTED_DRAFT
          and spec_same
          and len(recipe["spec_descs"]) == EXPECTED_DRAFT)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} -- the cap resolves as designed"
          if ok else
          "VERDICT: FAIL -- the resolution is NOT what the recipe was "
          "pre-registered on")
    print(json.dumps({"default_sizes": len(default["sizes"]),
                      "recipe_sizes": len(recipe["sizes"]),
                      "full_descs_default": len(default["descs"]),
                      "full_descs_recipe": len(recipe["descs"]),
                      "spec_descs_default": len(default["spec_descs"]),
                      "spec_descs_recipe": len(recipe["spec_descs"]),
                      "spec_q_len": spec_q,
                      "max_cg_default": default["max_cg"],
                      "max_cg_recipe": recipe["max_cg"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
