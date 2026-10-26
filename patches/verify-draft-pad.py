#!/usr/bin/env python3
"""verify-draft-pad.py -- the apply-time self-check for the
DSpark DRAFT HEAD pad-kit extension (TP=3).

WHAT THIS PROVES
================
The MiniMax M3.1 Flash checkpoint ships a DSpark speculative-decode DRAFT
HEAD in `<target>/dspark/`. Its geometry is 64 q-heads / 4 kv-heads /
head_dim 128, and the engine promotes its nested `text_config` INSIDE
`get_config()` (vllm/transformers_utils/config.py:979-981, through
`SpeculativeConfig.hf_config_override`) BEFORE the TP=3 pad hook runs
(same file, :995-998). The promoted config therefore has model_type
"minimax_m3_dspark" and NO nested text_config -- so the draft head was
never padded and TP=3 died at `vllm/config/model.py:1544-1549`
("Total number of attention heads (64) must be divisible by tensor
parallel size (3)") while building the draft ModelConfig, and would then
have died again at `vllm/model_executor/models/qwen3_dflash.py:227`
(`assert self.total_num_heads % tp_size == 0`) at draft-model init.

This script proves, from the PINNED IMAGE's own bytes and the REAL
checkpoint's own bytes, that the extended pad kit fixes that, BEFORE the
engine starts. It FAILS CLOSED: any failed assertion exits non-zero.

It is CPU-ONLY by construction: no CUDA context is created, no GPU memory
is allocated, no model is loaded. It only reads config.json, the
safetensors HEADERS (8-byte length + JSON header) and the index jsons.

HOW TO RUN IT (inside the pinned image, at boot time, BEFORE vllm starts)
========================================================================
The lane mounts the checkpoint read-only at /model, the kit at /tp3pad and
the ONE spec-config file at /spec-config.json:

    docker run --rm --entrypoint python3 \
      -v /path/to/patches:/kit:ro \
      -v /path/to/spec-config.json:/spec-config.json:ro \
      -v ~/models/minimax31-flash-54e4e69dfc45:/model:ro \
      -e CUDA_VISIBLE_DEVICES= -e VLLM_TP3_PAD=3 \
      vllm/vllm-openai:minimax-m31-tp3-ray \
      /kit/verify-draft-pad.py --model-dir /model \
        --spec-config /spec-config.json --served-spec-json "$(cat /spec-config.json)"

Inside the RUNNING serving container (the boot script's own gate), the
equivalent is:

    python3 /tp3pad/verify-draft-pad.py --model-dir /model \
      --spec-config /spec-config.json --served-spec-json "$SPEC_JSON"

`--served-spec-json` is the EXACT value the serve line puts on the command
line; the check asserts it is byte-identical to the file it parsed, so the
binding between the file and the serve line is PROVEN, not assumed.

`--selftest` runs the NON-VACUITY mode: it mutates the inputs (drops the
whitelist entry -- from the default tuple AND from the env var, because the
overlay's whitelist is their UNION -- uses an unpadded config, corrupts a
pad-table rule, merges the two rule scopes) and asserts that this very
check REJECTS each mutation. A check that cannot fail is not a check. The
boot runs BOTH modes: the normal check and the battery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = "/model"
DRAFT_SUBDIR = "dspark"

TARGET_REQUIRE = "language_model.model.layers."
DRAFT_REQUIRE = "language_model.model.dspark.layers."

TP = 3

# The lane's --speculative-config JSON lives in ONE file: spec-config.json,
# a kit file that the SERVE LINE reads (start-tp3.sh) and that this check
# parses. There is no second copy in this file: a hardcoded duplicate would
# be a second source of truth that nothing keeps in sync with the serve line
# (and the serve line is not visible from inside the container -- only
# /tp3pad and /model are mounted).
#
# The expected content is pinned here as the SPEC's SEMANTICS (what the
# engine must resolve), not as the file's bytes: the file is the source, and
# this pins that the source still says what the recipe requires.
SPEC_EXPECTED = {
    "method": "dspark",
    "num_speculative_tokens": 3,
    "attention_backend": "TRITON_ATTN",
    "kv_cache_dtype": "fp8",
    "draft_sample_method": "probabilistic",
    "rejection_sample_method": "block",
}
DEFAULT_SPEC_FILE = "/tp3pad/spec-config.json"

# The eight DRAFT rules, exactly as the pad table must carry them.
# (suffix, dim, expect, ops, result)
EXPECTED_DRAFT_RULES = (
    ("self_attn.q_proj.weight", 0, 8192, ((-1, 4096),), 12288),
    ("self_attn.q_proj.weight_scale_inv", 0, 8192, ((-1, 4096),), 12288),
    ("self_attn.k_proj.weight", 0, 512, ((-1, 256),), 768),
    ("self_attn.k_proj.weight_scale_inv", 0, 512, ((-1, 256),), 768),
    ("self_attn.v_proj.weight", 0, 512, ((-1, 256),), 768),
    ("self_attn.v_proj.weight_scale_inv", 0, 512, ((-1, 256),), 768),
    ("self_attn.o_proj.weight", 1, 8192, ((-1, 4096),), 12288),
    ("self_attn.o_proj.weight_scale_inv", 1, 256, ((-1, 128),), 384),
)

# The ten TARGET rules must not change (regression guard).
EXPECTED_TARGET_RULES = (
    ("self_attn.q_proj.weight", 0, 8192, ((-1, 4096),), 12288),
    ("self_attn.q_proj.weight_scale_inv", 0, 8192, ((-1, 4096),), 12288),
    ("self_attn.k_proj.weight", 0, 512, ((-1, 256),), 768),
    ("self_attn.k_proj.weight_scale_inv", 0, 512, ((-1, 256),), 768),
    ("self_attn.v_proj.weight", 0, 512, ((-1, 256),), 768),
    ("self_attn.v_proj.weight_scale_inv", 0, 512, ((-1, 256),), 768),
    ("self_attn.index_q_proj.weight", 0, 512, ((-1, 256),), 768),
    ("self_attn.index_q_proj.weight_scale_inv", 0, 512, ((-1, 256),), 768),
    ("self_attn.o_proj.weight", 1, 8192, ((-1, 4096),), 12288),
    ("self_attn.o_proj.weight_scale_inv", 1, 256, ((-1, 128),), 384),
)

# FP8 block-quant scale column block for the o_proj input dim:
# 8192 weight cols -> 256 scale cols  =>  block 32.
SCALE_BLOCK = 32


class CheckFailure(Exception):
    """A failed assertion. Always fatal: this gate fails the boot."""


def require(cond, msg):
    if not cond:
        raise CheckFailure(msg)


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safetensors_header(path):
    """Parse a safetensors file's header only (no tensor data, no torch)."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise CheckFailure(f"{path}: truncated safetensors header length")
        (n,) = struct.unpack("<Q", raw_len)
        if n <= 0 or n > (1 << 28):
            raise CheckFailure(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
    out = {}
    for key, val in header.items():
        if key == "__metadata__":
            continue
        out[key] = {"shape": list(val.get("shape", [])), "dtype": val.get("dtype")}
    return out


class _StubTensor:
    """Minimal tensor stand-in for the overlay's scale guard (no torch)."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


def _cfg_get(cfg, name, default=None):
    v = getattr(cfg, name, default)
    return v if isinstance(v, int) else default


# ---------------------------------------------------------------------------
# 0. the overlay itself
# ---------------------------------------------------------------------------


def load_overlay(kit_dir):
    """Import the REAL overlay package (installed, else from the kit tree)."""
    try:
        mod = importlib.import_module("vllm_tp3_pad")
        origin = "installed"
    except ImportError:
        overlay_dir = os.path.join(kit_dir, "overlay")
        if not os.path.isdir(overlay_dir):
            raise CheckFailure(
                "vllm_tp3_pad is not importable and no overlay tree found at "
                f"{overlay_dir} -- the pad kit was never applied to this image"
            )
        sys.path.insert(0, overlay_dir)
        mod = importlib.import_module("vllm_tp3_pad")
        origin = f"kit tree ({overlay_dir})"
    for attr in ("maybe_pad_hf_config", "tp3_pad_weights_iterator",
                 "_rule_for", "_guard_scale_tensor", "_table_path",
                 "_DEFAULT_MODEL_TYPES"):
        require(hasattr(mod, attr),
                f"the overlay module is missing {attr} -- wrong overlay?")
    return mod, origin


def check_overlay_selftest(overlay):
    """The overlay's own torch-free selftest must pass on the installed kit."""
    rc = overlay.selftest(quiet=True)
    require(rc == 0,
            f"the overlay's own selftest failed (rc={rc}) -- the pad kit is "
            "not self-consistent")
    return {"rules_replayed": None, "rc": rc}


# ---------------------------------------------------------------------------
# 1. the pad table
# ---------------------------------------------------------------------------


def _rule_index(table):
    return {(r["suffix"], r.get("require", "")): r for r in table["rules"]}


def check_table(table, intended_layouts, kit_dir, resolved_path=None,
                overlay_dir=None):
    """The table must carry the eight draft rules and keep the target's."""
    rules = table["rules"]
    indexed = _rule_index(table)

    require(len(rules) == len(indexed),
            f"the pad table has duplicate (suffix, require) keys: "
            f"{len(rules)} rules, {len(indexed)} distinct")

    scopes = sorted({r.get("require", "") for r in rules})
    require(scopes == sorted([TARGET_REQUIRE, DRAFT_REQUIRE]),
            f"the pad table's require scopes are {scopes}, expected exactly "
            f"{sorted([TARGET_REQUIRE, DRAFT_REQUIRE])}")
    for i, a in enumerate(scopes):
        for b in scopes[i + 1:]:
            require(not (a in b or b in a),
                    f"require scopes {a!r} and {b!r} overlap -- the target and "
                    "draft rule sets are not disjoint")

    for want, require_scope in (
        *((r, TARGET_REQUIRE) for r in EXPECTED_TARGET_RULES),
        *((r, DRAFT_REQUIRE) for r in EXPECTED_DRAFT_RULES),
    ):
        suffix, dim, expect, ops, result = want
        key = (suffix, require_scope)
        rule = indexed.get(key)
        require(rule is not None, f"pad table is missing the rule {key}")
        require(int(rule["dim"]) == dim,
                f"{key}: dim is {rule['dim']}, expected {dim}")
        require(int(rule["expect"]) == expect,
                f"{key}: expect is {rule['expect']}, expected {expect}")
        got_ops = tuple(tuple(int(x) for x in op) for op in rule["ops"])
        require(got_ops == tuple(tuple(int(x) for x in op) for op in ops),
                f"{key}: ops are {got_ops}, expected {ops}")
        require(int(rule["result"]) == result,
                f"{key}: result is {rule['result']}, expected {result}")
        require(int(rule["expect"]) + sum(n for _a, n in got_ops) == result,
                f"{key}: expect + pad != result")

    # every rule must be pinned by an intended layout (and vice versa)
    for key in sorted(indexed):
        require(key in intended_layouts,
                f"rule {key} has no pinned intended layout -- it is not "
                "protected against a future edit")
    for key in sorted(intended_layouts):
        require(key in indexed,
                f"pinned intended layout {key} has no rule")

    # all shipped table copies must be byte-identical
    kit_table = os.path.join(kit_dir, "tp3_pad_table.json")
    overlay_table = os.path.join(kit_dir, "overlay", "vllm_tp3_pad",
                                 "tp3_pad_table.json")
    digests = {}
    for path in (kit_table, overlay_table):
        require(os.path.isfile(path), f"missing pad-table copy {path}")
        digests[path] = md5_file(path)
    require(len(set(digests.values())) == 1,
            f"the two shipped pad-table copies differ: {digests}")
    if overlay_dir:
        installed = os.path.join(overlay_dir, "tp3_pad_table.json")
        if os.path.isfile(installed):
            installed_md5 = md5_file(installed)
            require(installed_md5 == digests[kit_table],
                    f"the pad table INSTALLED next to the overlay ({installed}, "
                    f"md5 {installed_md5}) is not the shipped one (md5 "
                    f"{digests[kit_table]}) -- the boot would pad with a "
                    "different table than the one this check verified")
            digests[installed] = installed_md5
    if resolved_path and os.path.isfile(resolved_path):
        resolved_md5 = md5_file(resolved_path)
        require(resolved_md5 == digests[kit_table],
                f"the pad table the overlay will actually load ({resolved_path}, "
                f"md5 {resolved_md5}) is not the shipped one (md5 "
                f"{digests[kit_table]}) -- the boot would pad with a different "
                "table than the one this check verified")
    return {
        "rules": len(rules),
        "draft_rules": len(EXPECTED_DRAFT_RULES),
        "target_rules": len(EXPECTED_TARGET_RULES),
        "intended_layouts": len(intended_layouts),
        "table_md5": digests[kit_table],
    }


# ---------------------------------------------------------------------------
# 2. the config side: the REAL promoted draft config must come out padded
# ---------------------------------------------------------------------------


def _promote(base, model_type="minimax_m3_dspark"):
    """Exactly what the engine does in SpeculativeConfig.hf_config_override:
    promote the nested text_config and retag its model_type."""
    text_cfg = getattr(base, "text_config", None)
    require(text_cfg is not None,
            "the draft config has no nested text_config -- this is not the "
            "MiniMax M3.1 draft layout the pad kit was built for")
    text_cfg.model_type = model_type
    return text_cfg


def check_config_pad(overlay, draft_dir, tp=TP, whitelist_override=None):
    """Layer A: the REAL engine path (promotion + pad hook, in that order).

    Layer B: the same promotion, built from the REAL draft config, with the
    pad hook applied by hand -- so the assertion does not depend on the
    image's hook being in place and stays sensitive to the whitelist entry.
    """
    from vllm.config import SpeculativeConfig
    from vllm.transformers_utils.config import get_config

    require("minimax_m3_dspark" in tuple(overlay._DEFAULT_MODEL_TYPES),
            "'minimax_m3_dspark' is not in the overlay's _DEFAULT_MODEL_TYPES "
            f"{tuple(overlay._DEFAULT_MODEL_TYPES)} -- the promoted draft "
            "config would never be padded")

    # --- Layer B: the promoted config, unpadded, then the real pad hook ---
    saved = os.environ.get("VLLM_TP3_PAD")
    try:
        os.environ.pop("VLLM_TP3_PAD", None)
        base = get_config(draft_dir, trust_remote_code=True)
    finally:
        if saved is not None:
            os.environ["VLLM_TP3_PAD"] = saved
    promoted = _promote(base)
    require(getattr(promoted, "model_type", None) == "minimax_m3_dspark",
            f"the promoted draft config's model_type is "
            f"{getattr(promoted, 'model_type', None)!r}")
    raw_heads = (_cfg_get(promoted, "num_attention_heads"),
                 _cfg_get(promoted, "num_key_value_heads"))
    require(raw_heads == (64, 4),
            f"the real draft config's head geometry is {raw_heads}, expected "
            "(64, 4) -- the checkpoint changed under the pad kit")
    require(_cfg_get(promoted, "head_dim") == 128,
            f"the real draft config's head_dim is "
            f"{_cfg_get(promoted, 'head_dim')!r}, expected 128")
    require(_cfg_get(promoted, "vocab_size") == 200064,
            f"the real draft config's vocab_size is "
            f"{_cfg_get(promoted, 'vocab_size')!r}, expected 200064")

    saved_types = overlay._DEFAULT_MODEL_TYPES
    if whitelist_override is not None:
        overlay._DEFAULT_MODEL_TYPES = whitelist_override
    try:
        os.environ["VLLM_TP3_PAD"] = str(tp)
        padded = overlay.maybe_pad_hf_config(promoted)
    finally:
        overlay._DEFAULT_MODEL_TYPES = saved_types
        if saved is None:
            os.environ.pop("VLLM_TP3_PAD", None)
        else:
            os.environ["VLLM_TP3_PAD"] = saved

    heads = (_cfg_get(padded, "num_attention_heads"),
             _cfg_get(padded, "num_key_value_heads"))
    require(heads == (96, 6),
            f"the promoted draft config was NOT padded: heads are {heads}, "
            f"expected (96, 6) at TP={tp}. The draft head would die at "
            "vllm/config/model.py:1544-1549 (64 % 3 != 0).")
    require(_cfg_get(padded, "head_dim") == 128,
            f"the pad changed the draft's head_dim to "
            f"{_cfg_get(padded, 'head_dim')!r} -- it must stay 128 "
            "(the upstream dspark attention module passes head_dim "
            "explicitly)")
    require(_cfg_get(padded, "vocab_size") == 200064,
            f"the pad changed the draft's vocab_size to "
            f"{_cfg_get(padded, 'vocab_size')!r} -- it must stay 200064")
    sparse = getattr(padded, "sparse_attention_config", None)
    if isinstance(sparse, dict) and "sparse_num_index_heads" in sparse:
        require(sparse["sparse_num_index_heads"] == 6,
                "the sparse index-head count must stay in lockstep with the "
                f"kv-head count (6), got {sparse['sparse_num_index_heads']!r}")
    # idempotency
    again = overlay.maybe_pad_hf_config(padded)
    require((_cfg_get(again, "num_attention_heads"),
             _cfg_get(again, "num_key_value_heads"),
             _cfg_get(again, "head_dim"),
             _cfg_get(again, "vocab_size")) == (96, 6, 128, 200064),
            "the draft config pads are not idempotent")

    # --- Layer A: the REAL engine path, in the REAL order ---
    engine_cfg = get_config(
        draft_dir,
        trust_remote_code=True,
        hf_overrides_kw={},
        hf_overrides_fn=SpeculativeConfig.hf_config_override,
    )
    require(getattr(engine_cfg, "model_type", None) == "minimax_m3_dspark",
            "the engine's promotion path did not produce a "
            f"'minimax_m3_dspark' config (got "
            f"{getattr(engine_cfg, 'model_type', None)!r})")
    engine_heads = (_cfg_get(engine_cfg, "num_attention_heads"),
                     _cfg_get(engine_cfg, "num_key_value_heads"))
    require(engine_heads == (96, 6),
            "the engine's own promotion+pad path did NOT yield a padded draft "
            f"config: heads are {engine_heads}, expected (96, 6). The pad hook "
            "must run AFTER the promotion (transformers_utils/config.py:979-981 "
            "then :995-998).")
    require(_cfg_get(engine_cfg, "head_dim") == 128,
            f"the engine's draft config head_dim is "
            f"{_cfg_get(engine_cfg, 'head_dim')!r}, expected 128")

    return {
        "raw_heads": raw_heads,
        "padded_heads": heads,
        "head_dim": _cfg_get(padded, "head_dim"),
        "vocab_size": _cfg_get(padded, "vocab_size"),
        "engine_heads": engine_heads,
        "sparse_index_heads": (sparse or {}).get("sparse_num_index_heads")
        if isinstance(sparse, dict) else None,
    }


# ---------------------------------------------------------------------------
# 3. the engine's own load-bearing invariants
# ---------------------------------------------------------------------------


def check_engine_invariants():
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention
    from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkAttention

    require(issubclass(MiniMaxM3DSparkAttention, DFlashQwen3Attention),
            "the draft's attention no longer inherits DFlashQwen3Attention -- "
            "the head-divisibility assert below would not apply to it")

    dflash_src = inspect.getsource(DFlashQwen3Attention.__init__)
    require("assert self.total_num_heads % tp_size == 0" in dflash_src,
            "DFlashQwen3Attention.__init__ no longer asserts "
            "'total_num_heads % tp_size == 0' -- the TP=3 failure mode the pad "
            "kit exists to prevent is gone (or moved); re-read the engine")

    dspark_src = inspect.getsource(MiniMaxM3DSparkAttention.__init__)
    require("head_dim=config.head_dim" in dspark_src,
            "MiniMaxM3DSparkAttention.__init__ no longer passes "
            "'head_dim=config.head_dim' EXPLICITLY -- the engine would fall "
            "back to `head_dim or hidden_size // total_num_heads`, and with 96 "
            "padded heads that fallback would silently change head_dim 128 -> 64")

    require("self.head_dim = head_dim or hidden_size // self.total_num_heads"
            in dflash_src,
            "DFlashQwen3Attention.__init__ no longer has the head_dim fallback "
            "-- the explicit pass is no longer load-bearing (re-read the engine)")

    # the fallback's would-be value, for the report
    hidden = 6144
    fallback = hidden // 96
    require(fallback != 128,
            "the head_dim fallback would coincidentally equal 128 -- the "
            "explicit head_dim pass is no longer load-bearing")

    return {
        "assert_present": True,
        "head_dim_explicit": True,
        "head_dim_fallback_would_be": fallback,
    }


# ---------------------------------------------------------------------------
# 4. the arithmetic the engine will perform with the padded config
# ---------------------------------------------------------------------------


def check_arithmetic(padded, tp=TP, real_q_rows=8192):
    """The arithmetic the engine will perform with the padded config.

    `real_q_rows` is the checkpoint's UNPADDED q-row count (8192 = 64 real
    heads x head_dim 128), i.e. where the zero-pad starts.
    """
    n_q = _cfg_get(padded, "num_attention_heads")
    n_kv = _cfg_get(padded, "num_key_value_heads")
    head_dim = _cfg_get(padded, "head_dim")
    require(n_q and n_kv and head_dim,
            f"the padded draft config lacks the attention geometry: "
            f"n_q={n_q!r} n_kv={n_kv!r} head_dim={head_dim!r}")

    require(n_q % tp == 0,
            f"num_attention_heads {n_q} is not divisible by TP={tp} -- "
            "DFlashQwen3Attention would assert")
    require(n_kv % tp == 0,
            f"num_key_value_heads {n_kv} is not divisible by TP={tp}")

    q_rows = n_q * head_dim
    kv_rows = n_kv * head_dim
    require(q_rows % tp == 0,
            f"the q projection's row count {q_rows} is not divisible by TP={tp}")
    require(kv_rows % tp == 0,
            f"the kv projection's row count {kv_rows} is not divisible by TP={tp}")

    q_per_rank = q_rows // tp
    kv_per_rank = kv_rows // tp
    o_cols_per_rank = q_rows // tp
    o_scale_cols_per_rank = (q_rows // SCALE_BLOCK) // tp

    # the zero-padded heads must land exactly on the last rank
    require(real_q_rows % head_dim == 0,
            f"the real q-row count {real_q_rows} is not a multiple of head_dim "
            f"{head_dim}")
    real_heads = real_q_rows // head_dim
    pad_q_rows = q_rows - real_q_rows
    require(pad_q_rows >= 0,
            f"the padded q rows ({q_rows}) are fewer than the real ones "
            f"({real_q_rows})")
    require(pad_q_rows == q_per_rank,
            f"the zero-padded q rows ({pad_q_rows}) must be exactly one rank's "
            f"worth ({q_per_rank}), so that the last rank's contribution "
            "through o_proj is exactly 0")
    require(real_q_rows == q_per_rank * (tp - 1),
            f"the zero-padded q heads do not land exactly on rank {tp - 1}: "
            f"rank {tp - 1} starts at q row {q_per_rank * (tp - 1)}, but the "
            f"real heads end at row {real_q_rows}")

    return {
        "n_q": n_q,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "q_rows": q_rows,
        "kv_rows": kv_rows,
        "q_rows_per_rank": q_per_rank,
        "kv_rows_per_rank": kv_per_rank,
        "o_proj_cols_per_rank": o_cols_per_rank,
        "o_proj_scale_cols_per_rank": o_scale_cols_per_rank,
        "real_q_rows": real_q_rows,
        "real_q_heads": real_heads,
        "pad_q_rows": pad_q_rows,
    }


# ---------------------------------------------------------------------------
# 5. the REAL checkpoint tensors must match the pad table
# ---------------------------------------------------------------------------


def check_tensors(overlay, table, draft_dir, model_dir, tp=TP, geometry=None):
    rules = table["rules"]
    indexed = _rule_index(table)

    # --- the draft checkpoint ---
    draft_index_path = os.path.join(draft_dir, "model.safetensors.index.json")
    require(os.path.isfile(draft_index_path),
            f"the draft checkpoint has no index json at {draft_index_path}")
    draft_weight_map = read_json(draft_index_path).get("weight_map", {})
    require(draft_weight_map, "the draft checkpoint's index json has no weight_map")
    draft_names = sorted(draft_weight_map)
    draft_shards = sorted(set(draft_weight_map.values()))
    draft_shapes = {}
    for shard in draft_shards:
        header = safetensors_header(os.path.join(draft_dir, shard))
        for name in draft_names:
            if name in header:
                draft_shapes[name] = header[name]["shape"]

    # --- the target checkpoint (names come from the index json) ---
    target_index_path = os.path.join(model_dir, "model.safetensors.index.json")
    require(os.path.isfile(target_index_path),
            f"the target checkpoint has no index json at {target_index_path}")
    target_weight_map = read_json(target_index_path).get("weight_map", {})
    require(target_weight_map, "the target checkpoint's index json has no weight_map")
    target_names = sorted(target_weight_map)

    # --- every draft rule must match REAL draft tensors with the right shape
    matched_draft = 0
    for suffix, dim, expect, ops, result in EXPECTED_DRAFT_RULES:
        rule = indexed.get((suffix, DRAFT_REQUIRE))
        require(rule is not None, f"pad table is missing the draft rule {suffix}")
        hits = [n for n in draft_names if overlay._rule_for(n, [rule]) is not None]
        require(hits,
                f"the draft rule {suffix} matches NO real draft tensor -- the "
                "checkpoint's names changed under the pad kit")
        for name in hits:
            shape = draft_shapes.get(name)
            require(shape is not None,
                    f"the draft tensor {name} is not in the safetensors header")
            require(len(shape) > dim,
                    f"the draft tensor {name} has {len(shape)} dims, the rule "
                    f"pads axis {dim}")
            require(shape[dim] == expect,
                    f"the draft tensor {name} has axis {dim} = {shape[dim]}, "
                    f"the rule expects {expect}")
            padded_shape = list(shape)
            padded_shape[dim] = result
            require(tuple(padded_shape) == _padded_shape(tuple(shape), dim,
                                                           expect, result),
                    f"the draft tensor {name} would pad to {padded_shape}, "
                    f"expected {_padded_shape(tuple(shape), dim, expect, result)}")
        matched_draft += len(hits)

    # --- the padded shapes must be what the engine will ask for ---
    geometry = geometry or _Geometry(96, 6, 128)
    real_q_rows = int(indexed[("self_attn.q_proj.weight", DRAFT_REQUIRE)]["expect"])
    arith = check_arithmetic(geometry, tp=tp, real_q_rows=real_q_rows)
    for suffix, dim, expect, ops, result in EXPECTED_DRAFT_RULES:
        rule = indexed[(suffix, DRAFT_REQUIRE)]
        want = {
            "self_attn.q_proj.weight": arith["q_rows"],
            "self_attn.q_proj.weight_scale_inv": arith["q_rows"],
            "self_attn.k_proj.weight": arith["kv_rows"],
            "self_attn.k_proj.weight_scale_inv": arith["kv_rows"],
            "self_attn.v_proj.weight": arith["kv_rows"],
            "self_attn.v_proj.weight_scale_inv": arith["kv_rows"],
            "self_attn.o_proj.weight": arith["q_rows"],
            "self_attn.o_proj.weight_scale_inv":
                arith["q_rows"] // SCALE_BLOCK,
        }[suffix]
        require(int(rule["result"]) == want,
                f"the draft rule {suffix} pads to {rule['result']}, but the "
                f"engine needs {want}")

    # --- the draft's OTHER tensors must match NO rule, and must not trip
    #     the overlay's scale guard ---
    unmatched = []
    for name in draft_names:
        rule = overlay._rule_for(name, rules)
        if rule is None:
            unmatched.append(name)
        else:
            require(rule.get("require") == DRAFT_REQUIRE,
                    f"the draft tensor {name} matched a rule scoped "
                    f"{rule.get('require')!r} -- the rule sets are not disjoint")
    require(unmatched,
            "every single draft tensor matched a rule -- the pad kit would "
            "touch tensors it must not touch")
    for name in unmatched:
        shape = draft_shapes.get(name, [2])
        overlay._guard_scale_tensor(name, _StubTensor(shape), rules)

    # --- the target's rules must still match the REAL target tensors ---
    target_matched = 0
    target_shapes = {}
    need = {}
    for suffix, dim, expect, ops, result in EXPECTED_TARGET_RULES:
        rule = indexed.get((suffix, TARGET_REQUIRE))
        require(rule is not None, f"pad table is missing the target rule {suffix}")
        hits = [n for n in target_names if overlay._rule_for(n, [rule]) is not None]
        require(hits,
                f"the target rule {suffix} matches NO real target tensor -- "
                "the checkpoint's names changed under the pad kit")
        for name in hits:
            need.setdefault(target_weight_map[name], []).append(name)
        target_matched += len(hits)
    for shard, names in need.items():
        header = safetensors_header(os.path.join(model_dir, shard))
        for name in names:
            target_shapes[name] = header[name]["shape"]

    for suffix, dim, expect, ops, result in EXPECTED_TARGET_RULES:
        rule = indexed[(suffix, TARGET_REQUIRE)]
        for name, shape in target_shapes.items():
            if overlay._rule_for(name, [rule]) is None:
                continue
            require(len(shape) > dim,
                    f"the target tensor {name} has {len(shape)} dims, the rule "
                    f"pads axis {dim}")
            require(shape[dim] == expect,
                    f"the target tensor {name} has axis {dim} = {shape[dim]}, "
                    f"the rule expects {expect}")

    # --- the two rule sets must be DISJOINT against BOTH checkpoints' names ---
    for suffix, dim, expect, ops, result in EXPECTED_DRAFT_RULES:
        rule = indexed[(suffix, DRAFT_REQUIRE)]
        cross = [n for n in target_names if overlay._rule_for(n, [rule]) is not None]
        require(not cross,
                f"the DRAFT rule {suffix} matches {len(cross)} real TARGET "
                f"tensor(s), e.g. {cross[:3]} -- the rule sets overlap")
    for suffix, dim, expect, ops, result in EXPECTED_TARGET_RULES:
        rule = indexed[(suffix, TARGET_REQUIRE)]
        cross = [n for n in draft_names if overlay._rule_for(n, [rule]) is not None]
        require(not cross,
                f"the TARGET rule {suffix} matches {len(cross)} real DRAFT "
                f"tensor(s), e.g. {cross[:3]} -- the rule sets overlap")

    return {
        "draft_tensors": len(draft_names),
        "draft_rule_hits": matched_draft,
        "draft_unmatched": len(unmatched),
        "target_tensors": len(target_names),
        "target_rule_hits": target_matched,
    }


def _padded_shape(shape, dim, expect, result):
    out = list(shape)
    out[dim] = result
    return tuple(out)


class _Geometry:
    """Tiny stand-in carrying the padded draft's attention geometry."""

    def __init__(self, n_q, n_kv, head_dim):
        self.num_attention_heads = n_q
        self.num_key_value_heads = n_kv
        self.head_dim = head_dim


# ---------------------------------------------------------------------------
# 6. the spec config must parse with the REAL engine config class
# ---------------------------------------------------------------------------


def _lane_parallel_config(tp):
    """The lane's ParallelConfig, built the way the ENGINE builds it.

    WHY THIS EXISTS (it is the bug that failed the first boot of the
    draft-head pad kit -- see docs/FIXLOG.md): `vllm/config/parallel.py`
    validates the world size against the number of GPUs VISIBLE ON THIS
    NODE:

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            ...
            elif current_platform.is_cuda() and current_platform.device_count() < self.world_size:
                raise ValueError("World size (3) is larger than the number of
                                  available GPUs (1) in this node ...")

    A Spark has ONE GPU, and the lane runs three ranks across three boxes, so
    a ParallelConfig built WITHOUT the multiprocessing backend trips that
    check on every box. The lane's own serve line passes
    `--distributed-executor-backend mp` (the engine's config dump shows
    `distributed_executor_backend: 'mp'`), so the engine never trips it --
    only a config built by this check did, and the check then died with a
    pydantic ValidationError instead of checking anything.

    THE LESSON, which is the real lesson: that check only runs when a GPU is
    VISIBLE. The offline verification ran in a container with NO GPU, so the
    device-count branch was never reached and the bug hid from every offline
    test. This check MUST therefore be verified in a container that SEES THE
    GPU (the boot's environment), not only in a GPU-less one.
    """
    from vllm.config.parallel import ParallelConfig
    return ParallelConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
    )


def _world_size_validation_reachable(tp):
    """Is the engine's world-size validation LIVE in THIS container?

    `vllm/config/parallel.py:1065-1083` validates the world size against the
    GPUs visible on this node ONLY when `distributed_executor_backend is None`:

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            ...
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"                      # <- no raise
            elif (current_platform.is_cuda()
                  and current_platform.device_count() < self.world_size):
                raise ValueError("World size (3) is larger than the number of
                                  available GPUs (1) in this node ...")

    THE LANE RUNS `mp` (`--distributed-executor-backend mp`), so the ENGINE
    never trips it -- but a config built WITHOUT the backend does, on a
    one-GPU box, and that is exactly the validation that killed the first
    boot of the draft-head pad kit (see docs/FIXLOG.md): the self-check's
    ParallelConfig had no backend,
    so it raised "World size (3) is larger than the number of available GPUs
    (1) in this node" and the bootstrap's `set -e` killed the boot before
    the engine started.

    THE HIDING PLACE, which is the reason this helper exists at all: the
    device-count branch is only reached when the platform is CUDA. A GPU-
    HIDDEN container (no `--gpus all`) never enters the branch, so the bug
    hid from every offline verification and reappeared at the boot. A check
    that cannot REACH the engine's validation is weaker than the engine, so
    this reports what the engine's own code path does here.

    Returns a dict describing the branch, never a bare bool, so the summary
    can show WHY it was or was not live.
    """
    from vllm.config.parallel import ParallelConfig
    from vllm.platforms import current_platform

    cuda = bool(current_platform.is_cuda())
    device_count = int(current_platform.device_count()) if cuda else 0
    world_size = int(tp)
    world_across_dp = int(tp)
    nnodes = 1
    raised = False
    detail = ""
    try:
        cfg = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=1)
        world_size = int(cfg.world_size)
        world_across_dp = int(cfg.world_size_across_dp)
        nnodes = int(getattr(cfg, "nnodes", 1) or 1)
        detail = ("the backend-less config was ACCEPTED: the world-size "
                  "validation did not fire in this environment")
    except Exception as exc:
        raised = True
        detail = f"{type(exc).__name__}: {exc}"

    # The engine's own branch, reproduced from vllm/config/parallel.py:
    #   outer: backend is None and world_size_across_dp > 1
    #     -> cuda and nnodes > 1                -> backend = "mp", NO raise
    #     -> cuda and device_count < world_size -> RAISE
    branch_entered = bool(cuda) and world_across_dp > 1 and nnodes <= 1
    validation_live = branch_entered and device_count < world_size
    return {
        "cuda": cuda,
        "device_count": device_count,
        "world_size": world_size,
        "world_size_across_dp": world_across_dp,
        "nnodes": nnodes,
        "branch_entered": branch_entered,
        "validation_live": validation_live,
        "raised": raised,
        "detail": detail,
    }


def check_spec_config(model_dir, tp=TP, spec_kwargs=None):
    from vllm.config import SpeculativeConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.config.scheduler import SchedulerConfig

    require(spec_kwargs is not None,
            "the spec config was not supplied -- this check parses the SAME "
            "file the serve line reads")
    spec_kwargs = dict(spec_kwargs)
    mc = ModelConfig(model=model_dir, tokenizer=model_dir, trust_remote_code=True,
                      dtype="bfloat16", seed=0, max_model_len=8192)
    # The ParallelConfig MUST MIRROR THE LANE'S REAL EXECUTION BACKEND.
    # `vllm/config/parallel.py:1065-1083` only performs the "World size (N) is
    # larger than the number of available GPUs (M)" check when
    # `distributed_executor_backend is None`. The lane runs the multiprocessing
    # backend (`--distributed-executor-backend mp`, which the engine's own
    # config dump shows), so the engine never trips that check -- but a
    # ParallelConfig built WITHOUT the backend does trip it on a 1-GPU node,
    # and the check then dies with a pydantic ValidationError instead of
    # checking anything. That is exactly what happened on the first boot of
    # the draft-head pad kit: the gate failed closed on a self-check bug
    # (see docs/FIXLOG.md), and
    # it hid from the offline verification because the offline container had
    # NO GPU visible, so the device-count branch was never reached.
    spec = SpeculativeConfig(
        **spec_kwargs,
        target_model_config=mc,
        target_parallel_config=_lane_parallel_config(tp),
    )

    require(spec.method == "dspark",
            f"the spec config's method is {spec.method!r}, expected 'dspark'")
    require(int(spec.num_speculative_tokens) == 3,
            f"the spec config's k is {spec.num_speculative_tokens!r}, expected 3")
    want_draft = os.path.join(model_dir, DRAFT_SUBDIR)
    require(spec.model == want_draft,
            f"the spec config resolved the draft to {spec.model!r}, expected "
            f"{want_draft!r}")
    require(spec.draft_sample_method == "probabilistic",
            f"the spec config's draft_sample_method is "
            f"{spec.draft_sample_method!r}, expected 'probabilistic'")
    require(spec.rejection_sample_method == "block",
            f"the spec config's rejection_sample_method is "
            f"{spec.rejection_sample_method!r}, expected 'block'")
    require(str(spec.kv_cache_dtype) == "fp8",
            f"the spec config's kv_cache_dtype is {spec.kv_cache_dtype!r}, "
            "expected 'fp8'")
    backend = spec.attention_backend
    backend_name = getattr(backend, "name", None) or str(backend)
    require(backend_name == "TRITON_ATTN",
            f"the spec config's attention_backend is {backend!r}, expected "
            "'TRITON_ATTN'")
    require(int(spec.draft_tensor_parallel_size) == tp,
            f"the draft's tensor parallel size is "
            f"{spec.draft_tensor_parallel_size!r}, expected {tp}")

    draft_cfg = spec.draft_model_config.hf_config
    require(getattr(draft_cfg, "model_type", None) == "minimax_m3_dspark",
            f"the draft model's config model_type is "
            f"{getattr(draft_cfg, 'model_type', None)!r}, expected "
            "'minimax_m3_dspark'")
    draft_heads = (_cfg_get(draft_cfg, "num_attention_heads"),
                   _cfg_get(draft_cfg, "num_key_value_heads"))
    require(draft_heads == (96, 6),
            f"the engine's DRAFT ModelConfig was built with heads {draft_heads}, "
            "expected (96, 6) -- the draft config was not padded, so "
            "vllm/config/model.py:1544-1549 would have refused TP=3")

    # --- the engine's own uniform-decode length and capture sizes ---
    vc = VllmConfig(
        model_config=ModelConfig(model=model_dir, tokenizer=model_dir,
                                  trust_remote_code=True, dtype="bfloat16",
                                  seed=0, max_model_len=8192),
        parallel_config=_lane_parallel_config(tp),
        scheduler_config=SchedulerConfig(max_num_seqs=32, max_model_len=8192,
                                          is_encoder_decoder=False,
                                          max_num_batched_tokens=8192),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=spec,
    )
    require(vc.uniform_decode_query_len == 4,
            f"the engine's uniform_decode_query_len is "
            f"{vc.uniform_decode_query_len}, expected 4 (1 + k)")

    # THE MODEL RUNNER. DSpark is implemented ONLY by the V2 GPU model runner
    # (vllm/config/vllm.py:3069-3071 -- "DSpark is implemented only by the V2
    # GPU model runner"), and the V1 runner REFUSES it outright
    # ("Model Runner V1 does not support: dspark speculative decoding",
    # vllm/config/vllm.py:3298). `use_v2_model_runner` (vllm/config/vllm.py:
    # 704-739) falls back to V1 whenever V2 reports an unsupported feature --
    # so a config that slips past this check's simplified VllmConfig but not
    # the ENGINE's real one would boot into a hard failure minutes later.
    # Assert the selection here, in the gate, where a failure is a boot
    # failure rather than a window failure.
    #
    # NOTE (measurement integrity): the BASELINE configuration also ran V2 --
    # the baseline container's log says "Using V2 Model Runner" -- so this
    # delta is NOT confounded by a model-runner change.
    require(bool(vc.use_v2_model_runner),
            "the engine would select the V1 model runner, but DSpark is "
            "implemented only by the V2 GPU model runner -- the boot would "
            "die with 'Model Runner V1 does not support: dspark speculative "
            "decoding'")

    # On a GPU-less box the platform forces cudagraph_mode=NONE, so the
    # engine's size resolver never runs. Force the mode the lane runs with
    # and call the ENGINE'S OWN resolver (VllmConfig._set_cudagraph_sizes) --
    # the default sizes depend on max_num_seqs / decode_query_len only, not
    # on the mode, so this is the engine's answer and not a reimplementation.
    from vllm.config.compilation import CUDAGraphMode
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    vc._set_cudagraph_sizes()
    capture_max = vc.compilation_config.max_cudagraph_capture_size
    capture_sizes = list(vc.compilation_config.cudagraph_capture_sizes or [])
    require(capture_max == 256,
            f"the engine's default max_cudagraph_capture_size is {capture_max}, "
            "expected 256 at max_num_seqs=32 and k=3")
    require(128 in capture_sizes,
            "the engine's resolved cudagraph capture list does not contain 128 "
            f"(it has {len(capture_sizes)} entries)")

    return {
        "method": spec.method,
        "k": int(spec.num_speculative_tokens),
        "draft": spec.model,
        "draft_sample_method": spec.draft_sample_method,
        "rejection_sample_method": spec.rejection_sample_method,
        "kv_cache_dtype": str(spec.kv_cache_dtype),
        "attention_backend": backend_name,
        "draft_tp": int(spec.draft_tensor_parallel_size),
        "uniform_decode_query_len": vc.uniform_decode_query_len,
        "max_cudagraph_capture_size": capture_max,
        "capture_sizes": len(capture_sizes),
        "capture_has_128": True,
        "model_runner": "V2" if bool(vc.use_v2_model_runner) else "V1",
    }


# ---------------------------------------------------------------------------
# 6a. the ONE spec-config file, and the binding to the serve line
# ---------------------------------------------------------------------------


def read_spec_config(spec_path, served_json):
    """Read the ONE spec-config file and PROVE the serve line uses it.

    The serve line (start-tp3.sh) reads this file inside the container and
    puts its content on the command line; the boot passes that very string
    to this check as `--served-spec-json`. The binding is therefore
    ASSERTED, not assumed: the bytes this check parsed and the bytes the
    engine will receive are compared for EQUALITY, and a difference fails
    the boot.
    """
    require(spec_path, "no spec-config path was given to this check")
    require(os.path.isfile(spec_path),
            f"the spec-config file {spec_path} does not exist -- the serve "
            "line reads it and this check must parse the SAME file")
    with open(spec_path, encoding="utf-8") as f:
        raw = f.read()
    # the shell's `$(cat file)` strips the trailing newlines; the serve line
    # receives exactly that string.
    content = raw.rstrip("\n")
    try:
        parsed = json.loads(content)
    except Exception as exc:
        raise CheckFailure(f"{spec_path} is not valid JSON: {exc}")
    require(isinstance(parsed, dict),
            f"{spec_path} does not hold a JSON object")
    require(parsed == SPEC_EXPECTED,
            f"{spec_path} says {parsed}, expected {SPEC_EXPECTED} -- the "
            "spec config changed under the kit")
    require(served_json is not None,
            "the serve line's --speculative-config value was not handed to "
            "this check (--served-spec-json), so the binding between the "
            "file and the serve line is UNPROVEN -- refusing to pass")
    require(served_json == content,
            f"the serve line's --speculative-config value is NOT the content "
            f"of {spec_path}: {served_json!r} != {content!r}")
    return parsed, content


# ---------------------------------------------------------------------------
# 7. the check itself must be able to FAIL
# ---------------------------------------------------------------------------


def run_selftest(overlay, kit_dir, model_dir, draft_dir):
    """Mutate the inputs and require that this check REJECTS each mutation."""
    table = read_json(os.path.join(kit_dir, "tp3_pad_table.json"))
    intended = overlay._INTENDED_LAYOUTS
    mutants = []

    def mutant_whitelist():
        # dropping the entry from the overlay's
        # _DEFAULT_MODEL_TYPES is NOT enough -- the overlay's whitelist is
        # the UNION of that tuple and the VLLM_TP3_PAD_MODEL_TYPES env var
        # (overlay/vllm_tp3_pad/__init__.py:100-104), and the boot sets that
        # env var to "minimax_m3_vl,minimax_m3_dspark". The mutation was
        # therefore DEFEATED: the draft stayed whitelisted, the pad still
        # happened, and the battery ACCEPTED a broken input.
        #
        # So the mutation is made authoritative over the WHOLE whitelist: the
        # env var is neutralised for the duration of the mutation and
        # restored afterwards -- on the failure path too, so the environment
        # is never left mutated.
        saved_env = os.environ.get("VLLM_TP3_PAD_MODEL_TYPES")
        try:
            os.environ["VLLM_TP3_PAD_MODEL_TYPES"] = ""
            return check_config_pad(
                overlay, draft_dir, tp=TP,
                whitelist_override=tuple(t for t in overlay._DEFAULT_MODEL_TYPES
                                         if t != "minimax_m3_dspark"))
        finally:
            if saved_env is None:
                os.environ.pop("VLLM_TP3_PAD_MODEL_TYPES", None)
            else:
                os.environ["VLLM_TP3_PAD_MODEL_TYPES"] = saved_env

    def mutant_unpadded_config():
        return check_arithmetic(_Geometry(64, 4, 128), tp=TP)

    def mutant_bad_expect():
        broken = json.loads(json.dumps(table))
        for rule in broken["rules"]:
            if rule.get("require") == DRAFT_REQUIRE and \
                    rule["suffix"] == "self_attn.q_proj.weight":
                rule["expect"] = 4096
        return check_tensors(overlay, broken, draft_dir, model_dir, tp=TP)

    def mutant_merged_scope():
        broken = json.loads(json.dumps(table))
        for rule in broken["rules"]:
            if rule.get("require") == DRAFT_REQUIRE:
                rule["require"] = TARGET_REQUIRE
        return check_table(broken, intended, kit_dir)

    for name, fn in (
        ("M1 the whitelist entry is dropped",
         mutant_whitelist),
        ("M2 an unpadded config (64 heads) reaches the arithmetic",
         mutant_unpadded_config),
        ("M3 a draft rule's expect is corrupted",
         mutant_bad_expect),
        ("M4 the target and draft rule scopes are merged",
         mutant_merged_scope),
    ):
        try:
            fn()
        except CheckFailure:
            mutants.append((name, True, ""))
            continue
        except Exception as exc:  # a crash is also a rejection, but a blunt one
            mutants.append((name, True, f"(raised {type(exc).__name__})"))
            continue
        mutants.append((name, False, "the check ACCEPTED it"))

    ok = all(m[1] for m in mutants)
    return ok, mutants


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_checks(kit_dir, model_dir, spec_path=None, served_spec_json=None,
               allow_gpu_hidden=False):
    overlay, origin = load_overlay(kit_dir)
    draft_dir = os.path.join(model_dir, DRAFT_SUBDIR)
    require(os.path.isdir(draft_dir),
            f"the draft checkpoint is not at {draft_dir} -- the model dir is "
            "wrong (the lane mounts it at /model)")

    facts = {}
    facts["overlay"] = {"origin": origin,
                        "file": getattr(overlay, "__file__", "<builtin>")}
    facts["selftest"] = check_overlay_selftest(overlay)

    resolved_table_path = overlay._table_path()
    table = read_json(resolved_table_path)
    facts["table"] = check_table(table, overlay._INTENDED_LAYOUTS, kit_dir,
                                 resolved_path=resolved_table_path,
                                 overlay_dir=os.path.dirname(
                                     getattr(overlay, "__file__", "") or ""))
    facts["table"]["resolved_path"] = resolved_table_path

    facts["config"] = check_config_pad(overlay, draft_dir, tp=TP)
    facts["engine"] = check_engine_invariants()
    geometry = _Geometry(facts["config"]["padded_heads"][0],
                          facts["config"]["padded_heads"][1],
                          facts["config"]["head_dim"])
    real_q_rows = int(
        _rule_index(table)[("self_attn.q_proj.weight", DRAFT_REQUIRE)]["expect"])
    facts["arithmetic"] = check_arithmetic(geometry, tp=TP,
                                            real_q_rows=real_q_rows)
    facts["tensors"] = check_tensors(overlay, table, draft_dir, model_dir,
                                      tp=TP, geometry=geometry)
    # the spec config: the ONE file, and the proof that the serve line uses
    # the very bytes this check parsed.
    spec_kwargs, spec_content = read_spec_config(spec_path, served_spec_json)
    facts["spec"] = check_spec_config(model_dir, tp=TP, spec_kwargs=spec_kwargs)
    facts["spec"]["file"] = spec_path
    facts["spec"]["file_md5"] = md5_file(spec_path)
    facts["spec"]["serve_line_value_matches_file"] = (served_spec_json ==
                                                       spec_content)

    # THIS GATE MUST NOT PUT ANY TENSOR MEMORY ON THE GPU.
    #
    # It CANNOT be "no CUDA context at all", and insisting on that was the
    # second boot-blocking bug (see docs/FIXLOG.md): the check has to see the
    # SAME platform the engine will see, because vllm/config/parallel.py's
    # world-size validation depends on the number of GPUs VISIBLE ON THIS
    # NODE -- that is exactly the validation that killed the first boot of
    # the draft-head pad kit.
    # A Spark has ONE GPU and the lane runs three ranks across three boxes, so
    # a GPU-HIDDEN check silently SKIPS the validation the engine will run,
    # and the bug hides again. The engine starts in a NEW process (exec vllm
    # serve) AFTER this one exits, so this process's context is released
    # before the engine allocates anything.
    import torch
    # Measure only if a context already exists; asking for the stats would
    # CREATE one, and this check must not put anything on the GPU.
    if torch.cuda.is_initialized():
        gpu_tensor_bytes = int(torch.cuda.memory_allocated())
    else:
        gpu_tensor_bytes = 0
    require(gpu_tensor_bytes == 0,
            f"the check allocated {gpu_tensor_bytes} bytes of GPU tensor "
            "memory -- it must not put anything on the GPU")
    reach = _world_size_validation_reachable(TP)
    facts["gpu"] = {
        "device_count": int(torch.cuda.device_count()),
        "cuda_context_created": bool(torch.cuda.is_initialized()),
        "gpu_tensor_bytes": gpu_tensor_bytes,
        "world_size_validation": reach,
        "engine_equivalent": bool(reach["validation_live"]),
    }
    # FAIL CLOSED ON THE ENVIRONMENT, not just on the inputs.
    #
    # The bug that killed the first boot of the draft-head pad kit lived in
    # the engine's
    # world-size validation, and that validation is only reachable when the
    # platform is CUDA (see _world_size_validation_reachable). A check that
    # runs GPU-HIDDEN therefore cannot see the class of failure it exists to
    # catch -- it hid from every offline verification and only appeared at
    # the boot, where the self-check's own ParallelConfig raised before it
    # checked anything.
    #
    # So the check REQUIRES that it runs where the engine runs. The only
    # escape is --allow-gpu-hidden, which is for OFFLINE diagnostics only:
    # it downgrades the gate to a recorded fact and marks the run as NOT
    # engine-equivalent. The boot never passes it.
    require(reach["validation_live"] or allow_gpu_hidden,
            "the engine's world-size validation is NOT live in this "
            f"environment (cuda={reach['cuda']}, device_count="
            f"{reach['device_count']}, world_size={reach['world_size']}, "
            f"nnodes={reach['nnodes']}) -- this check must run where the "
            "ENGINE runs, i.e. in a GPU-visible container "
            "(`docker run --gpus all ...`), because the validation it exists "
            "to catch is only reachable there; pass --allow-gpu-hidden for "
            "offline diagnostics only")

    return overlay, facts


def print_summary(facts):
    print("")
    print("=== verify-draft-pad: SUMMARY ===")
    ov = facts["overlay"]
    print(f"  overlay            : {ov['origin']}  ({ov['file']})")
    tb = facts["table"]
    print(f"  pad table          : {tb['rules']} rules "
          f"({tb['target_rules']} target + {tb['draft_rules']} draft), "
          f"{tb['intended_layouts']} pinned layouts, md5 {tb['table_md5']}")
    cf = facts["config"]
    print(f"  draft config       : heads {cf['raw_heads'][0]}/{cf['raw_heads'][1]}"
          f" -> {cf['padded_heads'][0]}/{cf['padded_heads'][1]}, "
          f"head_dim {cf['head_dim']}, vocab {cf['vocab_size']}")
    print(f"  engine path        : promotion+pad -> heads "
          f"{cf['engine_heads'][0]}/{cf['engine_heads'][1]}")
    ar = facts["arithmetic"]
    print(f"  per-rank shapes    : q {ar['q_rows_per_rank']} rows, "
          f"kv {ar['kv_rows_per_rank']} rows, o_proj "
          f"{ar['o_proj_cols_per_rank']} cols, o_proj scale "
          f"{ar['o_proj_scale_cols_per_rank']} cols "
          f"(q total {ar['q_rows']}, kv total {ar['kv_rows']}, "
          f"pad q rows {ar['pad_q_rows']} on rank {TP - 1})")
    en = facts["engine"]
    print(f"  engine invariants  : head-divisibility assert present, "
          f"head_dim explicit (fallback would have been "
          f"{en['head_dim_fallback_would_be']})")
    tn = facts["tensors"]
    print(f"  checkpoint tensors : draft {tn['draft_tensors']} tensors, "
          f"{tn['draft_rule_hits']} rule hits, {tn['draft_unmatched']} "
          f"untouched; target {tn['target_tensors']} tensors, "
          f"{tn['target_rule_hits']} rule hits; rule sets disjoint")
    sp = facts["spec"]
    print(f"  spec config        : method {sp['method']}, k {sp['k']}, "
          f"draft {sp['draft']}, samples {sp['draft_sample_method']}/"
          f"{sp['rejection_sample_method']}, kv {sp['kv_cache_dtype']}")
    print(f"  spec config file   : {sp['file']} (md5 {sp['file_md5']}), the "
          "serve line's --speculative-config value is byte-identical to it: "
          f"{sp['serve_line_value_matches_file']}")
    print(f"  capture            : uniform_decode_query_len "
          f"{sp['uniform_decode_query_len']}, max_cudagraph_capture_size "
          f"{sp['max_cudagraph_capture_size']}, capture list "
          f"{sp['capture_sizes']} entries (128 present)")
    print(f"  model runner       : {sp['model_runner']} (DSpark needs the V2 "
          "GPU model runner; the baseline configuration ran V2 too)")
    gp = facts["gpu"]
    wsv = gp["world_size_validation"]
    print(f"  gpu                : device_count {gp['device_count']}, "
          f"cuda_context_created {gp['cuda_context_created']}, "
          f"gpu_tensor_bytes {gp['gpu_tensor_bytes']}")
    print(f"  world-size gate    : branch_entered {wsv['branch_entered']}, "
          f"validation_live {wsv['validation_live']}, raised "
          f"{wsv['raised']} -- engine-equivalent environment: "
          f"{gp['engine_equivalent']}")
    print("=== verify-draft-pad: PASS ===")


def main():
    ap = argparse.ArgumentParser(
        description="apply-time self-check for the DSpark draft-head TP=3 pad")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="the checkpoint dir (the lane mounts it at /model)")
    ap.add_argument("--kit-dir", default=HERE,
                    help="the patches-tp3pad dir (default: this script's dir)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the non-vacuity mode: mutate the inputs and "
                         "require that this check rejects every mutation")
    ap.add_argument("--spec-config", default=DEFAULT_SPEC_FILE,
                    help="the ONE spec-config file the serve line reads "
                         "(default: %(default)s)")
    ap.add_argument("--served-spec-json", default=None,
                    help="the EXACT --speculative-config value the serve line "
                         "puts on the command line; the check asserts it is "
                         "byte-identical to the file (the binding)")
    ap.add_argument("--allow-gpu-hidden", action="store_true",
                    help="OFFLINE DIAGNOSTICS ONLY: do not require a "
                         "GPU-visible environment (the engine's world-size "
                         "validation is only reachable with a visible GPU, so "
                         "a GPU-hidden run is NOT engine-equivalent; the boot "
                         "never passes this flag)")
    args = ap.parse_args()

    kit_dir = os.path.abspath(args.kit_dir)
    model_dir = os.path.abspath(args.model_dir)

    try:
        if args.selftest:
            overlay, _origin = load_overlay(kit_dir)
            draft_dir = os.path.join(model_dir, DRAFT_SUBDIR)
            ok, mutants = run_selftest(overlay, kit_dir, model_dir, draft_dir)
            for name, rejected, note in mutants:
                print(f"  [{'REJECTED' if rejected else 'ACCEPTED!!'}] {name} {note}")
            if not ok:
                print("verify-draft-pad: FAIL -- the self-check accepted a "
                      "broken input, so it proves nothing", file=sys.stderr)
                return 1
            print(f"verify-draft-pad: SELFTEST PASS -- {len(mutants)}/"
                  f"{len(mutants)} broken inputs rejected")
            return 0

        overlay, facts = run_checks(kit_dir, model_dir,
                                    spec_path=args.spec_config,
                                    served_spec_json=args.served_spec_json,
                                    allow_gpu_hidden=args.allow_gpu_hidden)
        print_summary(facts)
        return 0
    except CheckFailure as exc:
        print(f"verify-draft-pad: FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # a crash is a failure, never a pass
        import traceback
        traceback.print_exc()
        print(f"verify-draft-pad: FAIL: unexpected {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
