# TP=3 sharding pad overlay for MiniMax-M3.1-Flash on the pinned vLLM image
# (minimax-m31-flash-arm64-cu134 lineage).
#
# Two entry points (both no-ops unless VLLM_TP3_PAD is set):
#
#   maybe_pad_hf_config(config)      -- called from the get_config() anchor in
#                                       vllm/transformers_utils/config.py
#   tp3_pad_weights_iterator(it)     -- called from the _get_weights_iterator()
#                                       anchor in
#                                       vllm/model_executor/model_loader/default_loader.py
#
# Design and evidence: docs/FIXLOG.md. Summary:
#
#  * attention  : num_attention_heads  64 -> 96, num_key_value_heads 4 -> 6
#                 (RATIO-PRESERVING: 16 Q heads per KV head before and after,
#                  so the q->kv association of every real head is unchanged;
#                  zero-padded heads contribute exactly 0 through the
#                  zero-padded o_proj columns)
#  * index heads: the sparse-attention index heads are padded in lockstep
#                 with the KV heads (4 -> 6), so the index geometry stays
#                 consistent with the padded KV geometry
#  * vocab      : no pad (200064 is divisible by 3 AND by 64, so the
#                 VocabParallelEmbedding pad alone already divides by 3)
#  * MoE        : no pad (3072 % 3 == 0; dense intermediates 12288 % 3 == 0)
#  * mamba-2    : also handled (head/group ratio preserved) when a config
#                 exposes mamba_num_heads / n_groups; this model's config
#                 does not, so that branch is inert here
#
# Weight tensors are zero-padded at load time. Every rule in this recipe's
# table is APPEND-ONLY (each op is a [-1, n] append), so the padded rows and
# columns land at the END of the axis, and the block-quantized scale tensors
# are padded on their own row grid (block size 32 for o_proj, on COLUMNS).
# Insert positions are ORIGINAL (pre-insert) coordinates and are applied
# TAIL-FIRST (position descending) -- see ordered_ops() and the --selftest
# guard.
#
# Idempotent by construction: every pad is a "round up to a multiple" of a
# value that is already rounded up on a second pass.
#
# The DSpark DRAFT HEAD is covered by the
# SAME geometry: the draft's promoted config ("minimax_m3_dspark") carries
# 64 q-heads / 4 kv-heads / head_dim 128, so the config pad is identical
# (64 -> 96 / 4 -> 6) and its weight rules are the target's rules with
# "language_model.model.dspark.layers." as the require scope. The draft's
# checkpoint names are <prefix>.layers.N.decoder_layer.self_attn.{q,k,v,o}_
# proj.weight[+_scale_inv]; the target's are
# language_model.model.layers.N.decoder_layer.self_attn... The two scopes
# are disjoint as SUBSTRINGS ("language_model.model.layers." is not a
# substring of "language_model.model.dspark.layers.N..."), which the
# selftest pins.

from __future__ import annotations

import json
import os
import sys

_PADS_LOG_PREFIX = "[tp3-pad]"

# Model types whose configs get padded. Other model types are deliberately
# NOT whitelisted: their geometry is unverified (see docs/FIXLOG.md).
# Extend via VLLM_TP3_PAD_MODEL_TYPES="a,b".
#
# "minimax_m3_dspark" is the PROMOTED DSpark draft head. The engine promotes
# the draft's nested text_config INSIDE
# get_config() (vllm/transformers_utils/config.py:979-981, via
# SpeculativeConfig.hf_config_override) and the pad hook runs AFTER that
# (same file, :995-998), so the hook sees a config whose model_type is
# "minimax_m3_dspark" and which has NO nested text_config. Without this
# entry the draft head is never padded: 64 q-heads % 3 != 0, and the
# engine dies at vllm/config/model.py:1544-1549 ("Total number of
# attention heads (64) must be divisible by tensor parallel size (3)")
# while building the draft ModelConfig -- before the draft model is even
# instantiated. The draft's geometry is the target's: 64 q-heads / 4 kv-
# heads / head_dim 128, so the SAME ratio-preserving pad applies
# (64 -> 96 q-heads, 4 -> 6 kv-heads, q-per-kv 16 preserved).
_DEFAULT_MODEL_TYPES = (
    "nemotron_h_omni",
    "minimax_m3_vl",
    "minimax_m3_dspark",
)

_TABLE = None


def _log(*args) -> None:
    print(_PADS_LOG_PREFIX, *args, file=sys.stderr, flush=True)


def _env_tp_size():
    raw = os.environ.get("VLLM_TP3_PAD", "")
    try:
        tp = int(raw)
    except ValueError:
        return None
    return tp if tp > 0 else None


def _model_type_whitelist():
    extra = os.environ.get("VLLM_TP3_PAD_MODEL_TYPES", "")
    types = set(_DEFAULT_MODEL_TYPES)
    types.update(t.strip() for t in extra.split(",") if t.strip())
    return types


def _table_path():
    explicit = os.environ.get("VLLM_TP3_PAD_TABLE", "")
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    installed = os.path.join(here, "tp3_pad_table.json")
    if os.path.isfile(installed):
        return installed
    # source tree (before apply_tp3_pads.py copies the table into the
    # package): the table ships two levels up, in patches/
    return os.path.join(os.path.dirname(os.path.dirname(here)), "tp3_pad_table.json")


def _load_table():
    global _TABLE
    if _TABLE is None:
        path = _table_path()
        with open(path) as f:
            _TABLE = json.load(f)
        _log("pad table loaded from", path)
    return _TABLE


# ---------------------------------------------------------------------------
# config-side pads
# ---------------------------------------------------------------------------


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _pad_text_config(cfg, tp):
    """Apply ratio-preserving TP pads to one (text) config object. Idempotent."""

    def geti(name, default=None):
        v = getattr(cfg, name, default)
        return v if isinstance(v, int) else default

    # --- attention: preserve num_attention_heads // num_key_value_heads -------
    n_q = geti("num_attention_heads")
    n_kv = geti("num_key_value_heads")
    if n_q and n_kv:
        q_per_kv = n_q // n_kv
        if n_q % n_kv != 0:
            raise ValueError(
                f"tp3-pad: unexpected GQA geometry {n_q=} {n_kv=} "
                "(num_attention_heads must divide evenly by num_key_value_heads)"
            )
        kv2 = _round_up(n_kv, tp)
        q2 = q_per_kv * kv2
        if (q2, kv2) != (n_q, n_kv):
            if q2 % tp != 0:
                raise ValueError(f"tp3-pad: derived q head count {q2} not divisible by {tp}")
            _log(
                f"attention heads {n_q} -> {q2}, kv heads {n_kv} -> {kv2} "
                f"(q-per-kv {q_per_kv} preserved)"
            )
            setattr(cfg, "num_attention_heads", q2)
            setattr(cfg, "num_key_value_heads", kv2)

            # --- MSA lightning-indexer heads: MUST equal the KV head count ------
            # MinimaxM3QKVParallelLinearWithIndexer asserts
            # total_num_index_heads == total_num_kv_heads and shards index_q on
            # the KV path; index_k is a single replicated head (never padded).
            # The head count lives in the nested sparse_attention_config DICT
            # (the indexer metadata builder reads sparse_cfg["sparse_num_index_
            # heads"] by key), so the dict is patched alongside the attr, and
            # they are kept identical if both exist.
            sparse = getattr(cfg, "sparse_attention_config", None)
            idx_attr = geti("sparse_num_index_heads")
            if isinstance(sparse, dict) and "sparse_num_index_heads" in sparse:
                cur = sparse["sparse_num_index_heads"]
                if isinstance(cur, int) and cur == n_kv and cur != kv2:
                    sparse["sparse_num_index_heads"] = kv2
                    _log(f"sparse_attention_config.sparse_num_index_heads {cur} -> {kv2} (lockstep with kv heads)")
            elif isinstance(idx_attr, int) and idx_attr == n_kv and idx_attr != kv2:
                setattr(cfg, "sparse_num_index_heads", kv2)
                _log(f"sparse_num_index_heads {idx_attr} -> {kv2} (lockstep with kv heads)")

    # --- mamba-2: preserve mamba_num_heads // n_groups ------------------------
    m_h = geti("mamba_num_heads")
    n_g = geti("n_groups")
    if m_h and n_g:
        heads_per_group = m_h // n_g
        if m_h % n_g != 0:
            raise ValueError(
                f"tp3-pad: unexpected mamba geometry {m_h=} {n_g=} "
                "(mamba_num_heads must divide evenly by n_groups)"
            )
        g2 = _round_up(n_g, tp)
        h2 = heads_per_group * g2
        if (h2, g2) != (m_h, n_g):
            if h2 % tp != 0:
                raise ValueError(f"tp3-pad: derived mamba head count {h2} not divisible by {tp}")
            _log(
                f"mamba heads {m_h} -> {h2}, n_groups {n_g} -> {g2} "
                f"(heads-per-group {heads_per_group} preserved)"
            )
            setattr(cfg, "mamba_num_heads", h2)
            setattr(cfg, "n_groups", g2)

    # --- vocab: multiple of tp AND of the engine's 64-row vocab padding -------
    # (vllm/model_executor/layers/vocab_parallel_embedding.py:
    #  DEFAULT_VOCAB_PADDING_SIZE = 64; num_embeddings_per_partition =
    #  divide(padded_vocab, tp). A vocab that is already a multiple of 64
    #  leaves the stock pad doing nothing, so the pad has to make the total
    #  divisible by tp as well.)
    vocab = geti("vocab_size")
    if vocab:
        vocab2 = _round_up(_round_up(vocab, tp), 64)
        if vocab2 != vocab:
            _log(f"vocab_size {vocab} -> {vocab2} (logical size kept for logits masking)")
            setattr(cfg, "tp3_org_vocab_size", vocab)
            setattr(cfg, "vocab_size", vocab2)
        elif getattr(cfg, "tp3_org_vocab_size", None) is None:
            setattr(cfg, "tp3_org_vocab_size", vocab)


def maybe_pad_hf_config(config):
    """Anchor hook: pad the HF config (and its text_config, if nested) for TP.

    No-op unless VLLM_TP3_PAD=<tp_size> is set and the config's model_type is
    whitelisted. Returns the config either way.
    """
    tp = _env_tp_size()
    if tp is None:
        return config
    if config is None:
        return config

    whitelist = _model_type_whitelist()

    targets = []
    own_type = getattr(config, "model_type", None)
    if own_type in whitelist:
        targets.append(config)
    text_cfg = getattr(config, "text_config", None)
    # Target the nested text config when EITHER its own model_type is
    # whitelisted OR the PARENT is: MiniMax's remote config code coerces
    # text_config to a MixtralConfig (model_type "mixtral" via
    # _coerce_sub_config's fallback default), so keying on the child's own
    # type alone never fires (see docs/FIXLOG.md).
    if text_cfg is not None and (
        getattr(text_cfg, "model_type", None) in whitelist or own_type in whitelist
    ):
        targets.append(text_cfg)

    if not targets:
        return config

    # Idempotency across repeated get_config() calls in one process.
    if getattr(config, "_tp3_pads_applied", False):
        return config

    for cfg in targets:
        _pad_text_config(cfg, tp)

    # Remember the logical vocab on the top config too: the VL wrapper's
    # language model is built from config.text_config, but anything reading
    # the top level (e.g. arch-config vocab) sees the padded value by design.
    if text_cfg is not None and getattr(text_cfg, "tp3_org_vocab_size", None):
        setattr(config, "tp3_org_vocab_size", text_cfg.tp3_org_vocab_size)

    setattr(config, "_tp3_pads_applied", True)
    _log(f"config pads applied for tp={tp}")
    return config


# ---------------------------------------------------------------------------
# checkpoint-side pads (zero rows / columns, some interior)
# ---------------------------------------------------------------------------


def ordered_ops(ops):
    """Insert ops sorted TAIL-FIRST: original-coordinate position DESCENDING.

    Pad-table / --emit-table positions are in ORIGINAL (pre-insert)
    coordinates (in_proj block ends 8192/16384/17408/18432; -1 = append).
    Each insert's position is only correct while everything after it is
    still un-shifted, so application MUST go tail-first. Ascending
    application against the already-shifted tensor re-bins every later
    block (the demonstrated interior-insert defect: z:8192|PAD:1024|x:7168|
    PAD:1152|x:896|...). The append op (-1) sorts last.
    """
    return sorted(ops, key=lambda op: int(op[0]), reverse=True)


def layout_segments(expect, ops):
    """Enumerate the FINAL segment layout of [(at, n)] inserts on one axis.

    Returns tokens 'orig[a:b)' (original-coordinate row ranges) and 'PAD:n'
    (inserted zero rows), left to right. Positions are original
    coordinates, so ascending order enumerates the final layout directly
    (the -1 append op sorts last, at position `expect`). Shared by
    the offline plan validator and the selftest so the offline replay and the
    runtime _apply_ops cannot drift apart.
    """
    segments = []
    prev = 0
    order = sorted(ops, key=lambda op: expect if int(op[0]) < 0 else int(op[0]))
    for at, n in order:
        at = expect if int(at) < 0 else int(at)
        if at > prev:
            segments.append(f"orig[{prev}:{at})")
        segments.append(f"PAD:{int(n)}")
        prev = at
    if prev < expect:
        segments.append(f"orig[{prev}:{expect})")
    return segments


def _apply_ops(tensor, ops, dim):
    """Apply [(at, n), ...] zero-row inserts on `dim`; at == -1 appends.

    TAIL-FIRST (see ordered_ops): positions are original-coordinate, so
    ops are applied position-descending against the current tensor.
    """
    import torch

    for at, n in ordered_ops(ops):
        n = int(n)
        at = int(at)
        shape = list(tensor.shape)
        shape[dim] = n
        zeros = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        if at < 0:
            at = tensor.shape[dim]
        if dim == 0:
            tensor = torch.cat((tensor[:at], zeros, tensor[at:]), dim=0)
        else:
            tensor = torch.cat(
                (tensor[:, :at], zeros, tensor[:, at:]), dim=dim
            )
    return tensor


def _rule_for(name, rules):
    for rule in rules:
        suffix = rule["suffix"]
        require = rule.get("require")
        if require and require not in name:
            continue
        if name == suffix or name.endswith("." + suffix):
            return rule
    return None


_SCALE_TOKENS = ("weight_scale", "weight_scale_inv", "weight_global_scale")


def _guard_scale_tensor(name, tensor, rules):
    """A rowwise quant scale for a padded tensor must itself have a rule."""
    prefix, _, last = name.rpartition(".")
    if not prefix or not any(tok in last for tok in _SCALE_TOKENS):
        return
    if tensor.numel() <= 1:
        return  # per-tensor scalar scale: nothing to pad
    weight_name = prefix + ".weight"
    if _rule_for(weight_name, rules) is not None and _rule_for(name, rules) is None:
        raise RuntimeError(
            f"tp3-pad: {name} has {tuple(tensor.shape)} rows (rowwise scale) but "
            f"the pad table has no rule for it while its weight {weight_name} IS "
            f"padded. Regenerate the table with the plan validator "
            f"(--emit-table)."
        )


def _transform(name, tensor, rules):
    rule = _rule_for(name, rules)
    if rule is None:
        _guard_scale_tensor(name, tensor, rules)
        return tensor

    dim = rule["dim"]
    expect = rule["expect"]
    ops = rule["ops"]
    if tensor.dim() <= dim:
        raise RuntimeError(
            f"tp3-pad: {name} has dim {tensor.dim()}, rule expects axis {dim}"
        )
    actual = tensor.shape[dim]
    result = rule["result"]
    if actual == result:
        return tensor  # already padded (idempotent)
    if actual != expect:
        raise RuntimeError(
            f"tp3-pad: {name} axis {dim} is {actual}, table expects {expect} "
            "(checkpoint does not match the pad table; re-run the plan "
            "validator against the staged checkpoint)"
        )
    padded = _apply_ops(tensor, ops, dim)
    got = padded.shape[dim]
    if got != result:
        raise RuntimeError(f"tp3-pad: {name} padded to {got}, expected {result}")
    _log(f"padded {name} axis {dim}: {actual} -> {got}")
    return padded


def tp3_pad_weights_iterator(weights_iterator):
    """Anchor hook: yield (name, zero-padded tensor) per the pad table."""
    if _env_tp_size() is None:
        yield from weights_iterator
        return

    table = _load_table()
    rules = table["rules"]

    for name, tensor in weights_iterator:
        yield name, _transform(name, tensor, rules)


# ---------------------------------------------------------------------------
# selftest (stdlib-only): python3 -m vllm_tp3_pad --selftest
#
# Guards the ops-ordering class of bug forever: insert positions are
# ORIGINAL-coordinate and must be applied TAIL-FIRST. Replays every pad
# rule with a list-based mirror of _apply_ops' exact slicing and asserts
# the exact intended segment layout for EVERY pad-table rule -- including
# the in-place [[0,n]] rewrite of an append-only op (e.g. o_proj
# [[-1,2048]] -> [[0,2048]]), which passed every gate silently before
# because the originally-asserted rules were exactly the multi-insert
# ones; the append-only rules are now pinned by their exact one-pad
# layouts too -- plus config-pad arithmetic and idempotency, the
# target/draft rule-scope disjointness, and the coverage of every rule by
# an intended layout (a rule with no pinned layout is not protected against
# a future edit).
# ---------------------------------------------------------------------------

_INTENDED_LAYOUTS = {
    # (rule suffix, rule require) -> exact final layout
    ("self_attn.q_proj.weight", "language_model.model.layers."): "orig[0:8192)|PAD:4096",
    ("self_attn.q_proj.weight_scale_inv", "language_model.model.layers."): "orig[0:8192)|PAD:4096",
    ("self_attn.k_proj.weight", "language_model.model.layers."): "orig[0:512)|PAD:256",
    ("self_attn.k_proj.weight_scale_inv", "language_model.model.layers."): "orig[0:512)|PAD:256",
    ("self_attn.v_proj.weight", "language_model.model.layers."): "orig[0:512)|PAD:256",
    ("self_attn.v_proj.weight_scale_inv", "language_model.model.layers."): "orig[0:512)|PAD:256",
    ("self_attn.index_q_proj.weight", "language_model.model.layers."): "orig[0:512)|PAD:256",
    ("self_attn.index_q_proj.weight_scale_inv", "language_model.model.layers."): "orig[0:512)|PAD:256",
    ("self_attn.o_proj.weight", "language_model.model.layers."): "orig[0:8192)|PAD:4096",
    ("self_attn.o_proj.weight_scale_inv", "language_model.model.layers."): "orig[0:256)|PAD:128",
    # --- DSpark DRAFT HEAD --------------------------------------------------
    # Same geometry as the target, same pads, disjoint require scope.
    ("self_attn.q_proj.weight", "language_model.model.dspark.layers."): "orig[0:8192)|PAD:4096",
    ("self_attn.q_proj.weight_scale_inv", "language_model.model.dspark.layers."): "orig[0:8192)|PAD:4096",
    ("self_attn.k_proj.weight", "language_model.model.dspark.layers."): "orig[0:512)|PAD:256",
    ("self_attn.k_proj.weight_scale_inv", "language_model.model.dspark.layers."): "orig[0:512)|PAD:256",
    ("self_attn.v_proj.weight", "language_model.model.dspark.layers."): "orig[0:512)|PAD:256",
    ("self_attn.v_proj.weight_scale_inv", "language_model.model.dspark.layers."): "orig[0:512)|PAD:256",
    ("self_attn.o_proj.weight", "language_model.model.dspark.layers."): "orig[0:8192)|PAD:4096",
    ("self_attn.o_proj.weight_scale_inv", "language_model.model.dspark.layers."): "orig[0:256)|PAD:128",
}


def _fake_apply(expect, ops):
    """Mirror _apply_ops' slicing on a plain list (torch-free) so the
    selftest exercises the exact same tail-first ordering and cut
    semantics: rows[:at] + PAD*n + rows[at:]."""
    rows = list(range(int(expect)))
    for at, n in ordered_ops(ops):
        at = len(rows) if int(at) < 0 else int(at)
        rows = rows[:at] + [None] * int(n) + rows[at:]
    return rows


def _segments_from_rows(rows):
    """Fold a _fake_apply result back into layout_segments tokens."""
    segments = []
    i = 0
    while i < len(rows):
        j = i
        if rows[j] is None:
            while j < len(rows) and rows[j] is None:
                j += 1
            segments.append(f"PAD:{j - i}")
        else:
            while j < len(rows) and rows[j] is not None:
                j += 1
            block = rows[i:j]
            if block != list(range(block[0], block[0] + len(block))):
                raise AssertionError(
                    "original rows not contiguous after inserts (mis-ordered ops)"
                )
            segments.append(f"orig[{block[0]}:{block[0] + len(block)})")
        i = j
    return segments


def selftest(quiet: bool = False) -> int:
    """Torch-free regression guard; returns 0 on pass, 1 on any failure."""
    problems = []
    table = _load_table()
    # Keyed by (suffix, require): the target and the DSpark draft head share
    # the same SUFFIXES (same geometry, different checkpoint scope), so a
    # suffix-keyed dict would silently drop one of the two rule sets.
    rules = {(r["suffix"], r.get("require", "")): r for r in table["rules"]}

    # 1. every rule: tail-first replay reaches `result` AND matches the
    #    enumerated final layout (ordering + cut semantics).
    for (suffix, require), rule in sorted(rules.items()):
        rows = _fake_apply(rule["expect"], rule["ops"])
        total = rule["expect"] + sum(int(n) for _a, n in rule["ops"])
        if len(rows) != rule["result"] or total != rule["result"]:
            problems.append(
                f"{suffix} [{require}]: replayed {len(rows)} rows / arith {total}, "
                f"table result {rule['result']}"
            )
        try:
            replayed = "|".join(_segments_from_rows(rows))
        except AssertionError as e:
            problems.append(f"{suffix} [{require}]: {e}")
            continue
        enumerated = "|".join(layout_segments(rule["expect"], rule["ops"]))
        if replayed != enumerated:
            problems.append(
                f"{suffix} [{require}]: tail-first replay layout != enumerated layout\n"
                f"    replayed:   {replayed}\n"
                f"    enumerated: {enumerated}"
            )

    # 2. the exact intended block layouts (the ops-ordering defect guard).
    table_rules = {(r["suffix"], r.get("require", "")): r for r in table["rules"]}
    for (suffix, require), want in sorted(_INTENDED_LAYOUTS.items()):
        rule = table_rules.get((suffix, require))
        if rule is None:
            problems.append(f"intended-layout rule ({suffix}, {require}) missing from pad table")
            continue
        got = "|".join(_segments_from_rows(_fake_apply(rule["expect"], rule["ops"])))
        if got != want:
            problems.append(
                f"({suffix}, {require}): segment layout != intended\n"
                f"    got:  {got}\n    want: {want}"
            )

    # 3. an interior insert that could be mis-ordered must not exist here
    # (all minimax rules are single appends; this guard stays vacuous-safe).
    multi = [r for r in table["rules"] if len(r["ops"]) > 1]
    if multi:
        problems.append(f"unexpected multi-op rule in an append-only table: {multi[0]['suffix']}")

    # 3b. EVERY rule must be pinned by an intended layout, and every pinned
    #     layout must belong to a rule. A rule with no pin is not protected
    #     against a future edit (the ops-ordering defect class).
    for key in sorted(rules):
        if key not in _INTENDED_LAYOUTS:
            problems.append(f"rule {key} has no pinned intended layout")
    for key in sorted(_INTENDED_LAYOUTS):
        if key not in rules:
            problems.append(f"pinned intended layout {key} has no rule")

    # 3c. the target and the DSpark draft head must use DISJOINT require
    #     scopes: neither scope may be a substring of the other, or one
    #     rule set would silently match the other checkpoint's tensors.
    scopes = sorted({r.get("require", "") for r in table["rules"]})
    for i, a in enumerate(scopes):
        for b in scopes[i + 1:]:
            if a in b or b in a:
                problems.append(
                    f"require scopes {a!r} and {b!r} overlap (one is a substring "
                    "of the other) -- the rule sets are not disjoint"
                )

    # 3d. the promoted DSpark draft config must be whitelisted, otherwise
    #     the draft head is never padded and TP=3 dies on 64 % 3 != 0.
    if "minimax_m3_dspark" not in _DEFAULT_MODEL_TYPES:
        problems.append(
            "the promoted DSpark draft config type 'minimax_m3_dspark' is not "
            f"in _DEFAULT_MODEL_TYPES {tuple(_DEFAULT_MODEL_TYPES)} -- the draft "
            "head would never be padded"
        )

    # 4. config-pad arithmetic and idempotency (torch-free) — minimax M3.1
    # geometry incl. the lockstep sparse-index head count in the nested dict.
    cfg = type("SelftestConfig", (), {})()
    cfg.num_attention_heads = 64
    cfg.num_key_value_heads = 4
    cfg.vocab_size = 200064
    cfg.sparse_attention_config = {"sparse_num_index_heads": 4}
    _pad_text_config(cfg, 3)
    want = {
        "num_attention_heads": 96,
        "num_key_value_heads": 6,
        "vocab_size": 200064,
    }
    for k, v in want.items():
        if getattr(cfg, k, None) != v:
            problems.append(f"config pad {k} == {v} (got {getattr(cfg, k, None)!r})")
    if cfg.sparse_attention_config.get("sparse_num_index_heads") != 6:
        problems.append(
            "sparse_attention_config.sparse_num_index_heads != 6 "
            f"(got {cfg.sparse_attention_config.get('sparse_num_index_heads')!r})"
        )
    _pad_text_config(cfg, 3)  # second pass must change nothing
    if (cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.sparse_attention_config["sparse_num_index_heads"]) != (96, 6, 6):
        problems.append("config pads not idempotent on second application")

    # 4b. the DSpark DRAFT HEAD config.
    #     The draft's promoted config is the TARGET's geometry (64 q-heads /
    #     4 kv-heads / head_dim 128 / vocab 200064) with no sparse indexer:
    #     the same ratio-preserving pad applies, and head_dim MUST NOT move
    #     (the upstream dspark attention module passes
    #     head_dim=config.head_dim EXPLICITLY, so the
    #     engine's `head_dim or hidden_size // total_num_heads` fallback
    #     never fires -- but if it did, 6144 // 96 would silently become 64).
    dcfg = type("SelftestDraftConfig", (), {})()
    dcfg.num_attention_heads = 64
    dcfg.num_key_value_heads = 4
    dcfg.head_dim = 128
    dcfg.vocab_size = 200064
    dcfg.sparse_attention_config = None
    _pad_text_config(dcfg, 3)
    want_draft = {
        "num_attention_heads": 96,
        "num_key_value_heads": 6,
        "head_dim": 128,
        "vocab_size": 200064,
    }
    for k, v in want_draft.items():
        if getattr(dcfg, k, None) != v:
            problems.append(
                f"draft config pad {k} == {v} (got {getattr(dcfg, k, None)!r})"
            )
    if dcfg.sparse_attention_config is not None:
        problems.append(
            "draft config sparse_attention_config must stay None "
            f"(got {dcfg.sparse_attention_config!r})"
        )
    _pad_text_config(dcfg, 3)  # second pass must change nothing
    if (dcfg.num_attention_heads, dcfg.num_key_value_heads,
            dcfg.head_dim, dcfg.vocab_size) != (96, 6, 128, 200064):
        problems.append("draft config pads not idempotent on second application")

    if not quiet:
        for p in problems:
            print(f"[tp3-pad-selftest] FAIL {p}", file=sys.stderr)
        verdict = "FAIL" if problems else "PASS"
        print(
            f"[tp3-pad-selftest] {verdict}: {len(rules)} rules replayed, "
            f"{len(_INTENDED_LAYOUTS)} exact intended layouts asserted, "
            "ascending-order regression guard, config pads idempotent "
            "(target + DSpark draft head)",
            file=sys.stderr,
        )
    return 1 if problems else 0
