#!/usr/bin/env python3
"""apply_mm31_tp3pads.py -- inject the TP=3 pad overlay into the MiniMax
day-0 vLLM tree (minimax-m31-flash-arm64-cu134 lineage).

Adapted from an earlier TP=3 pad script (a proven pattern): stdlib only,
marker-idempotent, --revert restores originals, --verify compile-checks +
imports the installed package + runs its selftest.

FOUR stock files are edited (a fifth candidate -- the org-vocab LogitsProcessor
pass -- is deliberately DROPPED: MiniMax vocab 200064 is divisible by 3
and 64, so no vocab pad exists):

  vllm/transformers_utils/config.py
      -> vllm_tp3_pad.maybe_pad_hf_config at the END of get_config()
         (anchor located dynamically: the LAST function-level
         `return config` inside get_config -- this build's function is
         ~200 lines and the alternate anchor text does not exist here)
  vllm/model_executor/model_loader/default_loader.py
      -> wraps the weights iterator with tp3_pad_weights_iterator
         (this anchor, verified present verbatim in this build)
  vllm/model_executor/models/vision.py
      -> the DP-sharded vision path's EMPTY-rank branch must contribute the
         same dtype as the rank that actually ran the ViT, or NCCL's byte
         accounting desyncs and the ring deadlocks (see docs/FIXLOG.md).
         The ViT keeps its patch_embedding in fp32 ON PURPOSE
         (vision_tower.py:62-64) and the forward returns pixel_values.dtype
         (vision_tower.py:76); the empty branch used
         next(parameters()).dtype == fp32, so the allgather went out as
         4 bytes/elem on the image-less ranks and 2 bytes/elem on the rank
         that held the image. Fix: the empty branch uses pixel_values.dtype
         -- byte-for-byte the dtype the ViT's forward returns.
  vllm/reasoning/minimax_m3_reasoning_parser.py
      -> is_reasoning_end() scans the WHOLE prompt for the marker pair. The
         M3 chat template writes the literal markers into the SYSTEM turn as
         documentation ("wrap your reasoning in <think></think> tags"),
         UNCONDITIONALLY. So the scan always finds a closed pair and declares
         the model's reasoning finished before the first token: the streaming
         path then skips reasoning extraction entirely and emits the raw
         markers as delta.content, while non-streaming (which never consults
         is_reasoning_end) splits them correctly. Fix: scope the scan to the
         FINAL assistant turn -- the M3 template opens each assistant turn
         with bos_token ~ "ai" ~ "\n", so only markers after that header
         belong to the model.
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = "# tp3-pad-overlay"
OVERLAY_PKG = "vllm_tp3_pad"
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- reasoning patch
# The reasoning-parser edit is defined ONCE, here, as an (anchor, replacement)
# pair. apply() applies it; verify_reasoning_patch() REVERSES it on the patched
# file and requires the result to be byte-identical to the pristine original.
# That single equality proves the patch changed EXACTLY these two places --
# in particular that is_reasoning_end_streaming (the structured-output bounds
# path) and every other method are byte-for-byte untouched.
_REASON_ANCHOR_INIT = (
    "        self._start_token_ids = self._encode_marker(self.start_token)\n"
    "        self._end_token_ids = self._encode_marker(self.end_token)\n"
)
_REASON_REPL_INIT = _REASON_ANCHOR_INIT + (
    f"        {MARKER}: scope the prompt scan to the final assistant turn\n"
    "        # The M3 chat template opens each assistant turn with\n"
    "        # bos_token ~ \"ai\" ~ \"\\n\"; only markers after that header\n"
    "        # belong to the model, not to the template's documentation.\n"
    "        self._assistant_header_ids = self._encode_marker(\"ai\\n\")\n"
    "        # Set ONLY by set_prompt_token_ids(). The structured-output path\n"
    "        # decides on a FRESH parser instance that never receives the prompt,\n"
    "        # so that path keeps the original whole-prompt semantics -- and with\n"
    "        # it, JSON-schema requests stay constrained from the FIRST token\n"
    "        # (ConstraintBounds(0,0)), exactly as they are today.\n"
    "        self._prompt_ids_known = False\n"
)
_REASON_ANCHOR_ISR = (
    "    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n"
    "        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)\n"
    "        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)\n"
    "        if end_index < 0:\n"
    "            return False\n"
    "        if start_index < 0:\n"
    "            return True\n"
    "        return end_index > start_index\n"
)
_REASON_REPL_ISR = (
    "    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:\n"
    f"        {MARKER}: the caller handed us the rendered prompt, so\n"
    "        # is_reasoning_end may scope its scan to the model's own turn.\n"
    "        super().set_prompt_token_ids(prompt_token_ids)\n"
    "        self._prompt_ids_known = True\n"
    "\n"
    "    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n"
    f"        {MARKER}: scope the scan to the final assistant turn.\n"
    "        # The M3 chat template puts the literal marker pair in the SYSTEM\n"
    "        # turn as documentation, unconditionally. A whole-prompt scan\n"
    "        # therefore always finds a closed pair and declares the model's\n"
    "        # reasoning finished before the first token -- which is why\n"
    "        # STREAMING emitted the raw markers as content while\n"
    "        # non-streaming (which never consults this method) split them.\n"
    "        if not self._prompt_ids_known:\n"
    "            # No prompt handed to us: keep the upstream whole-prompt\n"
    "            # semantics. The structured-output decision path uses a fresh\n"
    "            # parser instance that never gets set_prompt_token_ids, and it\n"
    "            # relies on this answer to keep JSON-schema requests\n"
    "            # constrained from the first token.\n"
    "            start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)\n"
    "            end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)\n"
    "            if end_index < 0:\n"
    "                return False\n"
    "            if start_index < 0:\n"
    "                return True\n"
    "            return end_index > start_index\n"
    "        ids = list(input_ids)\n"
    "        turn_start = self._rfind_token_sequence(ids, self._assistant_header_ids)\n"
    "        if turn_start < 0:\n"
    "            return False\n"
    "        region = ids[turn_start + len(self._assistant_header_ids):]\n"
    "        start_index = self._rfind_token_sequence(region, self._start_token_ids)\n"
    "        end_index = self._rfind_token_sequence(region, self._end_token_ids)\n"
    "        if end_index < 0:\n"
    "            return False\n"
    "        if start_index < 0:\n"
    "            return True\n"
    "        return end_index > start_index\n"
)


def fail(msg):
    print(f"apply_mm31_tp3pads: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_target(pkg: Path, rel: str) -> Path:
    if (pkg / "vllm").is_dir():
        return pkg / rel
    if (pkg / "__init__.py").is_file() and (pkg / rel.split("/", 1)[1]).parent.is_dir():
        return pkg / rel.split("/", 1)[1]
    return None


def site_root(pkg: Path) -> Path:
    """The dir that should receive the vllm_tp3_pad package."""
    if (pkg / "vllm").is_dir():
        return pkg
    return pkg.parent


def find_get_config_return(text: str) -> tuple[str, int]:
    """Locate get_config() and return (anchor_line_text, count) for its LAST
    function-level `    return config` line."""
    lines = text.splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("def get_config("):
            start = i
            break
    if start is None:
        fail("get_config() not found in transformers_utils/config.py")
    anchor = None
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.startswith((" ", ")", "#")):
            break  # next top-level statement: function ended
        if ln == "    return config\n":
            anchor = j  # keep the LAST one
    if anchor is None:
        fail("no function-level `return config` found inside get_config()")
    count = sum(1 for ln in lines if ln == "    return config\n")
    return anchor, count


def apply_reasoning_parser_patch(rp: Path) -> bool:
    """Edit 4 -- scope the M3 reasoning parser's prompt scan.

    The M3 chat template writes the literal marker pair into the SYSTEM turn
    as documentation ("wrap your reasoning in <think></think> tags"),
    UNCONDITIONALLY. is_reasoning_end() scans the WHOLE prompt for a closed
    pair, so it always finds one and declares the model's reasoning finished
    before the first token. The streaming path then skips reasoning
    extraction entirely and emits the raw markers as delta.content, while
    non-streaming (which never consults this method) splits them correctly.

    The fix scopes the scan to the FINAL assistant turn. The M3 template opens
    each assistant turn with bos_token ~ "ai" ~ "\n", so only markers after
    that header belong to the model.

    The scoping is CONDITIONAL on set_prompt_token_ids() having been called:
      * the STREAMING path calls it (abstract_parser.parse_delta -> the
        upstream-documented hook) before is_reasoning_end, so it gets the
        scoped answer;
      * the STRUCTURED-OUTPUT decision path calls is_reasoning_end on a FRESH
        parser instance that never receives the prompt, so it keeps the
        original whole-prompt answer -- which is what keeps JSON-schema
        requests constrained from the FIRST token (ConstraintBounds(0,0))
        instead of leaving the first token free (ConstraintBounds(1,1), where
        the model could emit "<think>" and produce invalid JSON).

    Idempotent: a second run sees the marker and does nothing. The pristine
    original is kept beside the file as <name>.tp3-orig (the revert path).
    """
    src = rp.read_text()
    if MARKER in src:
        print("reasoning parser: marker present (already applied)")
        return False

    rp.with_name(rp.name + ".tp3-orig").write_text(src)  # revert backup

    if src.count(_REASON_ANCHOR_INIT) != 1:
        fail(f"reasoning parser: __init__ anchor count != 1 "
             f"({src.count(_REASON_ANCHOR_INIT)})")
    if src.count(_REASON_ANCHOR_ISR) != 1:
        fail(f"reasoning parser: is_reasoning_end anchor count != 1 "
             f"({src.count(_REASON_ANCHOR_ISR)})")

    src = src.replace(_REASON_ANCHOR_INIT, _REASON_REPL_INIT)
    src = src.replace(_REASON_ANCHOR_ISR, _REASON_REPL_ISR)
    rp.write_text(src)
    print("reasoning parser: is_reasoning_end scoped to the final assistant turn")
    return True


def apply(pkg: Path) -> int:
    sp = resolve_target(pkg, "vllm/transformers_utils/config.py")
    dl = resolve_target(pkg, "vllm/model_executor/model_loader/default_loader.py")
    if sp is None or dl is None:
        fail(f"could not resolve vllm tree under {pkg}")

    # --- edit 1: config.py (dynamic anchor) --------------------------------
    src = sp.read_text()
    if MARKER not in src:
        sp.with_name(sp.name + ".tp3-orig").write_text(src)  # revert backup
        anchor_idx, count = find_get_config_return(src)
        lines = src.splitlines(keepends=True)
        hook = (
            f"    {MARKER}: pad sharding geometry for TP=3\n"
            "    from vllm_tp3_pad import maybe_pad_hf_config\n"
            "\n"
            "    config = maybe_pad_hf_config(config)\n"
            "\n"
        )
        new_lines = lines[:anchor_idx] + [hook] + lines[anchor_idx:]
        sp.write_text("".join(new_lines))
        print(f"config.py: hook inserted before get_config's return config (line {anchor_idx + 1})")
    else:
        print("config.py: marker present (already applied)")

    # --- edit 2: default_loader.py (this anchor, verified in this build) ---
    src = dl.read_text()
    if MARKER not in src:
        dl.with_name(dl.name + ".tp3-orig").write_text(src)  # revert backup
        anchor = (
            "        # Apply the prefix.\n"
            '        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)\n'
        )
        if src.count(anchor) != 1:
            fail(f"default_loader anchor count != 1 ({src.count(anchor)})")
        repl = (
            f"        {MARKER}: zero-pad checkpoint tensors for TP=3\n"
            "        from vllm_tp3_pad import tp3_pad_weights_iterator\n"
            "\n"
            "        weights_iterator = tp3_pad_weights_iterator(weights_iterator)\n"
            "        # Apply the prefix.\n"
            '        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)\n'
        )
        dl.write_text(src.replace(anchor, repl))
        print("default_loader.py: iterator wrapped")
    else:
        print("default_loader.py: marker present (already applied)")

    # --- edit 3: models/vision.py (collective dtype symmetry) --------------
    vis = resolve_target(pkg, "vllm/model_executor/models/vision.py")
    if vis is None:
        fail("could not resolve vllm/model_executor/models/vision.py")
    src = vis.read_text()
    if MARKER not in src:
        vis.with_name(vis.name + ".tp3-orig").write_text(src)  # revert backup
        anchor = (
            "                device=pixel_values.device,\n"
            "                dtype=embed_dtype,\n"
        )
        n = src.count(anchor)
        if n != 2:
            fail(f"vision.py: expected exactly 2 empty-branch dtype sites, "
                 f"found {n} -- refusing to patch a partial set")
        repl = (
            "                device=pixel_values.device,\n"
            f"                dtype=pixel_values.dtype,  {MARKER}: rank-symmetric\n"
            "                # the ViT's forward returns pixel_values.dtype; the empty\n"
            "                # branch must match it or the allgather's byte count\n"
            "                # differs per rank and NCCL deadlocks (see docs/FIXLOG.md)\n"
        )
        vis.write_text(src.replace(anchor, repl))
        print(f"vision.py: empty-branch collective dtype fixed ({n} site(s))")
    else:
        print("vision.py: marker present (already applied)")

    # --- edit 4: reasoning/minimax_m3_reasoning_parser.py (streamed reasoning)
    # The M3 chat template names both markers in the SYSTEM turn as
    # documentation, so is_reasoning_end()'s whole-prompt scan always reports
    # "reasoning finished" and the streaming path never extracts the thinking
    # block. Scope the scan to the final assistant turn.
    rp = resolve_target(pkg, "vllm/reasoning/minimax_m3_reasoning_parser.py")
    if rp is None:
        fail("could not resolve vllm/reasoning/minimax_m3_reasoning_parser.py")
    apply_reasoning_parser_patch(rp)

    # --- install the package + table ----------------------------------------
    root = site_root(pkg)
    dst = root / OVERLAY_PKG
    if dst.exists():
        shutil.rmtree(dst)
    # never ship stale bytecode or desktop cruft into site-packages: a .pyc
    # from another interpreter (or an older source) must never shadow the .py
    # being installed, and a .DS_Store has no business in a site-packages
    # tree.
    shutil.copytree(HERE / "overlay" / OVERLAY_PKG, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                  ".DS_Store"))
    shutil.copyfile(HERE / "tp3_pad_table.json", dst / "tp3_pad_table.json")
    stale = resolve_target(pkg, "vllm/vllm_tp3_pad")
    if stale is not None and stale.is_dir():
        shutil.rmtree(stale)
    print(f"installed {dst}")
    return 0


def revert(pkg: Path) -> int:
    ok = 0
    for rel, bak_suffix in [
        ("vllm/transformers_utils/config.py", ".tp3-orig"),
        ("vllm/model_executor/model_loader/default_loader.py", ".tp3-orig"),
        ("vllm/model_executor/models/vision.py", ".tp3-orig"),
        ("vllm/reasoning/minimax_m3_reasoning_parser.py", ".tp3-orig"),
    ]:
        p = resolve_target(pkg, rel)
        if p is not None and p.with_name(p.name + bak_suffix).is_file():
            shutil.copyfile(p.with_name(p.name + bak_suffix), p)
            p.with_name(p.name + bak_suffix).unlink()
            print(f"reverted {rel}")
            ok += 1
    dst = site_root(pkg) / OVERLAY_PKG
    if dst.is_dir():
        shutil.rmtree(dst)
        print(f"removed {dst}")
        ok += 1
    return 0 if ok else 1


def verify_vision_patch(site: Path) -> None:
    """Self-check: prove, from the PINNED IMAGE's bytes, that the
    dtype patch actually reaches the collective.

    Fails the BOOT (this runs before vllm starts), never the window. Checks:
      1. the original had exactly 2 `dtype=embed_dtype,` sites and BOTH were
         the empty-shard (0, ...) allocation -- never the image-holding path;
      2. the patched file has no residual `dtype=embed_dtype`;
      3. the padding block inherits image_embeds_local.dtype, so no
         empty-path-only fp32 operand can survive the patch (and torch.cat
         would reject mismatched dtypes anyway);
      4. the gathered operand is the dtype-consistent cat result;
      5. the ViT's forward returns pixel_values.dtype -- i.e. the dtype the
         patch uses is BY CONSTRUCTION the dtype the image-holding rank
         contributes, so every rank's collective carries the same bytes.
    """
    import re

    vis = resolve_target(site, "vllm/model_executor/models/vision.py")
    tower = resolve_target(site, "vllm/models/minimax_m3/common/vision_tower.py")
    if vis is None or tower is None:
        fail("verify_vision_patch: vision.py / vision_tower.py not found")
    patched = vis.read_text()
    orig_p = vis.with_name(vis.name + ".tp3-orig")
    if not orig_p.is_file():
        fail("verify_vision_patch: no .tp3-orig backup to compare against")
    orig = orig_p.read_text()
    tw = tower.read_text()

    if orig.count("dtype=embed_dtype,") != 2:
        fail("verify_vision_patch: original does not have exactly 2 embed_dtype sites")
    empty = re.findall(
        r"image_embeds_local = torch\.empty\(\s*\(0,[^)]*\),[^)]*dtype=embed_dtype,",
        orig, re.S)
    if len(empty) != 2:
        fail(f"verify_vision_patch: only {len(empty)} of the 2 embed_dtype sites "
             "are empty-shard allocations")
    if "dtype=embed_dtype" in patched:
        fail("verify_vision_patch: patched file still contains dtype=embed_dtype")
    if patched.count(MARKER) != 2:
        fail(f"verify_vision_patch: expected 2 marker sites, found {patched.count(MARKER)}")
    pads = re.findall(r"padding = torch\.empty\(.*?dtype=([A-Za-z_][\w\.]*)", patched, re.S)
    if not pads or any(p != "image_embeds_local.dtype" for p in pads):
        fail(f"verify_vision_patch: padding dtype sources {set(pads)} -- an "
             "empty-path-only fp32 operand would survive the patch")
    if "tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)" not in patched:
        fail("verify_vision_patch: gathered operand is not the patched tensor")
    if "return out.reshape(x.shape[0], -1).to(pixel_values.dtype)" not in tw:
        fail("verify_vision_patch: the ViT's forward does not return "
             "pixel_values.dtype -- the patched dtype would not match the "
             "image-holding rank")
    print("verify_vision_patch: empty-branch dtype == the ViT's output dtype; "
          "no empty-path-only fp32 operand survives; gathered operand is the "
          "dtype-consistent cat result")


class _VerificationError(Exception):
    """A self-check that must FAIL THE BOOT (never the window)."""


class _Vocab(dict):
    """A vocab that hands out a fresh id for any key it does not have.

    The real BaseThinkingReasoningParser.__init__ does
        start_token_id = self.vocab.get(self.start_token)
    and RAISES if that is None -- so a fake tokenizer whose get_vocab()
    returns a plain dict without the markers makes the REAL base class
    refuse to construct. (Not hypothetical: an earlier revision of this
    check shipped a fake with no get_vocab() at all, and the BOOT failed at
    this very gate. The gate worked; the fixture was wrong.)
    """

    def get(self, key, default=None):
        if key not in self:
            self[key] = 200000 + len(self)
        return dict.get(self, key, default)


class _FakeTok:
    """A deterministic stand-in for the pinned image's tokenizer.

    Two things are pinned to the IMAGE's truth and must not drift:

      * the assistant header -- the pinned tokenizer encodes the two
        characters 'ai' + newline as [1361, 10], and the rendered prompt
        carries exactly that pair as its LAST two tokens. They are FIXED
        here on purpose: if the test built its prompt from the parser's OWN
        header ids, a patch that encoded the header wrongly would be
        self-consistently accepted -- the check would be circular and would
        pass a broken patch. Anchoring the header to the image's real ids
        makes the test an independent cross-check of the patch against the
        model it serves.
      * the marker strings -- the M3 start/end markers, built from chr()
        so no invisible character can slip into the fixture. The check
        ASSERTS them against the parser under test, so an image that
        changed its markers fails loudly instead of silently testing the
        wrong thing.
    """

    HEADER_IDS = (1361, 10)
    START_TOKEN = chr(0x3C) + "mm:think" + chr(0x3E)
    END_TOKEN = chr(0x3C) + "/mm:think" + chr(0x3E)

    def __init__(self):
        self.vocab = _Vocab()

    def _id_for(self, key: str) -> int:
        return self.vocab.setdefault(key, 200000 + len(self.vocab))

    def encode(self, text, add_special_tokens=False):
        if text == "ai" + chr(10):
            return list(self.HEADER_IDS)
        # a whole vocabulary entry (the markers are single entries in the
        # real tokenizer), so the id matches get_vocab()[text]
        return [self._id_for(text)]

    def get_vocab(self):
        return self.vocab


def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _method_text(src: str, name: str) -> str:
    """The source text of one method (4-space indented) in a class body."""
    marker = f"    def {name}("
    i = src.find(marker)
    if i < 0:
        return ""
    j = src.find("\n    def ", i + 1)
    return src[i:] if j < 0 else src[i:j]


def _reasoning_behaviour(patched_src: str, orig_src: str, tmp: Path) -> dict:
    """Run the behavioural assertions over a (patched, original) source pair.

    Raises _VerificationError on any failure -- that is what makes the
    mutation test below meaningful: a BROKEN patch must make this raise.

    The assertions:

      case 1 -- the system turn NAMES the marker pair (the template's
                documentation) and the prompt ends with an assistant header:
                  original -> True   (the bug: reasoning "already ended")
                  patched  -> False  (the fix)
      case 2 -- the FINAL assistant turn contains a CLOSED reasoning block:
                  patched  -> True   (unchanged: that block really is closed)
      hook   -- the scoped answer appears ONLY after set_prompt_token_ids()
      s-out  -- a FRESH instance that never received the prompt keeps the
                ORIGINAL whole-prompt answer. This is the structured-output
                invariant: the structured-output decision path calls
                is_reasoning_end on exactly such a fresh instance
                (protocol.resolve_reasoning_ended), so it still reports
                "reasoning ended" -> reasoning_ended=True ->
                _CONSTRAINED_FROM_START (ConstraintBounds(0,0)) -> the grammar
                constrains the FIRST generated token -> JSON-schema requests
                stay valid. If this ever went False, the grammar would open at
                token 1, the model could emit "<think>", and the response
                would be "<think>{...}" -- invalid JSON.
    """
    patched_py = tmp / "patched_reasoning_parser.py"
    patched_py.write_text(patched_src)
    orig_py = tmp / "orig_reasoning_parser.py"
    orig_py.write_text(orig_src)
    try:
        pmod = _load_module(patched_py, "tp3_verify_patched")
        omod = _load_module(orig_py, "tp3_verify_orig")
    except Exception as exc:  # noqa: BLE001
        raise _VerificationError(f"could not import the parser modules: {exc}")

    tok = _FakeTok()
    try:
        po = omod.MiniMaxM3ReasoningParser(tok)
        pp = pmod.MiniMaxM3ReasoningParser(tok)
    except Exception as exc:  # noqa: BLE001
        raise _VerificationError(f"could not instantiate the parsers: {exc}")

    start_ids = list(po._start_token_ids)
    end_ids = list(po._end_token_ids)
    # The markers themselves are pinned to the image: if they change, the
    # fixture must change with them, and this check says so loudly instead
    # of silently testing the wrong strings.
    if po.start_token != _FakeTok.START_TOKEN or po.end_token != _FakeTok.END_TOKEN:
        raise _VerificationError(
            f"the M3 markers changed in the image (start={po.start_token!r}, "
            f"end={po.end_token!r}) -- the fixture must be updated to match")
    # The header comes from the PINNED IMAGE, not from the parser under test.
    header_ids = list(_FakeTok.HEADER_IDS)
    if not start_ids or not end_ids:
        raise _VerificationError("the markers did not encode to any tokens")
    if list(pp._assistant_header_ids) != header_ids:
        raise _VerificationError(
            f"the assistant-turn header is encoded as "
            f"{list(pp._assistant_header_ids)!r}, but the pinned image's "
            f"tokenizer produces {header_ids!r} for 'ai\\n' -- the scan would "
            f"never find the model's turn")

    # case 1: the system turn NAMES the pair (documentation), then an
    # assistant header. The original wrongly reports "reasoning ended".
    doc_prompt = list(start_ids) + list(end_ids) + list(header_ids)
    got_orig = po.is_reasoning_end(doc_prompt)
    if got_orig is not True:
        raise _VerificationError(
            "the ORIGINAL did not report the documented-pair prompt as ended -- "
            "the premise of the fix does not hold on this image")

    # s-out: a FRESH instance that never received the prompt keeps the
    # ORIGINAL whole-prompt answer. This is the structured-output invariant:
    # the structured-output decision path calls is_reasoning_end on exactly
    # such a fresh instance (protocol.resolve_reasoning_ended), so it still
    # reports "reasoning ended" -> reasoning_ended=True ->
    # _CONSTRAINED_FROM_START (ConstraintBounds(0,0)) -> the grammar
    # constrains the FIRST generated token -> JSON-schema requests stay
    # valid. If this ever went False, the grammar would open at token 1, the
    # model could emit "<think>", and the response would be "<think>{...}"
    # -- invalid JSON.
    frontend = pmod.MiniMaxM3ReasoningParser(tok)
    if frontend.is_reasoning_end(doc_prompt) is not True:
        raise _VerificationError(
            "a fresh parser instance must keep the original whole-prompt answer "
            "-- the structured-output path decides on such an instance, and if "
            "it reported 'not ended' the JSON-schema grammar would open at "
            "token 1 and the model could emit '<think>' (invalid JSON)")

    # case 1 (streaming path): the caller hands the rendered prompt over
    # FIRST (abstract_parser.parse_delta -> the upstream-documented hook),
    # and only then is the scan scoped -- so the template's documentation can
    # no longer close the model's reasoning.
    pp = pmod.MiniMaxM3ReasoningParser(tok)
    pp.set_prompt_token_ids(doc_prompt)
    got_patch = pp.is_reasoning_end(doc_prompt)
    if got_patch is not False:
        raise _VerificationError(
            f"the PATCHED parser still reports the documented-pair prompt as "
            f"ended (got {got_patch!r}) -- the scoping did not take effect")

    # case 2: the FINAL assistant turn holds a CLOSED reasoning block. It
    # must still be reported as ended -- the patch must not change that.
    closed_prompt = (list(start_ids) + list(end_ids) + list(header_ids)
                     + list(start_ids) + list(end_ids))
    pp3 = pmod.MiniMaxM3ReasoningParser(tok)
    pp3.set_prompt_token_ids(closed_prompt)
    if pp3.is_reasoning_end(closed_prompt) is not True:
        raise _VerificationError(
            "a CLOSED reasoning block in the final assistant turn must still be "
            "reported as ended -- the patch changed a case it must not change")

    # hook: the scoped answer appears ONLY once the prompt was handed over.
    pp2 = pmod.MiniMaxM3ReasoningParser(tok)
    if pp2.is_reasoning_end(doc_prompt) is not True:
        raise _VerificationError(
            "the structured-output fallback (no prompt handed over) must keep "
            "the original whole-prompt answer")
    pp2.set_prompt_token_ids(doc_prompt)
    if pp2.is_reasoning_end(doc_prompt) is not False:
        raise _VerificationError(
            "after set_prompt_token_ids the scan must be scoped -- the hook is "
            "not wired")

    return {
        "orig_doc_prompt": got_orig,
        "patched_doc_prompt": got_patch,
        "patched_closed_block": True,
        "fresh_instance_whole_prompt": True,
    }


# The four mutants below are the ways this patch could be WRONG. Each must be
# REJECTED by _reasoning_behaviour; if one were accepted, the self-check would
# be vacuous (it would pass a semantically broken patch).
_REASON_MUTANTS = (
    (
        "M1 no-scoping (the scan covers the whole prompt again)",
        "region = ids[turn_start + len(self._assistant_header_ids):]",
        "region = ids",
    ),
    (
        "M2 never-ends (the assistant-turn boundary is never found)",
        "turn_start = self._rfind_token_sequence(ids, self._assistant_header_ids)",
        "turn_start = -1",
    ),
    (
        "M3 always-ends (the scoped branch reports 'ended' unconditionally)",
        "        ids = list(input_ids)\n"
        "        turn_start = self._rfind_token_sequence(ids, self._assistant_header_ids)\n"
        "        if turn_start < 0:\n"
        "            return False\n",
        "        return True\n",
    ),
    (
        "M4 wrong-header (the assistant-turn header is encoded wrongly, so the "
        "boundary is never found)",
        "        self._assistant_header_ids = self._encode_marker(\"ai\\n\")\n",
        "        self._assistant_header_ids = self._encode_marker(\"zz\\n\")\n",
    ),
)


def verify_reasoning_patch(site: Path) -> None:
    """Self-check for the streaming-reasoning fix.

    Proves, from the PINNED IMAGE's bytes, that

      (1) the patch changed EXACTLY the two intended places and nothing else
          -- verified by REVERSING it on the patched file and requiring the
          result to be byte-identical to the pristine original. In
          particular is_reasoning_end_streaming (the structured-output
          bounds path) and every other method are untouched;
      (2) the behaviour is right -- the template's DOCUMENTATION of the marker
          pair (which lives in the system turn) can no longer make the engine
          believe the model's reasoning already ended, a closed block in the
          final assistant turn still does, and a fresh instance keeps the
          original answer so JSON-schema requests stay constrained from the
          FIRST token;
      (3) the self-check is not vacuous -- four mutated patches (no scoping /
          never ends / always ends / wrong header) must each be REJECTED.

    Textual assertions alone are not enough: a patch that merely CONTAINED
    the right strings (while still scanning the whole prompt) would pass
    them. So this instantiates the patched class and the original class over
    a synthetic prompt with a fake tokenizer and asserts the actual ANSWERS.

    Fails the BOOT (this runs before vllm starts), never the window.
    """
    import tempfile

    rp = resolve_target(site, "vllm/reasoning/minimax_m3_reasoning_parser.py")
    if rp is None:
        fail("verify_reasoning_patch: minimax_m3_reasoning_parser.py not found")
    patched = rp.read_text()
    orig_p = rp.with_name(rp.name + ".tp3-orig")
    if not orig_p.is_file():
        fail("verify_reasoning_patch: no .tp3-orig backup to compare against")
    orig = orig_p.read_text()

    # --- (1) the patch touches EXACTLY the two intended places -------------
    if patched.count(_REASON_REPL_ISR) != 1:
        fail("verify_reasoning_patch: the scoped is_reasoning_end replacement "
             "is not present exactly once")
    if patched.count(_REASON_REPL_INIT) != 1:
        fail("verify_reasoning_patch: the __init__ replacement is not present "
             "exactly once")
    reversed_src = patched.replace(_REASON_REPL_ISR, _REASON_ANCHOR_ISR)
    reversed_src = reversed_src.replace(_REASON_REPL_INIT, _REASON_ANCHOR_INIT)
    if reversed_src != orig:
        fail("verify_reasoning_patch: the patched file differs from the pristine "
             "original by MORE than the two intended edits -- something else "
             "was changed (is_reasoning_end_streaming, which the "
             "structured-output bounds path depends on, must stay untouched)")
    if _method_text(orig, "is_reasoning_end_streaming") != \
            _method_text(patched, "is_reasoning_end_streaming"):
        fail("verify_reasoning_patch: is_reasoning_end_streaming was modified "
             "-- it drives the structured-output constraint bounds")
    expected_markers = (_REASON_REPL_INIT.count(MARKER)
                        + _REASON_REPL_ISR.count(MARKER))
    if patched.count(MARKER) != expected_markers:
        fail(f"verify_reasoning_patch: expected {expected_markers} marker sites, "
             f"found {patched.count(MARKER)}")

    # --- (2) the behaviour --------------------------------------------------
    tmp = Path(tempfile.mkdtemp())
    try:
        summary = _reasoning_behaviour(patched, orig, tmp)
    except _VerificationError as exc:
        fail(f"verify_reasoning_patch: {exc}")

    # --- (3) the self-check is not vacuous ---------------------------------
    for name, needle, replacement in _REASON_MUTANTS:
        if patched.count(needle) != 1:
            fail(f"verify_reasoning_patch: the mutation anchor for {name} is "
                 f"not present exactly once -- the mutation test cannot run")
        mutant = patched.replace(needle, replacement)
        try:
            _reasoning_behaviour(mutant, orig, tmp)
        except _VerificationError:
            continue  # good: the broken patch was rejected
        fail(f"verify_reasoning_patch: the self-check ACCEPTED a broken patch "
             f"({name}) -- it would pass a semantically broken patch, so it "
             f"proves nothing")

    print("verify_reasoning_patch: the patch touches only __init__ and "
          "is_reasoning_end (is_reasoning_end_streaming is byte-identical); "
          "behaviourally the template's documentation no longer closes the "
          "model's reasoning, a closed block in the final assistant turn "
          "still does, and a fresh instance keeps the original whole-prompt "
          "answer so JSON-schema stays constrained from the first token; the "
          f"assistant header matches the pinned image's ids "
          f"{list(_FakeTok.HEADER_IDS)}; "
          f"{len(_REASON_MUTANTS)}/{len(_REASON_MUTANTS)} broken patches rejected")
def verify(pkg: Path) -> int:
    for rel in ["vllm/transformers_utils/config.py",
                "vllm/model_executor/model_loader/default_loader.py",
                "vllm/model_executor/models/vision.py",
                "vllm/reasoning/minimax_m3_reasoning_parser.py"]:
        p = resolve_target(pkg, rel)
        if p is None:
            fail(f"missing {rel}")
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            fail(f"{rel} does not compile: {e}")
        if MARKER not in p.read_text():
            fail(f"{rel} missing the {MARKER} marker")
    dst = site_root(pkg) / OVERLAY_PKG
    if not (dst / "__init__.py").is_file():
        fail(f"{dst} not installed")
    # prove the engine-side dtype claim from the image's bytes
    verify_vision_patch(pkg)
    # prove the streaming-reasoning fix from the image's bytes
    verify_reasoning_patch(pkg)
    # the real thing: import the package and run its selftest
    r = subprocess.run(
        [sys.executable, "-c",
         "import vllm_tp3_pad, sys;"
         " sys.exit(vllm_tp3_pad.selftest(quiet=False))"],
        cwd=str(site_root(pkg)), capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        fail("overlay selftest failed in-subprocess")
    print("verify: compiles, markers present, import + selftest PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_package_dir")
    ap.add_argument("cmd", choices=["apply", "revert", "verify"], nargs="?", default="apply")
    a = ap.parse_args()
    pkg = Path(a.vllm_package_dir).resolve()
    if a.cmd == "apply":
        rc = apply(pkg)
        return rc if rc == 0 else verify(pkg)
    if a.cmd == "revert":
        return revert(pkg)
    return verify(pkg)


if __name__ == "__main__":
    raise SystemExit(main())
