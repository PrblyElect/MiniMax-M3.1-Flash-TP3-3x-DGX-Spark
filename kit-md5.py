#!/usr/bin/env python3
"""kit-md5.py -- the repo<->box identity manifest for this recipe.

WHY: the lane is three boxes that must run BYTE-IDENTICAL code. A box that
carries a different kit than the repo would silently serve a different
recipe, and the symptom (a collective mismatch, a stale patch, a flag that
was never applied) appears much later and looks like a model problem.

So the kit carries its own manifest, and launch/boot-tp3.sh runs `--check`
on the box BEFORE it launches anything: a mismatch fails the BOOT, not the
window.

  --write   regenerate kit-md5.txt from the kit files next to this script
  --check   verify every kit file against kit-md5.txt (exit 1 on any mismatch)
  --check-container
            hash the files the pad kit TOUCHES INSIDE THE CONTAINER, as they
            are there AFTER the overlay ran, and fail loudly on any mismatch.
            This is the other half of the identity: the manifest proves the
            kit on the BOX, and this proves the overlay in the CONTAINER --
            which is a different tree, on a different filesystem, written by a
            script that can fail without anyone noticing.

Regenerate and commit the manifest whenever ANY kit file changes:

    python3 kit-md5.py --write

The file list is explicit on purpose: a kit file that is NOT listed would
ship unverified, and a listed file that is MISSING is a hard error rather
than a silent omission.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "kit-md5.txt"

# Every file that must be byte-identical between the repo and the box: the
# boot scripts, the patch kit, the speculative-decode config (the serve line
# reads it and the self-check parses it, so its bytes are load-bearing for
# the boot exactly like the overlay's are), and the tools that run ON the
# box.
KIT_FILES = [
    "kit-md5.py",
    "launch/boot-tp3.sh",
    "launch/start-tp3.sh",
    # The image build is part of the kit: three boxes that build from
    # different Dockerfiles produce different images, and the symptom shows
    # up as an engine difference rather than as a kit difference.
    "launch/Dockerfile",
    "config/spec-config.json",
    "patches/apply_mm31_tp3pads.py",
    "patches/verify-draft-pad.py",
    "patches/tp3_pad_table.json",
    "patches/overlay/vllm_tp3_pad/__init__.py",
    "patches/overlay/vllm_tp3_pad/__main__.py",
    "patches/overlay/vllm_tp3_pad/tp3_pad_table.json",
    "tools/guard-memory.sh",
    "tools/capture-config.sh",
    "tools/capture-base-config.sh",
    "tools/acceptance-check.py",
    "tools/assemble-evidence.py",
    "tools/bench-decode.py",
    # The CORRECTNESS gates. They ship with the kit because a correctness gate
    # that is not on the box cannot gate the boot: the reader runs them in a
    # boot window, and the kit they run has to be the kit the recipe ships.
    "tools/check-spec-decode-short-prompt.py",
    "tools/check-long-prompt-determinism.py",
    # The producer battery ships with the kit because the producers are part
    # of it: the bench and the assembler PUBLISH the evidence the rule judges,
    # and a kit that ships a producer without a way to prove the producer is
    # a kit that ships an unverified producer.
    "tools/_bench_producer_battery.py",
    # The RULE is this kit's main product -- it is the only thing here that
    # can say ACCEPT or REJECT -- and it shipped with no regression test
    # inside the artifact at all. These two ship for the same reason the
    # producer battery does: a battery that cannot fail a mutant is not a
    # proof, and a rule nobody can test is a rule nobody can trust.
    #   _rule_battery.py       the rule's own hostile cases and its mutants
    #   _capture_gate_battery.py  the two capture gates, which are producers
    #                           too and were the only ones with no test
    "tools/_rule_battery.py",
    "tools/_capture_gate_battery.py",
    "tools/resolve-capture-grid.py",
]

# --- the container's view of the kit ------------------------------------
#
# The pad kit changes the vLLM tree INSIDE the container. The manifest above
# cannot see that tree: it lives in the image's site-packages, on a
# filesystem the repo has never touched. So the manifest carries a second
# section, [container], naming the files the kit INSTALLS there and the repo
# file whose bytes they must be:
#
#     <path inside the vLLM site-packages> = <repo file>
#
# The expected hash is the repo file's own hash from the manifest above, so
# the recipe's bytes are the only accepted ones: a stale overlay left in the
# tree by an earlier boot, or one from a different kit, is rejected.
CONTAINER_FILES = [
    ("vllm_tp3_pad/__init__.py", "patches/overlay/vllm_tp3_pad/__init__.py"),
    ("vllm_tp3_pad/__main__.py", "patches/overlay/vllm_tp3_pad/__main__.py"),
    ("vllm_tp3_pad/tp3_pad_table.json", "patches/tp3_pad_table.json"),
]

# --- the stock files the kit PATCHES inside the container ---------------
#
# Their patched bytes are a function of the IMAGE's original bytes, so no
# expected hash can be written down here -- the image is not in the repo. The
# expected hash is RE-DERIVED in the container instead: the kit leaves the
# pristine original beside every file it edits (<name>.tp3-orig), and the
# check re-applies the kit's own edit to that backup and compares the
# result with the file that is actually there. That is byte-exact, and it
# needs nothing from the repo beyond the kit itself.
PATCHED_FILES = [
    "vllm/transformers_utils/config.py",
    "vllm/model_executor/model_loader/default_loader.py",
    "vllm/model_executor/models/vision.py",
    "vllm/reasoning/minimax_m3_reasoning_parser.py",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """-> (kit rows {repo path: digest}, container rows {in-container path:
    repo path}, patched rows [in-container path])."""
    kit: dict[str, str] = {}
    container: dict[str, str] = {}
    patched: list[str] = []
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section == "container":
            cpath, _, rpath = line.partition("=")
            container[cpath.strip()] = rpath.strip()
        elif section == "patched":
            patched.append(line)
        elif section is None:
            digest, _, rel = line.partition("  ")
            if not rel:
                digest, _, rel = line.partition(" ")
            kit[rel] = digest
        else:
            raise ValueError(f"kit-md5: unknown manifest section [{section}]")
    return kit, container, patched


def write() -> int:
    rows = []
    for rel in KIT_FILES:
        p = HERE / rel
        if not p.is_file():
            print(f"kit-md5: MISSING kit file: {rel}", file=sys.stderr)
            return 1
        rows.append(f"{md5(p)}  {rel}")
    body = [
        "# kit-md5.txt -- the manifest of every kit file this recipe ships.",
        "# Regenerate with `python3 kit-md5.py --write`; never hand-edit.",
        "# The digest of THIS FILE travels with the boot (KIT_MANIFEST_MD5 in",
        "# config/cluster.env) so a stale manifest cannot verify itself.",
        *rows,
        "",
        "# --- the container's view of the kit --------------------------------",
        "# The pad kit writes into the vLLM tree INSIDE the container, which",
        "# the rows above cannot see: that tree lives in the image's",
        "# site-packages. These rows name the files the kit INSTALLS there and",
        "# the repo file whose bytes they must be -- the recipe's own bytes",
        "# are the only accepted ones.",
        "[container]",
    ]
    for cpath, rpath in CONTAINER_FILES:
        body.append(f"{cpath} = {rpath}")
    body += [
        "",
        "# --- the stock files the kit PATCHES inside the container -----------",
        "# Their patched bytes are a function of the IMAGE's original bytes, so",
        "# no expected hash can be written down: the image is not in the repo.",
        "# The check RE-DERIVES it in the container, by re-applying the kit's",
        "# own edit to the pristine backup the kit left beside each file.",
        "[patched]",
    ]
    body += list(PATCHED_FILES)
    MANIFEST.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"kit-md5: wrote {MANIFEST.name} with {len(rows)} file(s)")
    for row in rows:
        print(f"  {row}")
    return 0


def check(expected_manifest_md5: str | None = None) -> int:
    if not MANIFEST.is_file():
        print(f"kit-md5: no manifest at {MANIFEST} -- run --write in the repo",
              file=sys.stderr)
        return 1
    kit_rows, container_rows, patched_rows = parse_manifest(MANIFEST)
    bad = []
    for rel, digest in kit_rows.items():
        p = HERE / rel
        if not p.is_file():
            bad.append(f"{rel}: MISSING on this box")
            continue
        got = md5(p)
        if got != digest:
            bad.append(f"{rel}: md5 {got} != manifest {digest}")
    for rel in KIT_FILES:
        if rel not in kit_rows:
            bad.append(f"{rel}: kit file is NOT in the manifest")
    # The container sections are part of the manifest: a manifest that lost
    # them would make the container check vacuous (it would have nothing to
    # compare against), so their absence is a hard error here, not there.
    for cpath, rpath in container_rows.items():
        if rpath not in kit_rows:
            bad.append(f"[container] {cpath}: names {rpath}, which is not a "
                       f"kit file in the manifest")
    for rel in patched_rows:
        if not rel:
            bad.append("[patched]: an empty path is not a file")
    # A manifest cannot contain its own hash, so the manifest itself is
    # verified over a SECOND channel: the repo's manifest hash travels with
    # the boot (KIT_MANIFEST_MD5). Without this, a box carrying a STALE kit
    # and a STALE manifest would happily verify against itself.
    if expected_manifest_md5:
        got_manifest = md5(MANIFEST)
        if got_manifest != expected_manifest_md5:
            bad.append(f"kit-md5.txt: md5 {got_manifest} != expected "
                       f"{expected_manifest_md5} -- this box carries a STALE "
                       f"manifest; the kit and its manifest must come from the "
                       f"repo together")
    if bad:
        print("kit-md5: MISMATCH -- this box does not carry the repo's kit:",
              file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"kit-md5: OK -- {len(kit_rows)} kit file(s) byte-identical to the manifest"
          + (f"; manifest md5 {expected_manifest_md5} confirmed" if expected_manifest_md5 else ""))
    return 0


# ---------------------------------------------------------------- container
#
# The container check answers ONE question: is the tree the engine is about
# to import the tree this recipe describes?
#
# It answers it in two halves, because the two halves have two different
# sources of truth:
#
#   * the INSTALLED package is compared against the manifest's own digests.
#     The expected bytes are the recipe's, and they are written down.
#   * the PATCHED stock files are compared against a hash RE-DERIVED from
#     the kit's own pristine backup. The expected bytes are a function of the
#     image, so they cannot be written down -- but they can be recomputed,
#     because the kit left the input it computed them from.
#
# Both halves fail LOUDLY: a boot whose overlay silently failed, or whose
# tree carries a stale overlay, must die here rather than serve.


def _rederive(tree: Path, kit_dir: Path) -> tuple[dict[str, str], str]:
    """Re-apply the kit's own edit to the pristine backups in `tree`.

    Returns (expected digest per patched path, the kit's output)."""
    apply_py = kit_dir / "apply_mm31_tp3pads.py"
    if not apply_py.is_file():
        return {}, f"the pad kit is not at {apply_py}"
    r = subprocess.run([sys.executable, str(apply_py), str(tree), "apply"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {}, (r.stderr or r.stdout).strip()
    out = {}
    for rel in PATCHED_FILES:
        p = tree / rel
        if not p.is_file():
            return {}, f"the kit did not produce {rel}"
        out[rel] = md5(p)
    return out, r.stdout.strip()


def check_container(site: Path, kit_dir: Path, manifest_path: Path) -> int:
    if not site.is_dir():
        print(f"kit-md5: no vLLM site-packages at {site} -- nothing to check",
              file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"kit-md5: no manifest at {manifest_path}", file=sys.stderr)
        return 1
    kit_rows, container_rows, patched_rows = parse_manifest(manifest_path)
    bad: list[str] = []
    checked = 0

    # --- half one: the installed package must be the recipe's own bytes ---
    if not container_rows:
        bad.append("the manifest's [container] section is EMPTY -- the check "
                   "would have nothing to compare against")
    for cpath, rpath in sorted(container_rows.items()):
        want = kit_rows.get(rpath)
        if want is None:
            bad.append(f"{cpath}: the manifest names {rpath}, which is not a "
                       f"kit file -- the expected hash is unknown")
            continue
        live = site / cpath
        if not live.is_file():
            bad.append(f"{cpath}: MISSING in the container -- the pad overlay "
                       f"did not land")
            continue
        got = md5(live)
        if got != want:
            bad.append(f"{cpath}: md5 {got} != the recipe's {want} ({rpath}) -- "
                       f"the installed overlay is not this recipe's")
        else:
            checked += 1

    # --- half two: the patched stock files must be what the kit produces ---
    if not patched_rows:
        bad.append("the manifest's [patched] section is EMPTY -- the check "
                   "would have nothing to compare against")
    missing_backup: list[str] = []
    for rel in patched_rows:
        live = site / rel
        if not live.is_file():
            bad.append(f"{rel}: MISSING in the container")
            continue
        backup = live.with_name(live.name + ".tp3-orig")
        if not backup.is_file():
            missing_backup.append(rel)
    if missing_backup:
        for rel in missing_backup:
            bad.append(f"{rel}: no pristine backup ({rel.rsplit('/', 1)[-1]}"
                       f".tp3-orig) -- the kit's edit did not run on this tree, "
                       f"so the expected bytes cannot be re-derived")
    elif patched_rows:
        with tempfile.TemporaryDirectory(prefix="tp3pad-rederive-") as tmp:
            tree = Path(tmp) / "site"
            for rel in patched_rows:
                live = site / rel
                backup = live.with_name(live.name + ".tp3-orig")
                target = tree / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(backup, target)
            expected, kit_out = _rederive(tree, kit_dir)
            if not expected:
                bad.append(f"the kit's edit could not be re-derived: {kit_out}")
            else:
                for rel in patched_rows:
                    live = site / rel
                    got = md5(live)
                    want = expected[rel]
                    if got != want:
                        bad.append(f"{rel}: md5 {got} != the kit's own edit "
                                   f"{want} -- the file in the container is not "
                                   f"what the kit produces")
                    else:
                        checked += 1

    if bad:
        print("kit-md5: MISMATCH -- the container's tree is not this recipe's:",
              file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"kit-md5: OK -- the container's patched tree is this recipe's "
          f"({checked} file(s) hashed in place)")
    return 0


# ---------------------------------------------------------------- selftest
#
# A gate that cannot fail proves nothing. The container check is proved
# falsifiable here, WITHOUT an image: a synthetic site tree is built from the
# kit's own anchors, the kit is applied to it, and the check must (a) accept
# it and (b) reject it once the tree is tampered with.
#
# The synthetic tree carries the ANCHORS the kit edits, not a working
# engine -- the check hashes bytes, it never imports the tree. If the kit's
# anchors move, this selftest fails loudly, which is the correct outcome:
# the fixture and the kit would then disagree about what the patch is.

_FIXTURE_CONFIG = '''def get_config(model, **kwargs):
    """Synthetic stand-in for the pinned image's get_config()."""
    config = {"model": model}
    return config
'''

_FIXTURE_LOADER = '''def _load_weights(source, weights_iterator):
    if True:
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
'''

_FIXTURE_VISION = '''def _empty_branch(pixel_values, embed_dtype):
    x = torch.empty(
        (0, 4),
                device=pixel_values.device,
                dtype=embed_dtype,
    )
    y = torch.empty(
        (0, 4),
                device=pixel_values.device,
                dtype=embed_dtype,
    )
'''

_FIXTURE_REASONING = '''class _Parser:
    def __init__(self):
        self._start_token_ids = self._encode_marker(self.start_token)
        self._end_token_ids = self._encode_marker(self.end_token)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index < 0:
            return False
        if start_index < 0:
            return True
        return end_index > start_index
'''

FIXTURE = {
    "vllm/transformers_utils/config.py": _FIXTURE_CONFIG,
    "vllm/model_executor/model_loader/default_loader.py": _FIXTURE_LOADER,
    "vllm/model_executor/models/vision.py": _FIXTURE_VISION,
    "vllm/reasoning/minimax_m3_reasoning_parser.py": _FIXTURE_REASONING,
}


def _build_fixture(tree: Path, kit_dir: Path) -> bool:
    """Write the synthetic originals and apply the kit to them."""
    for rel, body in FIXTURE.items():
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    apply_py = kit_dir / "apply_mm31_tp3pads.py"
    if not apply_py.is_file():
        print(f"kit-md5: selftest: the pad kit is not at {apply_py}", file=sys.stderr)
        return False
    r = subprocess.run([sys.executable, str(apply_py), str(tree), "apply"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("kit-md5: selftest: the kit refused the synthetic tree -- its "
              "anchors and the fixture disagree:", file=sys.stderr)
        print((r.stderr or r.stdout).strip(), file=sys.stderr)
        return False
    return True


def selftest_container(kit_dir: Path, manifest_path: Path) -> int:
    bad: list[str] = []
    with tempfile.TemporaryDirectory(prefix="tp3pad-selftest-") as tmp:
        tree = Path(tmp) / "site"
        tree.mkdir(parents=True)
        if not _build_fixture(tree, kit_dir):
            return 1
        # 1. the untouched tree must PASS.
        if check_container(tree, kit_dir, manifest_path) != 0:
            bad.append("the check REJECTED a tree the kit itself produced -- "
                       "the check is wrong, not the tree")
        # 2. a patched file that was edited behind the kit's back must FAIL.
        victim = tree / PATCHED_FILES[0]
        original = victim.read_text()
        victim.write_text(original + "\n# tampered\n")
        if check_container(tree, kit_dir, manifest_path) == 0:
            bad.append(f"the check ACCEPTED a tampered {PATCHED_FILES[0]} -- it "
                       f"cannot fail")
        victim.write_text(original)
        # 3. an installed overlay that is not the recipe's must FAIL.
        overlay = tree / "vllm_tp3_pad" / "__init__.py"
        keep = overlay.read_text()
        overlay.write_text(keep + "\n# tampered\n")
        if check_container(tree, kit_dir, manifest_path) == 0:
            bad.append("the check ACCEPTED a tampered installed overlay -- it "
                       "cannot fail")
        overlay.write_text(keep)
        # 4. a MISSING installed overlay must FAIL.
        overlay.unlink()
        if check_container(tree, kit_dir, manifest_path) == 0:
            bad.append("the check ACCEPTED a tree with no installed overlay -- "
                       "it cannot fail")
        overlay.write_text(keep)
        # 5. and the restored tree must PASS again.
        if check_container(tree, kit_dir, manifest_path) != 0:
            bad.append("the check REJECTED the restored tree -- the check is "
                       "not idempotent")
    if bad:
        print("kit-md5: selftest FAILED -- the container check is not "
              "falsifiable:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print("kit-md5: selftest OK -- the container check accepts a tree the kit "
          "produced and rejects a tampered file, a tampered overlay and a "
          "missing overlay")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate the manifest")
    ap.add_argument("--check", action="store_true", help="verify the kit against it")
    ap.add_argument("--check-container", action="store_true",
                    help="hash the kit's files as they are inside the container")
    ap.add_argument("--site", type=Path,
                    help="the vLLM site-packages dir inside the container")
    ap.add_argument("--kit-dir", type=Path,
                    help="the pad kit's dir (the one holding apply_mm31_tp3pads.py)")
    ap.add_argument("--manifest", type=Path, help="the manifest to read")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the container check can fail (no image needed)")
    ap.add_argument("expected_manifest_md5", nargs="?",
                    help="the manifest's own digest, carried over a second channel")
    args = ap.parse_args()
    if args.write:
        return write()
    if args.check_container or args.selftest:
        manifest = args.manifest or MANIFEST
        kit_dir = args.kit_dir or (HERE / "patches")
        if args.selftest:
            return selftest_container(kit_dir, manifest)
        if args.site is None:
            print("kit-md5: --check-container needs --site (the vLLM "
                  "site-packages dir inside the container)", file=sys.stderr)
            return 1
        return check_container(args.site, kit_dir, manifest)
    return check(args.expected_manifest_md5)


if __name__ == "__main__":
    sys.exit(main())
