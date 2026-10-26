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

Regenerate and commit the manifest whenever ANY kit file changes:

    python3 kit-md5.py --write

The file list is explicit on purpose: a kit file that is NOT listed would
ship unverified, and a listed file that is MISSING is a hard error rather
than a silent omission.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
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


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write() -> int:
    rows = []
    for rel in KIT_FILES:
        p = HERE / rel
        if not p.is_file():
            print(f"kit-md5: MISSING kit file: {rel}", file=sys.stderr)
            return 1
        rows.append(f"{md5(p)}  {rel}")
    MANIFEST.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"kit-md5: wrote {MANIFEST.name} with {len(rows)} file(s)")
    for row in rows:
        print(f"  {row}")
    return 0


def check(expected_manifest_md5: str | None = None) -> int:
    if not MANIFEST.is_file():
        print(f"kit-md5: no manifest at {MANIFEST} -- run --write in the repo",
              file=sys.stderr)
        return 1
    bad = []
    expected = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            digest, _, rel = line.partition(" ")
        expected += 1
        p = HERE / rel
        if not p.is_file():
            bad.append(f"{rel}: MISSING on this box")
            continue
        got = md5(p)
        if got != digest:
            bad.append(f"{rel}: md5 {got} != manifest {digest}")
    listed = {line.partition("  ")[2] for line in
              MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()}
    for rel in KIT_FILES:
        if rel not in listed:
            bad.append(f"{rel}: kit file is NOT in the manifest")
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
    print(f"kit-md5: OK -- {expected} kit file(s) byte-identical to the manifest"
          + (f"; manifest md5 {expected_manifest_md5} confirmed" if expected_manifest_md5 else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate the manifest")
    ap.add_argument("--check", action="store_true", help="verify the kit against it")
    ap.add_argument("expected_manifest_md5", nargs="?",
                    help="the manifest's own digest, carried over a second channel")
    args = ap.parse_args()
    if args.write:
        return write()
    return check(args.expected_manifest_md5)


if __name__ == "__main__":
    sys.exit(main())
