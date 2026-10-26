#!/usr/bin/env bash
# boot-tp3.sh -- launch the MiniMax-M3.1-Flash TP=3 ring on THIS box.
#
# Thin, reproducible wrapper around start-tp3.sh: it resolves this box's
# role/rank/IP from the hostname (overridable by env) and tees the launch
# output to logs-boot-<host>.out in the recipe root.
#
# Boot order: HEAD FIRST -- it owns the TCPStore rendezvous on
# DIST_MASTER_PORT -- then the workers, which join it.
#
# Usage:  bash boot-tp3.sh            (on the box)
# Env overrides: ROLE NODE_RANK NODE_IP HEAD_IP TP CTX GMU PORT
# Precedence: the ENVIRONMENT WINS over the hostname map. Set ROLE=worker
# (with the rank and the IP) and this box is a worker whatever its hostname
# says; leave ROLE unset and the hostname decides. See the resolution below.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # launch/
ROOT="$(dirname "$HERE")"                      # the recipe root (the kit)
HOST=$(hostname -s)
LOG="$ROOT/logs-boot-$HOST.out"

# --- resolve this box ------------------------------------------------------
# THE ENVIRONMENT WINS. The hostname map below is the DEFAULT, not the
# authority: an operator who exports ROLE=worker must get a worker even on a
# box whose hostname happens to match another pattern, because the silent
# alternative is two heads and a broken ring. So the map is consulted ONLY
# when ROLE is unset; ROLE=... in the environment bypasses it entirely.
#
# Each value is then REQUIRED. With the role set by hand and the rank or the
# IP missing, the boot stops here and says which one -- guessing a rank is
# worse than refusing to start.
if [ -z "${ROLE:-}" ]; then
  case "$HOST" in
    spark-head*) ROLE=head;   NODE_RANK="${NODE_RANK:-0}"; NODE_IP="${NODE_IP:-192.168.0.10}" ;;
    spark-a*)    ROLE=worker; NODE_RANK="${NODE_RANK:-1}"; NODE_IP="${NODE_IP:-192.168.0.11}" ;;
    spark-b*)    ROLE=worker; NODE_RANK="${NODE_RANK:-2}"; NODE_IP="${NODE_IP:-192.168.0.12}" ;;
    *) echo "unknown box host '$HOST'" >&2
       echo "  rename it spark-head*, spark-a* or spark-b*, OR set ROLE," >&2
       echo "  NODE_RANK and NODE_IP in the environment -- the environment wins." >&2
       exit 2 ;;
  esac
fi
: "${NODE_RANK:?set NODE_RANK (0, 1 or 2) -- the rank this box plays in the ring}"
: "${NODE_IP:?set NODE_IP -- the cluster-facing address of this box}"

export ROLE NODE_RANK NODE_IP
export HEAD_IP="${HEAD_IP:-192.168.0.10}"
export TP="${TP:-3}" CTX="${CTX:-196608}" GMU="${GMU:-0.78}" PORT="${PORT:-8100}"
# the capture-grid cap rides the same chain as the budget -- an environment
# argument, unset by default (the engine's own default grid).
export CGCAP="${CGCAP:-}"
export DIST_MASTER_PORT="${DIST_MASTER_PORT:-25000}" CLUSTER_SIZE="${CLUSTER_SIZE:-3}"

# --- refuse to double-launch ----------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx minimax31-flash-tp3; then
  echo "container minimax31-flash-tp3 already exists -- refusing to launch twice" >&2
  exit 3
fi

# tee the launch output so a failed canary leaves its own log on the box
# NOTE: the tee is installed BEFORE the banner and the kit census, so the
# rank-identity proof lands IN the log. The launch is normally started with
# stdout to /dev/null (setsid ... >/dev/null 2>&1), so anything echoed before
# the tee is discarded -- a proof nobody can find is not a proof (see
# docs/FIXLOG.md).
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u +%FT%TZ)  box=$HOST role=$ROLE rank=$NODE_RANK ip=$NODE_IP ==="
echo "=== TP=$TP CTX=$CTX GMU=$GMU PORT=$PORT head=$HEAD_IP:$DIST_MASTER_PORT ==="
if [ -n "$CGCAP" ]; then
  echo "=== capture-grid cap (CGCAP): $CGCAP ==="
else
  echo "=== capture-grid cap (CGCAP): none (the engine's own default grid) ==="
fi

# --- kit identity gate (repo <-> box, fail-closed) -------------------------
# The three boxes must run BYTE-IDENTICAL code. A box carrying a different
# kit than the repo would serve a different recipe and the symptom would
# surface much later, looking like a model problem. This runs BEFORE the
# launch, so a mismatch fails the BOOT, not the window (see docs/FIXLOG.md).
#
# THE MANIFEST IS VERIFIED OVER A SECOND CHANNEL WHEN ONE IS SUPPLIED.
# A manifest cannot contain its own hash, so a box carrying a STALE kit AND a
# STALE manifest would verify against itself -- and would then serve an OLDER
# recipe without saying so. KIT_MANIFEST_MD5 is the manifest's own digest
# (config/cluster.env publishes the one that ships with this recipe;
# `md5sum kit-md5.txt` in the repo is how you recompute it). With it set, a
# stale manifest is a BOOT FAILURE instead of a silent downgrade.
#
# NOTE WHAT THIS DOES NOT COVER: the digest travels inside the checkout, so a
# box whose kit, manifest AND config/cluster.env are all stale is
# self-consistent and boots. The recipe's README step 4 -- comparing each
# box's manifest digest against the repo's, from the repo -- is what catches
# that case.
if ! (cd "$ROOT" && python3 kit-md5.py --check "${KIT_MANIFEST_MD5:-}"); then
  echo "kit-md5 MISMATCH: this box does not carry the repo's kit -- refusing to boot" >&2
  exit 4
fi
if [ -z "${KIT_MANIFEST_MD5:-}" ]; then
  echo "NOTE: KIT_MANIFEST_MD5 is not set, so this box's manifest was verified" >&2
  echo "      only against ITSELF. A box carrying a stale kit AND a stale" >&2
  echo "      manifest would agree with itself and serve an older recipe." >&2
  echo "      Set it (md5sum kit-md5.txt in the repo) to close that hole." >&2
fi

echo "=== pad kit md5 (rank-identity proof) ==="
md5sum "$ROOT/patches/apply_mm31_tp3pads.py" \
       "$ROOT/patches/verify-draft-pad.py" \
       "$ROOT/patches/tp3_pad_table.json" \
       "$ROOT/patches/overlay/vllm_tp3_pad/__init__.py" \
       "$ROOT/patches/overlay/vllm_tp3_pad/__main__.py" \
       "$ROOT/patches/overlay/vllm_tp3_pad/tp3_pad_table.json" \
       "$ROOT/tools/guard-memory.sh" \
       "$ROOT/tools/resolve-capture-grid.py" \
       "$ROOT/kit-md5.txt" \
       "$ROOT/launch/start-tp3.sh" \
       "$ROOT/launch/boot-tp3.sh" 2>&1

bash "$ROOT/launch/start-tp3.sh"
rc=$?
echo "=== launch returned rc=$rc at $(date -u +%FT%TZ) ==="
exit $rc
