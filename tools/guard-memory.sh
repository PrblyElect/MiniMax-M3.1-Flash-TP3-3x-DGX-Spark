#!/usr/bin/env bash
# guard-memory.sh -- host-memory guard for the minimax31 TP=3 lane.
#
# Runs ON a Spark, nohup/setsid detached. It KILLS the lane container when
# the HOST's MemAvailable falls under the floor, so a CUDA-graph capture spike
# (or any other memory excursion) cannot take the box's other services down
# with it -- on a box that also carries other applications and co-tenant
# services, that is the difference between one dead service and all of them.
#
# Admission requirements this satisfies:
#   * admission-guarded, with a SINGLETON LOCK -- exactly one guard per box
#     (flock -n; a second instance exits silently);
#   * bounded cleanup -- it only kills the lane's own container and writes log
#     lines. It deletes NOTHING, so a failed canary and its evidence survive.
#
# It never stops, removes or reloads anything else, and it never touches a
# model suite other than this lane's own container.
#
# DEBOUNCE: a single low sample is not enough. A box's steady state is only
# ~2 GiB above its threshold, and a service's RSS grows after a restart, so a
# one-sample trip could kill a HEALTHY serving container. Three consecutive
# breaches are required.
#
# RING NOTE: this guard is LOCAL. If it trips on one box, the other two ranks
# may be wedged in a collective holding ~80 GiB each. The guard does NOT
# touch them, and it must not: an automatic restore of the previous model
# profile is the operator's decision, not the guard's. It logs the ring-abort
# notice instead.
#
# The floor and the debounce are PROFILE VALUES, not kit edits, so the kit
# stays byte-identical across a change: the floor comes from
# config/cluster.env (GUARD_MIN_HEAD_MB / GUARD_MIN_R1_MB / GUARD_MIN_R2_MB)
# and the debounce from GUARD_DEBOUNCE.
#
# Usage on a box:
#   bash tools/guard-memory.sh --container minimax31-flash-tp3 &
#   touch ~/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark/logs/guard/guard.stop
#                                     # disarm (the stop file is newer than the
#                                     # start marker, so the loop exits)
set -uo pipefail

# THE FLOOR IS RESOLVED IN THIS ORDER: the --floor ARGUMENT, then the PROFILE
# OF RECORD's floor for this box's ROLE (the profile carries one per role, and
# this box picks its own by ROLE first and then by hostname -- the same
# precedence the boot uses), then GUARD_MIN_MB. If NONE of them resolves, the
# guard REFUSES TO ARM. The profile's number is therefore the
# number the guard ARMS WITH: a guard that fell back to a built-in would be
# armed WEAKER than the floor this recipe was measured against, and a guard
# that is too weak does not hold.
MIN=""
DEBOUNCE=${GUARD_DEBOUNCE:-3}
CONTAINER=${GUARD_CONTAINER:-minimax31-flash-tp3}
# Where this recipe's checkout lives on the box -- the guard's own log
# directory is under it. Override it with LANE_DIR if you cloned elsewhere.
LANE_DIR=${LANE_DIR:-~/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark}
# A tilde inside a parameter-expansion default is not expanded by every
# shell -- expand it here, so the default means the operator's home.
case "$LANE_DIR" in "~/"*) LANE_DIR="$HOME/${LANE_DIR#\~/}" ;; esac
LOG_DIR="$LANE_DIR/logs/guard"

usage() {
  cat <<EOF
usage: bash tools/guard-memory.sh [--floor MB] [--debounce N] [--container NAME]

  --floor MB       free-host-memory floor in MiB (default: the profile of
                   record's floor for this box -- GUARD_MIN_HEAD_MB,
                   GUARD_MIN_R1_MB or GUARD_MIN_R2_MB, chosen by ROLE first
                   and then by hostname -- else GUARD_MIN_MB. The guard
                   REFUSES to start if none of them is set: arming at the
                   built-in would be arming weaker than every pinned floor.)
  --debounce N     consecutive breaches required before the kill (default ${GUARD_DEBOUNCE:-3})
  --container NAME the container to kill (default ${GUARD_CONTAINER:-minimax31-flash-tp3})
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --floor)     MIN="$2"; shift 2 ;;
    --debounce)  DEBOUNCE="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --help|-h)   usage; exit 0 ;;
    *) echo "guard-memory.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# No --floor? Take the profile's floor for THIS box.
#
# THE ROLE IS RESOLVED THE SAME WAY THE BOOT RESOLVES IT: the ENVIRONMENT
# FIRST, then the hostname map. The boot honours ROLE=worker whatever the
# box is called, so a guard that keyed only on the hostname would look up
# the wrong box's floor on exactly the boxes where the operator had to
# override the role.
#
# AND WHEN NO PROFILE FLOOR RESOLVES, THE GUARD DOES NOT ARM AT ALL. The
# built-in is WEAKER than every floor this recipe pins, so falling back to
# it silently would arm a guard that cannot hold -- which is the one failure
# this tool exists to prevent. Refusing to start, with the remedy, is the
# honest outcome.
if [ -z "$MIN" ]; then
  HOST_S=$(hostname -s)
  ROLE_GUARD="${ROLE:-}"
  # ROLE=head is unambiguous, and the environment wins over the map.
  if [ "$ROLE_GUARD" = "head" ]; then
    MIN="${GUARD_MIN_HEAD_MB:-}"
  fi
  # THE TWO WORKER RANKS ARE TOLD APART BY THEIR HOSTNAME -- the same
  # discriminator the boot uses for NODE_RANK -- because ROLE=worker alone
  # does not say WHICH worker this is, and the two carry different floors.
  if [ -z "$MIN" ]; then
    case "$HOST_S" in
      spark-b*)    MIN="${GUARD_MIN_R2_MB:-}" ;;
      spark-a*)    MIN="${GUARD_MIN_R1_MB:-}" ;;
      spark-head*) MIN="${GUARD_MIN_HEAD_MB:-}" ;;
    esac
  fi
  # A worker on a box with an unusual name: the role still resolves it, to
  # the first worker's floor. Be exact with --floor if that is not the box's.
  if [ -z "$MIN" ] && [ "$ROLE_GUARD" = "worker" ]; then
    MIN="${GUARD_MIN_R1_MB:-}"
  fi
  # The single-floor escape hatch.
  if [ -z "$MIN" ]; then
    MIN="${GUARD_MIN_MB:-}"
  fi
  if [ -z "$MIN" ]; then
    echo "guard-memory.sh: no floor resolved for this box." >&2
    echo "  The profile of record (config/cluster.env) carries one floor per box --" >&2
    echo "  GUARD_MIN_HEAD_MB, GUARD_MIN_R1_MB, GUARD_MIN_R2_MB -- and the guard" >&2
    echo "  arms with the one for this box's role. None was found, and the" >&2
    echo "  built-in fallback is weaker than every floor this recipe pins, so" >&2
    echo "  the guard refuses to arm rather than arm too weak." >&2
    echo "  Fix: source config/cluster.env before starting the guard, or pass" >&2
    echo "  --floor <MB> explicitly." >&2
    exit 2
  fi
fi

mkdir -p "$LOG_DIR"

# --- singleton lock: one guard per box ------------------------------------
exec 9>"$LOG_DIR/guard.lock"
flock -n 9 || { echo "a guard is already running on this box" >&2; exit 0; }

# --- rotate: THIS CYCLE'S EVIDENCE STARTS HERE ----------------------------
# The acceptance rule requires EXACTLY ONE arming line in the log it reads:
# that is what makes "the guard evidence is windowed to this cycle" a real
# statement rather than a claim. A log that is appended to forever carries
# one arming line per boot cycle, so on the second cycle the count is 2 and
# the rule fails a perfectly healthy box -- the evidence would span two
# cycles and the window could not be bounded.
#
# SO ARMING STARTS A FRESH LOG. The previous cycle's log is KEPT, renamed,
# so the operator still has the old evidence: nothing is deleted, and a
# failed cycle's log survives exactly as the guard's own contract promises
# (it deletes NOTHING).
#
# THIS HAPPENS AFTER THE SINGLETON LOCK, on purpose: a second guard
# instance that exits immediately must never rotate the RUNNING guard's log.
#
# THE ROTATED NAME CARRIES A TIMESTAMP, so NOTHING IS EVER OVERWRITTEN. A
# fixed name would be overwritten on the third arming, which destroys the
# second-oldest cycle's evidence -- and this guard's contract is that it
# deletes NOTHING, so a failed cycle's evidence survives. The counter
# handles two armings inside the same second.
_STAMP=$(date '+%Y%m%dT%H%M%S')
for _f in guard.log guard_mem.log; do
  if [ -s "$LOG_DIR/$_f" ]; then
    _dst="$LOG_DIR/${_f%.log}.prev-$_STAMP.log"
    _n=1
    while [ -e "$_dst" ]; do
      _dst="$LOG_DIR/${_f%.log}.prev-$_STAMP-$_n.log"
      _n=$((_n + 1))
    done
    mv "$LOG_DIR/$_f" "$_dst"
  fi
done

touch "$LOG_DIR/guard.started"
low=999999
n=0
breaches=0
echo "$(date '+%F %T') guard start min=${MIN}MB debounce=${DEBOUNCE} container=$CONTAINER pid=$$" >> "$LOG_DIR/guard.log"

while ! [ "$LOG_DIR/guard.stop" -nt "$LOG_DIR/guard.started" ]; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  up=$(docker ps -q -f "name=^${CONTAINER}$" 2>/dev/null)
  if [ -n "$up" ]; then
    [ -n "$avail" ] && [ "$avail" -lt "$low" ] && low=$avail
    if [ -n "$avail" ] && [ "$avail" -lt "$MIN" ]; then
      breaches=$((breaches + 1))
      echo "$(date '+%F %T') BREACH $((breaches))/$DEBOUNCE avail_mb=$avail min=${MIN}MB" >> "$LOG_DIR/guard.log"
      if [ "$breaches" -ge "$DEBOUNCE" ]; then
        # --- pre-kill evidence snapshot (diagnose forward) -----------------
        # docker stats can hang on a wedged container; bound it so the
        # snapshot can never block the kill that follows it.
        {
          echo "----- $(date '+%F %T') pre-kill snapshot -----"
          free -m
          timeout 5 docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.MemPerc}}' 2>&1 | head -20
          echo "--- dmesg tail ---"
          dmesg 2>/dev/null | tail -40 || echo "(dmesg unavailable)"
        } >> "$LOG_DIR/guard.log" 2>&1
        # --- the kill: do NOT swallow its output ---------------------------
        kill_out=$(docker kill "$CONTAINER" 2>&1)
        kill_rc=$?
        echo "$(date '+%F %T') KILL $CONTAINER rc=$kill_rc out=[${kill_out}] avail_mb=$avail" >> "$LOG_DIR/guard.log"
        # --- post-kill state verification (count observable state) ---------
        echo "$(date '+%F %T') post-kill containers: $(docker ps -a --format '{{.Names}} {{.Status}}' | grep -c "^${CONTAINER} " || true) $CONTAINER row(s); running=$(docker ps -q -f "name=^${CONTAINER}$" | wc -l)" >> "$LOG_DIR/guard.log"
        echo "$(date '+%F %T') RING ABORT REQUIRED: the other two ranks may be wedged in a collective holding ~80 GiB each. Do NOT auto-restore -- a failed boot is preserved and the restore is the operator's decision. Report and wait." >> "$LOG_DIR/guard.log"
        breaches=0
      fi
    else
      [ "$breaches" -gt 0 ] && echo "$(date '+%F %T') breach cleared avail_mb=$avail" >> "$LOG_DIR/guard.log"
      breaches=0
    fi
  else
    breaches=0
  fi
  n=$((n + 1))
  # THE SENTINEL IS NOT A MEASUREMENT. `low` stays at 999999 until the
  # container is actually up, and a sample that says low_mb=999999 would
  # read as "an enormous free-memory minimum" -- so the acceptance rule
  # would PASS a box on which nothing was ever measured. Until there is a
  # measurement, the field says so.
  lowmb=$low
  [ "$low" = "999999" ] && lowmb="-"
  # THE SAMPLE CARRIES THE FULL DATE, not just the time of day. The
  # assembler bounds "this cycle's window" by comparing the samples'
  # timestamps with the arming line's timestamp, and a time of day alone
  # cannot order the two across midnight: a sample at 00:04:11 sorts
  # BEFORE an arming at 23:58:02, so a cycle that crosses midnight would
  # lose every sample and the rule would fail a healthy box. With the
  # date on both sides the comparison is exact.
  [ "$n" -gt 0 ] && [ $((n % 10)) = 0 ] && echo "$(date '+%F %T') avail_mb=${avail:-?} low_mb=$lowmb up=${up:+1} breaches=$breaches" >> "$LOG_DIR/guard_mem.log"
  sleep 1
done

echo "$(date '+%F %T') guard exit low_mb=$low" >> "$LOG_DIR/guard.log"
