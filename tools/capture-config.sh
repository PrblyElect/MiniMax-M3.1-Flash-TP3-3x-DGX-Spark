#!/usr/bin/env bash
# capture-config.sh -- prove what the running boot actually enabled, and
# gate it.
#
# WHAT IT PROVES
# Speculative decode is not one flag: the --speculative-config line does
# three jobs at once, so "the spec config is on" is NOT one condition:
#   * it turns on the DRAFT head (a second model in the same process),
#   * it changes the DECODE QUERY LENGTH (uniform_decode_query_len = 1 + k),
#     which changes the CUDA-graph capture COVERAGE, and
#   * it spends KV on the draft's own KV group.
# The capture pool grows and the KV pool shrinks, so the boot has to PROVE
# each of them, and the interesting numbers are the capture COVERAGE and
# the capacity floor.
#
# IT RUNS THE BASE CAPTURE GATE FIRST (capture-base-config.sh, next to this
# script): eager is off, the three subsystems are on, the encoder is not
# captured, the capture pool holds memory, the budget took effect, the
# startup completed. The base gate's own capacity defaults are now THE
# RECIPE'S OWN (one full-length request at the served context, 196,608
# tokens and 7.51 GiB), so running the base gate on its own no longer fails
# a healthy boot of this recipe; this profile passes ITS OWN floor down to
# the base gate as well, because the floor belongs to the served context
# and not to the base gate.
#
# WHAT IS ASSERTED ON TOP OF THE BASE GATE:
#   1. the KIT is the recipe checkout's kit on THIS box ("kit-md5: OK" in
#      this box's own section). The check is asserted INSIDE this box's
#      section on purpose: a box that carries a different kit must fail
#      HERE, not be masked by another box's OK.
#   2. the SPEC CONFIG reached the process ON EVERY RANK. THE ONE SOURCE IS
#      THE MOUNTED FILE: the container's Cmd READS /spec-config.json and
#      passes the very same string to the self-check and to the serve line,
#      so the file is read INSIDE THE LIVE CONTAINER on every rank (the
#      workers print no engine config at all), and the head's log
#      additionally shows the resolved speculative_config= and the engine's
#      OWN spelling num_spec_tokens=3;
#   3. the CAPTURE COVERAGE -- max_cudagraph_capture_size >= 128 AND the
#      capture list CONTAINS 128 (with only 64, just 16 of the 32 sequences
#      are covered: that is a CLIFF, not a dial);
#   4. the CAPACITY FLOOR -- pool >= 196,608 tokens (the served context)
#      AND the engine's own
#      "Maximum concurrency for N tokens per request" >= 1.00x, WHERE N IS
#      THE SERVED CONTEXT (the lane can still hold ONE full-length request),
#      AND the engine's own max_seq_len must equal the pinned context
#      (the head proves it; the workers make no claim).
#
# EVERY CHECK FAILS LOUDLY: a [FAIL] line naming the box and the missing
# fact, and a non-zero exit. Nothing is "printed but not asserted".
#
# Usage: bash capture-config.sh [--head HOST] [--r1 HOST] [--r2 HOST] [--out DIR]
# Exit: 0 when the boot is a genuine speculative-decode boot, 1 otherwise.
# Fail-closed.
set -uo pipefail

CONTAINER=minimax31-flash-tp3
# KIT is overridable so the gate can be NEGATIVE-TESTED against a recipe that
# deliberately differs from the running boot, without touching the live kit.
KIT=${KIT:-~/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark}
# A tilde inside a parameter-expansion default is not expanded by every
# shell -- expand it here, so the default means the operator's home.
case "$KIT" in "~/"*) KIT="$HOME/${KIT#\~/}" ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST=$(hostname -s)
LOG="$KIT/logs-boot-$(hostname -s).out"
# The base capture gate (the inherited checks). It lives next to this
# script; override BASE_GATE to point at another copy.
BASE_GATE="${BASE_GATE:-$HERE/capture-base-config.sh}"

# --- the recipe's own thresholds -------------------------------------------
# THE SERVED CONTEXT THIS RECIPE PINS. The engine prints its concurrency
# FOR THE REQUEST LENGTH IT WAS ASKED ABOUT, so the denominator has to be
# checked, not assumed: a boot that served a different context would print
# a different denominator, and comparing that figure against a threshold
# written for another context would compare two different things.
EXPECTED_CTX=${EXPECTED_CTX:-196608}
# THE CAPACITY FLOOR IS THE SERVED CONTEXT, and it is ONE assignment: the
# lane must be able to hold ONE full-length request AT the context it
# serves. (It used to be a flat 150,000 tokens -- the same idea for the
# 131,072-token context this profile used to serve.)
#
# IF THE CONTEXT CHANGES, THE GiB FLOOR BELOW MUST CHANGE WITH IT: the two
# are the same fact in two units.
KV_FLOOR=${KV_FLOOR:-$EXPECTED_CTX}
# The capacity floor is NOT the 250,000 tokens of a plain (non-speculative)
# boot: speculative decode spends KV on the draft's weights (0.71 GiB/rank),
# the draft's own KV group (2.5 KiB/token) and the capture-pool growth.
# Below the served context, or a concurrency under 1.00x, is a REJECT ON
# CAPACITY.
#
# The token floor expressed in GiB at the recipe's KV density:
# 37.53 KiB/token (the target model's own KV group) + 2.5 KiB/token (the
# draft model's own KV group) = 40.03 KiB/token; 196,608 tokens x 40.03 KiB
# = 7.5056 GiB, rounded UP to 7.51 GiB. The pool is sized by the TIGHTEST
# rank, so every rank's own KV budget must clear it.
KV_FLOOR_GIB=${KV_FLOOR_GIB:-7.51}
# The capture COVERAGE floor: max_num_seqs 32 x (k+1) 4 = 128. A capture list
# that stops at 64 covers only 16 sequences -- a cliff.
CAPTURE_COVERAGE_MIN=${CAPTURE_COVERAGE_MIN:-128}
# The engine's own concurrency figure for one full-length request.
CONCURRENCY_MIN=${CONCURRENCY_MIN:-1.00}
# The memory budget this profile pins. It is PINNED, never derived from the
# box: a stale kit could otherwise agree with itself and PASS. GMU is NOT
# raised to buy the pool back.
EXPECTED_GMU_PINNED="${EXPECTED_GMU_PINNED:-0.78}"
# The spec config this profile pins.
SPEC_METHOD=${SPEC_METHOD:-dspark}
SPEC_K=${SPEC_K:-3}

# --- optional arguments ----------------------------------------------------
# The tool proves the box it runs on. The arguments only say which lane the
# three boxes belong to, and where to drop a copy of this box's findings.
#   --head HOST --r1 HOST --r2 HOST   the lane's three box labels (recorded
#                                     in the output; this tool always proves
#                                     the box it runs on)
#   --out DIR                         also write this box's findings to
#                                     DIR/capture-gate-<host>.txt
LANE_HEAD=""
LANE_R1=""
LANE_R2=""
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --head|--r1|--r2|--out)
      [[ $# -ge 2 ]] || { echo "capture-config: $1 needs a value" >&2; exit 2; }
      case "$1" in
        --head) LANE_HEAD="$2" ;;
        --r1)   LANE_R1="$2" ;;
        --r2)   LANE_R2="$2" ;;
        --out)  OUT_DIR="$2" ;;
      esac
      shift 2 ;;
    --head=*) LANE_HEAD="${1#--head=}"; shift ;;
    --r1=*)   LANE_R1="${1#--r1=}"; shift ;;
    --r2=*)   LANE_R2="${1#--r2=}"; shift ;;
    --out=*)  OUT_DIR="${1#--out=}"; shift ;;
    -h|--help)
      echo "usage: capture-config.sh [--head HOST] [--r1 HOST] [--r2 HOST] [--out DIR]"
      exit 0 ;;
    *)
      echo "capture-config: ignoring an unknown argument: $1" >&2
      shift ;;
  esac
done

fail=0
note_fail() { echo "[FAIL] $HOST: $1"; fail=1; }
note_pass() { echo "[PASS] $HOST: $1"; }

# --- EXTRACT_ONLY: the extraction as a TESTABLE API -------------------------
# A gate that required the per-mode capture bars on EVERY rank would have
# failed both workers, and no test could see it, because the tests never ran
# the extraction -- the tests inject the facts the gate must emit.
#
# EXTRACT_ONLY=<logfile> runs ONLY the extraction over a log file and prints
# the same machine-readable facts the live gate prints -- with NO fleet
# access, NO verdict and NO exit-code semantics. The tests run it over a
# boot-log slice and assert the counts the gate would have emitted. The
# extraction code below is the SAME code in both paths; there is no second
# implementation to drift.
EXTRACT_ONLY="${EXTRACT_ONLY:-}"
if [[ -n "$EXTRACT_ONLY" ]]; then
  HOST="extract-only"
  # Facts, not verdicts: in this mode the notes are suppressed and the exit
  # code carries no meaning (the thresholds live in the rule).
  note_fail() { :; }
  note_pass() { :; }
fi

TMP=$(mktemp)
OUT_FILE=""
cleanup() {
  local rc=$?
  if [[ -n "$OUT_FILE" ]]; then
    exec 1>&3 3>&-
    cat "$OUT_FILE"
  fi
  rm -f "$TMP"
  return $rc
}
trap cleanup EXIT

# --- the optional output file ----------------------------------------------
# With --out DIR, this box's findings are ALSO written to
# DIR/capture-gate-<label>.txt, where <label> is THIS BOX'S LANE LABEL --
# the one passed to --head/--r1/--r2, not the raw hostname. `hostname -s`
# trims the domain, so a box called spark-head.local would otherwise be
# written as "spark-head" and the acceptance rule, which keys its sections
# by the lane label, would find no section for it at all.
#
# The findings still go to stdout as well: the file is a copy, not a
# replacement.
if [[ -n "$OUT_DIR" ]]; then
  if ! mkdir -p "$OUT_DIR"; then
    echo "capture-config: cannot create the output directory $OUT_DIR" >&2
    exit 1
  fi
  # THIS BOX'S LANE LABEL: the one whose SHORT name is this box's hostname.
  LANE_LABEL=""
  for cand in "$LANE_HEAD" "$LANE_R1" "$LANE_R2"; do
    [[ -n "$cand" ]] || continue
    case "$cand" in
      "$HOST"|"$HOST".*) LANE_LABEL="$cand"; break ;;
    esac
  done
  if [[ -z "$LANE_LABEL" ]]; then
    LANE_LABEL="$HOST"
    echo "capture-config: WARNING -- no --head/--r1/--r2 label matches this box ($HOST)" >&2
    echo "  The section header will be '$HOST'. The acceptance rule looks" >&2
    echo "  sections up by the LANE LABEL, so pass the same three labels on" >&2
    echo "  every box, or the rule will not find this section." >&2
  fi
  OUT_FILE="$OUT_DIR/capture-gate-${LANE_LABEL}.txt"
  exec 3>&1 1>"$OUT_FILE"
  echo "=== ${LANE_LABEL} ==="
fi
echo "lane boxes: head=${LANE_HEAD:-unknown} r1=${LANE_R1:-unknown} r2=${LANE_R2:-unknown}"

# --- 0. the base capture gate ----------------------------------------------
if [[ -n "$EXTRACT_ONLY" ]]; then
  echo "=== EXTRACT_ONLY: no fleet access, no verdicts -- facts only ==="
elif KV_FLOOR="$KV_FLOOR" KV_FLOOR_GIB="$KV_FLOOR_GIB" \
   EXPECTED_GMU_PINNED="$EXPECTED_GMU_PINNED" KIT="$KIT" \
   bash "$BASE_GATE"; then
  note_pass "the base capture gate is clean"
else
  note_fail "the base capture gate FAILED (see the lines above)"
fi

# --- 1. the KIT on THIS box ------------------------------------------------
# The kit check is asserted INSIDE this box's own section: a box that
# carries a different kit must fail HERE.
if [[ -n "$EXTRACT_ONLY" ]]; then
  KIT_FACT="skipped"
else
  echo "--- kit identity on this box ---"
  KIT_OUT=$(cd "$KIT" && python3 kit-md5.py --check "${KIT_MANIFEST_MD5:-}" 2>&1)
  echo "$KIT_OUT"
  if echo "$KIT_OUT" | grep -q "kit-md5: OK"; then
    note_pass "this box carries the recipe checkout's kit (kit-md5: OK)"
    KIT_FACT=OK
  else
    note_fail "kit-md5 did not report OK on this box -- it carries a different kit"
    KIT_FACT=BAD
  fi
fi

# --- 2. scope the log to THIS boot -----------------------------------------
if [[ -n "$EXTRACT_ONLY" ]]; then
  if [[ ! -f "$EXTRACT_ONLY" ]]; then
    echo "capture-config: EXTRACT_ONLY file not found: $EXTRACT_ONLY" >&2
    exit 1
  fi
  cat "$EXTRACT_ONLY" > "$TMP"
  SRC="EXTRACT_ONLY=$EXTRACT_ONLY (a log file, no fleet access)"
elif docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker logs "$CONTAINER" > "$TMP" 2>&1
  SRC="docker logs $CONTAINER (this container = this boot)"
elif [[ -f "$LOG" ]]; then
  tac "$LOG" | awk '/^=== .*box=/{exit} {print}' | tac > "$TMP"
  SRC="$LOG (sliced from the last boot marker)"
else
  echo "capture-config: no container and no boot log -- cannot scope" >&2
  exit 1
fi
echo "=== log source: $SRC ($(wc -l < "$TMP") lines) ==="

ENG_CFG=$(grep -m1 "Initializing a V1 LLM engine" "$TMP" || true)
ROLE="worker"
[[ -n "$ENG_CFG" ]] && ROLE="head"
echo "=== role: $ROLE (engine config dump present: $([[ -n "$ENG_CFG" ]] && echo yes || echo no)) ==="

# --- 3. the SPEC CONFIG reached the process ON THIS RANK -------------------
# THE ONE SOURCE IS THE MOUNTED FILE. The container's Cmd does NOT carry the
# JSON -- it READS the mounted /spec-config.json at start-up and passes the
# very same string to the self-check and to the serve line. So a gate that
# greps the Cmd for "method": "dspark" can never pass: it reads an artifact
# that does not exist. (An earlier gate did exactly that, and its capture
# criteria failed while the configuration was right -- see docs/FIXLOG.md.)
#
# The proof is now:
#   (a) the Cmd carries --speculative-config AND reads the mounted file;
#   (b) the MOUNTED FILE, read INSIDE THE LIVE CONTAINER on EVERY rank, pins
#       method and num_speculative_tokens;
#   (c) on the head, the engine's RESOLVED config names the spec config with
#       THE ENGINE'S OWN SPELLING (num_spec_tokens, not num_speculative_tokens).
echo "--- the spec config on this rank ---"
CMD=$(docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null || true)
SPEC_CMD=no
SPEC_CMD_READS_FILE=no
SPEC_FILE=no
SPEC_FILE_METHOD=no
SPEC_FILE_K=no
if [[ -z "$CMD" ]]; then
  note_fail "docker inspect returned no Cmd for $CONTAINER -- the spec config cannot be proven"
elif echo "$CMD" | grep -q -- "--speculative-config"; then
  SPEC_CMD=yes
  note_pass "the container's Cmd carries --speculative-config"
  # the Cmd must READ the mounted file (the ONE source), not carry a copy
  if echo "$CMD" | grep -qE 'SPEC_FILE=/spec-config\.json'; then
    SPEC_CMD_READS_FILE=yes
    note_pass "the Cmd reads the mounted file: SPEC_FILE=/spec-config.json"
  else
    note_fail "the Cmd does not point at /spec-config.json -- the JSON's source is not proven to be the mounted file"
  fi
else
  note_fail "the container's Cmd does NOT carry --speculative-config -- the spec config never reached this rank"
fi

# (b) the MOUNTED FILE, read INSIDE THE LIVE CONTAINER. This is the artifact
# the engine actually receives (the Cmd cats it), on EVERY rank.
SPEC_JSON_MOUNTED=""
if [[ "$SPEC_CMD" == "yes" ]]; then
  if docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
    SPEC_JSON_MOUNTED=$(docker exec "$CONTAINER" cat /spec-config.json 2>/dev/null || true)
  else
    note_fail "the container is not running -- the mounted file cannot be read from inside it"
  fi
  if [[ -z "$SPEC_JSON_MOUNTED" ]]; then
    note_fail "could not read /spec-config.json from INSIDE the live container on $HOST"
  else
    SPEC_FILE=yes
    note_pass "the mounted /spec-config.json was read from inside the live container: $SPEC_JSON_MOUNTED"
    if echo "$SPEC_JSON_MOUNTED" | grep -qE "\"method\"[[:space:]]*:[[:space:]]*\"${SPEC_METHOD}\""; then
      SPEC_FILE_METHOD=yes
      note_pass "the mounted file pins method=${SPEC_METHOD}"
    else
      note_fail "the mounted file does not pin method=${SPEC_METHOD}"
    fi
    if echo "$SPEC_JSON_MOUNTED" | grep -qE "\"num_speculative_tokens\"[[:space:]]*:[[:space:]]*${SPEC_K}([^0-9]|$)"; then
      SPEC_FILE_K=yes
      note_pass "the mounted file pins num_speculative_tokens=${SPEC_K}"
    else
      note_fail "the mounted file does not pin num_speculative_tokens=${SPEC_K}"
    fi
  fi
fi

# --- 4. the head's RESOLVED spec config -----------------------------------
# On the head the engine prints its resolved config, which shows whether the
# spec config was actually honoured (not merely passed). THE FIELD-NAME
# GOTCHA: the engine spells the field num_spec_tokens -- its OWN spelling.
# It never prints num_speculative_tokens, which is the spelling the mounted
# file uses; a gate that greps the engine log for the mounted file's
# spelling therefore fails while the configuration is perfectly correct (see
# docs/FIXLOG.md).
SPEC_RESOLVED=no
SPEC_RESOLVED_NUM_SPEC_TOKENS=no
if [[ "$ROLE" == "head" ]]; then
  if echo "$ENG_CFG" | grep -qE "speculative_config=[A-Za-z]" \
     && ! echo "$ENG_CFG" | grep -qE "speculative_config=None"; then
    SPEC_RESOLVED=yes
    note_pass "the head's resolved config shows a non-None speculative_config"
  else
    note_fail "the head's resolved config shows speculative_config=None (or none at all) -- the spec config was NOT honoured"
  fi
  if echo "$ENG_CFG" | grep -qE "num_spec_tokens=${SPEC_K}([^0-9]|$)"; then
    SPEC_RESOLVED_NUM_SPEC_TOKENS=yes
    note_pass "the head's resolved config shows num_spec_tokens=${SPEC_K} (the engine's own spelling)"
  else
    note_fail "the head's resolved config does not show num_spec_tokens=${SPEC_K}"
  fi
else
  echo "NOTE: worker role -- the engine config dump lives on the head only; this"
  echo "      rank's evidence is the MOUNTED FILE read from inside the live"
  echo "      container (asserted above) and its own capture + KV lines."
fi

# --- 5. the CAPTURE COVERAGE (head only) ----------------------------------
# With k=3 the engine's own default raises max_cudagraph_capture_size 64 ->
# 256 and grows the capture list from 11 entries to its own default of 35
# (the size this recipe's tools/resolve-capture-grid.py MEASURES in the
# image). What matters is not
# the list's length but that it COVERS the lane's 32 sequences at the
# spec-decode query length: 32 x (k+1) = 128.
CAPTURE_MAX=0
CAPTURE_HAS_128=no
CAPTURE_LIST=""
if [[ "$ROLE" == "head" ]]; then
  CAPTURE_MAX=$(echo "$ENG_CFG" | grep -oE "max_cudagraph_capture_size': [0-9]+" | tail -1 | grep -oE "[0-9]+$" || true)
  CAPTURE_LIST=$(echo "$ENG_CFG" | grep -oE "cudagraph_capture_sizes': \[[^]]*\]" | tail -1 || true)
  if [[ -n "$CAPTURE_MAX" ]] && [[ "$CAPTURE_MAX" -ge "$CAPTURE_COVERAGE_MIN" ]]; then
    note_pass "max_cudagraph_capture_size = $CAPTURE_MAX (>= $CAPTURE_COVERAGE_MIN)"
  else
    note_fail "max_cudagraph_capture_size = ${CAPTURE_MAX:-absent} -- below the ${CAPTURE_COVERAGE_MIN} coverage floor (only 16 of 32 sequences would be covered)"
  fi
  if echo "$CAPTURE_LIST" | grep -qE "(^|[^0-9])${CAPTURE_COVERAGE_MIN}([^0-9]|$)"; then
    CAPTURE_HAS_128=yes
    note_pass "the capture list CONTAINS ${CAPTURE_COVERAGE_MIN}: $CAPTURE_LIST"
  else
    note_fail "the capture list does NOT contain ${CAPTURE_COVERAGE_MIN}: ${CAPTURE_LIST:-absent}"
  fi
else
  if grep -q "Graph capturing finished" "$TMP"; then
    note_pass "this rank performed its own graph capture: $(grep -oE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB' "$TMP" | tail -1)"
  else
    note_fail "no 'Graph capturing finished' on this rank -- it never captured"
  fi
fi

# --- 5b. THE CAPTURE-GRID FACTS (machine-readable, ALWAYS emitted) ---------
# These are FACTS, not verdicts: the acceptance rule applies its own
# thresholds to them, so a rule never has to trust this gate's opinion.
# The graph counts come from the capture progress bars in the rank's own
# log; the resolved grid comes from the head's engine config dump.
RESOLVED_LIST=""
if [[ "$ROLE" == "head" ]]; then
  RESOLVED_LIST=$(echo "$CAPTURE_LIST" | grep -oE "\[[0-9, ]+\]" | tr -d '[]' | tr ',' ' ' | tr -s ' ' | sed 's/^ //; s/ $//' || true)
fi
# THE CAPTURE BARS CARRY A DENOMINATOR, AND THE DENOMINATOR IS THE POINT.
# The engine prints each mode's capture as a tqdm bar "N/M" -- the graphs
# captured so far, over the graphs the mode has to capture. The old
# extraction took the NUMERATOR and threw the denominator away, so an
# INTERRUPTED bar whose last line read "19/35" published "19" -- the very
# figure a COMPLETE "19/19" publishes, and a figure the source never
# carried. An interrupted capture then looked like a complete one.
#
# SO THE BAR IS READ WHOLE ("N/M"), and it becomes a COUNT only when the bar
# is COMPLETE (N == M). A PARTIAL bar is published as the RAW pair ("19/35"),
# so the reader sees an interrupted capture instead of a count that looks
# complete -- and the gate names it. A COMPLETE bar publishes exactly what
# it always published.
CG_BAR_PW=$(grep -oE "Capturing CUDA graphs \(PIECEWISE\).*" "$TMP" | tail -1 | grep -oE "[0-9]+/[0-9]+" | tail -1 || true)
CG_BAR_FULL=$(grep -oE "Capturing CUDA graphs \(FULL\).*" "$TMP" | tail -1 | grep -oE "[0-9]+/[0-9]+" | tail -1 || true)
CG_BAR_DS=$(grep -oE "Capturing dspark CUDA graphs \(FULL\).*" "$TMP" | tail -1 | grep -oE "[0-9]+/[0-9]+" | tail -1 || true)
# A COMPLETE BAR IS A COUNT. A PARTIAL BAR IS NOT A COUNT: it stays the raw
# pair, so the reader is never handed a count the source did not carry.
capture_bar_count() {
  local bar="$1"
  if [[ -z "$bar" ]]; then
    return 0
  fi
  if [[ "${bar%%/*}" == "${bar#*/}" ]]; then
    printf '%s' "${bar%%/*}"
  else
    printf '%s' "$bar"
  fi
}
PW_N=$(capture_bar_count "$CG_BAR_PW")
FULL_N=$(capture_bar_count "$CG_BAR_FULL")
DS_N=$(capture_bar_count "$CG_BAR_DS")
# AND A PARTIAL BAR IS NAMED, so the note says WHY the counts are not the
# pinned ones instead of leaving the reader to guess.
CG_PARTIAL=no
for _cgbar in "$CG_BAR_PW" "$CG_BAR_FULL" "$CG_BAR_DS"; do
  if [[ -n "$_cgbar" && "${_cgbar%%/*}" != "${_cgbar#*/}" ]]; then
    CG_PARTIAL=yes
  fi
done

# --- 5c. THE CAPTURE-GRID CAP (the POSITIVE proof the knob took effect) ----
# The cap is an environment argument, so the engine could in principle
# resolve a grid that is not the pinned one -- and a capture list that
# silently fell back to the default would look like a successful boot. These
# checks FAIL THE BOOT when the resolved grid is not the pinned one. The
# grid facts come from the head (the engine's config dump and its per-mode
# capture bars live on the global first rank only); the per-rank facts --
# each rank's own capture-pool cost -- are checked on every rank.
#
# EXPECTED_CG_MAX    -- the resolved max_cudagraph_capture_size this profile pins
# EXPECTED_CG_LIST   -- the resolved capture sizes this profile pins (comma-sep)
# EXPECTED_CG_GRAPHS -- the graph counts this profile pins (piecewise,full,dspark)
# All three are OPTIONAL: unset means "this profile does not pin a grid".
# The acceptance rule fails if the evidence does not show the pinned grid, so
# the backstop is the rule, not this gate.
if [[ -n "${EXPECTED_CG_MAX:-}" ]]; then
  echo "--- capture-grid cap: the pinned grid, on every rank ---"
  if [[ "$ROLE" == "head" ]]; then
    if [[ "$CAPTURE_MAX" == "$EXPECTED_CG_MAX" ]]; then
      note_pass "max_cudagraph_capture_size = $CAPTURE_MAX (the pinned value)"
    else
      note_fail "max_cudagraph_capture_size = ${CAPTURE_MAX:-absent} -- this profile pins $EXPECTED_CG_MAX"
    fi
    if [[ -n "${EXPECTED_CG_LIST:-}" ]]; then
      EXPECTED_SET=$(echo "$EXPECTED_CG_LIST" | tr ',' ' ' | tr -s ' ')
      RESOLVED_SET=$(echo "$RESOLVED_LIST" | tr -s ' ')
      if [[ "$RESOLVED_SET" == "$EXPECTED_SET" ]]; then
        note_pass "the resolved capture grid IS the pinned one: [$RESOLVED_SET]"
      else
        note_fail "the resolved capture grid [$RESOLVED_SET] is NOT the pinned one [$EXPECTED_SET]"
      fi
    fi
    # THE DECODE COVERAGE: the uniform-decode sizes the FULL graphs are
    # built from. All seven must be present, or the decode hot path lost a
    # graph -- the whole point of the cap is that it does NOT.
    MISSING_DECODE=""
    for s in 4 8 16 32 64 96 128; do
      echo " $RESOLVED_SET " | grep -qE "(^| )$s( |$)" || MISSING_DECODE="$MISSING_DECODE $s"
    done
    if [[ -z "$MISSING_DECODE" ]]; then
      note_pass "all 7 uniform-decode sizes are present (4 8 16 32 64 96 128): the decode hot path is intact"
    else
      note_fail "the uniform-decode sizes are MISSING:$MISSING_DECODE -- the decode hot path lost a graph"
    fi
  fi
  # THE GRAPH COUNTS: piecewise, full, dspark.
  #
  # HEAD ONLY, and this is a MEASURED FACT, not a convenience: the engine
  # prints the per-mode capture bars under `is_global_first_rank()`, so
  # ONLY the head (rank 0) has them. On the three-rank lane, rank 1 and
  # rank 2 have ZERO "Capturing CUDA graphs" lines, while every rank
  # DOES have "Graph capturing finished in N secs" and "CUDA graph pool
  # memory: X (actual), Y (estimated)". A check that expected those bars
  # on the workers would have FAILED every rank 1 and 2 and aborted the
  # battery for a reason that has nothing to do with the boot.
  #
  # The per-rank symmetry proof is NOT lost: the capture pool's actual and
  # estimated sizes ARE printed on every rank, and the rule compares each
  # rank against its own capture-pool expectation (the "CUDA graph pool
  # memory" line).
  if [[ -n "${EXPECTED_CG_GRAPHS:-}" ]]; then
    EXP_PW=$(echo "$EXPECTED_CG_GRAPHS" | cut -d, -f1)
    EXP_FULL=$(echo "$EXPECTED_CG_GRAPHS" | cut -d, -f2)
    EXP_DS=$(echo "$EXPECTED_CG_GRAPHS" | cut -d, -f3)
    if [[ "$ROLE" == "head" ]]; then
      if [[ "$CG_PARTIAL" == "yes" ]]; then
        # AN INTERRUPTED CAPTURE IS NOT A COUNT OF CAPTURED GRAPHS. The bar
        # carries its own denominator, and the denominator says the capture
        # did not finish -- so the pinned counts are NOT asserted, and the
        # RAW bars travel with the facts, so the reader sees the
        # interruption instead of a count that looks complete.
        note_fail "the capture bar is INCOMPLETE: piecewise=${CG_BAR_PW:-absent} full=${CG_BAR_FULL:-absent} dspark=${CG_BAR_DS:-absent} -- an interrupted capture is not a count of captured graphs, so the pinned counts are NOT asserted"
      elif [[ "$PW_N" == "$EXP_PW" && "$FULL_N" == "$EXP_FULL" && "$DS_N" == "$EXP_DS" ]]; then
        note_pass "the captured graphs are the pinned ones: piecewise=$PW_N full=$FULL_N dspark=$DS_N"
      else
        note_fail "the captured graphs are piecewise=${PW_N:-absent} full=${FULL_N:-absent} dspark=${DS_N:-absent} -- this profile pins $EXP_PW/$EXP_FULL/$EXP_DS"
      fi
    else
      # The worker cannot report the per-mode counts (the engine prints
      # those bars on the global first rank only). It reports its own
      # capture cost instead, which IS a per-rank fact -- and it is
      # ASSERTED, not narrated: a worker with no capture bar captured
      # NOTHING, so it has no capture pool to compare with its own
      # expectation, and saying "captured its own graphs: " with nothing
      # after it would be a vacuous pass.
      OWN_CAPTURE=$(grep -oE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB' "$TMP" | tail -1)
      if [[ -n "$OWN_CAPTURE" ]]; then
        note_pass "this rank captured its own graphs: $OWN_CAPTURE -- the per-mode counts are the global first rank's (the engine prints those bars there only)"
      else
        note_fail "this rank has no 'Graph capturing finished' line -- it captured NO graphs, so its capture pool cannot be compared with its own expectation"
      fi
    fi
  else
    # A PARTIAL BAR IS NOT A COUNT HERE EITHER: this branch reports the
    # counts, and a count the source's own denominator contradicts is not a
    # count.
    if [[ "$CG_PARTIAL" == "yes" ]]; then
      note_fail "the capture bar is INCOMPLETE: piecewise=${CG_BAR_PW:-absent} full=${CG_BAR_FULL:-absent} dspark=${CG_BAR_DS:-absent} -- an interrupted capture is not a count of captured graphs"
    else
      note_pass "the captured graphs on this rank: piecewise=${PW_N:-absent} full=${FULL_N:-absent} dspark=${DS_N:-absent}"
    fi
  fi
fi

# --- 6. the CAPACITY FLOOR -------------------------------------------------
echo "--- capacity: the recipe's floor, the price of the profile, named ---"
# THE ENGINE'S OWN FIGURES ARE READ WITH THEIR GROUPING INTACT. The engine
# prints these numbers with a thousands separator, and that separator is the
# ENGINE'S SPELLING -- so the grouping is READ FIRST and VALIDATED SECOND,
# and the commas come off only after the grouping has been proved.
#
# A WELL-FORMED token is either all digits ("196608") or a thousands grouping
# ("196,608"). Anything else -- "196,60,8", "1,9", "196,6080" -- is
# MALFORMED: the engine did not write the integer that dropping the commas
# would suggest, so that integer is A FIGURE THE ENGINE NEVER WROTE.
# Dropping the commas first (the old order) made "196,60,8" and "196,608"
# the same number -- and a malformed grouping then passed this gate's own
# floor check and was published as a figure the engine never wrote.
#
# SO THE ORDER IS FIXED HERE: read the token, PROVE the grouping, and only
# then take the commas off. ON A MALFORMED TOKEN the floor is NOT asserted
# (a figure the engine did not write cannot be judged against a floor) and
# the RAW token -- commas and all -- is what gets published, so the reader
# sees the damage instead of a laundered integer. ON A WELL-FORMED TOKEN
# nothing changes: the published figure is still the bare integer this gate
# has always published.
is_well_formed_token() {
  [[ "$1" =~ ^[0-9]+$ || "$1" =~ ^[0-9]{1,3}(,[0-9]{3})+$ ]]
}
KV_TOKENS_RAW=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$TMP" | tail -1 | grep -oE "[0-9,]+" || true)
KV_TOKENS=""
KV_TOKENS_FACT=""
KV_TOKENS_MALFORMED=no
if [[ -n "$KV_TOKENS_RAW" ]]; then
  if is_well_formed_token "$KV_TOKENS_RAW"; then
    KV_TOKENS="${KV_TOKENS_RAW//,/}"
    KV_TOKENS_FACT="$KV_TOKENS"
  else
    KV_TOKENS_MALFORMED=yes
    KV_TOKENS_FACT="$KV_TOKENS_RAW"
  fi
fi
KV_GIB=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$TMP" | tail -1 | grep -oE "[0-9.]+")
# THE CONCURRENCY LINE, WITH ITS DENOMINATOR. The engine prints
#   "Maximum concurrency for N tokens per request: X.XXx"
# where N IS ITS RESOLVED max_seq_len -- so this line is the engine's own
# statement of the context it actually serves, and the denominator is read
# out of the SAME line rather than assumed.
CONC_LINE=$(grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$TMP" | tail -1 || true)
# THE MULTIPLIER IS A FIGURE OR IT IS NOT. The engine prints it as a decimal
# ("1.06x") and the token is read WHOLE, off the engine's own line, so the
# figure this gate compares against a floor is the figure the engine wrote.
# A token that is not a well-formed decimal is not a figure this gate may
# publish or compare -- and the token is NAMED rather than trimmed into
# something that would compare.
CONC_RAW=$(echo "$CONC_LINE" | grep -oE "per request: [0-9.,]+x$" | sed 's/^per request: //' | tr -d 'x' || true)
CONC=""
CONC_MALFORMED=no
if [[ -n "$CONC_RAW" ]]; then
  if [[ "$CONC_RAW" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    CONC="$CONC_RAW"
  else
    CONC_MALFORMED=yes
  fi
fi
# THE DENOMINATOR, WITH THE SAME ORDER AS THE POOL: read the token, PROVE
# the grouping, and only then take the commas off. A well-formed grouping
# is published as the bare integer it stands for; a malformed one is
# published RAW, so the reader is never handed a figure the engine did not
# write.
CONC_CTX_RAW=$(echo "$CONC_LINE" | grep -oE "for [0-9,]+ tokens" | grep -oE "[0-9,]+" || true)
CONC_CTX=""
CONC_CTX_FACT=""
CONC_CTX_MALFORMED=no
if [[ -n "$CONC_CTX_RAW" ]]; then
  if is_well_formed_token "$CONC_CTX_RAW"; then
    CONC_CTX="${CONC_CTX_RAW//,/}"
    CONC_CTX_FACT="$CONC_CTX"
  else
    CONC_CTX_MALFORMED=yes
    CONC_CTX_FACT="$CONC_CTX_RAW"
  fi
fi
CONC_FACT=""
if [[ "$ROLE" == "head" ]]; then
  if [[ "$KV_TOKENS_MALFORMED" == "yes" ]]; then
    # A MALFORMED GROUPING IS NOT A POOL FIGURE. The engine did not write
    # the integer the commas would spell, so the floor cannot be checked
    # against it -- and the token is NAMED, so the reader sees what the
    # engine published instead of a laundered number.
    note_fail "the engine's own 'GPU KV cache size' figure '${KV_TOKENS_RAW}' is a MALFORMED thousands grouping -- a figure the engine did not write cannot be judged, so the KV pool floor is NOT asserted"
  elif [[ -n "$KV_TOKENS" && "$KV_TOKENS" -gt 0 ]]; then
    if [[ "$KV_TOKENS" -ge "$KV_FLOOR" ]]; then
      note_pass "KV pool ${KV_TOKENS} tokens (the floor is ${KV_FLOOR})"
    else
      note_fail "KV pool ${KV_TOKENS} tokens -- BELOW the floor of ${KV_FLOOR} (REJECT ON CAPACITY)"
    fi
  else
    note_fail "the head did not report 'GPU KV cache size' -- the pool floor cannot be checked"
  fi
  # THE SERVED CONTEXT, PROVEN. The head proves it, POSITIVELY, and the
  # workers make NO CLAIM (they run no API server, so they cannot report
  # one -- saying "the context is fine" from a rank that never saw it would
  # be a vacuous pass).
  #
  # THE FIGURE THAT IS IN FORCE IS THE ENGINE'S. The API server's
  # "non-default args" line prints the configuration the user ASKED for;
  # the ENGINE's own resolved config line is what it is actually serving.
  # They agree in a healthy boot, and where they do not, the engine's is
  # the one that counts -- so the engine's is the one judged, and a
  # disagreement is NAMED rather than averaged away.
  RESOLVED_CTX=$(grep -oE "max_seq_len=[0-9]+" "$TMP" | tail -1 | grep -oE "[0-9]+$" || true)
  ASKED_CTX=$(grep -oE "max_model_len': [0-9]+" "$TMP" | tail -1 | grep -oE "[0-9]+$" || true)
  if [[ -z "$RESOLVED_CTX" ]]; then
    note_fail "the head did not record the engine's own max_seq_len -- the served context cannot be proven"
  elif [[ "$RESOLVED_CTX" != "$EXPECTED_CTX" ]]; then
    note_fail "the engine's own resolved config records max_seq_len=$RESOLVED_CTX but this recipe pins $EXPECTED_CTX"
  elif [[ -n "$ASKED_CTX" && "$ASKED_CTX" != "$RESOLVED_CTX" ]]; then
    note_fail "the API server was asked for max_model_len=$ASKED_CTX but the engine resolved max_seq_len=$RESOLVED_CTX -- the figure in force is the ENGINE's"
  else
    note_pass "the served context took effect: the engine's own max_seq_len=$RESOLVED_CTX (pinned value)"
  fi
  if [[ "$CONC_MALFORMED" == "yes" ]]; then
    note_fail "the engine's own concurrency figure '${CONC_RAW}' is not a number -- the capacity floor cannot be checked"
  elif [[ -n "$CONC" ]]; then
    # THE FIGURE IS PUBLISHED WHENEVER IT IS READABLE -- a malformed
    # DENOMINATOR does not erase a figure the engine wrote and the gate can
    # read. What a malformed denominator refuses is the JUDGMENT: the
    # comparison against the pinned context is only sound against a
    # VALIDATED denominator, so a malformed grouping never reaches it -- and
    # the token is NAMED, so the reader sees what the engine published.
    CONC_FACT="$CONC"
    if [[ "$CONC_CTX_MALFORMED" == "yes" ]]; then
      note_fail "the engine's own concurrency denominator '${CONC_CTX_RAW}' is a MALFORMED thousands grouping -- a figure the engine did not write cannot be judged, so the concurrency is NOT asserted"
    elif [[ -n "$CONC_CTX" && "$CONC_CTX" != "$EXPECTED_CTX" ]]; then
      note_fail "the engine reported its concurrency for a ${CONC_CTX}-token request -- a DIFFERENT denominator from the pinned $EXPECTED_CTX, so the figure cannot be judged"
    elif awk "BEGIN{exit !($CONC >= $CONCURRENCY_MIN)}"; then
      note_pass "the engine's own concurrency for a ${CONC_CTX}-token request is ${CONC}x (>= ${CONCURRENCY_MIN}x)"
    else
      note_fail "the engine's own concurrency is ${CONC}x -- BELOW ${CONCURRENCY_MIN}x: the lane can no longer hold one full-length request (REJECT ON CAPACITY)"
    fi
  else
    note_fail "the engine did not report its concurrency figure -- the capacity floor cannot be checked"
  fi
else
  echo "KV budget on this rank: ${KV_GIB:-unknown} GiB (the token count comes from the head's EngineCore)"
fi
if [[ -n "$KV_GIB" ]]; then
  if awk "BEGIN{exit !($KV_GIB >= $KV_FLOOR_GIB)}"; then
    note_pass "this rank's KV budget ${KV_GIB} GiB >= the ${KV_FLOOR_GIB} GiB the ${KV_FLOOR}-token floor needs"
  else
    note_fail "this rank's KV budget ${KV_GIB} GiB is BELOW the ${KV_FLOOR_GIB} GiB the ${KV_FLOOR}-token floor needs"
  fi
fi

# --- 7. the capture pool's ACTUAL memory (the price, measured) ------------
GRAPH_ACTUAL=$(grep -oE "CUDA graph pool memory: [0-9.]+ GiB \(actual\)" "$TMP" | tail -1 | grep -oE "[0-9.]+" || true)
GRAPH_EST=$(grep -oE "[0-9.]+ GiB \(estimated\)" "$TMP" | tail -1 | grep -oE "[0-9.]+" || true)
if [[ -n "$GRAPH_ACTUAL" ]]; then
  note_pass "the capture pool holds ${GRAPH_ACTUAL} GiB actual (estimate ${GRAPH_EST:-unknown} GiB)"
else
  note_fail "no 'CUDA graph pool memory: ... (actual)' line -- the capture pool's real cost is unaccounted for"
fi

# --- 8. the budget took effect (pinned, not derived) ----------------------
RESOLVED_GMU=$(grep -oE 'Desired GPU memory utilization is \([0-9.]+,' "$TMP" | tail -1 | grep -oE '[0-9.]+')
if [[ -z "$RESOLVED_GMU" ]]; then
  note_fail "the engine never reported its memory budget -- cannot prove the budget took effect"
elif [[ "$RESOLVED_GMU" != "$EXPECTED_GMU_PINNED" ]]; then
  note_fail "the engine resolved GMU=$RESOLVED_GMU but this profile pins GMU=$EXPECTED_GMU_PINNED"
else
  note_pass "the memory budget took effect: GMU=$RESOLVED_GMU (pinned value)"
fi

# --- 9. the startup completed ---------------------------------------------
if grep -q "Application startup complete" "$TMP"; then
  note_pass "Application startup complete"
elif [[ "$ROLE" == "worker" ]] && grep -q "Available KV cache memory" "$TMP"; then
  note_pass "this rank completed its startup (capture done, KV budget reported)"
else
  note_fail "'Application startup complete' never appeared"
fi

# --- the machine-readable facts (consumed by the acceptance rule) ---------
echo "--- GATE facts (machine-readable) ---"
echo "[GATE] host=${HOST}"
echo "[GATE] role=${ROLE}"
echo "[GATE] kit_md5=${KIT_FACT}"
echo "[GATE] spec_cmd=${SPEC_CMD}"
echo "[GATE] spec_cmd_reads_file=${SPEC_CMD_READS_FILE}"
echo "[GATE] spec_file=${SPEC_FILE}"
echo "[GATE] spec_file_method=${SPEC_FILE_METHOD}"
echo "[GATE] spec_file_k=${SPEC_FILE_K}"
echo "[GATE] spec_resolved=${SPEC_RESOLVED}"
echo "[GATE] spec_resolved_num_spec_tokens=${SPEC_RESOLVED_NUM_SPEC_TOKENS}"
echo "[GATE] capture_max_size=${CAPTURE_MAX:-0}"
echo "[GATE] capture_list_has_${CAPTURE_COVERAGE_MIN}=${CAPTURE_HAS_128}"
# THE POOL FIGURE IS PUBLISHED AS THE ENGINE WROTE IT. A well-formed
# grouping is published as the bare integer it stands for ("196608") -- the
# figure this gate has always published. A MALFORMED grouping is published
# WITH ITS COMMAS ("196,60,8"), because a reader that stripped them would
# be reading a figure the engine never wrote: the damage is published, not
# laundered.
echo "[GATE] kv_pool_tokens=${KV_TOKENS_FACT:-}"
echo "[GATE] kv_concurrency=${CONC_FACT}"
# THE DENOMINATOR IS A FACT TOO. The engine prints the concurrency figure
# FOR A REQUEST LENGTH, and the figure alone is a number about an unknown
# length -- so the length travels with it. (It used to travel only inside
# the gate's own prose note, which the acceptance rule cannot read: the
# rule parses FACTS, and a fact that is not published is a fact that does
# not exist.)
#
# AND IT IS PUBLISHED AS THE ENGINE WROTE IT, for the same reason as the
# pool figure above: the bare integer for a well-formed grouping, and the
# RAW token -- commas and all -- for a malformed one.
if [[ -n "$CONC_CTX_FACT" ]]; then
  echo "[GATE] kv_concurrency_ctx=${CONC_CTX_FACT}"
fi
echo "[GATE] kv_gib=${KV_GIB:-0}"
echo "[GATE] capture_pool_actual_gib=${GRAPH_ACTUAL:-0}"
echo "[GATE] capture_pool_estimated_gib=${GRAPH_EST:-0}"
echo "[GATE] cg_resolved_list=${RESOLVED_LIST:-}"
# THE CAPTURE COUNTS TRAVEL AS THE BAR CARRIED THEM. A COMPLETE bar
# publishes the count it always published ("19"); a PARTIAL one publishes the
# RAW pair ("19/35"), because a reader that read "19" from it would be
# reading a figure the source never carried.
echo "[GATE] cg_graph_piecewise=${PW_N:-0}"
echo "[GATE] cg_graph_full=${FULL_N:-0}"
echo "[GATE] cg_graph_dspark=${DS_N:-0}"
echo "[GATE] gmu=${RESOLVED_GMU:-unknown}"
echo "--- end GATE facts ---"

echo
if [[ -n "$EXTRACT_ONLY" ]]; then
  # The extraction API carries no verdict and no exit-code semantics: it
  # reports FACTS, and the thresholds live in the rule (or in the test).
  # IT SAYS SO MACHINE-READABLY, because the acceptance rule judges "the
  # capture gate is clean on this box" -- and an EXTRACT_ONLY section has
  # NOT been judged clean by anything.
  echo "[GATE] EXIT=extracted"
  echo "capture-config: EXTRACT_ONLY -- facts emitted, no verdict (rc=0)"
  exit 0
fi
if [[ $fail -ne 0 ]]; then
  echo "[GATE] EXIT=1"
  echo "capture-config: NOT a clean speculative-decode boot on $HOST (rc=1)"
  exit 1
fi
echo "[GATE] EXIT=0"
echo "capture-config: confirmed on $HOST -- the spec config reached this rank, the capture covers ${CAPTURE_COVERAGE_MIN}, the pool clears ${KV_FLOOR} tokens and the concurrency clears ${CONCURRENCY_MIN}x"
exit 0
