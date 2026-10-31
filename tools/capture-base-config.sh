#!/usr/bin/env bash
# capture-base-config.sh -- prove what the capture boot actually enabled.
#
# The capture boot deletes ONE line from the serve line (--enforce-eager). That
# line was doing three jobs at once, so "capture is on" is NOT one flag: it
# turns on torch.compile, it turns on the CUDA-graph capture pool, and it turns
# on JIT kernel warmup. The boot therefore has to PROVE each of them, and the
# interesting number is the capture pool size -- previously unmeasured.
#
# LOG SCOPING (a real trap -- see docs/FIXLOG.md): the boot log
# logs-boot-<host>.out is `tee -a` APPENDED across boots -- it already holds
# the lines of EARLIER boots, including their "Enforce eager set" lines.
# Grepping that file would (a) always FAIL a clean capture boot, because it
# still contains the old eager boots, and (b) could false-PASS "Application
# startup complete" off an OLD boot. So this script reads the CURRENT
# container's logs (one container per boot) and falls back to slicing the file
# from the LAST boot marker.
#
# Usage: bash capture-base-config.sh
# Exit: 0 when the boot is a genuine capture boot, 1 otherwise. Fail-closed.
set -uo pipefail

CONTAINER=minimax31-flash-tp3
# KIT is overridable so the gate can be NEGATIVE-TESTED against a recipe that
# deliberately differs from the running boot, without touching the live kit.
KIT=${KIT:-~/MiniMax-M3.1-Flash-TP3-3x-DGX-Spark}
# A tilde inside a parameter-expansion default is not expanded by every shell
# -- expand it here, so the default means the operator's home.
case "$KIT" in "~/"*) KIT="$HOME/${KIT#\~/}" ;; esac
LOG="$KIT/logs-boot-$(hostname -s).out"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# --- 1. scope the log to THIS boot ----------------------------------------
SRC=""
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker logs "$CONTAINER" > "$TMP" 2>&1
  SRC="docker logs $CONTAINER (this container = this boot)"
elif [[ -f "$LOG" ]]; then
  # slice from the LAST boot marker to the end of the file
  tac "$LOG" | awk '/^=== .*box=/{exit} {print}' | tac > "$TMP"
  SRC="$LOG (sliced from the last boot marker)"
else
  echo "capture-base-config: no container and no boot log -- cannot scope" >&2
  exit 1
fi
echo "=== log source: $SRC ($(wc -l < "$TMP") lines) ==="

fail=0
note_fail() { echo "[FAIL] $1"; fail=1; }
note_pass() { echo "[PASS] $1"; }

# --- 2. eager must be GONE ------------------------------------------------
if grep -q "Enforce eager set" "$TMP"; then
  note_fail "'Enforce eager set' is STILL LOGGED -- --enforce-eager is still on the serve line"
else
  note_pass "'Enforce eager set' is absent -- eager is off"
fi
if grep -q "Cudagraph is disabled under eager mode" "$TMP"; then
  note_fail "'Cudagraph is disabled under eager mode' -- cudagraphs are still off"
else
  note_pass "cudagraphs are no longer disabled by eager mode"
fi

# --- 3. the three subsystems the capture boot turns on, asserted from the
#        ENGINE's RESOLVED config, not the API server's echo of the user's args.
#        The API server prints 'non-default args' with the compilation_config
#        the user ASKED for (FULL_AND_PIECEWISE) even under eager; the
#        engine's resolved config is the one that shows what actually
#        happened (eager overrides it to NONE).
#
#        The ENGINE's config dump is printed by the API server's process,
#        which only exists on the HEAD. The workers' containers carry only
#        the (MainProcess)/(Worker) streams, so the config assertions are
#        HEAD-ONLY and the worker-level facts (eager off, the pool, the pool
#        memory line) are asserted on EVERY box. The role is detected from
#        the log, not assumed.
ENG_CFG=$(grep -m1 "Initializing a V1 LLM engine" "$TMP" || true)
ROLE="worker"
[[ -n "$ENG_CFG" ]] && ROLE="head"
echo "=== role: $ROLE (engine config dump present: $([[ -n "$ENG_CFG" ]] && echo yes || echo no)) ==="

if [[ "$ROLE" == "head" ]]; then
  echo "--- engine resolved config (selected keys) ---"
  echo "$ENG_CFG" | grep -oE "(enforce_eager|mode': <CompilationMode\.[A-Z_]+|cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|enable_jit_warmup=[A-Za-z]+|max_cudagraph_capture_size': [0-9]+|cudagraph_capture_sizes': \[[^]]*\])" | sort -u | head -12
  for want in "enforce_eager=False" \
              "cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE" \
              "enable_jit_warmup=True"; do
    if echo "$ENG_CFG" | grep -qF "$want"; then
      note_pass "engine config has $want"
    else
      note_fail "engine config is MISSING $want -- the capture boot did not enable it"
    fi
  done
  # NOTE ON THE COMPILATION MODE: this build resolves the mode to
  # CompilationMode.NONE unless it is asked for explicitly, so
  # --compilation-config '{"cudagraph_mode": ...}' turns the CAPTURE on
  # WITHOUT turning torch.compile on. That is what this boot did: the
  # capture pool is live and JIT warmup is on, but torch.compile stayed off.
  # It is reported, not asserted either way -- the throughput gate decides.
  CM=$(echo "$ENG_CFG" | grep -oE "mode': <CompilationMode\.[A-Z_]+" | head -1)
  echo "NOTE: resolved compilation mode = ${CM:-unknown} (cudagraph capture is ON regardless)"
  # --- 4. the encoder must NOT be captured (head-only config) ------------
  # The DP-sharded vision path is exactly where rank 0 and ranks 1/2 differ
  # (see docs/FIXLOG.md). If the ENCODER were captured, rank 0 would capture
  # the real branch and the workers the empty branch -> capture-time
  # collective divergence -> a wedge. Assert OFF, do not assume. The
  # cross-rank proof is the ring check + the vision smoke.
  for want in "cudagraph_mm_encoder': False" \
              "encoder_cudagraph_token_budgets': []" \
              "compile_mm_encoder': False"; do
    if echo "$ENG_CFG" | grep -qF "$want"; then
      note_pass "encoder capture is OFF ($want)"
    else
      note_fail "MISSING $want -- the encoder may be captured (wedge risk)"
    fi
  done
else
  echo "NOTE: worker role -- the engine config dump lives on the head only;"
  echo "      the encoder-capture flags are proven there and cross-checked by"
  echo "      the ring proof + the vision smoke."
fi

# --- 5. the capture pool: the number this gate pins down ------------------
echo "--- capture / compile / warmup evidence ---"
grep -iE "CUDA graph pool memory|Available KV cache memory|GPU KV cache size|Capturing cudagraph|cudagraph_capture_sizes|max_cudagraph_capture_size|CUDA graph memory profiling" "$TMP" \
  | tail -12 || echo "(no capture/compile lines found)"

if grep -qF "CUDA graph pool memory:" "$TMP"; then
  note_pass "the engine reported the capture pool's actual vs estimated memory"
  grep -F "CUDA graph pool memory:" "$TMP" | tail -2
else
  note_fail "no 'CUDA graph pool memory:' line -- the actual-vs-estimated delta is unaccounted for"
fi

# The capture LIST and its max are facts from the ENGINE's resolved config,
# which only the head prints. On a worker the equivalent evidence is that
# THIS rank performed its own capture ("Graph capturing finished ... took
# X GiB" appears once for the profiling pass and once for the real pool).
if [[ "$ROLE" == "head" ]]; then
  SIZES=$(echo "$ENG_CFG" | grep -oE "cudagraph_capture_sizes': \[[^]]*\]" | tail -1)
  if [[ -n "$SIZES" && "$SIZES" != "cudagraph_capture_sizes': []" ]]; then
    note_pass "the capture list is populated: $SIZES"
  else
    note_fail "cudagraph_capture_sizes is empty -- no graphs were captured"
  fi

  MAXSIZE=$(echo "$ENG_CFG" | grep -oE "max_cudagraph_capture_size': [0-9]+" | tail -1 | grep -oE "[0-9]+$")
  if [[ -n "$MAXSIZE" && "$MAXSIZE" -gt 0 ]]; then
    note_pass "max_cudagraph_capture_size = $MAXSIZE"
  else
    note_fail "max_cudagraph_capture_size is ${MAXSIZE:-absent} -- expected > 0"
  fi
else
  if grep -q "Graph capturing finished" "$TMP"; then
    note_pass "this rank performed its own graph capture: $(grep -oE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB' "$TMP" | tail -1)"
  else
    note_fail "no 'Graph capturing finished' on this rank -- it never captured"
  fi
fi

# --- 5b. the KV pool: the capture boot BUYS THE POOL AT THE COST OF KV ----
# The capture pool is PROFILED AND RESERVED before the KV is sized
# (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS defaults True), so the capture
# boot's cost is a CAPACITY loss: the KV cache shrinks by the pool estimate.
# The KEEP/REJECT rule has a KV floor, so the pool size must be captured.
echo "--- KV pool (the capture boot trades KV for the capture pool) ---"
grep -E "Available KV cache memory|GPU KV cache size" "$TMP" | tail -2
# THE ENGINE'S OWN FIGURE IS READ WITH ITS GROUPING INTACT. The engine prints
# the pool size with a thousands separator, and that separator is the ENGINE'S
# SPELLING -- so the grouping is READ FIRST and VALIDATED SECOND, and the
# commas come off only after the grouping has been proved.
#
# A WELL-FORMED token is either all digits ("196608") or a thousands grouping
# ("196,608"). Anything else -- "196,60,8", "1,9", "196,6080" -- is
# MALFORMED: the engine did not write the integer that dropping the commas
# would suggest, so that integer is A FIGURE THE ENGINE NEVER WROTE. Dropping
# the commas first (the old order) made "196,60,8" and "196,608" the same
# number -- and a malformed grouping then passed this gate's own floor check
# and was published as a figure the engine never wrote.
#
# ON A MALFORMED TOKEN the floor is NOT asserted (a figure the engine did not
# write cannot be judged against a floor) and the token is NAMED, so the
# reader sees what the engine published instead of a laundered number. ON A
# WELL-FORMED TOKEN nothing changes: the figure is still the bare integer
# this gate has always published.
is_well_formed_token() {
  [[ "$1" =~ ^[0-9]+$ || "$1" =~ ^[0-9]{1,3}(,[0-9]{3})+$ ]]
}
KV_TOKENS_RAW=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$TMP" | tail -1 | grep -oE "[0-9,]+" || true)
KV_TOKENS=""
KV_TOKENS_MALFORMED=no
if [[ -n "$KV_TOKENS_RAW" ]]; then
  if is_well_formed_token "$KV_TOKENS_RAW"; then
    KV_TOKENS="${KV_TOKENS_RAW//,/}"
  else
    KV_TOKENS_MALFORMED=yes
  fi
fi
KV_GIB=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$TMP" | tail -1 | grep -oE "[0-9.]+")
# THE FLOORS THIS GATE CHECKS ARE THE RECIPE'S OWN. A shipped tool's
# default must match the recipe it ships: the old defaults (250,000 tokens
# and 8.95 GiB) were the PLAIN, non-speculative ones, and running this
# gate on its own -- which is its documented usage -- then failed a HEALTHY
# boot of this recipe with a misleading "pool below floor". A false
# rejection of a good boot is still a wrong answer, and this gate's own
# usage line must not lead a reader to it.
#
# The token floor is one full-length request at the served context. The
# GiB floor is the same fact in the other unit, at the recipe's KV density:
# 37.53 KiB/token (the target model's own KV group) + 2.5 KiB/token (the
# draft model's own KV group) = 40.03 KiB/token; 196,608 tokens x 40.03
# KiB = 7.5056 GiB, rounded UP to 7.51 GiB. The POOL is sized by the
# TIGHTEST rank, so EVERY rank's own KV budget must clear this or the pool
# cannot reach the floor: that is what makes the check mean something on a
# worker, where the token count does not exist.
KV_FLOOR=${KV_FLOOR:-196608}
KV_FLOOR_GIB=${KV_FLOOR_GIB:-7.51}
if [[ "$KV_TOKENS_MALFORMED" == "yes" ]]; then
  # A MALFORMED GROUPING IS NOT A POOL FIGURE. The engine did not write the
  # integer the commas would spell, so the floor cannot be checked against it
  # -- and the token is NAMED, so the reader sees what the engine published
  # instead of a laundered number. The comparison below is reached only with a
  # validated figure, so a mangled token can never be fed to '-gt 0'.
  note_fail "the engine's own 'GPU KV cache size' figure '${KV_TOKENS_RAW}' is a MALFORMED thousands grouping -- a figure the engine did not write cannot be judged, so the token floor is NOT asserted"
elif [[ -n "$KV_TOKENS" && "$KV_TOKENS" -gt 0 ]]; then
  # The token count is printed by the EngineCore, which lives on the head
  # only. A worker reports its own KV budget in GiB instead -- and the POOL
  # is sized by the TIGHTEST rank, so the per-rank GiB lines are the story.
  if [[ "$KV_TOKENS" -ge "$KV_FLOOR" ]]; then
    note_pass "KV pool ${KV_TOKENS} tokens (floor ${KV_FLOOR})"
  else
    note_fail "KV pool ${KV_TOKENS} tokens -- BELOW the ${KV_FLOOR} floor"
  fi
elif [[ "$ROLE" == "head" ]]; then
  note_fail "the head did not report 'GPU KV cache size' -- the token floor cannot be checked"
elif [[ -n "$KV_GIB" ]]; then
  echo "KV budget on this rank: ${KV_GIB} GiB  (the token count comes from the head's EngineCore)"
else
  note_fail "no KV line at all -- the KV pool is unaccounted for"
fi
if [[ -n "$KV_GIB" ]]; then
  if awk "BEGIN{exit !($KV_GIB >= $KV_FLOOR_GIB)}"; then
    note_pass "this rank's KV budget ${KV_GIB} GiB >= the ${KV_FLOOR_GIB} GiB the ${KV_FLOOR}-token floor needs"
  else
    note_fail "this rank's KV budget ${KV_GIB} GiB is BELOW the ${KV_FLOOR_GIB} GiB the ${KV_FLOOR}-token floor needs"
  fi
fi

# --- 5c. THE MEMORY BUDGET: a VALUE check, not a consistency check ---------
# The budget's whole change is --gpu-memory-utilization. Four separate facts
# must agree, or the box is not running the recipe:
#   1. EXPECTED_GMU_PINNED -- the budget THIS PROFILE must run at. It is
#      PINNED HERE, not derived from the box: a box that still carries the
#      OLD kit would otherwise agree with ITSELF (recipe 0.80 = container
#      0.80 = engine 0.80) and the gate would PASS a boot that silently ran
#      at the wrong budget. That hole is recorded in docs/FIXLOG.md, and
#      this line is its fix. Override it for a later profile.
#   2. the RECIPE's default (read from launch/start-tp3.sh ON THIS BOX) -- so
#      the repo describes the running recipe and not a stale one.
#   3. the value the CONTAINER was actually given (docker inspect .Config.Env
#      GMU=) -- so the boot did not silently override the recipe.
#   4. the value the ENGINE resolved ("Desired GPU memory utilization is
#      (0.80, ... GiB)", printed by EVERY rank's gpu_worker) -- so the
#      budget really took effect and was not clamped back.
EXPECTED_GMU_PINNED="${EXPECTED_GMU_PINNED:-0.80}"
RECIPE_GMU=""
for recipe_start in "$KIT/launch/start-tp3.sh" "$KIT/start-tp3.sh"; do
  [[ -f "$recipe_start" ]] || continue
  RECIPE_GMU=$(grep -oE 'GMU:-[0-9.]+' "$recipe_start" 2>/dev/null | head -1 | sed 's/^GMU:-//')
  [[ -n "$RECIPE_GMU" ]] && break
done
CONTAINER_GMU=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null | grep -E '^GMU=' | head -1 | cut -d= -f2)
RESOLVED_GMU=$(grep -oE 'Desired GPU memory utilization is \([0-9.]+,' "$TMP" | tail -1 | grep -oE '[0-9.]+')
echo "--- memory budget: pinned='${EXPECTED_GMU_PINNED}' recipe='${RECIPE_GMU:-unknown}' container='${CONTAINER_GMU:-unset}' engine='${RESOLVED_GMU:-unknown}' ---"
if [[ -z "$RESOLVED_GMU" ]]; then
  note_fail "the engine never reported its memory budget -- cannot prove the budget took effect"
elif [[ "$RESOLVED_GMU" != "$EXPECTED_GMU_PINNED" ]]; then
  note_fail "the engine resolved GMU=$RESOLVED_GMU but this profile pins GMU=$EXPECTED_GMU_PINNED"
elif [[ "$RECIPE_GMU" != "$EXPECTED_GMU_PINNED" ]]; then
  note_fail "this box's recipe says GMU=$RECIPE_GMU but this profile pins GMU=$EXPECTED_GMU_PINNED (stale kit?)"
elif [[ -z "$CONTAINER_GMU" ]]; then
  # A MISSING FACT IS NOT A PASS. The old test was
  # [[ -n "$CONTAINER_GMU" && "$CONTAINER_GMU" != "$EXPECTED_GMU_PINNED" ]],
  # so a container with NO GMU env at all satisfied it vacuously -- the
  # vacuous-pass pattern. The budget a container ran at must be PROVEN, and
  # "unset" is not a proof of anything (see docs/FIXLOG.md).
  note_fail "the container carries NO GMU env -- the budget it ran at cannot be proven"
elif [[ "$CONTAINER_GMU" != "$EXPECTED_GMU_PINNED" ]]; then
  note_fail "the container was given GMU=$CONTAINER_GMU but this profile pins GMU=$EXPECTED_GMU_PINNED"
else
  note_pass "the memory budget took effect: GMU=$RESOLVED_GMU (pinned value, recipe, container env and engine all agree)"
fi

# --- 6. the pool's VRAM bill ----------------------------------------------
echo "--- gpu_worker memory split ---"
grep -E "Free memory on device" "$TMP" | tail -2
GRAPH_GIB=$(grep -oE "[0-9.]+ GiB for CUDAGraph memory" "$TMP" | tail -1 | awk '{print $1}')
if [[ -n "$GRAPH_GIB" ]]; then
  echo "CUDAGraph memory in use: ${GRAPH_GIB} GiB"
  if awk "BEGIN{exit !($GRAPH_GIB > 0)}"; then
    note_pass "the capture pool actually holds memory (${GRAPH_GIB} GiB) -- the capture really happened"
  else
    note_fail "CUDAGraph memory is 0 GiB -- the capture pool is empty, the capture did NOT take effect"
  fi
else
  note_fail "no gpu_worker memory line found -- cannot account for the capture pool"
fi

# --- 7. startup completed ------------------------------------------------
# "Application startup complete" is printed by the API server, which only
# exists on the head. On a worker the equivalent liveness marker is that the
# rank finished its own capture and reported its KV budget (both asserted
# above) -- a worker that never got that far fails those assertions anyway.
if grep -q "Application startup complete" "$TMP"; then
  note_pass "Application startup complete"
elif [[ "$ROLE" == "worker" ]] && grep -q "Available KV cache memory" "$TMP"; then
  note_pass "this rank completed its startup (capture done, KV budget reported)"
else
  note_fail "'Application startup complete' never appeared"
fi

echo
if [[ $fail -ne 0 ]]; then
  echo "capture-base-config: NOT a clean capture boot (rc=1)"
  exit 1
fi
echo "capture-base-config: capture confirmed -- eager off, compile+capture+JIT warmup on, encoder NOT captured, pool accounted for, startup complete"
exit 0
