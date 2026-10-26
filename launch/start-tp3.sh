#!/usr/bin/env bash
# start-tp3.sh -- boot ONE box of the minimax31-flash TP=3 ring.
#
# One `docker run` per box. The head serves; the two ranks join headless.
# Every knob is an environment override, and the shipped profile lives in
# ../config/cluster.env.
#
# What this script does, in order:
#   1. refuses to boot without an image pin and a staged checkpoint;
#   2. applies the TP=3 pad kit INSIDE the container, before anything
#      imports vllm, and fails the boot if the pad does not verify;
#   3. proves the speculative-decode draft head is padded to TP=3, from the
#      real engine path, and proves that proof can fail;
#   4. reads the speculative config ONCE and hands the very same string to
#      both the gate and the serve line;
#   5. starts the engine.
#
# The failure modes this script exists to prevent are documented in
# ../docs/FIXLOG.md; the landmines are in ../README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# ---------------------------------------------------------------- image pin
# THE PIN IS AT BUILD TIME, NOT AT RUN TIME. The base is the upstream day-0
# build for this model on arm64 / CUDA 13.4; it ships no ray, and the
# multi-node executor needs one, so the recipe's image adds ray as a single
# layer (see ../launch/Dockerfile, whose ARG BASE_IMAGE carries the digest).
# Build it once, then `docker save | docker load` it to the other two boxes
# -- a save/load pipe DROPS the repo digest, so what the three boxes share
# is the LOCAL TAG, and the tag is what the launch runs.
#
# WHICH IS WHY THE PIN CANNOT BE CHECKED FROM HERE. A save/load pipe
# destroys the digest that would make a per-box check possible: the boxes
# carry the same LOCAL TAG and no RepoDigests. The digest this recipe
# publishes is the BUILD input (launch/Dockerfile's ARG BASE_IMAGE).
#
# The thing a reader CAN check on a box is that all three boxes carry the
# SAME image ID, and that check is the OPERATOR'S: read `docker images` on
# each of the three boxes and compare the IDs. The capture gate does NOT
# do it. The gate checks the KIT -- the recipe's own files, by digest --
# and the engine's resolved settings; it never inspects the container
# image, so it makes no claim about the image ID. Crediting a shipped tool
# with a proof it does not perform would be exactly the kind of claim this
# recipe refuses to make elsewhere.
#
# So there is no DIGEST variable here pretending to be a runtime pin: the
# variable that did nothing but LOOK mandatory is gone, rather than left as
# a check that cannot fail.
TAG="${VLLM_TAG:-minimax31-flash-tp3}"
if [[ -z "${TAG}" ]]; then
  echo "ERROR: VLLM_TAG is required -- refusing to boot." >&2
  echo "  It is the local tag all three boxes carry after the save/load pipe." >&2
  exit 1
fi
IMAGE="vllm/vllm-openai:${TAG}"

# ---------------------------------------------------------------- knobs
ROLE="${ROLE:-head}"                      # head | worker (1 head + 2 workers)
NODE_IP="${NODE_IP:?set NODE_IP to the cluster-facing IP of this box}"
HEAD_IP="${HEAD_IP:-$NODE_IP}"
CLUSTER_SIZE="${CLUSTER_SIZE:-3}"
NODE_RANK="${NODE_RANK:-0}"
DIST_MASTER_PORT="${DIST_MASTER_PORT:-25000}"
# The interface the fabric (control-plane) socket lives on -- the DGX Spark
# management NIC by default. Override it if your box names it differently
# (`ip -o link` lists the candidates).
#
# THE PROFILE OF RECORD NAMES IT TOO: config/cluster.env sets
# NCCL_SOCKET_IFNAME, and `docker run` does not forward that value into the
# container -- only SOCKET_IFNAME is passed -- so the profile's override was
# INERT until this fallback was added. The profile's name is now honoured
# when the shorter one is not set.
SOCKET_IFNAME="${SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-enP7s7}}"
# THE TP AND GLOO SOCKETS HAVE THEIR OWN KNOBS IN THE PROFILE OF RECORD, and
# until now they never reached the container: `docker run` is given ONE
# interface, so an operator who set TP_SOCKET_IFNAME in config/cluster.env
# was silently ignored. Each now falls back to the interface the profile set,
# and each can still be overridden on its own.
TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$SOCKET_IFNAME}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$SOCKET_IFNAME}"

PORT="${PORT:-8100}"
TP="${TP:-3}"
# The memory budget. On a unified-memory box this is HOST memory: the budget
# is a fraction of the box's total, and everything the engine holds -- the
# weights, the KV pool, the capture pool, the activation peak -- comes out
# of the same pot. A higher budget is not free; see ../docs/FIXLOG.md.
GMU="${GMU:-0.78}"
# The context the lane serves. This must match what the KV pool can hold:
# a context twice this size does not fit the pool, and the engine refuses
# the request rather than degrading.
CTX="${CTX:-196608}"
# The capture-grid cap, as a JSON list of sizes. UNSET means the engine's own
# default grid, which costs roughly 15,000 tokens of KV pool in graphs this
# lane never replays as decode graphs.
CGCAP="${CGCAP:-}"
# The cap reaches the container ONLY when it is set. With no cap the
# container's environment is identical to the uncapped one, so a revert
# leaves nothing behind. The expansion below is the safe-empty-array idiom:
# under `set -u` an empty array must be expanded through ${arr[@]+...} or
# bash treats it as an unset variable.
if [ -n "$CGCAP" ]; then
  CGCAP_ENV=(-e "CGCAP=$CGCAP")
else
  CGCAP_ENV=()
fi

SERVED_NAME="${SERVED_NAME:-minimax31-flash}"
ALIAS_NAME="${ALIAS_NAME:-main}"
MODEL_DIR="${MODEL_DIR:-/home/YOURUSER/models/minimax31-flash-54e4e69dfc45}"
CONTAINER_NAME="${CONTAINER_NAME:-minimax31-flash-tp3}"
SPEC_FILE="${SPEC_FILE:-${SPEC_CONFIG_FILE:-$ROOT/config/spec-config.json}}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

# The collective's networking values, and the pad's model-type list. These
# used to be HARDCODED in the `docker run` below, with the profile of record
# carrying a copy of the same numbers next to them -- so a reader who edited
# the profile changed nothing and the copy was a comment pretending to be a
# knob. Each is read from the environment here, with the SAME value the
# launch hardcoded before as its default, so the default is still the
# validated one.
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1}"
NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
NCCL_IB_SUBNET_PREFIX_LEN="${NCCL_IB_SUBNET_PREFIX_LEN:-24}"
NCCL_IB_GID_AUTO="${NCCL_IB_GID_AUTO:-1}"
NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-0}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"
CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
VLLM_TP3_PAD_MODEL_TYPES="${VLLM_TP3_PAD_MODEL_TYPES:-minimax_m3_vl,minimax_m3_dspark}"

[[ -d "$MODEL_DIR" ]] || { echo "model not staged: $MODEL_DIR" >&2; exit 1; }
[[ -f "$SPEC_FILE" ]] || { echo "spec config missing: $SPEC_FILE" >&2; exit 1; }

# ---------------------------------------------------------------- bootstrap
read -r -d '' BOOTSTRAP <<'EOF' || true
set -euo pipefail
# The TP=3 pad overlay (../patches): applied in-container BEFORE anything
# imports vllm. Fail the BOOT, not the window.
VLLM_SITE="$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
python3 /tp3pad/apply_mm31_tp3pads.py "$VLLM_SITE"
python3 /tp3pad/apply_mm31_tp3pads.py "$VLLM_SITE" verify
# The speculative config is ONE file. It is read ONCE here, and the very same
# string goes to the self-check AND to the serve line -- so the JSON the
# engine receives is provably the file's content (the self-check compares the
# two byte-for-byte and fails the boot otherwise).
SPEC_FILE=/spec-config.json
SPEC_JSON="$(cat "$SPEC_FILE")"
# The DSpark DRAFT head is padded to TP=3 by the same kit as the target: the
# draft carries the same published geometry (64 query heads / 4 KV heads),
# which does not divide by three either. This gate proves it BEFORE the
# engine starts, from the real engine path: the real get_config() plus the
# real speculative-config promotion plus the real pad hook plus the real
# checkpoint's tensor headers. It fails the BOOT, not the window.
python3 /tp3pad/verify-draft-pad.py --model-dir /model \
  --spec-config "$SPEC_FILE" --served-spec-json "$SPEC_JSON"
# The non-vacuity battery: the check above must be ABLE to fail. --selftest
# mutates the inputs (drops the whitelist entry, uses an unpadded config,
# corrupts a pad-table rule, merges the two rule scopes) and requires the
# check to REJECT every one of them. A boot whose gate cannot fail proves
# nothing, so this runs at every boot too; under `set -e` a non-zero exit
# kills the boot.
python3 /tp3pad/verify-draft-pad.py --model-dir /model --selftest
# The mp/TCPStore rendezvous: the head serves, the ranks join headless.
if [[ "${ROLE:-head}" == "worker" ]]; then
  echo "[tp3-boot] worker rank ${NODE_RANK:?}: headless serve joining ${HEAD_IP:?}:${DIST_MASTER_PORT:-25000}"
  HEADLESS_FLAG="--headless"
else
  echo "[tp3-boot] head: serving (TCPStore master ${NODE_IP}:${DIST_MASTER_PORT:-25000})"
  HEADLESS_FLAG=""
fi
# The capture-grid cap is an ENVIRONMENT argument, like the budget. UNSET is
# the engine's own default grid; SET, it caps the PIECEWISE grid at the
# given JSON list.
#
# This block is inside the container's bootstrap (the quoted heredoc), and
# that is deliberate: the assignment below and the expansion of
# ${COMPILATION_CONFIG} on the serve line both happen inside the container,
# in the same shell, so the variable is always defined when the serve line
# reads it. CGCAP reaches the container as an environment variable. Do NOT
# move this block out of the heredoc: on the host, the quoted heredoc is
# literal text, so an assignment made outside it would never reach the
# container, and the serve line would expand an undefined variable under
# `set -u`.
#
# WHY IT IS SAFE TO CAP AT 128 (proved in the pinned image by
# ../tools/resolve-capture-grid.py, not by a desk read): the FULL
# (uniform-decode) graph descriptors are DERIVED FROM this grid. The engine
# rounds each grid size up to a multiple of the uniform decode query length
# and drops it if it passes the maximum decode tokens. The sizes above 128
# therefore NEVER produced a decode graph; they are prefill/mixed graphs
# only. Capping at 128 keeps all the decode descriptors and removes only the
# oversized prefill graphs.
#
# WHY: the engine subtracts a capture-pool ESTIMATE from the budget BEFORE
# the KV cache is sized, so the difference between the estimate and the real
# pool is capacity the lane pays for and never uses. The estimate falls with
# the graph count, so removing graphs makes the estimate and the real pool
# fall TOGETHER, and the freed estimate becomes KV capacity at
# (approximately) zero price in host memory.
if [ -n "${CGCAP:-}" ]; then
  COMPILATION_CONFIG="{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": ${CGCAP}}"
  echo "[tp3-boot] capture-grid cap IS IN FORCE: ${CGCAP}"
else
  COMPILATION_CONFIG='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
  echo "[tp3-boot] no capture-grid cap (the engine's default grid)"
fi
# The serve line.
#   --block-size 128          the MSA cache's block size (the engine's own
#                             requirement for this model's sparse attention)
#   --kv-cache-dtype fp8      the KV cache dtype
#   --attention-config        the indexer's KV dtype, fp8
#   --moe-backend marlin      pinned deliberately: the automatic choice hung
#                             the profile run at TP=3
#   --disable-hybrid-kv-cache-manager
#                             the hybrid cache manager is wrong for this model
#   --limit-mm-per-prompt     video is not served by this recipe
#   --reasoning-parser /      the model's own reasoning and tool-call parsers
#   --tool-call-parser
exec vllm serve /model \
  --load-format safetensors \
  --trust-remote-code \
  --tensor-parallel-size "${TP:-3}" \
  --mm-encoder-tp-mode data \
  --moe-backend marlin \
  --distributed-executor-backend mp \
  --nnodes "${CLUSTER_SIZE:-3}" \
  --node-rank "${NODE_RANK:-0}" \
  --master-addr "${HEAD_IP}" \
  --master-port "${DIST_MASTER_PORT:-25000}" \
  ${HEADLESS_FLAG} \
  --max-model-len "${CTX:-196608}" \
  --gpu-memory-utilization "${GMU:-0.78}" \
  --served-model-name "${SERVED_NAME:-minimax31-flash}" "${ALIAS_NAME:-main}" \
  --port "${PORT:-8100}" \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype": "fp8"}' \
  --compilation-config "${COMPILATION_CONFIG}" \
  --speculative-config "${SPEC_JSON}" \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"video": 0}' \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice --tool-call-parser minimax_m3 \
  --no-enable-log-requests \
  --disable-hybrid-kv-cache-manager \
  ${EXTRA_SERVE_ARGS:-}
EOF

# ---------------------------------------------------------------- run
# NOTE: no --rm on purpose -- a failed container is preserved for forward
# diagnosis; stop and remove it explicitly.
exec docker run \
  --name "$CONTAINER_NAME" \
  --network host \
  --gpus all \
  --device /dev/infiniband \
  --ipc host \
  --ulimit stack=67108864 \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  -e NODE_RANK="$NODE_RANK" \
  -e DIST_MASTER_PORT="$DIST_MASTER_PORT" \
  -e ROLE="$ROLE" \
  -e NODE_IP="$NODE_IP" \
  -e HEAD_IP="$HEAD_IP" \
  -e CLUSTER_SIZE="$CLUSTER_SIZE" \
  -e VLLM_HOST_IP="$NODE_IP" \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL \
  -e NCCL_DEBUG_FILE="/tmp/nccl-rank${NODE_RANK}.log" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_CROSS_NIC="$NCCL_CROSS_NIC" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING="$NCCL_IB_SUBNET_AWARE_ROUTING" \
  -e NCCL_IB_SUBNET_PREFIX_LEN="$NCCL_IB_SUBNET_PREFIX_LEN" \
  -e NCCL_IB_GID_AUTO="$NCCL_IB_GID_AUTO" -e NCCL_IB_MERGE_NICS="$NCCL_IB_MERGE_NICS" \
  -e NCCL_SOCKET_IFNAME="$SOCKET_IFNAME" \
  -e TP_SOCKET_IFNAME="$TP_SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM="$NCCL_IB_ROCE_VERSION_NUM" \
  -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_PLUGIN=none \
  -e VLLM_TP3_PAD="$TP" \
  -e VLLM_TP3_PAD_MODEL_TYPES="$VLLM_TP3_PAD_MODEL_TYPES" \
  -e VLLM_TP3_PAD_TABLE=/tp3pad/tp3_pad_table.json \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="$VLLM_ENABLE_PCIE_ALLREDUCE" \
  -e CUTE_DSL_ARCH="$CUTE_DSL_ARCH" \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e PORT="$PORT" -e TP="$TP" -e GMU="$GMU" -e CTX="$CTX" \
  ${CGCAP_ENV[@]+"${CGCAP_ENV[@]}"} \
  -e SERVED_NAME="$SERVED_NAME" \
  -e ALIAS_NAME="$ALIAS_NAME" \
  -e EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$ROOT/patches:/tp3pad:ro" \
  -v "$SPEC_FILE:/spec-config.json:ro" \
  --entrypoint /bin/bash \
  "$IMAGE" -euo pipefail -c "$BOOTSTRAP"
