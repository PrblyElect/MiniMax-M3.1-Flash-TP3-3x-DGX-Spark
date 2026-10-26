# The recipe's runtime image.
#
# The upstream day-0 build for MiniMax M3.1 Flash on arm64 / CUDA 13.4 ships
# NO ray, and the multi-node executor needs one. Everything else the recipe
# needs is applied at boot time by ../patches (the TP=3 pad kit), so this
# image is the base plus exactly one layer.
#
#   docker build -t vllm/vllm-openai:minimax31-flash-tp3 launch/
#   docker save vllm/vllm-openai:minimax31-flash-tp3 | \
#     ssh spark-b.local docker load
#
# A save/load pipe drops repo digests, so every box carries the identical
# build under the same local tag. Keep the tag and the base digest in
# config/cluster.env in step with what you actually shipped.
#
# THE TWO BUILD INPUTS ARE ARGUMENTS, so the profile of record's copy of them
# is the one that decides what gets built. They used to be written here as
# literals, which made the profile's BASE_IMAGE and RAY_VERSION decorative.
# The defaults below are the validated ones, so a plain `docker build` with
# no --build-arg still produces exactly the image this recipe was measured on.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:5d76347628973fb80519a919a2f2740b6645b62cba3fe2bd87b25f111bd0e948
ARG RAY_VERSION=2.59.0

FROM ${BASE_IMAGE}

ARG RAY_VERSION
RUN pip install --no-cache-dir "ray==${RAY_VERSION}"
