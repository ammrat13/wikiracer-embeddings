ARG CS229_CARGO_BUILD_PROFILE=dev
# Build the project in release mode?
ARG CS229_CARGO_BUILD_DIR=debug
# Name of the build directory under target/

ARG CS229_LIBTORCH_CONTAINER=ammrat13/wikiracer-embeddings-libtorch:latest
# Container which has a copy of LibTorch

# A small container to hold the LibTorch download. This way, we can download it
# once and use it in multiple stages.
FROM ${CS229_LIBTORCH_CONTAINER} AS libtorch

# First container to build the project. Note that we have to install a few
# libraries for bindgen to work, according to Perplexity.
FROM rust:1.82.0 AS builder
ARG CS229_CARGO_BUILD_PROFILE
USER root
WORKDIR /work/
COPY --from=libtorch /libtorch/ /libtorch/
COPY . .
ENV LIBTORCH=/libtorch/
RUN apt update && \
    apt install -y llvm-dev libclang-dev clang && \
    rm -rf /var/lib/apt/lists/* && \
    cargo build --profile ${CS229_CARGO_BUILD_PROFILE}

# Second container for Memgraph
FROM memgraph/memgraph-mage:1.20-memgraph-2.20 AS memgraph
ARG CS229_CARGO_BUILD_DIR
USER root
COPY --from=libtorch /libtorch/ /libtorch/
ENV LIBTORCH=/libtorch/
USER memgraph
COPY --from=builder \
    /work/target/${CS229_CARGO_BUILD_DIR}/libgraphsearch.so \
    /usr/lib/memgraph/plugins/wikiracer_embeddings_graph_search.so
