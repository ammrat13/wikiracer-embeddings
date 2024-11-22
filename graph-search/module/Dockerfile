FROM ammrat13/wikiracer-embeddings-memgraph-base:latest
# See: Dockerfile.base

ARG CS229_BUILD_TYPE=Debug

USER root
COPY . rust/wikiracer_embeddings/
RUN sed -e 's@^rsmgp-sys.*$@rsmgp-sys = { path = "../rsmgp-sys" }@' -i rust/wikiracer_embeddings/Cargo.toml && \
    python3 setup build --path /usr/lib/memgraph/query_modules/ --lang rust --type ${CS229_BUILD_TYPE}
USER memgraph
