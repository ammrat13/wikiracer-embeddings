FROM ammrat13/wikiracer-embeddings-memgraph-base:latest
# See: Dockerfile.base

USER root
COPY . rust/wikiracer_embeddings_graph_search/
RUN python3 setup build --path /usr/lib/memgraph/query_modules/ --lang rust
USER memgraph
