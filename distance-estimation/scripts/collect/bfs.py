"""
Generate training data for the distance estimation task.

We pick a random node according to the pagerank distribution, then run BFS from
that node as far as possible. We use zero as the label for unconnected nodes,
and we don't include the node itself in the dataset.
"""

import argparse
import os
import sys
from typing import Any

import h5py
from neo4j import GraphDatabase
import numpy as np
from tqdm import tqdm
import yaml


def main(
    driver: GraphDatabase,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:

    # Compute the number of nodes in the graph
    n_response, _, _ = driver.execute_query("MATCH (n:Page) RETURN count(n) AS count")
    N = n_response[0]["count"]

    # Create the output file. We could try to extend a file if it already
    # exists, but that gets into issues with Ctrl-C and other interruptions. So,
    # we'll just overwrite the file if it already exists.
    output_directory = config["training-data"]["distance-estimation"]
    output_file = h5py.File(os.path.join(output_directory, args.dataset_name), "w")
    output_file.create_dataset("source-idx", (0,), maxshape=(None,), dtype=np.uint32)
    output_file.create_dataset(
        "target-idx", (0, N - 1), maxshape=(None, N - 1), dtype=np.uint32
    )
    output_file.create_dataset(
        "distance", (0, N - 1), maxshape=(None, N - 1), dtype=np.uint8
    )

    # Hoist these variables out of the loop to avoid re-creating them every time
    IDX_VECTOR = np.arange(N, dtype=np.uint32)
    PAGERANK_PROP = (
        "uniform_cdf"
        if args.damping_factor == 0
        else f"pagerank_cdf_{args.damping_factor}"
    )
    SOURCE_QUERY = f"""
        MATCH (n:Page)
        WHERE n.{PAGERANK_PROP} >= rand()
        RETURN n.idx AS idx
        LIMIT 1
    """
    BFS_RELATION = "-[:GOES_TO *BFS]-" if args.undirected else "-[:GOES_TO *BFS]->"
    BFS_QUERY = f"""
        MATCH path = (s :Page {{idx: $idx}}){BFS_RELATION}(t)
        RETURN t.idx AS idx, size(path) AS distance
    """

    # Make sure to skip nodes we've already seen
    source_idx_set = set()
    for it in range(args.num_source_nodes):

        # Pick a random node according to the pagerank distribution. The user
        # specifies what the damping factor is.
        source_idx = None
        while source_idx is None or source_idx in source_idx_set:
            source_response, _, _ = driver.execute_query(SOURCE_QUERY)
            if len(source_response) == 0:
                continue
            else:
                source_idx = source_response[0]["idx"]
        source_idx_set.add(source_idx)

        # Create an array to store the resulting distances
        y = np.zeros((N,), dtype=np.uint8)

        # Run BFS from the source node
        with driver.session() as session:
            result = session.run(BFS_QUERY, idx=source_idx)
            for record in tqdm(result, desc=f"BFS Number {it}", total=N):
                y[record["idx"]] = record["distance"]

        # Extend the output dataset
        assert output_file["source-idx"].shape[0] == it
        assert output_file["target-idx"].shape[0] == it
        assert output_file["distance"].shape[0] == it
        output_file["source-idx"].resize(it + 1, axis=0)
        output_file["target-idx"].resize(it + 1, axis=0)
        output_file["distance"].resize(it + 1, axis=0)

        # Write
        output_file["source-idx"][it] = source_idx
        output_file["target-idx"][it, :source_idx] = IDX_VECTOR[:source_idx]
        output_file["target-idx"][it, source_idx:] = IDX_VECTOR[source_idx + 1 :]
        output_file["distance"][it, :source_idx] = y[:source_idx]
        output_file["distance"][it, source_idx:] = y[source_idx + 1 :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data for distance estimation"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-s",
        "--num-source-nodes",
        type=int,
        help="Number of source nodes to run BFS from",
        default=1000,
    )
    parser.add_argument(
        "-d",
        "--damping-factor",
        type=int,
        help="Damping factor of the PageRank distribution to use",
        default=80,
    )
    parser.add_argument(
        "-u",
        "--undirected",
        action="store_true",
        help="Treat edges as undirected",
    )
    parser.add_argument(
        "-n",
        "--dataset-name",
        type=str,
        help="Name of the file to write",
        default="dataset.h5",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    if args.damping_factor != 0 and args.damping_factor not in range(70, 100, 5):
        raise ValueError("Must have computed PageRank for this damping factor")

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver, args, config)
