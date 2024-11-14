"""
Generate training data for the distance estimation task.

We pick random edges from the graph and use them as distance-one examples. For
the undirected case, we randomly flip them. Note that we don't sample nodes
according to the damping factor. Instead, we sample them uniformly at random.
"""

import argparse
import os
import random
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

    # Compute the number of edges in the graph. Cap this at the number of edges
    # to collect.
    k_response, _, _ = driver.execute_query(
        "MATCH ()-[r:GOES_TO]->() RETURN count(r) AS count"
    )
    K = min(args.num_edges, k_response[0]["count"])

    # Create the output file. We could try to extend a file if it already
    # exists, but that gets into issues with Ctrl-C and other interruptions. So,
    # we'll just overwrite the file if it already exists.
    output_directory = config["training-data"]["distance-estimation"]
    output_file = h5py.File(os.path.join(output_directory, args.dataset_name), "w")
    output_group = output_file.create_group("edge")
    output_group.create_dataset("pairs", (K, 2), maxshape=(None, 2), dtype=np.uint32)

    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Page)-[:GOES_TO]->(t:Page)
            WITH s, t, rand() AS rd
            ORDER BY rd
            RETURN s.idx AS st, t.idx AS ed
            LIMIT $k
            """,
            k=K,
        )

        for i, record in tqdm(enumerate(result), desc="Edges", total=K):
            st, ed = record["st"], record["ed"]
            if args.undirected and random.choice([True, False]):
                st, ed = ed, st
            output_group["pairs"][i] = np.array([st, ed], dtype=np.uint32)


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
        "-e",
        "--num-edges",
        type=int,
        help="Number of edges to collect",
        default=1000000,
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

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver, args, config)
