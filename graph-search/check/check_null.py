"""
Perform an end-to-end test of the null heuristic. It should always return a path
of the same length as the shortest path.
"""

import argparse
import random
from typing import Any

from neo4j import GraphDatabase
import yaml
from tqdm import tqdm


def main(
    driver: GraphDatabase, args: argparse.Namespace, config: dict[str, Any]
) -> None:

    for _ in tqdm(range(args.num_tests), desc="Random tests"):

        # Select two nodes uniformly at random
        nodes = []
        for _ in range(2):
            response, _, _ = driver.execute_query(
                """
                MATCH (n:Page)
                WHERE n.uniform_cdf >= $rng
                RETURN n.idx AS idx
                LIMIT 1
                """,
                rng=random.random(),
            )
            nodes.append(response[0]["idx"])
        nodes = tuple(nodes)
        s_idx, t_idx = nodes

        # Find the length of the true shortest path
        bfs_response, _, _ = driver.execute_query(
            """
            MATCH (s :Page {idx: $s_idx}), (t :Page {idx: $t_idx})
            MATCH p = (s)-[:GOES_TO *BFS]->(t)
            RETURN size(p) AS distance
            """,
            s_idx=s_idx,
            t_idx=t_idx,
        )
        true_distance = None if len(bfs_response) == 0 else bfs_response[0]["distance"]

        # Find the length of the null heuristic path
        heur_response, _, _ = driver.execute_query(
            """
            MATCH (s :Page {idx: $s_idx}), (t :Page {idx: $t_idx})
            CALL wikiracer_embeddings.graph_search_null(s, t)
            PROCEDURE MEMORY UNLIMITED
            YIELD path AS p
            RETURN size(p) AS distance
            """,
            s_idx=s_idx,
            t_idx=t_idx,
        )
        heur_distance = heur_response[0]["distance"]

        if true_distance != heur_distance:
            print(f"Found a discrepancy for nodes {nodes}:")
            print(f"True distance: {true_distance}")
            print(f"Null heuristic distance: {heur_distance}")
            return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check the null heuristic.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-n",
        "--num-tests",
        type=int,
        help="Number of times to sample",
        default=1000,
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver, args, config)
