"""
Collect the graph from Memgraph and output it locally.

We don't want to have to keep querying the database while we're running A*. So,
we serialize all the data we need into Smile. This is the graph's adjacency
list.

We could serialize the embeddings too, but it turns out that doing this in
Python is very memory intensive.

The schema for the file is as follows: ```
{
    "num_nodes": int,
    "adjacency_list": list[list[int]],
}
```
"""

import argparse
from typing import Any

from bson import BSON
from neo4j import GraphDatabase
import numpy as np
from tqdm import tqdm
import yaml


def main(
    driver: GraphDatabase, args: argparse.Namespace, config: dict[str, Any]
) -> None:

    ret = {}

    # Compute the number of nodes in the graph
    num_nodes_response, _, _ = driver.execute_query(
        "MATCH (n :Page) RETURN count(n) AS count"
    )
    ret["num_nodes"] = num_nodes_response[0]["count"]

    # Read the adjacency list
    print("Reading adjacency list...")
    ret["adjacency_list"] = [[] for _ in range(ret["num_nodes"])]
    with driver.session() as session:
        edges_result = session.run(
            "MATCH (s :Page)-[:GOES_TO]->(t :Page) RETURN s.idx AS s, t.idx AS t"
        )
        for record in tqdm(edges_result, desc="Edges"):
            s = record["s"]
            t = record["t"]
            assert s < ret["num_nodes"] and t < ret["num_nodes"]
            ret["adjacency_list"][s].append(t)

    print("Writing to file...")
    args.output.write(BSON.encode(ret))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data for graph search")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("wb"),
        help="Path to output file",
        default="graph.bson",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver, args, config)
