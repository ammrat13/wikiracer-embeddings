"""Collect a set of node pairs to test on, and dump it as CSV."""

import argparse
import csv
from typing import Any

from neo4j import GraphDatabase
from tqdm import tqdm
import yaml


def main(
    driver: GraphDatabase, args: argparse.Namespace, config: dict[str, Any]
) -> None:

    PAGERANK_PROP = (
        "uniform_cdf"
        if args.damping_factor == 0
        else f"pagerank_cdf_{args.damping_factor}"
    )
    RANDOM_NODE_QUERY = f"""
        MATCH (n:Page)
        WHERE n.{PAGERANK_PROP} >= rand()
        RETURN n.idx AS idx
        LIMIT 1
    """

    # Create the output file
    wr = csv.DictWriter(args.output, fieldnames=["source-index", "target-index"])
    wr.writeheader()

    for _ in tqdm(range(args.num_pairs), desc="Pairs"):
        source_idx = driver.execute_query(RANDOM_NODE_QUERY)[0][0]["idx"]
        target_idx = driver.execute_query(RANDOM_NODE_QUERY)[0][0]["idx"]
        wr.writerow({"source-index": source_idx, "target-index": target_idx})


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
        "-n",
        "--num-pairs",
        type=int,
        help="Number of node pairs to collect",
        default=500,
    )
    parser.add_argument(
        "-d",
        "--damping-factor",
        type=int,
        help="Damping factor of the PageRank distribution to use",
        default=80,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        help="Path to output file",
        default="test-set.csv",
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
