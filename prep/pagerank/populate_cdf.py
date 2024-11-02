"""
Add PageRank statistics to the database.

The statistics are added for a variety of damping factors, and the results are
stored as `pagerank_cdf_xx`.
"""

import argparse
import itertools
from typing import Iterator

from neo4j import GraphDatabase
import tqdm
import yaml

DAMPING_PCTS = list(range(70, 100, 5))
"""Damping factors to populate, times 100."""


def populate(
    driver: GraphDatabase, prop: str, idxs: Iterator[int], cum_ranks: Iterator[float]
) -> None:

    with driver.session() as session:
        session.run(f"CREATE CONSTRAINT ON (n:Page) ASSERT n.{prop} IS TYPED FLOAT")
        session.run(f"CREATE CONSTRAINT ON (n:Page) ASSERT n.{prop} IS UNIQUE")

        for idx, cum_rank in tqdm.tqdm(zip(idxs, cum_ranks)):
            session.run(
                f"""
                    MATCH (n:Page {{idx: $idx}})
                    SET n.{prop} = $cum_rank
                    """,
                idx=idx,
                cum_rank=cum_rank,
            )

        session.run(f"CREATE CONSTRAINT ON (n:Page) ASSERT EXISTS (n.{prop})")
        session.run(f"CREATE INDEX ON :Page({prop})")


def main(driver: GraphDatabase) -> None:

    for damping_pct in DAMPING_PCTS:

        records, _, _ = driver.execute_query(
            """
            CALL pagerank.get(100, $damping_factor, 1e-5)
            YIELD node, rank
            RETURN node.idx AS idx, rank
            ORDER BY rank DESCENDING
            """,
            damping_factor=damping_pct / 100,
        )

        idxs = map(lambda r: r["idx"], records)
        ranks = map(lambda r: r["rank"], records)
        cum_ranks = itertools.accumulate(ranks)

        prop = f"pagerank_cdf_{damping_pct:02d}"
        populate(driver, prop, idxs, cum_ranks)

    records, _, _ = driver.execute_query(
        """
        MATCH (n:Page)
        RETURN n.idx AS idx
        """
    )
    uniform_idxs = map(lambda r: r["idx"], records)
    uniform_cum_ranks = map(lambda r: (r["idx"] + 1) / len(records), records)
    populate(driver, "uniform_cdf", uniform_idxs, uniform_cum_ranks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get statistics on PageRank distribution"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    args = parser.parse_args()

    config = yaml.load(args.config, yaml.Loader)

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver)
