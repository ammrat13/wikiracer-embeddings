"""
Make a graph of the PageRank distribution for the graph in the database.

This runs the PageRank algorithm at different damping factors, and plots how
skewed the distribution is. It saves the plots in the current directory.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import yaml

from neo4j import GraphDatabase

DAMPING_FACTORS = np.linspace(0.95, 0.50, 10).tolist()
"""A list of the damping factors to plot"""


def main(uri: str, user: str, password: str) -> None:

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()

        for damping_factor in DAMPING_FACTORS:
            damping_factor_int = round(100 * damping_factor)

            records, _, _ = driver.execute_query(
                """
                CALL pagerank.get(100, $damping_factor, 1e-5)
                YIELD node, rank
                RETURN rank
                ORDER BY rank DESCENDING
                """,
                damping_factor=damping_factor,
            )

            ranks = np.array([record["rank"] for record in records], dtype=np.float64)
            cum_ranks = np.cumsum(ranks)

            print(f"Damping = {damping_factor_int}%")
            for cutoff in [0.50, 0.90, 0.95, 0.99]:
                idx = np.searchsorted(cum_ranks, cutoff)
                print(f"\t{int(100 * cutoff)} = {idx}")

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(cum_ranks)
            ax.set_title(f"PageRank Distribution at \\(d = {damping_factor_int}\\%\\)")
            ax.set_ylabel("Cumulative Probability")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Rank")

            fig.savefig(f"pagerank-damping-{damping_factor_int}.png")


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
    plt.style.use(config["plotting"]["style"])
    main(
        config["data"]["memgraph"]["uri"],
        config["data"]["memgraph"]["user"],
        config["data"]["memgraph"]["pass"],
    )
