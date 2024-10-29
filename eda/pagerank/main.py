"""
Make a graph of the PageRank distribution for the graph in the database.

This runs the PageRank algorithm at different damping factors, and plots how
skewed the distribution is. It saves the plots in the current directory.
"""

import matplotlib.pyplot as plt
import numpy as np

from neo4j import GraphDatabase

DAMPING_FACTORS = np.linspace(0.95, 0.50, 10).tolist()
"""A list of the damping factors to plot"""

GDB_URI = "bolt://localhost:7687"
GDB_USER = "cs229-simplewiki-data"
GDB_PASS = "cs229-simplewiki-data-pw"


def main():

    with GraphDatabase.driver(GDB_URI, auth=(GDB_USER, GDB_PASS)) as driver:
        driver.verify_connectivity()

        for damping_factor in DAMPING_FACTORS:

            records, _, _ = driver.execute_query(
                """
                CALL pagerank.get(100, $damping_factor, 1e-5)
                YIELD node, rank
                RETURN node.idx, rank
                ORDER BY rank DESCENDING
                """,
                damping_factor=damping_factor,
            )

            nodes = np.array([record["node.idx"] for record in records], dtype=np.int32)
            ranks = np.array([record["rank"] for record in records], dtype=np.float64)
            np.save(f"pagerank-node-{damping_factor:.2f}.npy", nodes)
            np.save(f"pagerank-rank-{damping_factor:.2f}.npy", ranks)

            cum_ranks = np.cumsum(ranks)
            print(f"Damping = {damping_factor:.2f}")
            for cutoff in [0.50, 0.90, 0.95, 0.99]:
                idx = np.searchsorted(cum_ranks, cutoff)
                print(f"\t{int(100 * cutoff)} = {idx}")

            for plt_type in ["log"]:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(ranks)
                ax.set_title(f"Damping = {damping_factor:.2f}")
                ax.set_ylabel("PageRank")

                if plt_type == "nrm":
                    ax.set_ylim(0, 1e-2)
                if plt_type == "log":
                    ax.set_yscale("log")
                    ax.set_ylim(1e-7, 1e-2)

                fig.savefig(f"pagerank-{plt_type}-damping-{damping_factor:.2f}.png")


if __name__ == "__main__":
    main()
