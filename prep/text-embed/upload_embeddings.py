import argparse

from neo4j import GraphDatabase
import numpy as np
import tqdm
import yaml


def main(driver: GraphDatabase, embeddings: np.ndarray) -> None:
    for idx, embedding in tqdm.tqdm(enumerate(embeddings), total=embeddings.shape[0]):
        driver.execute_query(
            """
            MATCH (n:Page {idx: $idx})
            SET n.text_embedding = $text_embedding
            """,
            idx=idx,
            text_embedding=embedding.tolist(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload embeddings to MemGraph")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "embeddings",
        type=argparse.FileType("rb"),
        help="Path to embeddings file",
    )
    args = parser.parse_args()

    config = yaml.load(args.config, yaml.Loader)
    embeddings = np.load(args.embeddings)

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver, embeddings)
