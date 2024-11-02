"""
Generate training data for the link prediction task.

For positive examples, we randomly sample the edges of the graph. For negative
examples, we randomly sample a pair of nodes, and filter to those that have no
edge between them.
"""

import argparse
import os

from neo4j import GraphDatabase
import numpy as np
import tensorflow as tf
import tqdm
import yaml


def main(driver: GraphDatabase, args: argparse.Namespace) -> None:
    collect_positive(driver, args)
    collect_negative(driver, args)


def collect_positive(driver: GraphDatabase, args: argparse.Namespace) -> None:

    positive_path = os.path.join(args.output, "positive.tfrecord")
    with driver.session() as session, tf.io.TFRecordWriter(positive_path) as writer:

        result = session.run(
            """
            MATCH (s:Page)-[:GOES_TO]->(t:Page)
            WITH s, t, rand() AS rd
            ORDER BY rd
            RETURN s.text_embedding[0..$l] AS st, t.text_embedding[0..$l] AS ed
            LIMIT $n
            """,
            l=args.embedding_length,
            n=args.positive_samples,
        )

        for record in tqdm.tqdm(
            result, desc="Positive samples", total=args.positive_samples
        ):
            write_record(record, 1, writer)


def collect_negative(driver: GraphDatabase, args: argparse.Namespace) -> None:

    negative_path = os.path.join(args.output, "negative.tfrecord")
    with driver.session() as session, tf.io.TFRecordWriter(negative_path) as writer:

        result = session.run(
            """
            WITH range(0, $n) AS iter
            UNWIND iter AS it
            WITH it, rand() AS rs, rand() AS rt
            CALL {
                WITH rs
                MATCH (s:Page)
                WHERE s.uniform_cdf >= rs
                RETURN s
                LIMIT 1
            }
            CALL {
                WITH rt
                MATCH (t:Page)
                WHERE t.uniform_cdf >= rt
                RETURN t
                LIMIT 1
            }
            CALL {
                WITH s, t
                MATCH (s), (t)
                WHERE NOT exists((s)-[:GOES_TO]->(t))
                RETURN true as no_edge
            }
            RETURN s.text_embedding[0..$l] AS st, t.text_embedding[0..$l] AS ed
            """,
            l=args.embedding_length,
            n=args.positive_samples,
        )

        # Note that we don't necessarily get the number of negative samples we
        # asked for, because we're filtering for nodes that don't have an edge.
        # We could retry until we get the right number, but this is good enough.

        for record in tqdm.tqdm(
            result, desc="Negative samples", total=args.positive_samples
        ):
            write_record(record, 0, writer)


def write_record(record: dict, label: int, writer: tf.io.TFRecordWriter) -> None:

    st = np.array(record["st"], dtype=np.float32)
    ed = np.array(record["ed"], dtype=np.float32)
    st = st / np.linalg.norm(st)
    ed = ed / np.linalg.norm(ed)

    writer.write(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "source": tf.train.Feature(float_list=tf.train.FloatList(value=st)),
                    "target": tf.train.Feature(float_list=tf.train.FloatList(value=ed)),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])
                    ),
                }
            )
        ).SerializeToString()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data for link prediction"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-l",
        "--embedding-length",
        type=int,
        help="Length of text embeddings",
        default=256,
    )
    parser.add_argument(
        "-p",
        "--positive-samples",
        type=int,
        help="How many positive samples to generate",
        default=1000000,
    )
    parser.add_argument(
        "-n",
        "--negative-samples",
        type=int,
        help="How many negative samples to generate",
        default=1000000,
    )
    parser.add_argument("-o", "--output", help="Path to output directory", default=".")
    args = parser.parse_args()

    config = yaml.load(args.config, yaml.Loader)

    gdb_uri = config["data"]["memgraph"]["uri"]
    gdb_user = config["data"]["memgraph"]["user"]
    gdb_pass = config["data"]["memgraph"]["pass"]
    with GraphDatabase.driver(uri=gdb_uri, auth=(gdb_user, gdb_pass)) as driver:
        main(driver, args)
