"""
Postprocess the data collected by `bfs.py` and `edge.py`. This script combines
the two data sets, then splits the result into train, validation, and test sets.
"""

import argparse
import os
from typing import Any

import h5py
import numpy as np
import yaml

TRAIN_FRAC = 0.85
VAL_FRAC = 0.1
TEST_FRAC = 0.05


def main(args: argparse.Namespace, config: dict[str, Any]):

    data_dir = config["training-data"]["distance-estimation"]

    bfs_file = h5py.File(os.path.join(data_dir, args.bfs))
    edge_file = h5py.File(os.path.join(data_dir, args.edge))
    bfs = bfs_file["bfs"]
    edge = edge_file["edge"]

    assert bfs["distance"].shape[0] == bfs["source-idx"].shape[0]
    assert bfs["distance"].shape[0] == bfs["target-idx"].shape[0]
    assert bfs["distance"].shape[1] == bfs["target-idx"].shape[1]
    assert edge["pairs"].shape[1] == 2

    N = bfs["distance"].shape[1] + 1
    T = bfs["distance"].shape[0]
    K = edge["pairs"].shape[0]

    def copychunks(src, dst, offsets, src_sel):
        for cnk in src.iter_chunks(sel=src_sel):
            ocnk = tuple(
                [
                    slice(slc.start - off, slc.stop - off)
                    for slc, off in zip(cnk, offsets)
                ]
            )
            dst[ocnk] = src[cnk]

    def write(name, tslice, kslice):
        with h5py.File(os.path.join(data_dir, args.output, name), "w") as f:
            tlen = tslice.stop - tslice.start
            klen = kslice.stop - kslice.start
            obfs = f.create_group("bfs")
            oedge = f.create_group("edge")

            obfs.create_dataset("source-idx", (tlen,), dtype=np.uint32)
            obfs.create_dataset("target-idx", (tlen, N - 1), dtype=np.uint32)
            obfs.create_dataset("distance", (tlen, N - 1), dtype=np.uint8)
            oedge.create_dataset("pairs", (klen, 2), dtype=np.uint32)

            copychunks(
                bfs["source-idx"], obfs["source-idx"], [tslice.start], np.s_[tslice]
            )
            copychunks(
                bfs["target-idx"],
                obfs["target-idx"],
                [tslice.start, 0],
                np.s_[tslice, 0 : N - 1 : 1],
            )
            copychunks(
                bfs["distance"],
                obfs["distance"],
                [tslice.start, 0],
                np.s_[tslice, 0 : N - 1 : 1],
            )
            copychunks(
                edge["pairs"],
                oedge["pairs"],
                [kslice.start, 0],
                np.s_[kslice, 0:2:1],
            )

    T_train = int(T * TRAIN_FRAC)
    T_val = int(T * VAL_FRAC)
    Tslice_train = slice(0, T_train)
    Tslice_val = slice(T_train, T_train + T_val)
    Tslice_test = slice(T_train + T_val, T)

    K_train = int(K * TRAIN_FRAC)
    K_val = int(K * VAL_FRAC)
    Kslice_train = slice(0, K_train, 1)
    Kslice_val = slice(K_train, K_train + K_val, 1)
    Kslice_test = slice(K_train + K_val, K, 1)

    write("train.h5", Tslice_train, Kslice_train)
    write("val.h5", Tslice_val, Kslice_val)
    write("test.h5", Tslice_test, Kslice_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess the collected data.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output directory name", default="."
    )
    parser.add_argument("bfs", type=str, help="Name of the BFS dataset")
    parser.add_argument("edge", type=str, help="Name of the edge dataset")

    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    main(args, config)
