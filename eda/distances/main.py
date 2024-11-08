"""Plot a histogram of node distances in the `distance-estimation` dataset."""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot node distance histogram.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "-n",
        "--file-name",
        type=str,
        help="Name of the `distance-estimation` file",
        default="dataset.h5",
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        help="Number of bins in the histogram",
        default=12,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        help="Output file",
        default="hist.png",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    plt.style.use(config["plotting"]["style"])

    data_directory = config["training-data"]["distance-estimation"]
    data_file = h5py.File(os.path.join(data_directory, args.file_name))

    ret = np.zeros(args.bins, dtype=np.uint64)
    dset = data_file["distance"]
    for chunk in dset.iter_chunks():
        arr = dset[chunk]
        num, _ = np.histogram(arr, bins=np.arange(args.bins + 1))
        ret += num.astype(np.uint64)

    print(ret)

    plt.bar(np.arange(args.bins), ret, width=1.0, align="edge")
    plt.title("BFS Path-Length Histogram")
    plt.xlabel("Length")
    plt.ylabel("Number of Node Pairs")
    plt.savefig(args.output.name)
