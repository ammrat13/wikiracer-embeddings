"""Find how often a greedy walk finds the target node."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main(args: argparse.Namespace):
    trace_df = pd.read_csv(args.trace)["steps"]

    failures = trace_df.isna().sum()
    success_rate = 1 - failures / len(trace_df)

    avg_steps = trace_df.mean()

    print(f"Success Rate:  {100 * success_rate:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two A* runs.")
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        help="Path to config file",
        default="config.yaml",
    )
    parser.add_argument(
        "trace",
        type=argparse.FileType("r"),
        help="Results of greedy walking on the test set.",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    plt.style.use(config["plotting"]["style"])

    main(args)
