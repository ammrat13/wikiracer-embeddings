import argparse
import sys
import time

import openai


def get_file_by_index(client: openai.Client, index: int) -> openai.File:
    """Get the file object corresponding to a request index.

    When we created the request files, we named them
    `simplewiki-embed-{index}.jsonl`. We want to feed each of them into the
    embedding API in turn.
    """
    files = client.files.list()
    for file in files:
        if file.filename == f"simplewiki-embed-{index}.jsonl":
            return file
    return None


def main(args: argparse.Namespace) -> None:

    client = openai.Client()
    batch_monitor = args.last_batch
    next_idx = args.next_file

    while True:

        if batch_monitor is not None:
            batch = client.batches.retrieve(batch_monitor)
            if batch.status in ["failed", "expired", "cancelling", "cancelled"]:
                print(f"Batch {batch_monitor} failed", file=sys.stderr)
                return
            if batch.status != "completed":
                time.sleep(30)
                continue

        file = get_file_by_index(client, next_idx)
        if file is None:
            return

        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        print(f"Running file {file.filename} in batch {batch.id}")
        batch_monitor = batch.id
        next_idx += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Push the SimpleWikipedia batches through"
    )
    parser.add_argument(
        "-l",
        "--last-batch",
        type=str,
        default=None,
        help="The batch to monitor for completion before starting",
    )
    parser.add_argument(
        "-n",
        "--next-file",
        type=int,
        default=0,
        help="The index of the next file to run",
    )
    args = parser.parse_args()

    main(args)
