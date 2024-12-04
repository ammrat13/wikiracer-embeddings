import argparse
import re

import openai

INPUT_REGEX = re.compile(r"simplewiki-embed-(\d+).jsonl")
"""Format of the input file names

This regex is used to detect which batches we should download the results for.
We skip any files that don't match this pattern.
"""


def main(args: argparse.Namespace) -> None:

    client = openai.Client()
    batches = client.batches.list()

    for i, batch in enumerate(batches):

        if batch.status != "completed":
            continue

        try:
            input_file = client.files.retrieve(batch.input_file_id)
        except openai.NotFoundError:
            continue
        if INPUT_REGEX.fullmatch(input_file.filename) is None:
            continue

        output_file_resp = client.files.content(batch.output_file_id)
        output_file_content = output_file_resp.text

        if not output_file_content.endswith("\n"):
            output_file_content += "\n"

        args.output.write(output_file_content)
        print(f"Downloaded batch: ({i}) {batch.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the embeddings from all the batches"
    )
    parser.add_argument(
        "output",
        type=argparse.FileType("w"),
    )
    args = parser.parse_args()

    main(args)
