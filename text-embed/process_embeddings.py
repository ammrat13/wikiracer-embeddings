import argparse

import jsonlines
import numpy as np
import tqdm


def main(args: argparse.Namespace) -> None:

    embeddings = {}

    reader = jsonlines.Reader(args.input)
    max_idx = None
    dimensions = None
    for line in tqdm.tqdm(reader, desc="Reading embeddings"):

        idx = int(line["custom_id"].removeprefix("simplewiki-"))
        emb = line["response"]["body"]["data"][0]["embedding"]

        embeddings[idx] = np.array(emb, dtype=np.float32)

        if max_idx is None:
            max_idx = idx
        max_idx = max(max_idx, idx)

        if dimensions is None:
            dimensions = len(emb)
        else:
            assert dimensions == len(emb)

    ret = np.zeros((max_idx + 1, dimensions), dtype=np.float32)
    for idx, emb in tqdm.tqdm(embeddings.items(), desc="Converting to numpy array"):
        ret[idx] = emb

    print(f"Saving to {args.output.name}")
    print(f"Shape: {ret.shape}")
    np.save(args.output, ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the JSONL embeddings to a numpy array"
    )
    parser.add_argument(
        "input",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "output",
        type=argparse.FileType("wb"),
    )
    args = parser.parse_args()

    main(args)
