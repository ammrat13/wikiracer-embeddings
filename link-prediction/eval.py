"""
Evaluate a link prediction model.
"""

import argparse
import os

import tensorflow as tf

import util


def main(
    args: argparse.Namespace, positive: tf.data.Dataset, negative: tf.data.Dataset
) -> None:

    decoder = util.example_decoder(args.embedding_length)
    train, val, test = util.get_datasets(
        positive.map(decoder), negative.map(decoder), args.num_train_samples
    )

    model = tf.keras.models.load_model(args.model, compile=False)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )

    metrics_train = model.evaluate(train.batch(util.BATCH_SIZE), return_dict=True)
    metrics_val = model.evaluate(val.batch(util.BATCH_SIZE), return_dict=True)
    metrics_test = model.evaluate(test.batch(util.BATCH_SIZE), return_dict=True)

    print("\nTraining set metrics:")
    for name, value in metrics_train.items():
        print(f"\t{name}: {value:.4f}")

    print("\nValidation set metrics:")
    for name, value in metrics_val.items():
        print(f"\t{name}: {value:.4f}")

    print("\nTest set metrics:")
    for name, value in metrics_test.items():
        print(f"\t{name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a link prediction model")
    parser.add_argument(
        "-t",
        "--num-train-samples",
        type=int,
        help="Number of samples used when training the model",
        default=10000,
    )
    parser.add_argument(
        "-l",
        "--embedding-length",
        type=int,
        help="Length of text embeddings",
        default=256,
    )
    parser.add_argument("model", type=str, help="Model to evaluate")
    parser.add_argument("data", type=str, help="Path to training data")
    args = parser.parse_args()

    positive_dataset = tf.data.TFRecordDataset(
        os.path.join(args.data, "positive.tfrecord")
    )
    negative_dataset = tf.data.TFRecordDataset(
        os.path.join(args.data, "negative.tfrecord")
    )

    main(args, positive_dataset, negative_dataset)
