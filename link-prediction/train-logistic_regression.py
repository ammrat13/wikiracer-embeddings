"""
Train a logistic regression model to do link prediction.
"""

import argparse
import os

import tensorflow as tf


def main(
    args: argparse.Namespace, positive: tf.data.Dataset, negative: tf.data.Dataset
) -> None:

    # Compute how many samples to put in training and validation. Note that we
    # don't have a test set here. We don't compute any metrics on it, so we
    # don't need it here. We still reserve space for it though.
    N = args.num_samples
    N_train = int(0.8 * N)
    N_val = int(0.1 * N)

    # The datasets are already in a random order, so there's no need to shuffle
    # them
    positive = positive.map(decode_example).take(N)
    negative = negative.map(decode_example).take(N)

    # Split into training, validation, and test
    positive_train = positive.take(N_train)
    positive_val = positive.skip(N_train).take(N_val)
    negative_train = negative.take(N_train)
    negative_val = negative.skip(N_train).take(N_val)

    # Until now, we've kept the positive and negative examples separate in order
    # to have a balanced dataset. Now we'll combine them.
    dataset_train = positive_train.concatenate(negative_train)
    dataset_val = positive_val.concatenate(negative_val)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2 * args.embedding_length,)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    model.fit(
        dataset_train.batch(1024),
        validation_data=dataset_val.batch(1024),
        epochs=1000,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                min_delta=1e-6,
                patience=10,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                args.output,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ],
    )


def decode_example(serialized_example: bytes) -> tuple[tf.Tensor, tf.Tensor]:
    example = tf.io.parse_single_example(
        serialized_example,
        {
            "source": tf.io.FixedLenFeature([args.embedding_length], tf.float32),
            "target": tf.io.FixedLenFeature([args.embedding_length], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        },
    )
    return (
        tf.concat([example["source"], example["target"]], axis=0),
        example["label"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model to do link prediction."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        help="Number of samples to truncate the data to",
        default=10000,
    )
    parser.add_argument(
        "-l",
        "--embedding-length",
        type=int,
        help="Length of text embeddings",
        default=256,
    )
    parser.add_argument(
        "-o", "--output", help="Path to output model", default="model.keras"
    )
    parser.add_argument("data", type=str, help="Path to training data")
    args = parser.parse_args()

    positive_dataset = tf.data.TFRecordDataset(
        os.path.join(args.data, "positive.tfrecord")
    )
    negative_dataset = tf.data.TFRecordDataset(
        os.path.join(args.data, "negative.tfrecord")
    )

    main(args, positive_dataset, negative_dataset)
