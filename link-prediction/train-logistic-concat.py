"""
Train a logistic regression model to do link prediction.

This file takes the source and target vectors and concatenates them. Logistic
regression is done on the result.
"""

import argparse
import os

import tensorflow as tf

import util


def main(
    args: argparse.Namespace, positive: tf.data.Dataset, negative: tf.data.Dataset
) -> None:

    decoder = util.example_decoder(args.embedding_length)
    train, val, _ = util.get_datasets(
        positive.map(decoder), negative.map(decoder), args.num_samples
    )

    if args.continue_training:
        model = tf.keras.models.load_model(args.output, compile=True)
    else:
        input_s = tf.keras.layers.Input(shape=(args.embedding_length,))
        input_t = tf.keras.layers.Input(shape=(args.embedding_length,))
        combine = tf.keras.layers.Concatenate()([input_s, input_t])
        output = tf.keras.layers.Dense(1, activation="sigmoid")(combine)
        model = tf.keras.Model(inputs=[input_s, input_t], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

    model.fit(
        train.batch(1024),
        validation_data=val.batch(1024),
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
        "-c",
        "--continue-training",
        action="store_true",
        help="Continue training from a saved checkopoint, using `-o` as the path",
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
