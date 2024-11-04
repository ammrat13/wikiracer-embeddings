"""
Train a logistic regression model to do link prediction.

This file takes the source and target vectors and concatenates them. Logistic
regression is done on the result.
"""

import argparse
import os

import tensorflow as tf
import yaml

import util


def main(args: argparse.Namespace) -> None:

    config = yaml.safe_load(args.config)
    train, val, _ = util.get_datasets(
        config["training-data"]["link-prediction"],
        args.num_samples,
        util.example_decoder(args.embedding_length),
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
        train.batch(util.BATCH_SIZE),
        validation_data=val.batch(util.BATCH_SIZE),
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
    args = util.TRAIN_PARSER.parse_args()
    main(args)
