"""
Train a logistic regression model to do link prediction.

This file takes the source and target vectors and computes their outer product.
Logistic regression is done on the result.
"""

import argparse
import os

import tensorflow as tf

import util


@tf.keras.utils.register_keras_serializable()
class OuterProductLayer(tf.keras.Layer):
    """
    From ChatGPT:
    > Write a `keras` layer which takes two equal-length vectors as input and
    > outputs their outer product. The layer must not have any learnable
    > parameters.
    """

    def __init__(self, **kwargs):
        super(OuterProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2, "Input must consist of two tensors"
        assert input_shape[0] == input_shape[1], "Both inputs must have the same shape"
        assert len(input_shape[0]) == 2, "Input tensors must be 2D"
        assert input_shape[0][0] is None, "Batch size must be None"

    def call(self, inputs):
        s, t = inputs
        return tf.keras.ops.einsum("bi,bj->bij", s, t)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        vector_size = input_shape[0][1]
        return (batch_size, vector_size, vector_size)


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
        combine = OuterProductLayer()([input_s, input_t])
        flatten = tf.keras.layers.Flatten()(combine)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(flatten)
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
