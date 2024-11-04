"""Utility methods for training and evaluating link prediction models."""

import argparse
import os
from typing import Callable

import tensorflow as tf

TrainingExample = tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]

TRAIN_FRAC: float = 0.8
VAL_FRAC: float = 0.1
TEST_FRAC: float = 0.1

BATCH_SIZE: int = 1024

TRAIN_PARSER: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Train a logistic regression model to do link prediction."
)
"""Argument parser that's used by all the training scripts."""

TRAIN_PARSER.add_argument(
    "-n",
    "--num-samples",
    type=int,
    help="Number of samples to truncate the data to",
    default=10000,
)
TRAIN_PARSER.add_argument(
    "-l",
    "--embedding-length",
    type=int,
    help="Length of text embeddings",
    default=256,
)
TRAIN_PARSER.add_argument(
    "-t",
    "--continue-training",
    action="store_true",
    help="Continue training from a saved checkopoint, using `-o` as the path",
)
TRAIN_PARSER.add_argument(
    "-c",
    "--config",
    type=argparse.FileType("r"),
    help="Path to config file",
    default="config.yaml",
)
TRAIN_PARSER.add_argument(
    "-o", "--output", help="Path to output model", default="model.keras"
)


def example_decoder(length: int) -> Callable[[bytes], TrainingExample]:
    """
    Decode a serialized example from a TFRecord file.

    We need to know the length of the embeddings in order to decode the example,
    so we take that and pass it via currying. The examples in the file are a
    fixed length (1536) but we truncate.
    """

    def decode_example(serialized_example: bytes) -> TrainingExample:
        EMBEDDING_LENGTH = 1536
        example = tf.io.parse_single_example(
            serialized_example,
            {
                "source": tf.io.FixedLenFeature([EMBEDDING_LENGTH], tf.float32),
                "target": tf.io.FixedLenFeature([EMBEDDING_LENGTH], tf.float32),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            },
        )

        source = example["source"][0:length]
        target = example["target"][0:length]

        source = source / tf.norm(source)
        target = target / tf.norm(target)

        return ((source, target), example["label"])

    return decode_example


def get_datasets(
    data_dir: str, num_samples: int, decoder: Callable[[bytes], TrainingExample]
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Construct the training, validation, and test sets.

    The datasets will not be decoded by this function, nor will they be
    shuffled. We assume the data is already shuffled.
    """

    # Load the positive and negative datasets
    positive = tf.data.TFRecordDataset(os.path.join(data_dir, "positive.tfrecord"))
    negative = tf.data.TFRecordDataset(os.path.join(data_dir, "negative.tfrecord"))

    # Compute how many samples to put in training and validation. Note that we
    # don't have a test set here. We don't compute any metrics on it, so we
    # don't need it here. We still reserve space for it though.
    num_train = int(TRAIN_FRAC * num_samples)
    num_val = int(VAL_FRAC * num_samples)
    num_test = num_samples - num_train - num_val

    # Truncate the datasets to the number of samples we want.
    pos = positive.take(num_samples)
    neg = negative.take(num_samples)

    # Split into training, validation, and test
    pos_train = pos.take(num_train)
    pos_val = pos.skip(num_train).take(num_val)
    pos_test = pos.skip(num_train + num_val).take(num_test)
    neg_train = neg.take(num_train)
    neg_val = neg.skip(num_train).take(num_val)
    neg_test = neg.skip(num_train + num_val).take(num_test)

    # Until now, we've kept the positive and negative examples separate in order
    # to have a balanced dataset. Now we'll combine them.
    train = pos_train.concatenate(neg_train)
    val = pos_val.concatenate(neg_val)
    test = pos_test.concatenate(neg_test)

    # Decode the examples
    train = train.map(decoder)
    val = val.map(decoder)
    test = test.map(decoder)

    return train, val, test


@tf.keras.utils.register_keras_serializable(package="CS229")
class OuterProductLayer(tf.keras.Layer):
    """
    From ChatGPT:
    > Write a `keras` layer which takes two equal-length vectors as input and
    > outputs their outer product. The layer must not have any learnable
    > parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def get_config(self):
        return super().get_config()
