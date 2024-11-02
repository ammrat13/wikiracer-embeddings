"""Utility methods for training and evaluating link prediction models."""

from typing import Callable

import tensorflow as tf

TrainingExample = tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]


def example_decoder(length: int) -> Callable[[bytes], TrainingExample]:
    """
    Decode a serialized example from a TFRecord file.

    We need to know the length of the embeddings in order to decode the example,
    so we take that and pass it via currying.
    """

    def decode_example(serialized_example: bytes) -> TrainingExample:
        example = tf.io.parse_single_example(
            serialized_example,
            {
                "source": tf.io.FixedLenFeature([length], tf.float32),
                "target": tf.io.FixedLenFeature([length], tf.float32),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            },
        )
        return (
            (example["source"], example["target"]),
            example["label"],
        )

    return decode_example


TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def get_datasets(
    positive: tf.data.Dataset, negative: tf.data.Dataset, num_samples: int
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Construct the training, validation, and test sets from the positive and
    negative examples.

    The datasets will not be decoded by this function, nor will they be
    shuffled. We assume the data is already shuffled.
    """

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

    return train, val, test
