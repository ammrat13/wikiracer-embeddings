"""
Evaluate a link prediction model.
"""

import argparse

import tensorflow as tf

import util


def main(args: argparse.Namespace) -> None:

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

    input_shape = model.input_shape
    assert len(input_shape) == 2
    assert input_shape[0] == input_shape[1]
    assert len(input_shape[0]) == 2
    assert input_shape[0][0] is None

    train, val, test = util.get_datasets(
        args.data, args.num_train_samples, util.example_decoder(input_shape[0][1])
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
    parser.add_argument("model", type=str, help="Model to evaluate")
    parser.add_argument("data", type=str, help="Path to training data")
    args = parser.parse_args()

    main(args)
