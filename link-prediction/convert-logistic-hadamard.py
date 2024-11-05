"""Convert a logistic regression model to the PyTorch format."""

import tensorflow as tf
import torch

import util


class Model(torch.nn.Module):

    embedding_length: int

    def __init__(self, embedding_length: int):
        super(Model, self).__init__()
        self.embedding_length = embedding_length
        self.linear = torch.nn.Linear(embedding_length, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = s * t
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    args = util.CONVERT_PARSER.parse_args()

    keras_model = tf.keras.models.load_model(args.source)
    assert len(keras_model.input_shape) == 2
    assert keras_model.input_shape[0] == keras_model.input_shape[1]
    assert keras_model.input_shape[0][0] is None

    keras_w = keras_model.get_weights()[0]
    keras_b = keras_model.get_weights()[1]

    torch_model = Model(embedding_length=keras_model.input_shape[0][1])
    with torch.no_grad():
        torch_model.linear.weight.data = torch.tensor(keras_w.T)
        torch_model.linear.bias.data = torch.tensor(keras_b)

    script_model = torch.jit.script(torch_model)
    script_model.save(args.dest)
