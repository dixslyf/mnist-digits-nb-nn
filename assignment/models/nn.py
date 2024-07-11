from typing import Sequence

import torch
import torch.nn as nn


class MnistNN(nn.Module):
    """
    A PyTorch convolutional neural network for the MNIST data set.
    """

    _input_dims = torch.tensor([28, 28])

    def __init__(
        self,
        activation: nn.Module,
        conv_params: Sequence[tuple[int, int]],
        pool_kernel_size: int,
        conv_dropout_p: float,
        linear_dropout_p: float,
        linear_out_features: Sequence[int],
    ):
        super().__init__()

        self._activation = activation

        # Keeps track of the dimensions of the image after each convolution and pool.
        first_linear_dims = self._input_dims.clone()

        # Set up the convolution layers.
        self._conv_layers = nn.ModuleList()
        for idx, conv_param in enumerate(conv_params):
            # The first convolution layer will see the original images,
            # which only have 1 channel. Each subsequent convolution layer
            # will see the output channels of the previous layer.
            in_channels = 1 if idx == 0 else self._conv_layers[idx - 1].out_channels
            out_channels = conv_param[0]
            kernel_size = conv_param[1]
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # Move the kernel filter by one each time.
            )
            self._conv_layers.append(conv)

            # After the layer has been applied, the dimensions of the images get reduced.
            first_linear_dims -= kernel_size - 1

        self._pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # After pooling, each dimension of the image gets divided by the pooling kernel size.
        first_linear_dims //= pool_kernel_size

        # The number of inputs the first linear layer will see is the
        # number of output channels of the last convolutional layer
        # multiplied by the number of pixels of the output image.
        first_linear_in_features = (
            self._conv_layers[-1].out_channels * first_linear_dims.prod()
        )

        # If the convolutional and pooling kernel sizes are too large, we can
        # end up reducing the dimensions of the images to 0.
        if first_linear_in_features == 0:
            raise ValueError("Encountered 0 input features for the first linear layer")

        # Use dropout for regularisation.
        self._conv_dropout = nn.Dropout(conv_dropout_p)
        self._linear_dropout = nn.Dropout(linear_dropout_p)

        # Set up the linear layers.
        # Similar to setting up the convolutional layers.
        # However, the last layer needs to be handled separately
        # since the number of outputs must be 10.
        self._linear_layers = nn.ModuleList()
        for idx, out_features in enumerate(linear_out_features):
            in_features = (
                first_linear_in_features
                if idx == 0
                else self._linear_layers[idx - 1].out_features
            )

            linear = nn.Linear(in_features=in_features, out_features=out_features)
            self._linear_layers.append(linear)

        # The last layer needs to have 10 output features (corresponding to the digits).
        self._linear_layers.append(
            nn.Linear(in_features=self._linear_layers[-1].out_features, out_features=10)
        )

    def forward(self, x):
        for conv_layer in self._conv_layers:
            x = conv_layer(x)
            x = self._activation(x)
            x = self._conv_dropout(x)

        x = self._pool(x)

        # The convolution layers spit out 3-dimensional data
        # (4-dimensional if you count the rows),
        # which need to be flattened before passing to the linear layers.
        x = torch.flatten(x, 1)

        for linear_layer in self._linear_layers[:-1]:
            x = linear_layer(x)
            x = self._activation(x)
            x = self._linear_dropout(x)

        x = self._linear_layers[-1](x)

        return x
