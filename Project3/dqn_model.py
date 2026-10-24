#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    # def __init__(self, in_channels=4, num_actions=4):
    #     """
    #     Parameters:
    #     -----------
    #     in_channels: number of channel of input.
    #             i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
    #     num_actions: number of action-value to output, one-to-one correspondence to action in game.

    #     You can add additional arguments as you need.
    #     In the constructor we instantiate modules and assign them as
    #     member variables.
    #     """
    #     super(DQN, self).__init__()
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #

    # def forward(self, x):
    #     """
    #     In the forward function we accept a Tensor of input data and we must return
    #     a Tensor of output data. We can use Modules defined in the constructor as
    #     well as arbitrary operators on Tensors.
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #

    #     ###########################
    #     return x

    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the flattened features after
        # the conv layers
        # Assuming input image size of 84x84
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        size = 84  # Input image size (height and width)
        size = conv2d_size_out(size, kernel_size=8, stride=4)  # After conv1
        size = conv2d_size_out(size, kernel_size=4, stride=2)  # After conv2
        size = conv2d_size_out(size, kernel_size=3, stride=1)  # After conv3

        linear_input_size = size * size * 64  # 64 is the number of filters from conv3

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # Input x has shape (batch_size, in_channels, height, width)
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from convolutional layers
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, flattened_features)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  

        return x  # Returns Q-values for each action
