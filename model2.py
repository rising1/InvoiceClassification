import torch
from torch import tensor
from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(71, 512),
            # nn.Linear(4544, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Linear(512, 61),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        x = x.numpy()
        x = x.astype(np.float)
        x = tensor(x ,dtype=torch.float32)
        logits = self.linear_relu_stack(x)
        return logits