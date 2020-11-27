import torch
import numpy as np


def convertToTensorInput(input, input_size):
    input = np.reshape(input, [1, input_size])
    return torch.FloatTensor(input)
