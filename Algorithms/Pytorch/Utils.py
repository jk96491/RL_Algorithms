import torch
import numpy as np


def convertToTensorInput(input, input_size, batsize=1):
    input = np.reshape(input, [batsize, input_size])
    return torch.FloatTensor(input)
