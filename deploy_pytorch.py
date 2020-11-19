import numpy as np
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from torch.autograd import Variable
from model.mlp import MLP
os.environ['CUDA_LAUNCH_BLOCKING']='1'

class pytorch_nn():
    def __init__(self):
        weights_pth = os.path.join('./checkpoints','nn_weights.pth')
        self.model = MLP(input_size=5, output_size=3)
        self.model.load_state_dict(torch.load(weights_pth, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, in_vector):    
        in_vector = torch.from_numpy(np.array(in_vector))
        in_vector =  Variable(in_vector).float()
        outputs = self.model.forward(in_vector)
        prob, pred = outputs.max(0, keepdim=True)
        return outputs, prob, pred


if __name__ == '__main__':
    model = pytorch_nn()
    a = [1 ,567, 4, 1, 0]
    outputs = model.predict(a)
    print(outputs)