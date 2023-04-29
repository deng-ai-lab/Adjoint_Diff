import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)
        #self.init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)
    
    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
               
                layer.weight.data.normal_(mean = 0, std = 0.01)
                layer.bias.data.fill_(0.0)

        