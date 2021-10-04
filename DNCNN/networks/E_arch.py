import torch
import torch.nn as nn

class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        model = []
        model+= [nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=(3//2), bias=False)]
        model += [nn.ReLU()]
        for i in range(4):
            model+= [nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(3//2), bias=False)]
            model += [nn.ReLU()]
        model+= [nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=(3//2), bias=False)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, lr):
        return self.model(lr)