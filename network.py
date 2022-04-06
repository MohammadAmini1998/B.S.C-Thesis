
import torch
from torch import nn



device = torch.device("cuda:0")
dtype = torch.float

class Network(nn.Module):
    def __init__(self,input_channels,output_dim):
        super().__init__()
        self.input_channels=input_channels
        self.output_dim=output_dim

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )



    def forward(self,state):
        return self.network(state)