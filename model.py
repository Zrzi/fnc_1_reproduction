import torch.nn as nn
import torch.nn.functional as F


class FNCModel(nn.Module):
    def __init__(self):
        super(FNCModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=10001, out_features=100),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=4),
            nn.Dropout(p=0.6)
        )

    def forward(self, x):
        output = self.layers(x)
        return F.log_softmax(output)

