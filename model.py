import torch.nn as nn
import torch.nn.functional as F


class FNCModel(nn.Module):
    def __init__(self, in_features, dropout):
        super(FNCModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=100),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=4),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        output = self.layers(x)
        return F.log_softmax(output)