"""
Implementation of models from paper.
"""
import torch
import torch.nn as nn
from nice.layers import AdditiveCouplingLayer


class NICEModel(nn.Module):
    """
    Replication of model from the paper:
      "Nonlinear Independent Components Estimation",
      Laurent Dinh, David Krueger, Yoshua Bengio (2014)
      https://arxiv.org/abs/1410.8516

    Has the following components:
    * four additive coupling layers with nonlinearity functions consisting of
      five-layer RELUs;
    * a diagonal scaling matrix output layer;
    * (...)
    """
    def __init__(self, input_dim):
        super(NICEModel, self).__init__()
        self.input_dim = input_dim
        self.layer1 = AdditiveCouplingLayer(
            input_dim,
            'even',
            nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
            )
        )
        self.layer2 = AdditiveCouplingLayer(
            input_dim,
            'odd',
            nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
            )
        )
        self.layer3 = AdditiveCouplingLayer(
            input_dim,
            'even',
            nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
            )
        )
        self.layer4 = AdditiveCouplingLayer(
            input_dim,
            'odd',
            nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.BatchNorm1d(input_dim),
            )
        )
        self.scaling_matrix = nn.Parameter(torch.eye(input_dim, requires_grad=True))


    def forward(self, xs):
        """..."""
        out = self.layer1(xs)
        out = self.layer2(xs)
        out = self.layer3(xs)
        out = self.layer4(xs)
        out = self.scaling_matrix * out
        return out


    def inverse(self, ys):
        """..."""
        with torch.enable_grad(False):
            # TODO
