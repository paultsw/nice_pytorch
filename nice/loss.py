"""
Implementation of NICE log-likelihood loss.
"""
import torch
import torch.nn as nn
import numpy as np

# ===== ===== Loss Function Implementations ===== =====
"""
We assume that we final output of the network are components of a multivariate distribution that
factorizes, i.e. the output is (y1,y2,...,yK) ~ p(Y) s.t. p(Y) = p_1(Y1) * p_2(Y2) * ... * p_K(YK),
with each individual component's prior distribution coming from a standardized family of
distributions, i.e. p_i == Gaussian(mu,sigma) for all i in 1..K, or p_i == Logistic(mu,scale).
"""
def gaussian_nice_loss(h, diag, size_average=True):
    """
    Definition of maximum (log-)likelihood loss function with a Gaussian prior, as in the paper.
    
    Args:
    * h: float tensor of shape (N,D). First dimension is batch dim, second dim consists of components
      of a factorized probability distribution.
    * diag: scaling diagonal of shape (D,).
    * size_average: if True, average over the batch dimension; if False, sum over batch dimension.

    Returns:
    * loss: torch float tensor.
    """
    if size_average:
        # AVERAGE { (1/2) * \sum^D_i  h_i**2 }
        #         + (D/2) * log(2\pi)
        #         + \sum^D_i S_{ii}
        return (
            torch.mean(0.5*torch.sum(torch.pow(h,2),dim=1)) + \
            h.size(1)*0.5*torch.log(torch.tensor(2*np.pi)) + \
            torch.sum(diag)
        )
    else:
        #   (1/2) * \sum_^N_n \sum^D_i h_i**2
        # + (N*D/2) * log(2\pi)
        # + N * \sum^D_i S_{ii}
        return (
            (0.5*torch.sum(torch.pow(h,2))) + \
            h.size(0)*h.size(1)*0.5*torch.log(torch.tensor(2*np.pi)) + \
            h.size(0) * torch.sum(diag)
        )

def logistic_nice_loss(h, diag, size_average=True):
    """Definition of maximum (log-)likelihood loss function with a Logistic prior."""
    if size_average:
        return (
            torch.mean(torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)), dim=1)) + \
            torch.sum(diag)
        )

    else:
        return (
            torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h))) + \
            h.size(0) * torch.sum(diag)
        )

# wrap above loss functions in Modules:
class GaussianPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(GaussianPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        return gaussian_nice_loss(fx, diag, size_average=self.size_average)

class LogisticPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogisticPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        return logistic_nice_loss(fx, diag, size_average=self.size_average)
