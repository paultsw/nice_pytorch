"""
Implementation of NICE log-likelihood loss and bijective triangular-jacobian layers.

(This should be refactored into self-contained, re-usable components.)
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
def gaussian_nice_loss(h, size_average=True):
    """
    Definition of maximum (log-)likelihood loss function with a Gaussian prior, as in the paper.
    
    Args:
    * h: float tensor of shape (N,D). First dimension is batch dim, second dim consists of components
      of a factorized probability distribution.
    * size_average: if True, average over the batch dimension; if False, sum over batch dimension.

    Returns:
    * loss: torch float tensor.
    """
    if size_average:
        return (torch.mean(0.5*torch.sum(torch.pow(h,2),dim=1)) + (h.size(1)*0.5*torch.log(torch.tensor(2*np.pi))))
    else:
        return (0.5*torch.sum(torch.pow(h,2)) + ((h.size(0)+h.size(1))*0.5*torch.log(torch.tensor(2*np.pi))))

def logistic_nice_loss(h, size_average=True):
    """Definition of maximum (log-)likelihood loss function with a Logistic prior."""
    if size_average:
        return torch.mean(torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)), dim=1))
    else:
        return torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)))

# wrap above loss functions in Modules:
class GaussianPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(GaussianPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx):
        return gaussian_nice_loss(fx, size_average=self.size_average)

class LogisticPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogisticPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx):
        return logistic_nice_loss(fx, size_average=self.size_average)

# ===== ===== Coupling Layer Implementations ===== =====

_get_even = lambda xs: xs[:,0::2]
_get_odd = lambda xs: xs[:,1::2]

def _interleave(y_evens, y_odds):
    """Given 2 rank-2 tensors with same batch dimension, interleave their columns."""
    raise NotImplementedError() # TODO

class _BaseCouplingLayer(nn.Module):
    def __init__(self, dim, partition, activation):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module.

        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))
        
        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module. 
        * activation: an instance of torch.nn.Module.
        """
        super(BaseCouplingLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in ['even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"
        self.partition = partition
        if (partition == 'even'): # TODO: double-check these
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        # store activation module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        self.add_module('activation', activation)

    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        return _interleave(
            self._first(x),
            self.coupling_law(self._second(x), self.activation(self._first(x)))
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        return _interleave(
            self._first(y),
            self.anticoupling_law(self._second(y), self.activation(self._first(y)))
        )

    def coupling_law(self, a, b):
        # (a,b) --> g(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")

    def anticoupling_law(self, a, b):
        # (a,b) --> g^{-1}(a,b)
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")


class AdditiveCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a + b."""
    def coupling_law(self, a, b):
        return (a + b)
    def anticoupling_law(self, a, b):
        return None # TODO


class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b."""
    def coupling_law(self, a, b):
        return torch.mul(a,b)
    def anticoupling_law(self, a, b):
        return None # TODO

class AffineCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b1 + b2, where (b1,b2) is a partition of b."""
    def coupling_law(self, a, b):
        return torch.mul(a, _get_odd(b)) + _get_even(b2) # TODO: fix this to be based on self.partition
    def anticoupling_law(self, a, b):
        return None # TODO
