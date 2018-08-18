"""
Training loop for NICEModel. Attempts to replicate the conditions in the NICE paper.

Supports the following datasets:
* MNIST (LeCun & Cortes, 1998);
* Toronto Face Dataset (Susskind et al, 2010);
* CIFAR-10 (Krizhevsky, 2010);
* Street View House Numbers (Netzer et al, 2011).

We apply a dequantization for MNIST, TFD, SVHN as follows (following the NICE authors):
1. Add uniform noise ~ Unif([0, 1/256]);
2. Rescale data to be in [0,1] in each dimension.

For CIFAR10, we instead do:
1. Add uniform noise ~ Unif([-1/256, 1/256]);
2. Rescale data to be in [-1,1] in each dimensions.

Additionally, we perform:
* approximate whitening for TFD;
* exact ZCA on SVHN, CIFAR10;
* no additional preprocessing for MNIST.

Finally, images are flattened from (H,W) to (H*W,).
"""
# numeric/nn libraries:
import torch
import torchvision
import torch.optim as optim
import torch
import torch.utils.data as data
import numpy as np
# models/losses:
from models import NICEModel
from nice.loss import LogisticPriorNICELoss, GaussianPriorNICELoss
# python/os utils:
import argparse
import os
from tqdm import tqdm, trange

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Dataset loaders: each of these helper functions does the following:
# 1) downloads the corresponding dataset into a folder (if not already downloaded);
# 2) adds the corresponding whitening & rescaling transforms;
# 3) returns a dataloader for that dataset.

def _collate_images(batch):
    # TODO: collect [(input, target)] ==> torch.stack([x for x in inputs], dim=0)
    return None

def load_mnist(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess MNIST dataset."""
    mnist_transform = torchvision.transforms.Compose([
        # 1. TODO: uniform noise ~ [0,128]
        # 2. TODO: rescale to [0,1]
        # 3. TODO: unravel to 1 dimension
        torchvision.transforms.ToTensor(),
    ])
    return data.DataLoader(
        torchvision.datasets.MNIST(root="./datasets/mnist", train=train, transform=mnist_transform, download=True),
        batch_size=batch_size,
        collate_fn=collate_images,
        pin_memory=False,
        drop_last=True
    )

def load_svhn(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess SVHN dataset."""
    svhn_transform = torchvision.transforms.Compose([
        # 1. TODO
        # 2. TODO
        # 3. TODO
        torchvision.transforms.ToTensor(),
    ])
    _mode = 'train' if train else 'test'
    return data.DataLoader(
        torchvision.datasets.SVHN(root="./datasets/mnist", split=_mode, transform=svhn_transform, download=True),
        batch_size=batch_size,
        collate_fn=_collate_images,
        pin_memory=False,
        drop_last=True
    )

def load_cifar10(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess CIFAR10 dataset."""
    cifar10_transform = torchvision.transforms.Compose([
        # 1. TODO
        # 2. TODO
        # 3. TODO
        torchvision.transforms.ToTensor(),
    ])
    return data.DataLoader(
        torchvision.datasets.CIFAR10(root="./datasets/cifar", train=train, transform=cifar10_transform, download=True),
        batch_size=batch_size,
        collate_fn=_collate_images,
        pin_memory=False,
        drop_last=True
    )

def load_tfd(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess TFD dataset."""
    raise NotImplementedError("[load_tfd] Toronto Faces Dataset unsupported right now. Sorry!")

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Training loop: return a NICE model trained over a number of epochs.
def train(args):
    """Construct a NICE model and train over a number of epochs."""
    # === choose which dataset to build:
    if args.dataset == 'mnist':
        dataloader_fn = load_mnist
        input_dim = 24*24
    if args.dataset == 'svhn':
        dataloader_fn = load_svhn
        input_dim = 32*32
    if args.dataset == 'cifar10':
        dataloader_fn = load_cifar10
        input_dim = 32*32
    if args.dataset == 'tfd':
        dataloader_fn = load_tfd
        input_dim = None # TODO
        
    # === choose which loss function to build:
    if args.prior == 'logistic':
        loss_fn = LogisticPriorNICELoss(size_average=True)
    else:
        loss_fn = GaussianPriorNICELoss(size_average=True)
    
    # === build model:
    model = NICEModel(input_dim, args.nhidden, args.nlayers)

    # === train over a number of epochs; perform validation after each:
    for _ in trange(args.num_epochs):
        dataloader = dataloader_fn(train=True, batch_size=args.batch_size)

        # [[[ TODO: train/save for one epoch ]]]
        
        # save model to disk and delete dataloader to save memory::
        torch.save(None)
        del dataloader
        
        # perform validation loop:
        avg_val_loss = validate(model, dataloader_fn, loss_fn)
        # TODO: print validation loss?

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Validation loop: set gradient-tracking off with model in eval mode:
def validate(model, dataloader_fn, loss_fn):
    """Perform validation on a dataset."""
    # set model to eval mode (turns batch norm training off)
    model.eval()

    # build dataloader in eval mode:
    dataloader = dataloader_fn(train=True, batch_size=args.batch_size)

    # turn gradient-tracking off (for speed) during validation:
    validation_losses = []
    with torch.no_grad():
        for inputs in dataloader:
            validation_losses.append(loss_fn(model(inputs), model.scaling_diag).item())
    
    # delete dataloader to save memory:
    del dataloader

    # set model back in train mode:
    model.train()

    # return average validation loss:
    return np.mean(validation_losses)

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    # ----- parse training settings:
    parser = argparse.ArgumentParser(description="Train a fresh NICE model and save.")
    # configuration settings:
    parser.add_argument("--dataset", required=True, dest='dataset', choices=('tfd', 'cifar10', 'svhn', 'mnist'),
                        help="Dataset to train the NICE model on.")
    parser.add_argument("--epochs", dest='num_epochs', default=1500,
                        help="Number of epochs to train on. [1500]")
    parser.add_argument("--batch_size", dest="batch_size", default=16,
                        help="Number of examples per batch.")
    parser.add_argument("--savedir", dest='savedir', default="./saved_models",
                        help="Where to save the trained model. [./saved_models]")
    # model settings:
    parser.add_argument("--nonlinearity_layers", dest='nlayers', default=5,
                        help="...")
    parser.add_argument("--nonlinearity_hiddens", dest='nhidden', default=1000,
                        help="...")
    parser.add_argument("--prior", choices=('logistic', 'prior'), default="logistic",
                        help="...")
    # optimization settings:
    parser.add_argument("--lr", default=0.001, dest='lr',
                        help="Learning rate for ADAM optimizer. [0.001]")
    parser.add_argument("--beta1", default=0.9,  dest='beta1',
                        help="Momentum for ADAM optimizer. [0.9]")
    parser.add_argument("--beta2", default=0.01, dest='beta2',
                        help="Beta2 for ADAM optimizer. [0.01]")
    parser.add_argument("--eps", default=0.0001, dest='eps',
                        help="Epsilon for ADAM optimizer. [0.0001]")
    parser.add_argument("--lambda", default=1.0, dest='lambda',
                        help="L1 weight decay coefficient. [1.0]")
    args = parser.parse_args()
    # ----- run training loop over several epochs & save:
    model = train(args)
    _now = "2018_08_18_05h11m" # TODO: datetime
    _save_name = "nice.{0}.{1}.{2}.{3}.{4}.cpu.pt".format(args.dataset, args.epochs, args.nlayers, args.nhiddens, args.prior, _now)
    torch.save(model.state_dict, os.path.join(args.savedir, _save_name)
