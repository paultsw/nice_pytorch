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
# models/losses/image utils:
from nice.models import NICEModel
from nice.loss import LogisticPriorNICELoss, GaussianPriorNICELoss
from nice.utils import rescale, l1_norm
# python/os utils:
import argparse
import os
from tqdm import tqdm, trange

# set CUDA training on if detected:
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    CUDA = True
else:
    DEVICE = torch.device('cpu')
    CUDA = False

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Dataset loaders: each of these helper functions does the following:
# 1) downloads the corresponding dataset into a folder (if not already downloaded);
# 2) adds the corresponding whitening & rescaling transforms;
# 3) returns a dataloader for that dataset.

def load_mnist(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess MNIST dataset."""
    mnist_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        # rescale to [0,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x, 0., 1.))
    ])
    return data.DataLoader(
        torchvision.datasets.MNIST(root="./datasets/mnist", train=train, transform=mnist_transform, download=False),
        batch_size=batch_size,
        pin_memory=CUDA,
        drop_last=True
    )

def load_svhn(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess SVHN dataset."""
    # check if ZCA matrix exists on dataset yet:
    assert os.path.exists("./datasets/svhn/zca_matrix.pt"), \
        "[load_svhn] ZCA whitening matrix not built! Run `python make_dataset.py` first."
    zca_matrix = torch.load("./datasets/svhn/zca_matrix.pt")

    svhn_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
        # rescale to [0,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x, 0., 1.)),
        # exact ZCA:
        torchvision.transforms.LinearTransformation(zca_matrix)
    ])
    _mode = 'train' if train else 'test'
    return data.DataLoader(
        torchvision.datasets.SVHN(root="./datasets/svhn", split=_mode, transform=svhn_transform, download=False),
        batch_size=batch_size,
        pin_memory=CUDA,
        drop_last=True
    )

def load_cifar10(train=True, batch_size=1, num_workers=0):
    """Rescale and preprocess CIFAR10 dataset."""
    # check if ZCA matrix exists on dataset yet:
    assert os.path.exists("./datasets/cifar/zca_matrix.pt"), \
        "[load_cifar10] ZCA whitening matrix not built! Run `python make_datasets.py` first."
    zca_matrix = torch.load("./datasets/cifar/zca_matrix.pt")

    cifar10_transform = torchvision.transforms.Compose([
        # convert PIL image to tensor:
        torchvision.transforms.ToTensor(),
        # flatten:
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # add uniform noise ~ [-1/256, +1/256]:
        torchvision.transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(128.).add_(-1./256.))),
        # rescale to [-1,1]:
        torchvision.transforms.Lambda(lambda x: rescale(x,-1.,1.)),
        # exact ZCA:
        torchvision.transforms.LinearTransformation(zca_matrix)
    ])
    return data.DataLoader(
        torchvision.datasets.CIFAR10(root="./datasets/cifar", train=train, transform=cifar10_transform, download=False),
        batch_size=batch_size,
        pin_memory=CUDA,
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
        input_dim = 28*28
    if args.dataset == 'svhn':
        dataloader_fn = load_svhn
        input_dim = 32*32*3
    if args.dataset == 'cifar10':
        dataloader_fn = load_cifar10
        input_dim = 32*32*3
    if args.dataset == 'tfd':
        raise NotImplementedError("[train] Toronto Faces Dataset unsupported right now. Sorry!")
        dataloader_fn = load_tfd
        input_dim = None

    # === build model & optimizer:
    model = NICEModel(input_dim, args.nhidden, args.nlayers)
    if (args.model_path is not None):
        assert(os.path.exists(args.model_path)), "[train] model does not exist at specified location"
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), eps=args.eps)

    # === choose which loss function to build:
    if args.prior == 'logistic':
        nice_loss_fn = LogisticPriorNICELoss(size_average=True)
    else:
        nice_loss_fn = GaussianPriorNICELoss(size_average=True)
    def loss_fn(fx):
        """Compute NICE loss w/r/t a prior and add L1 regularization."""
        return nice_loss_fn(fx, model.scaling_diag) + args.lmbda*l1_norm(model, include_bias=True)

    # === train over a number of epochs; perform validation after each:
    for t in range(args.num_epochs):
        print("* Epoch {0}:".format(t))
        dataloader = dataloader_fn(train=True, batch_size=args.batch_size)
        for inputs, _ in tqdm(dataloader):
            opt.zero_grad()
            loss_fn(model(inputs.to(DEVICE))).backward()
            opt.step()
        
        # save model to disk and delete dataloader to save memory:
        _dev = 'cuda' if CUDA else 'cpu'
        _fn = "nice.{0}.l_{1}.h_{2}.p_{3}.e_{4}.{5}.pt".format(args.dataset, args.nlayers, args.nhidden, args.prior, t, _dev)
        torch.save(model.state_dict(), os.path.join(args.savedir, _fn))
        print(">>> Saved file: {0}".format(_fn))
        del dataloader
        
        # perform validation loop:
        vmin, vmed, vmean, vmax = validate(model, dataloader_fn, nice_loss_fn)
        print(">>> Validation Loss Statistics: min={0}, med={1}, mean={2}, max={3}".format(vmin,vmed,vmean,vmax))

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Validation loop: set gradient-tracking off with model in eval mode:
def validate(model, dataloader_fn, loss_fn):
    """Perform validation on a dataset."""
    # set model to eval mode (turns batch norm training off)
    model.eval()

    # build dataloader in eval mode:
    dataloader = dataloader_fn(train=False, batch_size=args.batch_size)

    # turn gradient-tracking off (for speed) during validation:
    validation_losses = []
    with torch.no_grad():
        for inputs,_ in tqdm(dataloader):
            validation_losses.append(loss_fn(model(inputs.to(DEVICE)), model.scaling_diag).item())
    
    # delete dataloader to save memory:
    del dataloader

    # set model back in train mode:
    model.train()

    # return validation loss summary statistics:
    return (np.amin(validation_losses),
            np.median(validation_losses),
            np.mean(validation_losses),
            np.amax(validation_losses))

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    # ----- parse training settings:
    parser = argparse.ArgumentParser(description="Train a fresh NICE model and save.")
    # configuration settings:
    parser.add_argument("--dataset", required=True, dest='dataset', choices=('tfd', 'cifar10', 'svhn', 'mnist'),
                        help="Dataset to train the NICE model on.")
    parser.add_argument("--epochs", dest='num_epochs', default=1500, type=int,
                        help="Number of epochs to train on. [1500]")
    parser.add_argument("--batch_size", dest="batch_size", default=16, type=int,
                        help="Number of examples per batch. [16]")
    parser.add_argument("--savedir", dest='savedir', default="./saved_models",
                        help="Where to save the trained model. [./saved_models]")
    # model settings:
    parser.add_argument("--nonlinearity_layers", dest='nlayers', default=5, type=int,
                        help="Number of layers in the nonlinearity. [5]")
    parser.add_argument("--nonlinearity_hiddens", dest='nhidden', default=1000, type=int,
                        help="Hidden size of inner layers of nonlinearity. [1000]")
    parser.add_argument("--prior", choices=('logistic', 'prior'), default="logistic",
                        help="Prior distribution of latent space components. [logistic]")
    parser.add_argument("--model_path", dest='model_path', default=None, type=str,
                        help="Continue from pretrained model. [None]")
    # optimization settings:
    parser.add_argument("--lr", default=0.001, dest='lr', type=float,
                        help="Learning rate for ADAM optimizer. [0.001]")
    parser.add_argument("--beta1", default=0.9,  dest='beta1', type=float,
                        help="Momentum for ADAM optimizer. [0.9]")
    parser.add_argument("--beta2", default=0.01, dest='beta2', type=float,
                        help="Beta2 for ADAM optimizer. [0.01]")
    parser.add_argument("--eps", default=0.0001, dest='eps', type=float,
                        help="Epsilon for ADAM optimizer. [0.0001]")
    parser.add_argument("--lambda", default=1.0, dest='lmbda', type=float,
                        help="L1 weight decay coefficient. [1.0]")
    args = parser.parse_args()
    # ----- run training loop over several epochs & save models for each epoch:
    model = train(args)
