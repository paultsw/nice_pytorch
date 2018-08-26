"""
Sample from latent space of independent gaussians.
"""
import torch
import torchvision
import torch.distributions as dist
from nice.models import NICEModel
from nice.utils import unflatten_images
import argparse


def sample(args):
    """
    Performs the following:
    1. construct model object & load state dict from saved model;
    2. make H x W samples from a set of gaussian or logistic prior on the latent space;
    3. save to disk as a grid of images.
    """
    # parse settings:
    if args.dataset == 'mnist':
        input_dim = 28*28
        img_height = 28
        img_width = 28
        img_depth = 1
    if args.dataset == 'svhn':
        input_dim = 32*32*3
        img_height = 32
        img_width = 32
        img_depth = 3
    if args.dataset == 'cifar10':
        input_dim = 32*32*3
        img_height = 32
        img_width = 32
        img_depth = 3
    if args.dataset == 'tfd':
        raise NotImplementedError("[sample] Toronto Faces Dataset unsupported right now. Sorry!")
        input_dim = None
        img_height = None
        img_width = None
        img_depth = None

    # shut off gradients for sampling:
    torch.set_grad_enabled(False)
    
    # build model & load state dict:
    nice = NICEModel(input_dim, args.nhidden, args.nlayers)
    if args.model_path is not None:
        nice.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    nice.eval()
    
    # sample a batch:
    if args.prior == 'logistic':
        logistic = dist.TransformedDistribution(
            dist.Uniform(0, 1),
            [dist.SigmoidTransform().inv, dist.AffineTransform(loc=0, scale=1)]
        )
        xs = nice.inverse(logistic.sample(torch.Size([args.nrows*args.ncols, input_dim])))
    if args.prior == 'gaussian':
        xs = nice.inverse(torch.randn(args.nrows*args.ncols, input_dim))

    # format sample into images of correct shape:
    image_batch = unflatten_images(xs, img_depth, img_height, img_width)
    
    # arrange into a grid and save to file:
    torchvision.utils.save_image(image_batch, args.save_image_path, nrow=args.nrows)


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample from a trained NICE model.")
    # --- sampling options:
    parser.add_argument("--dataset", dest='dataset', choices=('tfd', 'cifar10', 'svhn', 'mnist'), required=True,
                        help="Which dataset to use; determines height and width")
    parser.add_argument("--prior", choices=('logistic', 'prior'), default="logistic",
                        help="Prior distribution of latent space components. [logistic]")
    parser.add_argument("--nrows", dest='nrows', default=1, type=int,
                        help="Number of rows in grid of output images. [1]")
    parser.add_argument("--ncols", dest='ncols', default=1, type=int,
                        help="Number of columns in grid of output images. [1]")
    parser.add_argument("--save_image", default="./samples.png", dest='save_image_path',
                        help="Where to save the grid of samples. [./samples.png]")
    # --- model options:
    parser.add_argument("--model", dest='model_path', default=None,
                        help="Path to trained model. [None/untrained model]")
    parser.add_argument("--nonlinearity_layers", dest='nlayers', default=5, type=int,
                        help="Number of layers in the nonlinearity. [5]")
    parser.add_argument("--nonlinearity_hiddens", dest='nhidden', default=1000, type=int,
                        help="Hidden size of inner layers of nonlinearity. [1000]")
    args = parser.parse_args()
    sample(args)
