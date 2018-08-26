"""
Download all datasets and compute preprocessing structures (whitening matrices, etc).
"""
import torch
import torchvision
from nice.utils import rescale


def zca_matrix(data_tensor):
    """
    Helper function: compute ZCA whitening matrix across a dataset ~ (N, C, H, W).
    """
    # 1. flatten dataset:
    X = data_tensor.view(data_tensor.shape[0], -1)
    
    # 2. zero-center the matrix:
    X = rescale(X, -1., 1.)
    
    # 3. compute covariances:
    cov = torch.t(X) @ X

    # 4. compute ZCA(X) == U @ (diag(1/S)) @ torch.t(V) where U, S, V = SVD(cov):
    U, S, V = torch.svd(cov)
    return (U @ torch.diag(torch.reciprocal(S)) @ torch.t(V))


def main():
    ### download training datasets:
    print("Downloading CIFAR10...")
    cifar10 = torchvision.datasets.CIFAR10(root="./datasets/cifar", train=True,
                                           transform=torchvision.transforms.ToTensor(), download=True)
    print("Downloading SVHN...")
    svhn = torchvision.datasets.SVHN(root="./datasets/svhn", split='train',
                                     transform=torchvision.transforms.ToTensor(), download=True)
    print("Downloading MNIST...")
    mnist = torchvision.datasets.MNIST(root="./datasets/mnist", train=True,
                                       transform=torchvision.transforms.ToTensor(), download=True)

    ### save ZCA whitening matrices:
    print("Computing CIFAR10 ZCA matrix...")
    torch.save(zca_matrix(torch.cat([x for (x,_) in cifar10], dim=0)), "./datasets/cifar/zca_matrix.pt")
    print("Computing SVHN ZCA matrix...")
    torch.save(zca_matrix(torch.cat([x for (x,_) in svhn], dim=0)), "./datasets/svhn/zca_matrix.pt")

    print("...All done.")

if __name__ == '__main__':
    main()
