Nonlinear Independent Components Estimation
===========================================

An implementation of the NICE model from Dinh et al (2014) in PyTorch.

I was only able to find [the original theano-based repo](https://github.com/laurent-dinh/nice) from the first author,
and I figured it would be good practice to re-implement the architecture in PyTorch.

Please cite the paper by the original authors and credit them (not me or this repo) if any of the code in this repo
ends up being useful to you in a publication:

["NICE: Non-linear independent components estimation"](http://arxiv.org/abs/1410.8516), Laurent Dinh, David Krueger, Yoshua Bengio. ArXiv 2014.


Requirements
------------
* PyTorch 0.4.1+
* NumPy 1.14.5+
* tqdm 4.15.0+ (though any version should work --- we primarily just use the main tqdm and trange wrappers.)


Benchmarks
----------
We plan to use the same four datasets as in the original paper (MNIST, TFD, SVHN, and CIFAR-10) and attempt to reproduce the results in the paper. At present, MNIST, SVHN, and CIFAR10 are supported; TFD is a bit harder to get access to (due to privacy issues regarding the faces, etc.)

Running `python make_datasets.py` will download the relevant dataset and store it in the appropriate directory the first time
you run it; subsequent runs will not re-download the datasets if they already exist. Additionally, the ZCA matrices will be
computed for the relevant datasets that require them (CIFAR10, SVHN).

`(TBD: comparisons to original repo & paper results here --- once I find the time to run on 1500 epochs.)`


License
-------
The license for this repository is the 3-clause BSD, as in the theano-based implementation.


Status
------
* Training on MNIST, CIFAR10, SVHN currently work; trained models can be sampled via `python sample.py [--args]`.
* Training on GPU currently works. (Sampling is still CPU-only, but this is by design.)
* Benchmarks are still forthcoming.
* Toronto Face Dataset support is still something I'm considering if I can find a place to download it.

Future To-Do List
-----------------
+ Implement inpainting from trained model.
+ Toronto Face Dataset? (See remark about privacy issues above)
+ Implement affine coupling law
+ Allow arbitrary partitions of the input in coupling layers?
