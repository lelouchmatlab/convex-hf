# convex-hf
Sample code for hessian-free optimization on MNIST.

The CPU version is built from python linear algebra package. Simply by running python test.py will start Hessian-free training on a built in model with size (784,512,10) (one hidden layer plus soft-max).

The GPU version is base on Theano and folked from https://github.com/boulanni/theano-hf.git. The main modification is rewritting soft-max operation such that it can be used for classification with Hessian-free training. On a GPU machine with Theano installed, simply run bash run.sh to start Hessian-free training on MNIST.
