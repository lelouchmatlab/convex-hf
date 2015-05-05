import numpy
import cPickle
import gzip
import os



def sigmoid(X):
  return 1/(1+numpy.exp(-X))
  
def softmax(X):
  Y=numpy.exp(X)
  Normlization=numpy.sum(Y,axis=1)
  for i in xrange(Y.shape[0]):
    Y[i][:]=Y[i][:]/Normlization[i]
  return Y