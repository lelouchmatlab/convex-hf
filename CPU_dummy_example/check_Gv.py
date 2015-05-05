import numpy
import cPickle
import pickle
import gzip
from LR import Logisticlayer
from mlp import MLP

if __name__=="__main__":
  numpy.set_printoptions(threshold=numpy.nan)
  input_dim = 4
  output_dim = 3
  sample_size = 100
  #X=numpy.random.normal(0,1,(sample_size,input_dim))
  #temp,Y=numpy.nonzero(numpy.random.multinomial(1,[1.0/output_dim]*output_dim,size=sample_size))
 
  mlp = MLP(4,3,[10,10])
  with open('debug_nnet.pickle') as f:
    init_param = pickle.load(f)
  init_param = numpy.concatenate([i.flatten() for i in init_param])
  mlp.packParam(init_param)
  
  with open('debug_data.pickle') as f:
    data = pickle.load(f)
  X = data[0]
  Y = data[1]
  
  with open('HJv.pickle') as f:
    HJv_theano = pickle.load(f)
  num_param = numpy.sum(mlp.sizes)
  batch_size = 100
  
  grad,train_nll,train_error=mlp.get_gradient(X,Y,batch_size)
  
  
  d = 1.0*numpy.ones((num_param,))
  col = mlp.get_Gv(X, Y, batch_size, d)
  #print 'Some col:'
  #print col
  
  """  
  grad,train_nll,train_error=mlp.get_gradient(X,Y,2)
  
  v=numpy.zeros(num_param)
  mlp.forward(X)
  O = mlp.layers[-1].output
  S = mlp.layers[-1].linear_output
  #nll.append(mlp.Cost(Y))
  #error.append(mlp.error(Y))
  mlp.backprop(Y)
  
  G = numpy.zeros((num_param,num_param))
  for i in xrange(num_param):
    v[i]=1
    mlp.packV(v)
    mlp.forward_R()
    mlp.backprop_R()
    col = mlp.flatGv()
    G[:,i]=col
    v[i] = 0
  
  
  v=numpy.zeros(num_param)
  G2 = numpy.zeros((num_param,num_param))
  for i in xrange(num_param):
    v[i]=1
    col = mlp.get_Gv(X,Y,100,v)
    G2[:,i]=col
    v[i] = 0
  
  
  #res, next_init = mlp.cg(-grad,X,Y,2,v_init,1)
  
  
  
  param = mlp.flatParam()
  G_numerical = numpy.zeros((num_param,num_param))
  epsilon = 1e-6
  for index in xrange(0,sample_size):
    JM = numpy.zeros((output_dim,num_param))
    s_index = S[index,:]
    o_index = O[index,:]
    o_index_matrix = numpy.matrix(o_index)
    for i in xrange(num_param):
      param[i] = param[i] + epsilon
      mlp.packParam(param)
      mlp.forward(numpy.matrix(X[index,:]))
      sp = mlp.layers[-1].linear_output
      diff = sp - s_index
      JM[:,i] = diff/epsilon
      H = -numpy.dot(o_index_matrix.T,o_index_matrix)+numpy.diag(o_index)
      param[i] = param[i] - epsilon
    G_numerical = G_numerical + numpy.dot(JM.T,numpy.dot(H,JM))
  G_numerical = G_numerical/sample_size
  """
  
  
    
    
    
  
  