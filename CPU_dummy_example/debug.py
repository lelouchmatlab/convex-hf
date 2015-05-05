import numpy
import cPickle
import pickle
import gzip
from LR import Logisticlayer
from mlp import MLP

def load_mnist(dataset):
  f = gzip.open(dataset, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  return (train_set, valid_set, test_set)

if __name__=="__main__":
  numpy.set_printoptions(threshold=numpy.nan)
  (train_set, valid_set, test_set)=load_mnist('mnist.pkl.gz')
  train_gradient_X=train_set[0][0:500,:]
  train_gradient_Y=train_set[1][0:500]
  
  train_cg_X=train_gradient_X.copy()
  train_cg_Y=train_gradient_Y.copy()
  
  valid_X=valid_set[0]
  valid_Y=valid_set[1]
  
  mlp = MLP(784,10,[500])
  with open('init_nnet.pickle') as f:
    init_param = pickle.load(f)
  init_param = numpy.concatenate([i.flatten() for i in init_param])
  mlp.packParam(init_param)
  
  
  num_param = numpy.sum(mlp.sizes)
  
  batch_size = 500
  next_init = numpy.zeros((num_param,))
  #grad,train_nll,train_error = mlp.get_gradient(train_gradient_X, train_gradient_Y, batch_size)
  
  d = 1.0*numpy.ones((num_param,))
  col = mlp.get_Gv(train_cg_X, train_cg_Y, batch_size, d)
  print 'Some col:'
  print col[-100:]
  
  
  #delta, next_init, after_cost = mlp.cg(-grad, train_cg_X, train_cg_Y, batch_size, next_init, 1)
  
  
  """
  iters=1000
  batch_size = 500
  n_train_batches = train_gradient_X.shape[0]/batch_size
  
  n_valid_batches = valid_X.shape[0]/batch_size
  
  
  next_init = numpy.zeros((num_param,))
  cg_chunk_size = 10000
  cg_chunk_index = 0
  for i in xrange(iters):
    #numpy.random.seed(18877)
    #numpy.random.shuffle(train_gradient_X)
    #numpy.random.seed(18877)
    #numpy.random.shuffle(train_gradient_Y)
    
    
    if i%5==0:
      cg_chunk_index = 0
      numpy.random.seed(18877)
      numpy.random.shuffle(train_cg_X)
      numpy.random.seed(18877)
      numpy.random.shuffle(train_cg_Y)
      
    train_cg_X_cur = train_cg_X[cg_chunk_index*cg_chunk_size:(cg_chunk_index+1)*cg_chunk_size,:]
    train_cg_Y_cur = train_cg_Y[cg_chunk_index*cg_chunk_size:(cg_chunk_index+1)*cg_chunk_size]
    
    cg_chunk_index = cg_chunk_index+1
    
    nll=[]
    error=[]
    
    print "Iter: %d ..."%(i), "Lambda: %f"%(mlp._lambda)
    
    grad,train_nll,train_error = mlp.get_gradient(train_gradient_X, train_gradient_Y, batch_size)
    
    delta, next_init, after_cost = mlp.cg(-grad, train_cg_X_cur, train_cg_Y_cur, batch_size, next_init, 1)
    
    Gv = mlp.get_Gv(train_cg_X_cur,train_cg_Y_cur,batch_size,delta)
    
    delta_cost = numpy.dot(delta,grad+0.5*Gv)
    
    before_cost = mlp.quick_cost(numpy.zeros((num_param,)), train_cg_X_cur, train_cg_Y_cur, batch_size)
    
    l2norm = numpy.linalg.norm(Gv + mlp._lambda*delta + grad)
    
    print "Residual Norm: ",l2norm
    print 'Before cost: %f, After cost: %f'%(before_cost,after_cost)
    param = mlp.flatParam() + delta
    
    mlp.packParam(param)
    
    tune_lambda = (after_cost - before_cost)/delta_cost
    
    if tune_lambda < 0.25:
      mlp._lambda = mlp._lambda*1.5
    elif tune_lambda > 0.75:
      mlp._lambda = mlp._lambda/1.5

    print "Training   NNL: %f, Error: %f"%(train_nll,train_error)
    nll=[]
    error=[]
    for batch_index in xrange(n_valid_batches):
      X=valid_X[batch_index*batch_size:(batch_index+1)*batch_size,:]
      Y=valid_Y[batch_index*batch_size:(batch_index+1)*batch_size]
      mlp.forward(X)
      nll.append(mlp.Cost(Y))
      error.append(mlp.error(Y))
    print "Validation NNL: %f, Error: %f"%(numpy.mean(nll),numpy.mean(error))
  """
    
