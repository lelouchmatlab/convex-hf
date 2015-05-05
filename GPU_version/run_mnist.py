import numpy, sys
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cPickle
import pickle
import gzip
import os
import time
from hf import hf_optimizer, SequenceDataset
from collections import OrderedDict
from SdA import SdA
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer


def load_mnist(dataset):
  f = gzip.open(dataset, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  return (train_set, valid_set, test_set)

def load_cifar(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict



def sgd_optimizer(p, inputs, costs, train_set,valid_set, acceleration=True, lr_init=0.01,  momentum=0.5, logfile=None):
  '''SGD optimizer with a similar interface to hf_optimizer. Support Acceleration'''
  previous_validation = 1
  lr = lr_init
  v = [theano.shared(value=numpy.zeros_like(i.get_value(borrow=True),dtype=theano.config.floatX)) for i in p]
  g = [T.grad(costs[0], i) for i in p]
  learning_rate = T.fscalar('learning_rate')
  updates = OrderedDict('')
  
  if acceleration==True:
    for (v_ele,g_ele) in zip(v,g):
      updates[v_ele]=momentum * v_ele - learning_rate * g_ele
    for (p_ele, v_ele) in zip(p,v):
      updates[p_ele] = p_ele + v_ele
  else:
    for i,j in zip(p,g):
      updates[i] = i - learning_rate*j

  train_f = theano.function(inputs+[theano.Param(learning_rate, default = 0.01)], costs, updates=updates)
  valid_f = theano.function(inputs, costs)
  try:
    index = 1
    training_obj = []
    training_error = []
    validation_error = []
    training_time = []
    start_time = time.time()
    threshold = 0.02
    for index in xrange(1000):
    #while lr != 0:
      train_cost = []
      valid_cost = []
      for i in train_set.iterate(True):
        train_cost.append(train_f(*i+[lr]))
      for i in valid_set.iterate(False):
        valid_cost.append(valid_f(*i))
      print 'update %i, lr %f, training cost=' %(index,lr), numpy.mean(train_cost, axis=0), 'validation cost=',numpy.mean(valid_cost,axis=0)
      cur_time = time.time()
      current_validation = numpy.mean(valid_cost,axis=0)[1]
      training_obj.append(numpy.mean(train_cost, axis=0)[0])
      training_error.append(numpy.mean(train_cost, axis=0)[1])
      validation_error.append(numpy.mean(valid_cost, axis=0)[1])
      training_time.append(cur_time-start_time)
      
      if training_error[-1] < threshold:
        lr = lr * 0.5
        threshold = threshold * 0.5
      #TO DO: learning rate decay ...  
      #if index>100 and current_validation > previous_validation:
      #  lr = 0
      #  continue
      #diff = abs(current_validation-previous_validation)/previous_validation
      #if index>50 and diff < 0.05:
      #  lr = lr * 0.5
      #previous_validation = current_validation
      #index +=1
      sys.stdout.flush()
    if logfile is not None:
      logf = open(logfile,'w')
      logf.write(str(training_obj)[1:-1]+'\n')
      logf.write(str(training_error)[1:-1]+'\n')
      logf.write(str(validation_error)[1:-1]+'\n')
      logf.write(str(training_time)[1:-1]+'\n')



  except KeyboardInterrupt: 
    if logfile is not None:
      logf = open(logfile,'w')
      logf.write(str(training_obj)[1:-1]+'\n')
      logf.write(str(training_error)[1:-1]+'\n')
      logf.write(str(validation_error)[1:-1]+'\n')
      logf.write(str(training_time)[1:-1]+'\n')
    print 'Training interrupted.'

def softmax(x):
  e_x = T.exp(x - x.max(axis=1, keepdims=True))
  out = e_x / e_x.sum(axis=1, keepdims=True)
  return out

def CNN():
  x = T.matrix('x')
  y = T.ivector('y')
  t = T.cast(y,'int32')
  p = []
  rng = numpy.random.RandomState(23455)
  batch_size = 500
  nkerns=[20, 50]
  
  layer0_input = x.reshape((batch_size, 1, 28, 28))
  layer0 = LeNetConvPoolLayer(rng,input=layer0_input,image_shape=(batch_size, 1, 28, 28),filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
  p.extend(layer0.params)

  layer1 = LeNetConvPoolLayer(rng,input=layer0.output,image_shape=(batch_size, nkerns[0], 12, 12),filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))
  p.extend(layer1.params)

  layer2_input = layer1.output.flatten(2)
  layer2 = HiddenLayer(rng,input=layer2_input,n_in=nkerns[1] * 4 * 4,n_out=500,activation=T.tanh)
  p.extend(layer2.params)

  a = 500
  b = 10
  Wi = theano.shared((10./numpy.sqrt(a+b) * numpy.random.uniform(-1, 1, size=(a, b))).astype(theano.config.floatX))
  bi = theano.shared(numpy.zeros(b, dtype=theano.config.floatX))
  s = T.dot(layer2.output,Wi) + bi
  
  p = p+[Wi, bi]

  p_y_given_x = softmax(s)
  y_pred = T.argmax(p_y_given_x, axis=1)
  c =  -T.mean(T.log(p_y_given_x)[T.arange(t.shape[0]), t])
  acc = T.mean(T.neq(y_pred, t))
  return p, [x, t], s, [c, acc]



# feed-forward neural network with sigmoidal output
def DNN(sizes=(784, 100, 10)):
  x = T.matrix()
  t = T.ivector('y')
  t = T.cast(t,'int32')
  p = []
  y = x
  rng = numpy.random.RandomState(89677)
  with open('Globalmodel.pickle') as f:
    init_param = pickle.load(f)
  index = 0
  for i in xrange(len(sizes)-2):
    a, b = sizes[i:i+2]
    #Wi = theano.shared((init_param[index]).astype(theano.config.floatX))
    Wi = theano.shared((10./numpy.sqrt(a+b) * numpy.random.uniform(-1, 1, size=(a, b))).astype(theano.config.floatX))
    
    #W_value = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (a + b)),
    #    high=numpy.sqrt(6. / (a + b)),size=(a, b)), dtype=theano.config.floatX)
    #Wi = theano.shared((W_value).astype(theano.config.floatX))
    index = index +2
    bi = theano.shared(numpy.zeros(b, dtype=theano.config.floatX))
    p += [Wi, bi]

    s = T.dot(y,Wi) + bi
    y = T.nnet.sigmoid(s)
    
  a = sizes[-2]
  b = sizes[-1]
  #Wi = theano.shared((init_param[index]).astype(theano.config.floatX))
  Wi = theano.shared((10./numpy.sqrt(a+b) * numpy.random.uniform(-1, 1, size=(a, b))).astype(theano.config.floatX))
  #W_value = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (a + b)),
  #    high=numpy.sqrt(6. / (a + b)),size=(a, b)), dtype=theano.config.floatX)
  #Wi = theano.shared((W_value).astype(theano.config.floatX))
  bi = theano.shared(numpy.zeros(b, dtype=theano.config.floatX))
  p += [Wi, bi]
  s = T.dot(y,Wi) + bi
  p_y_given_x = softmax(s)
  y_pred = T.argmax(p_y_given_x, axis=1)
  
  c =  -T.mean(T.log(p_y_given_x)[T.arange(t.shape[0]), t])
  acc = T.mean(T.neq(y_pred, t))
  #c = (-t* T.log(y) - (1-t)* T.log(1-y)).mean()
  #acc = T.neq(T.round(y), t).mean()

  return p, [x, t], s, [c, acc]

#def Mnist_test():
if __name__=="__main__":
  print "Loading data ..." 
  (train_set, valid_set, test_set)=load_mnist('data/mnist.pkl.gz')
  train_dataset = [[], []]
  valid_dataset = [[], []]
  train_dataset[0].append(train_set[0])
  train_dataset[1].append(numpy.array(train_set[1],dtype='int32'))
  #train_dataset[1].append(T.cast(train_set[1],'int32'))
  valid_dataset[0].append(valid_set[0])
  valid_dataset[1].append(numpy.array(valid_set[1],dtype='int32'))
  gradient_dataset=SequenceDataset(train_dataset, batch_size=50000, number_batches=1)
  cg_dataset = SequenceDataset(train_dataset, batch_size=500, number_batches=100)
  valid_dataset = SequenceDataset(valid_dataset, batch_size=500, number_batches=20)
  
  #p, inputs, s, costs = My_simple_NN([784, 512, 512, 512, 10])
  p, inputs, s, costs = DNN([784, 1000, 500, 250, 30, 10])
  #p, inputs, s, costs = CNN()
  #numpy_rng = numpy.random.RandomState(89677)  
  #NN=SdA(numpy_rng=numpy_rng,
  #        n_ins=28 * 28,
  #        hidden_layers_sizes=[1000, 1000, 1000],
  #        n_outs=10)
  #p, inputs, s, costs = NN.model()


  #sgd_optimizer(p, inputs, costs, gradient_dataset, valid_dataset, acceleration=False, lr_init=0.08,  momentum=0.5, logfile='./Log/log_gd_test2')

  dnn = hf_optimizer(p, inputs, s, costs)
  print 'Done building model'
  dnn.train(gradient_dataset, cg_dataset, num_updates=500, initial_lambda=30, preconditioner=False, validation=valid_dataset,logfile='./Log/log_HF_DNN')
  #dnn.train_gd(gradient_dataset, valid_dataset, acceleration=False, lr_init=0.08, momentum=0.5, logfile='./Log/log_gd_noacceleration_0.08')
  #gradient=dnn.get_gradient(gradient_dataset)
  #dnn.mu=0
  #v= numpy.ones(sum(dnn.sizes),dtype=theano.config.floatX)
  #col = dnn.batch_Gv_test(v, cg_dataset, lambda_=0)
  #dnn.cg_dataset = cg_dataset
  #dnn.gradient_dataset = gradient_dataset
  #dnn.lambda_ = 30
  #dnn.preconditioner = False
  #dnn.max_cg_iterations = 250
  #dnn.global_backtracking = False
  #res=dnn.batch_Jv(v)
  #after_cost, flat_delta, backtracking, num_cg_iterations = dnn.cg(-gradient)
  
  #dnn.train(gradient_dataset, cg_dataset, num_updates=500, initial_lambda=30, preconditioner=False, validation=valid_dataset,logfile='./Log/log')
  #mlp=hf_optimizer(p, inputs, s, costs)
  #mlp.train(gradient_dataset, cg_dataset, num_updates=1, initial_lambda=30, preconditioner=False, validation=valid_dataset)
  #d= numpy.ones(sum(mlp.sizes),dtype=theano.config.floatX)
  #col = mlp.batch_Gv(d, lambda_=0)
#if __name__=="__main__":
#  Mnist_test()











