import numpy
import cPickle
import gzip
import os
import pickle
from LR import Logisticlayer
from Hidden import Hiddenlayer

class MLP(object):
  def __init__(self,n_in, n_out, hidden_size):
    self.layers=[]
    self.inputs = None
    self.n_hiddenlayers = len(hidden_size)
    self.param = []
    self.V = []
    self.Gv = []
    self._lambda = 30
    for i in xrange(0,len(hidden_size)):
      if i==0:
	layer_n_in = n_in
	layer_n_out = hidden_size[0]
      else:
	layer_n_in = hidden_size[i-1]
	layer_n_out = hidden_size[i]
      cur_layer = Hiddenlayer(layer_n_in,layer_n_out)
      self.param.append(cur_layer.W)
      self.param.append(cur_layer.b)
      self.V.append(cur_layer.V_w)
      self.V.append(cur_layer.V_b)
      self.Gv.append(cur_layer.Gv_w)
      self.Gv.append(cur_layer.Gv_b)
      self.layers.append(cur_layer)
      
    loglayer = Logisticlayer(hidden_size[-1],n_out)
    self.param.append(loglayer.W)
    self.param.append(loglayer.b)
    self.V.append(loglayer.V_w)
    self.V.append(loglayer.V_b)
    self.Gv.append(loglayer.Gv_w)
    self.Gv.append(loglayer.Gv_b)
    self.layers.append(loglayer)
    #with open('debug_nnet.pickle', 'w') as f:
    #  pickle.dump(self.param,f)
    
    self.shapes = [i.shape for i in self.param]
    self.sizes = map(numpy.prod,self.shapes)
    self.positions = numpy.cumsum([0] + self.sizes)[:-1]
  
  
  def quick_cost(self, v, train_cg_X, train_cg_Y, batch_size):
    param = self.flatParam()
    param_temp = param + v
    self.packParam(param_temp)
    n_train_batches = train_cg_X.shape[0]/batch_size
    cost = []
    for batch_index in xrange(n_train_batches):
      X=train_cg_X[batch_index*batch_size:(batch_index+1)*batch_size,:]
      Y=train_cg_Y[batch_index*batch_size:(batch_index+1)*batch_size]
      self.forward(X)
      cost.append(self.layers[-1].NNL(Y))
    self.packParam(param)
    return numpy.mean(cost)
  
  def flatV(self):
    l = []
    for layer in self.layers:
      l.append(layer.V_w.flatten())
      l.append(layer.V_b.flatten())
    return numpy.concatenate(l)
    #return numpy.concatenate([i.flatten() for i in self.V])
    
  def flatGv(self):
    l = []
    for layer in self.layers:
      l.append(layer.Gv_w.flatten())
      l.append(layer.Gv_b.flatten())
    return numpy.concatenate(l)
    #return numpy.concatenate([i.flatten() for i in self.Gv])
  
  def flatParam(self):
    l = []
    for layer in self.layers:
      l.append(layer.W.flatten())
      l.append(layer.b.flatten())
    return numpy.concatenate(l)
    
  def flatGrad(self):
    l = []
    for layer in self.layers:
      l.append(layer.W_grad.flatten())
      l.append(layer.b_grad.flatten())
    return numpy.concatenate(l)
  
  
  def packV(self,vector):
    pos = 0
    for layer in self.layers:
      layer.V_w = vector[pos:pos+layer.W_size].reshape(layer.W.shape)
      pos  = pos + layer.W_size
      layer.V_b = vector[pos:pos+layer.b_size].reshape(layer.b.shape)
      pos = pos +layer.b_size
  
  def packGv(self,vector):
    pos = 0
    for layer in self.layers:
      layer.Gv_w = vector[pos:pos+layer.W_size].reshape(layer.W.shape)
      pos  = pos + layer.W_size
      layer.Gv_b = vector[pos:pos+layer.b_size].reshape(layer.b.shape)
      pos = pos +layer.b_size
    
  def packParam(self,vector):
    pos = 0
    for layer in self.layers:
      layer.W = vector[pos:pos+layer.W_size].reshape(layer.W.shape)
      pos  = pos + layer.W_size
      layer.b = vector[pos:pos+layer.b_size].reshape(layer.b.shape)
      pos = pos +layer.b_size
  
  
  def forward(self,X):
    for i in xrange(0,self.n_hiddenlayers+1):
      if i ==0:
	layer_inputs = X
      else:
	layer_inputs = self.layers[i-1].output
      self.layers[i].forward(layer_inputs)
    self.output = self.layers[-1].output
    
  def forward_R(self):
    for i in xrange(0,self.n_hiddenlayers+1):
      if i==0:
	Ry_prev = numpy.zeros_like(self.layers[i].inputs)
	y_prev = self.layers[i].inputs
      else:
	Ry_prev = self.layers[i-1].Ry
	y_prev = self.layers[i].inputs
      self.layers[i].forward_R(Ry_prev,y_prev)
      
	
  
  def backprop(self,Y):
    #LR layer BP
    self.layers[-1].backprop_to_param(Y)
    self.layers[-1].backprop_to_prev(self.layers[-2])
    
    #All hidden layers except the first hidden layer BP 
    for i in xrange(len(self.layers)-2,0,-1):
      cur_layer = self.layers[i]
      cur_layer.backprop_to_param()
      cur_layer.backprop_to_prev(self.layers[i-1])
    
    #The first hidden layer BP (don't need to BP to prev since it is the first layer)
    self.layers[0].backprop_to_param()
    
  def backprop_R(self):
    self.layers[-1].backprop_to_param_R()
    self.layers[-1].backprop_to_prev_R(self.layers[-2])
    
    #All hidden layers except the first hidden layer BP 
    for i in xrange(len(self.layers)-2,0,-1):
      cur_layer = self.layers[i]
      cur_layer.backprop_to_param_R()
      cur_layer.backprop_to_prev_R(self.layers[i-1])
    
    #The first hidden layer BP (don't need to BP to prev since it is the first layer)
    self.layers[0].backprop_to_param_R()
  
  
  def get_gradient(self,train_grad_X,train_grad_Y,batch_size):
    print "Getting Grad"
    n_train_batches = train_grad_X.shape[0]/batch_size
    nll=[]
    error=[]
    nll.append(0)
    error.append(0)
    grad = numpy.zeros((numpy.sum(self.sizes),))
    for batch_index in xrange(n_train_batches):
      X=train_grad_X[batch_index*batch_size:(batch_index+1)*batch_size,:]
      Y=train_grad_Y[batch_index*batch_size:(batch_index+1)*batch_size]
      self.forward(X)
      nll.append(self.Cost(Y))
      error.append(self.error(Y))
      self.backprop(Y)
      grad = grad + self.flatGrad()
    grad = grad / n_train_batches
    
    return grad,numpy.mean(nll),numpy.mean(error)
    
  def get_Gv(self,train_cg_X,train_cg_Y,batch_size,vector):
    self.packV(vector)
    n_train_batches = train_cg_X.shape[0]/batch_size
    Gv = numpy.zeros((numpy.sum(self.sizes),))
    """
    H = numpy.zeros
    for batch_index in xrange(n_train_batches):
      X=train_cg_X[batch_index*batch_size:(batch_index+1)*batch_size,:]
      Y=train_cg_Y[batch_index*batch_size:(batch_index+1)*batch_size]
      self.forward(X)
    """  
    print "Getting GV, total number of batches:%d, batch size: %d"%(n_train_batches,batch_size)
    for batch_index in xrange(n_train_batches):
      X=train_cg_X[batch_index*batch_size:(batch_index+1)*batch_size,:]
      Y=train_cg_Y[batch_index*batch_size:(batch_index+1)*batch_size]
      self.forward(X)
      self.backprop(Y)
      self.forward_R()
      self.backprop_R()
      Gv = Gv + self.flatGv()
    Gv = Gv / n_train_batches
    return Gv
  
  def cg(self,b ,train_cg_X, train_cg_Y, batch_size, v_init, M):
    print "Start CG"
    max_iter = 250
    v = v_init + 0.0
    r = b - (self.get_Gv(train_cg_X, train_cg_Y, batch_size, v)+ self._lambda*v)
    
    
    d = M*r
    delta_new = numpy.dot(r,d)
    phi = []
    backtracking = []
    #print "Init d: "
    #print d[-100:]
    
    
    for i in xrange(1,1+max_iter):
      Ad = (self.get_Gv(train_cg_X, train_cg_Y, batch_size, d)+ self._lambda*d)
      #print '\nGd'
      #print Ad[-100:],'\n'
      dAd = numpy.dot(d,Ad)
      alpha = delta_new/dAd
      v = v + alpha*d
      r = r - alpha*Ad
      Mr_next = M*r
      delta_old = delta_new
      delta_new = numpy.dot(r,Mr_next)
      beta = delta_new/delta_old
      d = Mr_next + beta*d
      
      quick_cost = self.quick_cost(v,train_cg_X,train_cg_Y,batch_size)
      
      if i>=int(numpy.ceil(1.3**len(backtracking))):
	backtracking.append((quick_cost,v.copy(),i))
      
      phi_i = -0.5*numpy.dot(v,r+b)
      phi.append(phi_i)
      print 'CG iter %d,alpha = %f, phi = %+.5f, quick_cost = %+.5f'%(i,alpha,phi_i,quick_cost)
      
      k = max(10,i/10)
      if i>k and phi_i <0 and (phi_i - phi[-k-1])/phi_i < k*0.0005:
	break
      
    j = len(backtracking) - 1
    while j > 0 and backtracking[j-1][0] < backtracking[j][0]:
      print '###',backtracking[j-1][0], backtracking[j][0]
      j = j-1
    print "CG backtracking:",backtracking[j][2],"out of",i
    return (backtracking[j][1],backtracking[-1][1],backtracking[j][0])
	
  def update(self,lr,momentum):
    for layer in self.layers:
      layer.update(lr,momentum)
      
  
  def DebugForward(self):
    for layer in self.layers:
      print "Layer\n",layer.output
  def DebufBackward(self):
    for layer in self.layers:
      print "Layer\n",layer.W_grad
  def Cost(self,Y):
    return self.layers[-1].NNL(Y)
  def error(self,Y):
    return self.layers[-1].error(Y)

