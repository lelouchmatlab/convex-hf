import numpy
import cPickle
import gzip
import os
from utils import *

class Logisticlayer(object):
  def __init__(self,n_in,n_out):
    self.n_in = n_in
    self.n_out = n_out
    self.inputs = None
    self.rng = numpy.random.RandomState(89677)
    self.W = numpy.random.normal(0.0,0.01,(n_in,n_out))
    self.b = numpy.zeros((n_out,))
    self.W_size = n_out * n_in
    self.b_size = n_out
    self.W_grad = numpy.zeros((n_in,n_out))
    self.b_grad = numpy.zeros((n_out,))
    self.W_v = numpy.zeros((n_in,n_out))
    self.b_v = numpy.zeros((n_out,))
    
    #V_w and V_b are used in R forward operation
    self.V_w = numpy.zeros((n_in,n_out))
    self.V_b = numpy.zeros((n_out,))
    
    self.H = numpy.zeros((n_out,n_out))
    self.Rx = None
    self.Ry = None
    
    self.Gv_w = numpy.zeros((n_in,n_out))
    self.Gv_b = numpy.zeros((n_out,))
    
    self.dEdx = None
    self.RdEdx_pseudo = None
    self.output = None
    self.linear_output = None
  def update(self,lr,momentum):
    if momentum==0:
      self.W = self.W - lr*self.W_grad
      self.b = self.b - lr*self.b_grad
    else:
      self.W_v = momentum*self.W_v - lr*self.W_grad
      self.b_v = momentum*self.b_v - lr*self.b_grad
      self.W = self.W + self.W_v
      self.b = self.b + self.b_v
  def forward(self,X):
    self.inputs = X
    self.linear_output=numpy.dot(self.inputs,self.W)+self.b
    self.output = softmax(self.linear_output)
   
  def forward_R(self,Ry_prev,y_prev):
    self.Rx = numpy.dot(Ry_prev,self.W)+numpy.dot(y_prev,self.V_w)+self.V_b
    temp = numpy.sum(self.Rx*self.output,axis=1)
    temp2 = self.output.copy()
    for i in xrange(self.output.shape[0]):
      temp2[i,:] = temp2[i,:]*temp[i]
    
    self.Ry = self.Rx*self.output - temp2
    #self.Ry = self.output*(1-self.output)*self.Rx
    
  def NNL(self,Y):
    prob = self.output[numpy.arange(0,Y.shape[0]),Y]
    return -numpy.mean(numpy.log(prob))
  def error(self,Y):
    res = numpy.nonzero(numpy.argmax(self.output,axis=1)-Y)[0]
    return (res.shape[0]+0.0)/Y.shape[0]
  
  def backprop_to_param(self,Y):
    self.W_grad = numpy.zeros((self.n_in,self.n_out))
    self.b_grad = numpy.zeros((self.n_out,))
    self.dEdx = self.output.copy()
    for sample_index in xrange(self.output.shape[0]):
      self.dEdx[sample_index][Y[sample_index]]=self.dEdx[sample_index][Y[sample_index]]-1
    #W_grad = (input' * dEdx)/N   b_grad = sum(dEdx,axis=0)/N 
    self.W_grad =  (1.0/self.output.shape[0])*numpy.dot(numpy.transpose(self.inputs),self.dEdx)
    self.b_grad =  (1.0/self.output.shape[0])*numpy.sum(self.dEdx,axis=0)
  def backprop_to_prev(self, hidden_layer):
    hidden_layer.dEdx = numpy.dot(self.dEdx,numpy.transpose(self.W))*self.inputs*(1-self.inputs)

  def backprop_to_param_R(self):
    self.RdEdx_pseudo = numpy.zeros((self.output.shape[0],self.n_out))
    for i in xrange(0,self.output.shape[0]):
      y_i = numpy.matrix(self.output[i])
      Rx_i = self.Rx[i]
      HJv = numpy.dot((-numpy.dot(y_i.T,y_i)+numpy.diag(self.output[i])),Rx_i)
      #print numpy.array(HJv).shape
      #print numpy.matrix(self.linear_output[i]).shape
      #HJv = numpy.multiply(HJv, numpy.matrix(self.linear_output[i]))
      self.RdEdx_pseudo[i] = HJv
      #self.RdEdx_pseudo[i] = -self.Ry[i]
    
    self.Gv_w[:] = numpy.zeros((self.n_in,self.n_out))
    self.Gv_b[:] = numpy.zeros((self.n_out,))
    self.Gv_w[:] =  (1.0/self.output.shape[0])*numpy.dot(numpy.transpose(self.inputs),self.RdEdx_pseudo)
    self.Gv_b[:] =  (1.0/self.output.shape[0])*numpy.sum(self.RdEdx_pseudo,axis=0)
  
  def backprop_to_prev_R(self,hidden_layer):
    hidden_layer.RdEdx = numpy.dot(self.RdEdx_pseudo,numpy.transpose(self.W))*self.inputs*(1-self.inputs)
    
      
    
    
    
    



