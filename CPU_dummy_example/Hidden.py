import numpy
import cPickle
import gzip
import os
from utils import *

class Hiddenlayer(object):
  def __init__(self,n_in,n_out):
    self.rng = numpy.random.RandomState(89677)
    self.n_in = n_in
    self.n_out = n_out
    self.W = 4*numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),high=numpy.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)))
    self.b = numpy.zeros((n_out,))
    self.W_size = n_out * n_in
    self.b_size = n_out
    self.W_grad = numpy.zeros((n_in,n_out))
    self.b_grad = numpy.zeros((n_out,))
    #W_v and b_v is acceleration delta
    self.W_v = numpy.zeros((n_in,n_out))
    self.b_v = numpy.zeros((n_out,))
    
    #V_w and V_b are used in R forward operation
    self.V_w = numpy.zeros((n_in,n_out))
    self.V_b = numpy.zeros((n_out,))
    self.Rx = None
    self.Ry = None
    
    #Gv_w and Gv_b is used to store G*v
    self.Gv_w = numpy.zeros((n_in,n_out))
    self.Gv_b = numpy.zeros((n_out,))
    
    self.inputs = None
    self.output = None
    self.dEdx = None
    self.RdEdx = None
    
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
    linear_output=numpy.dot(self.inputs,self.W)+self.b
    self.output = sigmoid(linear_output)
    
  def forward_R(self,Ry_prev,y_prev):
    self.Rx = numpy.dot(Ry_prev,self.W)+numpy.dot(y_prev,self.V_w)+self.V_b
    self.Ry = self.output*(1-self.output)*self.Rx
    
  def backprop_to_param(self):
    self.W_grad = numpy.zeros((self.n_in,self.n_out))
    self.b_grad = numpy.zeros((self.n_out,))
    self.W_grad = (1.0/self.output.shape[0])*numpy.dot(numpy.transpose(self.inputs),self.dEdx)
    self.b_grad = (1.0/self.output.shape[0])*numpy.sum(self.dEdx,axis=0)
    
  def backprop_to_prev(self,hidden_layer):
    hidden_layer.dEdx = numpy.dot(self.dEdx,numpy.transpose(self.W))*self.inputs*(1-self.inputs)
  
  def backprop_to_param_R(self):
    self.Gv_w[:] = numpy.zeros((self.n_in,self.n_out))
    self.Gv_b[:] = numpy.zeros((self.n_out,))
    self.Gv_w[:] =  (1.0/self.output.shape[0])*numpy.dot(numpy.transpose(self.inputs),self.RdEdx)
    self.Gv_b[:] =  (1.0/self.output.shape[0])*numpy.sum(self.RdEdx,axis=0)
  
  def backprop_to_prev_R(self,hidden_layer):
    hidden_layer.RdEdx = numpy.dot(self.RdEdx,numpy.transpose(self.W))*self.inputs*(1-self.inputs)
  
  
  
  