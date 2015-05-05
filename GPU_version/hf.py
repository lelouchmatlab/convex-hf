# Author: Nicolas Boulanger-Lewandowski
# University of Montreal, 2012-2013


import numpy, sys
import theano
import theano.tensor as T
import cPickle
import time
import os



def gauss_newton_product(cost, p, v, s):  # this computes the product Gv = J'HJv (G is the Gauss-Newton matrix)
  Jv = T.Rop(s, p, v)
  HJv = T.grad(T.sum(T.grad(cost, s)*Jv), s, consider_constant=[Jv], disconnected_inputs='ignore')
  Gv = T.grad(T.sum(HJv*s), p, consider_constant=[HJv, Jv], disconnected_inputs='ignore')
  Gv = map(T.as_tensor_variable, Gv)  # for CudaNdarray
  #HJv = map(T.as_tensor_variable, HJv)
  return Gv

def Rs(cost, p, v, s):
  Jv = T.Rop(s, p, v)
  HJv = T.grad(T.sum(T.grad(cost, s)*Jv), s, consider_constant=[Jv], disconnected_inputs='ignore')
  Jv = map(T.as_tensor_variable, HJv[-1])
  return Jv


class hf_optimizer:
  '''Black-box Theano-based Hessian-free optimizer.
See (Martens, ICML 2010) and (Martens & Sutskever, ICML 2011) for details.

Useful functions:
__init__ :
    Compiles necessary Theano functions from symbolic expressions.
train :
    Performs HF optimization following the above references.'''

  def __init__(self, p, inputs, s, costs, h=None, ha=None):
    '''Constructs and compiles the necessary Theano functions.

  p : list of Theano shared variables
      Parameters of the model to be optimized.
  inputs : list of Theano variables
      Symbolic variables that are inputs to your graph (they should also
      include your model 'output'). Your training examples must fit these.
  s : Theano variable
    Symbolic variable with respect to which the Hessian of the objective is
    positive-definite, implicitly defining the Gauss-Newton matrix. Typically,
    it is the activation of the output layer.
  costs : list of Theano variables
      Monitoring costs, the first of which will be the optimized objective.
  h: Theano variable or None
      Structural damping is applied to this variable (typically the hidden units
      of an RNN).
  ha: Theano variable or None
    Symbolic variable that implicitly defines the Gauss-Newton matrix for the
    structural damping term (typically the activation of the hidden layer). If
    None, it will be set to `h`.'''

    self.p = p
    self.shapes = [i.get_value().shape for i in p]
    self.sizes = map(numpy.prod, self.shapes)
    self.positions = numpy.cumsum([0] + self.sizes)[:-1]
    print self.shapes
    print self.sizes
    print self.positions
    sys.stdout.flush()

    g = T.grad(costs[0], p)
    g = map(T.as_tensor_variable, g)  # for CudaNdarray
    self.Gmap = g
    print 'building functions'
    self.f_gc = theano.function(inputs, g + costs, on_unused_input='ignore')  # during gradient computation
    self.f_cost = theano.function(inputs, costs, on_unused_input='ignore')  # for quick cost evaluation
    

    g_s = T.grad(costs[0], s)
    print '###'
    self.Grad_s = theano.function(inputs, g_s, on_unused_input='ignore')
    symbolic_types = T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4

    v = [symbolic_types[len(i)]() for i in self.shapes]
    print 'Getting GN'
    Gv = gauss_newton_product(costs[0], p, v, s)
    sys.stdout.flush()
    self.Jv = T.Rop(s, p, v)
    H = T.grad(T.sum(T.grad(costs[0], s)), s, disconnected_inputs='ignore')
    self.getH = theano.function(inputs, H, on_unused_input='ignore')
    self.HJv = T.grad(T.sum(T.grad(costs[0], s)*self.Jv), s, consider_constant=[self.Jv], disconnected_inputs='ignore')
    self.Gv = T.grad(T.sum(self.HJv*s), p, consider_constant=[self.HJv, self.Jv], disconnected_inputs='ignore')
    #Jv = map(T.as_tensor_variable, Jv)
    self.jv = theano.function(inputs + v, self.Jv, on_unused_input='ignore')
    self.hjv = theano.function(inputs + v, self.HJv, on_unused_input='ignore')
    
    gs_Jv = T.grad(costs[0], s)*self.Jv

    self.gsjv = theano.function(inputs + v, gs_Jv, on_unused_input='ignore')
    
    
    
    
    
    coefficient = T.scalar()  # this is lambda*mu    
    if h is not None:  # structural damping with cross-entropy
      h_constant = symbolic_types[h.ndim]()  # T.Rop does not support `consider_constant` yet, so use `givens`
      structural_damping = coefficient * (-h_constant*T.log(h + 1e-10) - (1-h_constant)*T.log((1-h) + 1e-10)).sum() / h.shape[0]
      if ha is None: ha = h
      Gv_damping = gauss_newton_product(structural_damping, p, v, ha)
      Gv = [a + b for a, b in zip(Gv, Gv_damping)]
      givens = {h_constant: h}
    else:
      givens = {}

    self.function_Gv = theano.function(inputs + v + [coefficient], Gv, givens=givens,
                                       on_unused_input='ignore')

  def quick_cost(self, delta=0):
    # quickly evaluate objective (costs[0]) over the CG batch
    # for `current params` + delta
    # delta can be a flat vector or a list (else it is not used)
    if isinstance(delta, numpy.ndarray):
      delta = self.flat_to_list(delta)

    if type(delta) in (list, tuple):
      for i, d in zip(self.p, delta):
        i.set_value(i.get_value() + d)

    cost = numpy.mean([self.f_cost(*i)[0] for i in self.cg_dataset.iterate(update=False)])

    if type(delta) in (list, tuple):
      for i, d in zip(self.p, delta):
        i.set_value(i.get_value() - d)

    return cost


  def get_gradient(self,gradient_dataset):
    result = numpy.zeros(sum(self.sizes), dtype=theano.config.floatX)
    cost = []
    for inputs in gradient_dataset.iterate(update=False):
      temp = self.f_gc(*inputs)
      result = result + self.list_to_flat(temp[:len(self.p)])/ gradient_dataset.number_batches
      cost.append(temp[len(self.p):])    
    return result,numpy.mean(cost,axis=0)

  def train_gd(self,gradient_dataset,validation_dataset,acceleration=True, lr_init=0.01,  momentum=0.5, logfile=None):
    v = [theano.shared(value=numpy.zeros_like(i.get_value(),dtype=theano.config.floatX)) for i in self.p]
    learning_rate = lr_init
    training_obj = []
    training_error = []
    validation_error = []
    training_time = []
    start_time = time.time()
    threshold = 0.1
    try:
      for index in xrange(2000):
        print 'Iter %d ...'%(index+1)
        grad,cost = self.get_gradient(gradient_dataset)
        grad = self.flat_to_list(grad)
        if acceleration == True:
          for (v_ele,g_ele) in zip(v,grad):
            v_ele.set_value(momentum * v_ele.get_value() - learning_rate * g_ele)
          for (p_ele, v_ele) in zip(self.p,v):
            p_ele.set_value(p_ele.get_value() + v_ele.get_value())
        else:
          for p_ele,g_ele in zip(self.p,grad):
            p_ele.set_value(p_ele.get_value() - learning_rate * g_ele)
        training_obj.append(cost[0])
        training_error.append(cost[1])
        validation_error.append(numpy.mean([self.f_cost(*i)[1] for i in validation_dataset.iterate(update=False)]))
        print 'Training obj: %f, training error: %f, and validation error: %f'%(training_obj[-1],training_error[-1],validation_error[-1])
        cur_time = time.time()
        training_time.append(cur_time-start_time)
        if index > 1 and training_obj[-1] > training_obj[-2]:
          learning_rate = learning_rate * 0.8
          #threshold = threshold * 0.5
      if logfile is not None:
        logf = open(logfile,'w')
        logf.write(str(training_obj)[1:-1]+'\n')
        logf.write(str(training_error)[1:-1]+'\n')
        logf.write(str(validation_error)[1:-1]+'\n')
        logf.write(str(training_time)[1:-1]+'\n')
        logf.close()

    except KeyboardInterrupt:
      if logfile is not None:
        logf = open(logfile,'w')
        logf.write(str(training_obj)[1:-1]+'\n')
        logf.write(str(training_error)[1:-1]+'\n')
        logf.write(str(validation_error)[1:-1]+'\n')
        logf.write(str(training_time)[1:-1]+'\n')
        logf.close()
      print 'Training interrupted.'



  def cg(self, b):
    print 'Start CG ...'
    if self.preconditioner:
      M = self.lambda_ * numpy.ones_like(b)
      for inputs in self.cg_dataset.iterate(update=False):
        M += self.list_to_flat(self.f_gc(*inputs)[:len(self.p)])**2  #/ self.cg_dataset.number_batches**2
      #print 'precond~%.3f,' % (M - self.lambda_).mean(),
      M **= -0.75  # actually 1/M
      sys.stdout.flush()
    else:
      M = 1.0

    x = self.cg_last_x if hasattr(self, 'cg_last_x') else numpy.zeros_like(b)  # sharing information between CG runs
    #print 'Init v0:'
    #print x[-100:]
    r = b - self.batch_Gv(x)
    #print 'Init residual:'
    #print r[-100:]
    sys.stdout.flush()
    d = M*r
    delta_new = numpy.dot(r, d)
    phi = []
    backtracking = []
    backspaces = 0

    for i in xrange(1, 1 + self.max_cg_iterations):
      # adapted from http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf (p.51)
      q = self.batch_Gv(d)
      #print 'Get Gv q:'
      #print q[-100:]
      dq = numpy.dot(d, q)
      #assert dq > 0, 'negative curvature'
      alpha = delta_new / dq
      print 'delta_new and dq: ',delta_new, dq
      x = x + alpha*d
      r = r - alpha*q
      s = M*r
      delta_old = delta_new
      delta_new = numpy.dot(r, s)
      d = s + (delta_new / delta_old) * d

      if i >= int(numpy.ceil(1.3**len(backtracking))):
        backtracking.append((self.quick_cost(x), x.copy(), i))

      phi_i = -0.5 * numpy.dot(x, r + b)
      
      phi.append(phi_i)
 
      progress = ' [CG iter %i, ahpha= %f, phi=%+.5f, cost=%.5f]\n' % (i, alpha, phi_i, backtracking[-1][0])
      #sys.stdout.write('\b'*backspaces + progress)
      sys.stdout.write(progress)
      sys.stdout.flush()
      backspaces = len(progress)

      k = max(10, i/10)
      if i > k and phi_i < 0 and (phi_i - phi[-k-1]) / phi_i < k*0.0005:
        break

    self.cg_last_x = x.copy()

    if self.global_backtracking:
      j = numpy.argmin([b[0] for b in backtracking])
    else:
      j = len(backtracking) - 1
      while j > 0 and backtracking[j-1][0] < backtracking[j][0]:
        j -= 1
    print ' backtracked %i/%i' % (backtracking[j][2], i),
    sys.stdout.flush()

    return backtracking[j] + (i,)

  def flat_to_list(self, vector):
    return [vector[position:position + size].reshape(shape) for shape, size, position in zip(self.shapes, self.sizes, self.positions)]

  def list_to_flat(self, l):
    return numpy.concatenate([i.flatten() for i in l])

  def batch_Jv(self,vector):
    v = self.flat_to_list(vector)
    res = []
    for inputs in self.cg_dataset.iterate(False):
      Jv = self.jv(*(inputs + v))
      res.append(Jv)
    return res
  
  def grad_s(self):
    res = []  
    for inputs in self.cg_dataset.iterate(False):
      res.append(self.Grad_s(*(inputs)))
    return res
  

  def batch_gsjv(self,vector):
    v = self.flat_to_list(vector)
    res = []
    for inputs in self.cg_dataset.iterate(False):
      res.append(self.gsjv(*(inputs + v)))
    return res

  def batch_HJv(self,vector):
    v = self.flat_to_list(vector)
    res = []
    for inputs in self.cg_dataset.iterate(False):
      HJv = self.hjv(*(inputs + v))
      res.append(HJv)
    return res

  def batch_Gv(self, vector, lambda_=None):
    v = self.flat_to_list(vector)
    if lambda_ is None: lambda_ = self.lambda_
    result = lambda_*vector  # Tikhonov damping
    for inputs in self.cg_dataset.iterate(False):
      Gv = self.function_Gv(*(inputs + v + [lambda_*self.mu]))
      result += self.list_to_flat(Gv)/self.cg_dataset.number_batches
    return result
      #result += self.list_to_flat(self.function_Gv(*(inputs + v + [lambda_*self.mu]))) / self.cg_dataset.number_batches
    #return result
  def batch_Gv_test(self,vector,cg_dataset,lambda_=None):
      v = self.flat_to_list(vector)
      if lambda_ is None: lambda_ = self.lambda_
      result = lambda_*vector
      for inputs in cg_dataset.iterate(False):
          Gv = self.function_Gv(*(inputs + v + [lambda_*self.mu]))
          result += self.list_to_flat(Gv)/cg_dataset.number_batches
      return result


  def train(self, gradient_dataset, cg_dataset, initial_lambda=0.1, mu=0.03, global_backtracking=False, preconditioner=False, max_cg_iterations=250, num_updates=100, validation=None, logfile=None, validation_frequency=1, patience=numpy.inf, save_progress=None):
    '''Performs HF training.

  gradient_dataset : SequenceDataset-like object
      Defines batches used to compute the gradient.
      The `iterate(update=True)` method should yield shuffled training examples
      (tuples of variables matching your graph inputs).
      The same examples MUST be returned between multiple calls to iterator(),
      unless update is True, in which case the next batch should be different.
  cg_dataset : SequenceDataset-like object
      Defines batches used to compute CG iterations.
  initial_lambda : float
      Initial value of the Tikhonov damping coefficient.
  mu : float
      Coefficient for structural damping.
  global_backtracking : Boolean
      If True, backtracks as much as necessary to find the global minimum among
      all CG iterates. Else, Martens' heuristic is used.
  preconditioner : Boolean
      Whether to use Martens' preconditioner.
  max_cg_iterations : int
      CG stops after this many iterations regardless of the stopping criterion.
  num_updates : int
      Training stops after this many parameter updates regardless of `patience`.
  validation: SequenceDataset object, (lambda : tuple) callback, or None
      If a SequenceDataset object is provided, the training monitoring costs
      will be evaluated on that validation dataset.
      If a callback is provided, it should return a list of validation costs
      for monitoring, the first of which is also used for early stopping.
      If None, no early stopping nor validation monitoring is performed.
  validation_frequency: int
      Validation is performed every `validation_frequency` updates.
  patience: int
      Training stops after `patience` updates without improvement in validation
      cost.
  save_progress: string or None
      A checkpoint is automatically saved at this location after each update.
      Call the `train` function again with the same parameters to resume
      training.'''

    self.lambda_ = initial_lambda
    self.mu = mu
    self.global_backtracking = global_backtracking
    self.cg_dataset = cg_dataset
    self.preconditioner = preconditioner
    self.max_cg_iterations = max_cg_iterations
    best = [0, numpy.inf, None]  # iteration, cost, params
    first_iteration = 1
    

    #gradient = numpy.zeros(sum(self.sizes), dtype=theano.config.floatX)
    #for inputs in gradient_dataset.iterate(update=True):
    #  result = self.f_gc(*inputs)
    #  gradient += self.list_to_flat(result[:len(self.p)]) / gradient_dataset.number_batches
    #print gradient

    if isinstance(save_progress, str) and os.path.isfile(save_progress):
      save = cPickle.load(file(save_progress))
      self.cg_last_x, best, self.lambda_, first_iteration, init_p = save
      first_iteration += 1
      for i, j in zip(self.p, init_p): i.set_value(j)
      print '* recovered saved model'
    
    try:
      #d= numpy.ones(sum(self.sizes),dtype=theano.config.floatX)
      #Gv = self.batch_Gv(d, lambda_=0)
      #print Gv[-200:]

      training_obj = []
      training_error = []
      validation_error = []
      cg_iter = []
      training_time = []
      start_time = time.time()
      for u in xrange(first_iteration, 1 + num_updates):
        print '\nupdate %i/%i,' % (u, num_updates),
        sys.stdout.flush()

        gradient = numpy.zeros(sum(self.sizes), dtype=theano.config.floatX)
        costs = []
        for inputs in gradient_dataset.iterate(update=True):
          result = self.f_gc(*inputs)
          gradient += self.list_to_flat(result[:len(self.p)]) / gradient_dataset.number_batches
          costs.append(result[len(self.p):])
        #print "Gradient:"
        #print gradient[-100:]
        print 'cost=', numpy.mean(costs, axis=0),
        print 'lambda=%.5f,' % self.lambda_,
        training_obj.append(numpy.mean(costs, axis=0)[0])
        training_error.append(numpy.mean(costs, axis=0)[1])
        sys.stdout.flush()

        after_cost, flat_delta, backtracking, num_cg_iterations = self.cg(-gradient)
        delta_cost = numpy.dot(flat_delta, gradient + 0.5*self.batch_Gv(flat_delta, lambda_=0))
        before_cost = self.quick_cost()
        for i, delta in zip(self.p, self.flat_to_list(flat_delta)):
          i.set_value(i.get_value() + delta)
        cg_dataset.update()
         
        cg_iter.append(num_cg_iterations) 
        rho = (after_cost - before_cost) / delta_cost
        if rho < 0.25:
          self.lambda_ *= 1.5
        elif rho > 0.75:
          self.lambda_ /= 1.5
        
        if validation is not None and u % validation_frequency == 0:
          if hasattr(validation, 'iterate'):
            costs = numpy.mean([self.f_cost(*i) for i in validation.iterate()], axis=0)
          elif callable(validation):
            costs = validation()
          print 'validation=', costs,
          validation_error.append(costs[1])
          if costs[0] < best[1]:
            best = u, costs[0], [i.get_value().copy() for i in self.p]
            print '*NEW BEST',

        if isinstance(save_progress, str):
          # do not save dataset states
          save = self.cg_last_x, best, self.lambda_, u, [i.get_value().copy() for i in self.p]
          cPickle.dump(save, file(save_progress, 'wb'), cPickle.HIGHEST_PROTOCOL)
        
        if u - best[0] > patience:
          print 'PATIENCE ELAPSED, BAILING OUT'
          break
        cur_time = time.time()
        training_time.append(cur_time-start_time)
        print
        sys.stdout.flush()
      if logfile is not None:
        logf = open(logfile,'w')
        logf.write(str(training_obj)[1:-1]+'\n')
        logf.write(str(training_error)[1:-1]+'\n')
        logf.write(str(validation_error)[1:-1]+'\n')
        logf.write(str(cg_iter)[1:-1]+'\n')
        logf.write(str(training_time)[1:-1]+'\n')
        logf.close()
    except KeyboardInterrupt:
      if logfile is not None:
        logf = open(logfile,'w')
        logf.write(str(training_obj)[1:-1]+'\n')
        logf.write(str(training_error)[1:-1]+'\n')
        logf.write(str(validation_error)[1:-1]+'\n')
        logf.write(str(cg_iter)[1:-1]+'\n')
        logf.write(str(training_time)[1:-1]+'\n')
        logf.close()
      print 'Interrupted by user.'
    
    if best[2] is None:
      best[2] = [i.get_value().copy() for i in self.p]
    return best[2]
      

class SequenceDataset:
  '''Slices, shuffles and manages a small dataset for the HF optimizer.'''

  def __init__(self, data, batch_size, number_batches, minimum_size=10):
    '''SequenceDataset __init__

  data : list of lists of numpy arrays
    Your dataset will be provided as a list (one list for each graph input) of
    variable-length tensors that will be used as mini-batches. Typically, each
    tensor is a sequence or a set of examples.
  batch_size : int or None
    If an int, the mini-batches will be further split in chunks of length
    `batch_size`. This is useful for slicing subsequences or provide the full
    dataset in a single tensor to be split here. All tensors in `data` must
    then have the same leading dimension.
  number_batches : int
    Number of mini-batches over which you iterate to compute a gradient or
    Gauss-Newton matrix product.
  minimum_size : int
    Reject all mini-batches that end up smaller than this length.'''
    self.current_batch = 0
    self.number_batches = number_batches
    self.items = []

    for i_sequence in xrange(len(data[0])):
      if batch_size is None:
        self.items.append([data[i][i_sequence] for i in xrange(len(data))])
      else:
        for i_step in xrange(0, len(data[0][i_sequence]) - minimum_size + 1, batch_size):
          self.items.append([data[i][i_sequence][i_step:i_step + batch_size] for i in xrange(len(data))])
          
    self.shuffle()
  
  def shuffle(self):
    numpy.random.shuffle(self.items)

  def iterate(self, update=True):
    for b in xrange(self.number_batches):
      yield self.items[(self.current_batch + b) % len(self.items)]
    if update: self.update()

  def update(self):
    if self.current_batch + self.number_batches >= len(self.items):
      self.shuffle()
      self.current_batch = 0
    else:
      self.current_batch += self.number_batches




