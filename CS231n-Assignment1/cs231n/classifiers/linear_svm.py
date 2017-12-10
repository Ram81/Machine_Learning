import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    diff = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        #gradient update for incorrect rows
        diff += 1
        dW[:,j] += X[i] 
        loss += margin
    dW[:,y[i]] += -1*diff * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  delta = 1.0
  
  #for i in xrange(num_train):
  #  scores = X[i].dot(W)
  #  correct_score =scores[y[i]]
  #  margin = np.maximum(0,scores - correct_score + delta)
  #  margin[y[i]]=0;
  #  loss+=np.sum(margin)

  scores = X.dot(W)
  correct_score = scores[np.arange(num_train),y]
  
  margin = np.maximum(0, scores - correct_score[:,np.newaxis] + delta)
  margin[np.arange(num_train),y] = 0
  
  loss += np.sum(margin)
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_classes = W.shape[1]
  
  #incorrect_count = np.sum(margin > 0, axis=1)
  
  #for i in xrange(num_classes):
  #  wj = np.sum(X[margin[:, i] > 0], axis=0)
  #  wy = np.sum(-incorrect_count[y==i][:, np.newaxis] * X[y==i], axis=0)
  #  dW[:,i] = wj + wy
  
  X_mask = np.zeros(margin.shape)
  
  X_mask[margin > 0] = 1
  incorrect_count = np.sum(X_mask, axis=1)
  
  X_mask[np.arange(num_train), y] = -1 * incorrect_count
  
  dW = X.T.dot(X_mask)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
