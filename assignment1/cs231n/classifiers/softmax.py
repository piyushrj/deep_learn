import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_samples = X.shape[0]
  for i in range(n_samples):
      f = (X[i].dot(W)).T
      f -= np.max(f)
      p = np.exp(f) / np.sum(np.exp(f))
      loss_i = -1*np.log(p[y[i]])
      loss += loss_i
      # computing the gradient
      scores = X[i].dot(W).T
      for j in range(W.shape[1]):
          dW[:, j] += 1/np.sum(np.exp(scores))*np.exp(scores[j])*X[i]
          if j == y[i]:
              dW[:, j] -= X[i]
          
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= n_samples
  dW /= n_samples
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.max(f, axis=1)[:, np.newaxis]
  f = np.exp(f) / np.sum(np.exp(f), axis=1)[:, np.newaxis]
  losses = -1*np.log(f[np.arange(len(f)), y])
  loss = np.mean(losses)
  # vectorized code for gradient
  dW += ((X.T).dot(f))
  zeros_mat = np.zeros(shape=(X.shape[0], dW.shape[1]))
  zeros_mat[np.arange(len(zeros_mat)), y] = 1
  dW -= (X.T).dot(zeros_mat)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= X.shape[0]
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  return loss, dW

