import numpy as np
from random import shuffle


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        cnt = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                cnt -= 1
        dW[:, y[i]] += cnt*X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += 2*reg*W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    # loss = 0.0
    # dW = np.zeros(W.shape)  # initialize the gradient as zero
    scores = X.dot(W)  # NxC
    correct_labels = (np.array(range(len(y))), y)  # locations in scores where j = y[i]
    correct_class_scores = scores[correct_labels]  # Nx1

    M = scores - correct_class_scores[:, np.newaxis] + 1  # NxC margins
    not_thresh = np.where(M <= 0)
    Lij = M
    Lij[not_thresh] = 0
    Lij[correct_labels] = 0

    loss = np.sum(Lij)/X.shape[0] + 0.5*reg*np.sum(W * W)

    TMP = np.ones(M.shape)
    TMP[not_thresh] = 0
    TMP[correct_labels] = 0 # NxC; 1 where margin > 0
    dW = X.T.dot(TMP)  # np.einsum('ij,ik->kj',TMP,X)
    cnts = TMP.sum(axis=-1)
    num_train = X.shape[0]
    for i in xrange(num_train):
        dW[:, y[i]] -= cnts[i]*X[i]

    dW = dW/num_train + 2*reg*W
    return loss, dW
