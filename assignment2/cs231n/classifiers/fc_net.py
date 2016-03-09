import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(shape=hidden_dim, dtype=float)

        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(shape=num_classes, dtype=float)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h1, cache = affine_relu_forward(X, W1, b1) # layer 1
        scores = h1.dot(W2) + b2 # layer 2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dh2 = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))

        # layer 2
        dh1 = dh2.dot(W2.T)
        grads['W2'] = h1.T.dot(dh2) + self.reg*W2
        grads['b2'] = dh2.sum(axis=0)

        # layer 1
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dh1, cache)
        grads['W1'] += self.reg*W1

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims_list = [input_dim] + hidden_dims + [num_classes]
        for l, ndim in enumerate(dims_list[1:]):
            ind = str(l+1)
            self.params['W'+ind] = np.random.normal(loc=0.0, scale=weight_scale, size=(dims_list[l], ndim))
            self.params['b'+ind] = np.zeros(shape=ndim)

        if self.use_batchnorm:
            for l in range(len(hidden_dims)):
                ind = str(l+1)
                self.params['gamma'+ind] = np.ones(hidden_dims[l])
                self.params['beta'+ind] = np.zeros(hidden_dims[l])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        caches = []
        dropout_caches = []
        # in_ = X
        # if self.use_batchnorm:
        #     for l in range(1, self.num_layers):
        #         ind = str(l)
        #         out, cache = affine_batchnorm_relu_forward(in_, self.params['W'+ind], self.params['b'+ind],
        #                                                    self.params['gamma'+ind], self.params['beta'+ind],
        #                                                    self.bn_params[l-1])
        #         caches.append(cache)
        #         in_ = out
        # else:
        #     for l in range(1, self.num_layers):
        #         ind = str(l)
        #         out, cache = affine_relu_forward(in_, self.params['W'+ind], self.params['b'+ind])
        #         caches.append(cache)
        #         in_ = out
        # ind = str(self.num_layers)
        # scores, cache = affine_forward(in_, self.params['W'+ind], self.params['b'+ind])
        # caches.append(cache)

        in_ = X

        for l in range(1, self.num_layers):
            ind = str(l)

            if self.use_batchnorm:
                out, cache = affine_batchnorm_relu_forward(in_, self.params['W'+ind], self.params['b'+ind],
                                                           self.params['gamma'+ind], self.params['beta'+ind],
                                                           self.bn_params[l-1])
            else:
                out, cache = affine_relu_forward(in_, self.params['W'+ind], self.params['b'+ind])

            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                dropout_caches.append((dropout_cache))

            caches.append(cache)
            in_ = out
        ind = str(self.num_layers)
        scores, cache = affine_forward(in_, self.params['W'+ind], self.params['b'+ind])
        caches.append(cache)

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################


        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, dL = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum([np.sum(self.params['W'+str(ind+1)]**2) for ind in range(self.num_layers)])

        # output layer
        ind = str(self.num_layers)
        dL, grads['W'+ind], grads['b'+ind] = affine_backward(dL, caches[self.num_layers-1])
        grads['W'+ind] += self.reg*self.params['W'+ind]

        # hidden layers
        for l in range(self.num_layers-1, 0, -1):
            ind = str(l)
            if self.use_dropout:
                dL = dropout_backward(dL, dropout_caches[l-1])

            if self.use_batchnorm:
                dL, grads['W'+ind], grads['b'+ind], grads['gamma'+ind], grads['beta'+ind] = \
                    affine_batchnorm_relu_backward(dL, caches[l-1])
            else:
                dL, grads['W'+ind], grads['b'+ind] = affine_relu_backward(dL, caches[l-1])
            grads['W'+ind] += self.reg*self.params['W'+ind]

        return loss, grads
