import numpy as np
from scipy.signal import fftconvolve, convolve
from itertools import product

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    cache = (x, w, b)
    x = x.reshape(x.shape[0], -1)  # NxD
    out = x.dot(w) + b
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: biases, of shape (M,

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dout[np.where(x <= 0.)] = 0.
    dx = dout

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        var = np.var(x, axis=0)
        mean = x.mean(axis=0)
        x_shift = x - mean
        x_norm = x_shift/np.sqrt(var + eps)

        out = gamma*x_norm
        out += beta

        scale = (1. - momentum)
        running_mean *= momentum
        running_mean += scale*mean

        running_var *= momentum
        running_var += scale*var

        cache = (x_shift, x_norm, var+eps, gamma)
        # cache = (x, mean, var+eps, gamma)

    elif mode == 'test':
        out = (x - running_mean)/np.sqrt(running_var + eps)
        out *= gamma
        out += beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x_shift, x_norm, var_eps, gamma = cache
    dx_norm = gamma*dout

    dgamma = np.einsum('ij,ij->j', dout, x_norm)
    dbeta = np.sum(dout, axis=0)

    dx = dx_norm - np.mean(dx_norm, axis=0)
    dx -= x_shift*np.einsum('ij,ij->j', dx_norm, x_shift)/(var_eps*dx_norm.shape[0])
    dx /= np.sqrt(var_eps)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    x_shift, x_norm, var_eps, gamma = cache
    dx_norm = gamma*dout

    dgamma = np.einsum('ij,ij->j', dout, x_norm)
    dbeta = np.sum(dout, axis=0)

    dx = dx_norm - np.mean(dx_norm, axis=0)
    dx -= x_shift*np.einsum('ij,ij->j', dx_norm, x_shift)/(var_eps*dx_norm.shape[0])
    dx /= np.sqrt(var_eps)

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p)/p
        out = x*mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = mask*dout
    elif mode == 'test':
        dx = dout
    return dx


def flip_ndarray(x):
    """ Reverse the positions of the entries of the input array along all of its axes.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        out : numpy.ndarray
            A view of x with all of its axes reversed. Since a view is returned, this operation is O(1).

        Example
        ------
        x = array([[1, 2, 3],
                   [6, 5, 6]])
        flip_ndarray(x))
        >> array([[6, 5, 4],
                  [3, 2, 1]])"""
    loc = tuple(slice(None, None, -1) for i in xrange(x.ndim))
    return x[loc]


def multi_layer_conv(data, kernel, stride, pad, flip_kernel=True, outshape=None, channels=True):
    """ Perform a convolution on a collection of ndarrays of uniform shape with a kernel,
        using a specified stride and padding.

        The convolution

        Parameters
        ----------
        data : Iterable[numpy.ndarray, ...]
            Collection of arrays to be convolved over.
        kernel : numpy.ndarray
            Kernel used to perform convolution.
        stride : int ( > 0)
            Step size used while sliding the kernel during the convolution.
        pad : int ( >= 0)
            Specifies the number of zeroes added to every side of each input array.
        flip_kernel : bool, optional
            When True (default), reverse all of the axes of the kernel before performing convolutions.
        outshape : Union[NoneType, Tuple[int, ...]], optional
            Provide the known output shape of the convolution, allowing the function to bypass
            sanity checks and some initial computations.

        Returns
        -------
        out : numpy.ndarray, shape = (len(data), data[0].shape[0], data[0].shape[1], ...)
            An array of the resulting convolutions. The first axis runs over the results of
            the independent convolutions that were performed.
        """

    if flip_kernel:
        kernel = flip_ndarray(kernel)
    for n, dat in enumerate(data):
        if n == 0:  # initialize output and stride information
            if np.max(dat.shape) >= 500:
                conv = fftconvolve  # use fft method for large input arrays
            else:
                conv = convolve

            if outshape is None:
                npad = np.array([0]+[pad for i in xrange(dat.ndim-1)])
                outshape = (np.array(dat.shape)-np.array(kernel.shape)+2.*npad)/float(stride)+1.
                for num in outshape:
                    assert (num).is_integer(), num
                outshape = np.round(outshape).astype(int)

            if pad:
                if channels:
                    pad = [(0, 0)] + [(pad, pad) for i in xrange(dat.ndim-1)]

            total_out_shape = tuple([data.shape[0]]+[i for i in outshape])
            total_out = np.zeros(total_out_shape, dtype=dat.dtype)

            all_pos = zip(*product(*(stride*np.arange(i) for i in outshape)))

        if pad:
            dat = np.pad(dat, pad, mode='constant').astype(dat.dtype)

        full_conv = conv(dat, kernel, mode='valid')
        if stride == 1:
            total_out[n] = full_conv
        else:
            out = np.zeros(outshape, dtype=full_conv.dtype)
            out.flat = full_conv[all_pos]
            total_out[n] = out
    return total_out


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    tmp = 1. + (np.array(x.shape[2:]) + 2*conv_param['pad'] - np.array(w.shape[2:]))/conv_param['stride']
    out = np.zeros((x.shape[0], w.shape[0], int(round(tmp[0])), int(round(tmp[1]))))
    outshape = None
    for n, kernel in enumerate(w):
        out[:, n:n+1, :, :] = multi_layer_conv(x, kernel, conv_param['stride'], conv_param['pad'], outshape=outshape)
        out[:, n:n+1, :, :] += b[n]

    cache = (x, w, b, conv_param)
    return out, cache


def back_prop_dw(data, dout, kernel, stride, pad):
    dw = np.zeros_like(kernel, dtype=kernel.dtype)
    for n, dat in enumerate(data):
        dy = dout[n][np.newaxis, :, :]
        if n == 0:  # initialize output and stride information
            npad = np.array([0]+[pad for i in xrange(dat.ndim-1)])
            outshape = (np.array(dat.shape)-np.array(kernel.shape)+2.*npad)/float(stride)+1.
            outshape = np.round(outshape).astype(int)

            if pad:
                pad = [(0, 0)] + [(pad, pad) for i in xrange(dat.ndim-1)]

        all_pos = list(product(*[stride*np.arange(i) for i in outshape]))
        all_slices = [tuple(slice(start, start+kernel.shape[i]) for i,start in enumerate(x)) for x in all_pos]

        if pad:
            dat = np.pad(dat, pad, mode='constant').astype(dat.dtype)

        for i,slices in enumerate(all_slices):
            loc = np.unravel_index(i, outshape)
            dw += dy[loc]*dat[slices]
    return dw


def back_prop_dx(data, dout, kernel, stride, pad):
    for n, dat in enumerate(data):
        dy = dout[n][np.newaxis, :, :]
        out = np.zeros((dat.shape[0], dat.shape[1]+2*pad, dat.shape[2]+2*pad), dtype=data.dtype)
        if n == 0:  # initialize output and stride information
            npad = np.array([0]+[pad for i in xrange(dat.ndim-1)])
            outshape = (np.array(dat.shape)-np.array(kernel.shape)+2.*npad)/float(stride)+1.
            outshape = np.round(outshape).astype(int)

            total_out = np.zeros(data.shape, dtype=dat.dtype)
            all_pos = list(product(*[stride*np.arange(i) for i in outshape]))
            all_slices = [tuple(slice(start, start+kernel.shape[i]) for i,start in enumerate(x)) for x in all_pos]

        for i, slices in enumerate(all_slices):
            loc = np.unravel_index(i, outshape)

            out[slices] += dy[loc]*kernel

        if pad:
            out = out[:, pad:-pad, pad:-pad]

        total_out[n] = out[:]
    return total_out


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    x, w, b, conv_param = cache

    dx = np.zeros_like(x, dtype=x.dtype)
    dw = np.zeros_like(w, dtype=w.dtype)

    for n, kernel in enumerate(w):
        dx += back_prop_dx(x, dout[:,n,:,:], kernel, conv_param['stride'], conv_param['pad'])
        dw[n:n+1, :, :, :] = back_prop_dw(x, dout[:,n,:,:], kernel, conv_param['stride'], conv_param['pad'])
    db = np.sum(dout, axis=(0, 2, 3))
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
