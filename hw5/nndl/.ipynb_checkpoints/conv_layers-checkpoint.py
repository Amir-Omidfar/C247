import numpy as np
from nndl.layers import *
import pdb
""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

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
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

    # obtaining dimensions
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    # hieght and width of output
    HConv = 1 + (H + 2 * pad - HH) // stride
    WConv = 1 + (W + 2 * pad - WW) // stride
    
    # output dimensions and initialization
    out_shape = (N, F, HConv, WConv)
    out = np.zeros(out_shape)
    
    # padding
    npad = ((0,0), (0,0), (pad, pad), (pad, pad))
    x = np.pad(x, pad_width=npad, mode='constant')
    
    # for loops
    for i in np.arange(N):
        for j in np.arange(F):
                for h1 in np.arange(HConv):
                    for w1 in np.arange(WConv):
                        start_h = h1*stride
                        start_w = w1*stride
                        x_selected = x[i,:,start_h:(start_h + HH),start_w:(start_w + WW)]
                        w_selected = w[j,:,:,:]  # j th filter
                        b_selected = b[j] 
                        out[i, j, h1, w1] = np.sum(x_selected*w_selected) + b_selected
    
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
    cache = (x, w, b, conv_param)
    return out, cache


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
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache

    stride, pad = [conv_param['stride'], conv_param['pad']]
    xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    num_filts, _, f_height, f_width = w.shape

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #

    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
   
    N, C, H, W = x.shape
    
    HConv = 1 + (H - f_height) // stride
    WConv = 1 + (W - f_width) // stride
    
    for i in np.arange(N):
        for j in np.arange(num_filts):
            if (i == 0):
                db[j] += np.sum(dout[:, j, :, :])
            for h1 in np.arange(HConv):
                start_h = h1*stride
                for w1 in np.arange(WConv):
                    start_w = w1*stride

                    upstream_selected = dout[i,j,h1,w1]
                    x_selected = x[i,:,start_h:(start_h+f_height),start_w:(start_w+f_width)]
                    w_selected = w[j,:,:,:]  # j th filter
                 
                    dx[i,:,start_h:(start_h+f_height),start_w:(start_w+f_width)] += w[j,:,:,:] * dout[i,j,h1,w1] 
                    dw[j,:,:,:] += x[i,:,start_h:(start_h+f_height),start_w:(start_w+f_width)] * dout[i,j,h1,w1]
    
    dx = dx[:, :, pad:-pad, pad:-pad]
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 
    
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

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #

    f_height = pool_param['pool_height']
    f_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    
    # hieght and width of output
    HMaxPool = 1 + (H  - f_height) // stride
    WMaxPool = 1 + (W - f_width) // stride
    
    # output dimensions and initialization
    out_shape = (N, C, HMaxPool, WMaxPool)
    out = np.zeros(out_shape)
    
    # for loops
    for i in np.arange(N):
        for j in np.arange(C):
                for h1 in np.arange(HMaxPool):
                    for w1 in np.arange(WMaxPool):
                        start_h = h1*stride
                        start_w = w1*stride
                        x_selected = x[i,j,start_h:(start_h + f_height),start_w:(start_w + f_width)]
                        out[i, j, h1, w1] = np.max(x_selected)
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 
    
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
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #
    
    N, C, H, W = x.shape
    
    # hieght and width of output
    HMaxPool = 1 + (H  - pool_height) // stride
    WMaxPool = 1 + (W - pool_width) // stride
    
    dx = np.zeros(x.shape)
    
    # for loops
    for i in np.arange(N):
        for j in np.arange(C):
                for h1 in np.arange(HMaxPool):
                    for w1 in np.arange(WMaxPool):
                        start_h = h1*stride
                        start_w = w1*stride
                        
                        x_selected = x[i,j,start_h:(start_h + pool_height),start_w:(start_w + pool_width)]
                        upstream_selected = dout[i,j,h1,w1]
                        local_gradient = (x_selected==np.max(x_selected))
                        
                        dx[i,j,start_h:(start_h+pool_height),start_w:(start_w+pool_width)] += local_gradient * upstream_selected
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

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

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #

    
    N, C, H, W = x.shape
    x = x.reshape((N,H,W,C))
    x = x.reshape((N*H*W,C))
  
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.T 
    out = out.reshape(C,N,H,W)
    out = out.swapaxes(0,1)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

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

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #

    N, C, H, W = dout.shape
    
    dout = dout.swapaxes(0,1)
    dout = dout.reshape(C,N*H*W)
    dout = dout.T
    
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    
    dx = dx.reshape((N,C,H,W))
    dgamma = dgamma.reshape((C,))
    dbeta = dbeta.reshape((C,))

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx, dgamma, dbeta