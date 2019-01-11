pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that pefrorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_relu_dropout_forward(x, w, b, dropout_param):
    """
    Convenience layer that pefrorms an affine transform followed by a ReLU followed by Dropout

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - dropout_param: Dropout paramter
    
    Returns a tuple of:
    - out: Output from the ReLU followed by dropout
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    relu_out, relu_cache = relu_forward(a)
    out, dropout_cache = dropout_forward(relu_out, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
    return out, cache

   
def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu-dropout convenience layer
    """
    fc_cache, relu_cache, dropout_cache = cache
    drelu = dropout_backward(dout, dropout_cache)
    da = relu_backward(drelu, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
    

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Forward pass for affine-batchnorm-relu layer
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: scale parameter batchnorm
    - beta: shift parameter batchnorm
    - bn_param: batchnorm parameters

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for affine-batchnorm-relu layer
    """
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_batchnorm_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    """
    Forward pass for affine-batchnorm-relu-dropout layer
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: scale parameter batchnorm
    - beta: shift parameter batchnorm
    - bn_param: batchnorm parameters
    - dropout_param: dropout parameters

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(bn_out)
    out, dropout_cache = dropout_forward(relu_out, dropout_param)
    cache = (fc_cache, bn_cache, relu_cache, dropout_cache)
    return out, cache
    

def affine_batchnorm_relu_dropout_backward(dout, cache):
    """
    Backward pass for affine-batchnorm-relu-dropout layer
    """
    fc_cache, bn_cache, relu_cache, dropout_cache = cache
    drelu = dropout_backward(dout, dropout_cache)
    dbn = relu_backward(drelu, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
    

def affine_layernorm_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    Forward pass for affine-batchnorm-relu layer
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: scale parameter batchnorm
    - beta: shift parameter batchnorm
    - bn_param: batchnorm parameters

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    ln_out, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    out, relu_cache = relu_forward(ln_out)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache


def affine_layernorm_relu_backward(dout, cache):
    """
    Backward pass for affine-layernorm-relu layer
    """
    fc_cache, ln_cache, relu_cache = cache
    dln = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = layernorm_backward(dln, ln_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_layernorm_relu_dropout_forward(x, w, b, gamma, beta, ln_param, dropout_param):
    """
    Forward pass for affine-batchnorm-relu-dropout layer
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: scale parameter batchnorm
    - beta: shift parameter batchnorm
    - ln_param: layernorm parameters
    - dropout_param: dropout parameters

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    ln_out, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    relu_out, relu_cache = relu_forward(ln_out)
    out, dropout_cache = dropout_forward(relu_out, dropout_param)
    cache = (fc_cache, ln_cache, relu_cache, dropout_cache)
    return out, cache
    

def affine_layernorm_relu_dropout_backward(dout, cache):
    """
    Backward pass for affine-layernorm-relu-dropout layer
    """
    fc_cache, ln_cache, relu_cache, dropout_cache = cache
    drelu = dropout_backward(dout, dropout_cache)
    dln = relu_backward(drelu, relu_cache)
    da, dgamma, dbeta = layernorm_backward(dln, ln_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
    

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
