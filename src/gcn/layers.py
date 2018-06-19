from inits import *
import tensorflow as tf

import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

flags = tf.app.flags
FLAGS = flags.FLAGS



def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


# class Layer(object):
#     """Base layer class. Defines basic API for all layer objects.
#     Implementation inspired by keras (http://keras.io).

#     # Properties
#         name: String, defines the variable scope of the layer.
#         logging: Boolean, switches Tensorflow histogram logging on/off

#     # Methods
#         _call(inputs): Defines computation graph of layer
#             (i.e. takes input, returns output)
#         __call__(inputs): Wrapper for _call()
#         _log_vars(): Log all variables
#     """

#     def __init__(self, **kwargs):
#         allowed_kwargs = {'name', 'logging'}
#         for kwarg in kwargs.keys():
#             assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
#         name = kwargs.get('name')
#         if not name:
#             layer = self.__class__.__name__.lower()
#             name = layer + '_' + str(get_layer_uid(layer))
#         self.name = name
#         self.vars = {}
#         logging = kwargs.get('logging', False)
#         self.logging = logging
#         self.sparse_inputs = False

#     def _call(self, inputs):
#         return inputs

#     def __call__(self, inputs):
#         with tf.name_scope(self.name):
#             if self.logging and not self.sparse_inputs:
#                 tf.summary.histogram(self.name + '/inputs', inputs)
#             outputs = self._call(inputs)
#             if self.logging:
#                 tf.summary.histogram(self.name + '/outputs', outputs)
#             return outputs

#     def _log_vars(self):
#         for var in self.vars:
#             tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


# class Dense(Layer):
#     """Dense layer."""
#     def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
#                  act=tf.nn.relu, bias=False, featureless=False, **kwargs):
#         super(Dense, self).__init__(**kwargs)

#         if dropout:
#             self.dropout = placeholders['dropout']
#         else:
#             self.dropout = 0.

#         self.act = act
#         self.sparse_inputs = sparse_inputs
#         self.featureless = featureless
#         self.bias = bias

#         # helper variable for sparse dropout
#         self.num_features_nonzero = placeholders['num_features_nonzero']

#         with tf.variable_scope(self.name + '_vars'):
#             self.vars['weights'] = glorot([input_dim, output_dim],
#                                           name='weights')
#             if self.bias:
#                 self.vars['bias'] = zeros([output_dim], name='bias')

#         if self.logging:
#             self._log_vars()

#     def _call(self, inputs):
#         x = inputs

#         # dropout
#         if self.sparse_inputs:
#             x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
#         else:
#             x = tf.nn.dropout(x, 1-self.dropout)

#         # transform
#         output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

#         # bias
#         if self.bias:
#             output += self.vars['bias']

#         return self.act(output)


# class GraphConvolution(Layer):
#     """Graph convolution layer."""
#     def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
#                  sparse_inputs=False, act=tf.nn.relu, bias=False,
#                  featureless=False, **kwargs):
#         super(GraphConvolution, self).__init__(**kwargs)

#         if dropout:
#             self.dropout = placeholders['dropout']
#         else:
#             self.dropout = 0.

#         self.act = act
#         self.support = placeholders['support']
#         self.sparse_inputs = sparse_inputs
#         self.featureless = featureless
#         self.bias = bias

#         # helper variable for sparse dropout
#         self.num_features_nonzero = placeholders['num_features_nonzero']

#         with tf.variable_scope(self.name + '_vars'):
#             for i in range(len(self.support)):
#                 self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
#                                                         name='weights_' + str(i))
#             if self.bias:
#                 self.vars['bias'] = zeros([output_dim], name='bias')

#         if self.logging:
#             self._log_vars()

#     def _call(self, inputs):
#         x = inputs

#         # dropout
#         if self.sparse_inputs:
#             x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
#         else:
#             x = tf.nn.dropout(x, 1-self.dropout)

#         # convolve
#         supports = list()
#         for i in range(len(self.support)):
#             if not self.featureless:
#                 pre_sup = dot(x, self.vars['weights_' + str(i)],
#                               sparse=self.sparse_inputs)
#             else:
#                 pre_sup = self.vars['weights_' + str(i)]
#             support = dot(self.support[i], pre_sup, sparse=True)
#             supports.append(support)
#         output = tf.add_n(supports)

#         # bias
#         if self.bias:
#             output += self.vars['bias']

#         return self.act(output)



# class GraphConvolution_Norm(Layer):
#     """Graph convolution layer."""
#     def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
#                  sparse_inputs=False, act=tf.nn.relu, bias=False,
#                  featureless=False, **kwargs):
#         super(GraphConvolution_Norm, self).__init__(**kwargs)

#         if dropout:
#             self.dropout = placeholders['dropout']
#         else:
#             self.dropout = 0.

#         self.act = act
#         self.support = placeholders['support']
#         self.sparse_inputs = sparse_inputs
#         self.featureless = featureless
#         self.bias = bias

#         # helper variable for sparse dropout
#         self.num_features_nonzero = placeholders['num_features_nonzero']

#         with tf.variable_scope(self.name + '_vars'):
#             for i in range(len(self.support)):
#                 self.vars['weights_' + str(i)] = uniform([input_dim, output_dim], scale=0.001,
#                                                         name='weights_' + str(i))
#             if self.bias:
#                 self.vars['bias'] = zeros([output_dim], name='bias')

#         if self.logging:
#             self._log_vars()

#     def _call(self, inputs):
#         x = inputs

#         # dropout
#         if self.sparse_inputs:
#             x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
#         else:
#             x = tf.nn.dropout(x, 1-self.dropout)

#         # convolve
#         supports = list()
#         for i in range(len(self.support)):
#             if not self.featureless:
#                 pre_sup = dot(x, self.vars['weights_' + str(i)],
#                               sparse=self.sparse_inputs)
#             else:
#                 pre_sup = self.vars['weights_' + str(i)]
#             support = dot(self.support[i], pre_sup, sparse=True)
#             supports.append(support)
#         output = tf.add_n(supports)

#         # bias
#         if self.bias:
#             output += self.vars['bias']

#         return self.act(output)

class Dense(Module):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, support_num=0, dropout=0., bias=False):
        super(Dense, self).__init__()

        self.dropout = dropout

        self.weights = Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.bias = torch.zeros([output_dim])
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.constant_(p)

    def forward(self, inputs, adjs, num_features_nonzero=None):
        x = inputs

        # dropout
        x = F.dropout(x, self.dropout, self.training)

        # transform
        output = torch.matmul(x, self.weights)

        # bias
        if self.bias is not None:
            output = output + self.bias

        return output

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GraphConvolution(Module):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, support_num, dropout=0., bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.outptu_dim = output_dim
        self.dropout = dropout

        for i in range(support_num):
            setattr(self, 'weights_' + str(i), Parameter(torch.Tensor(input_dim, output_dim)))
        if bias:
            self.bias = torch.zeros([output_dim])
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.constant_(p)

    def forward(self, inputs, adjs, num_features_nonzero=None):
        x = inputs

        # dropout
        if x.is_sparse:
            assert False, 'Features can not be sparse'
            # x = sparse_dropout(x, self.dropout, num_features_nonzero, self.training)
        else:
            x = F.dropout(x, self.dropout, self.training)

        supports = []
        # convolve
        for i in range(len(adjs)):
            pre_sup = torch.mm(x, getattr(self, 'weights_' + str(i)))
            support = SparseMM(adjs[i])(pre_sup)
            supports.append(support)
        output = sum(supports)

        # bias
        if self.bias is not None:
            output = output + self.bias
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'

