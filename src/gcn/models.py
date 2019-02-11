import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def parse_hiddens(dim_in, dim_out):
    hidden_layers = FLAGS.hiddens    
    if len(hidden_layers) == 0:
        hidden_layers = str(dim_out)
    elif len(hidden_layers) == 1:
        hidden_layers = str(dim_out) + 'd'
    elif hidden_layers[-1] == ',' or hidden_layers[-2] == ',':
        # add last layer in.
        hidden_layers = hidden_layers[:hidden_layers.rfind(',')+1] \
                        + str(dim_out) \
                        + hidden_layers[hidden_layers.rfind(',')+1:]
    else:
        hidden_layers = hidden_layers + ',' + str(dim_out)
    hiddens = hidden_layers.split(',')

    for i in range(len(hiddens)):
        if hiddens[i][-1] == 'd':
            hiddens[i] = (dim_in, int(hiddens[i][:-1]), True)
        else:
            hiddens[i] = (dim_in, int(hiddens[i]), False)
        dim_in = hiddens[i][1]
    return hiddens

# class Model(object):
#     def __init__(self, **kwargs):
#         allowed_kwargs = {'name', 'logging'}
#         for kwarg in kwargs.keys():
#             assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
#         name = kwargs.get('name')
#         if not name:
#             name = self.__class__.__name__.lower()
#         self.name = name

#         logging = kwargs.get('logging', False)
#         self.logging = logging

#         self.vars = {}
#         self.placeholders = {}

#         self.layers = []
#         self.activations = []

#         self.inputs = None
#         self.outputs = None

#         self.loss = 0
#         self.accuracy = 0
#         self.optimizer = None
#         self.opt_op = None

#         self.decay = 0

#     def _build(self):
#         raise NotImplementedError

#     def build(self):
#         """ Wrapper for _build() """
#         with tf.variable_scope(self.name):
#             self._build()

#         # Build sequential layer model
#         self.activations.append(self.inputs)
#         for layer in self.layers:
#             hidden = layer(self.activations[-1])
#             self.activations.append(hidden)
#         self.outputs = self.activations[-1]

#         # Store model variables for easy access
#         variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
#         self.vars = {var.name: var for var in variables}

#         # Build metrics
#         self._loss()
#         self._accuracy()

#         self.opt_op = self.optimizer.minimize(self.loss)

#     def predict(self):
#         pass

#     def _loss(self):
#         raise NotImplementedError

#     def _accuracy(self):
#         raise NotImplementedError

#     def save(self, sess=None):
#         if not sess:
#             raise AttributeError("TensorFlow session not provided.")
#         saver = tf.train.Saver(self.vars)
#         save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
#         print("Model saved in file: %s" % save_path)

#     def load(self, sess=None):
#         if not sess:
#             raise AttributeError("TensorFlow session not provided.")
#         saver = tf.train.Saver(self.vars)
#         save_path = "tmp/%s.ckpt" % self.name
#         saver.restore(sess, save_path)
#         print("Model restored from file: %s" % save_path)


# class Model_dense(object):
#     def __init__(self, **kwargs):
#         allowed_kwargs = {'name', 'logging'}
#         for kwarg in kwargs.keys():
#             assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
#         name = kwargs.get('name')
#         if not name:
#             name = self.__class__.__name__.lower()
#         self.name = name

#         logging = kwargs.get('logging', False)
#         self.logging = logging

#         self.vars = {}
#         self.placeholders = {}

#         self.layers = []
#         self.activations = []

#         self.inputs = None
#         self.outputs = None

#         self.loss = 0
#         self.accuracy = 0
#         self.optimizer = None
#         self.opt_op = None

#         self.decay = 0

#     def _build(self):
#         raise NotImplementedError

#     def build(self):

#         # Build sequential layer model
#         self.activations.append(self.inputs)
#         for layer in self.layers:
#             hidden = layer(self.activations[-1])
#             self.activations.append(hidden)
#         self.outputs = self.activations[-1]

#         # Build metrics
#         self._loss()
#         self._accuracy()

#     def predict(self):
#         pass

#     def _loss(self):
#         raise NotImplementedError

#     def _accuracy(self):
#         raise NotImplementedError


class  Model_dense_mse(nn.Module):
    def __init__(self, layer_func, input_dim, output_dim, support_num, dropout, logging, features=None):
        super(Model_dense_mse, self).__init__()
        if FLAGS.trainable_embedding:
            self.register_parameter('features', nn.Parameter(torch.from_numpy(features).float()))

        self.layer_func = layer_func

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.logging = logging

        self.layers = nn.ModuleList()

        hiddens = parse_hiddens(self.input_dim, self.output_dim)
        for i in range(len(hiddens)):
            self.layers.append(self.layer_func(input_dim=hiddens[i][0],
                                                output_dim=hiddens[i][1],
                                                support_num=support_num,
                                                dropout=dropout if hiddens[i][2] else 0))

    def forward(self, features, adjs, labels = None, labels_mask=None):
        if FLAGS.trainable_embedding:
            features = self.features
            if FLAGS.normalize_embedding:
                features = F.normalize(features)

        inputs = features
        num_features_nonzero = features[1].shape

        # Build sequential layer model
        self.activations = []
        self.activations.append(inputs)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                act = lambda x: F.normalize(x, dim=1)
            else:
                act = lambda x: F.leaky_relu(x, FLAGS.lrelu_slope)
            hidden = act(layer(self.activations[-1], adjs, num_features_nonzero))
            self.activations.append(hidden)
        outputs = self.activations[-1]

        if labels is not None:
            # return loss and accuracy

            loss = mask_mse_loss(outputs, F.normalize(labels, dim=1),
                                    labels_mask)
            return loss
        else:
            return outputs

class GCN_dense_mse(Model_dense_mse):
    def __init__(self, *args, **kwargs):
        super(GCN_dense_mse, self).__init__(GraphConvolution, *args, **kwargs)

class Pure_dense_mse(Model_dense_mse):
    def __init__(self, *args, **kwargs):
        super(Pure_dense_mse, self).__init__(Dense, *args, **kwargs)
