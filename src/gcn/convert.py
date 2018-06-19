import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from collections import OrderedDict
import re
import torch
import numpy as np

import argparse

def tf_to_pth(tensorflow_model):

    reader = pywrap_tensorflow.NewCheckpointReader(tensorflow_model)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_dict = {k:reader.get_tensor(k) for k in var_to_shape_map.keys()}

    if 'beta1_power' in var_dict: del var_dict['beta1_power']
    if 'beta2_power' in var_dict: del var_dict['beta2_power']

    for k in list(var_dict.keys()):
        if 'Adam' in k:
            del var_dict[k]

    # dummy_replace = OrderedDict([
    #                 ('qgen/', ''),\
    #                 ('/', '.'),
    #                 ('image_embedding.fully_connected', 'image_embedding.0'),
    #                 ('decoder_output.fully_connected', 'decoder_output_mlp.0'),
    #                 ('rl_baseline.baseline_hidden', 'rl_baseline_mlp.0.0'),
    #                 ('rl_baseline.baseline_out', 'rl_baseline_mlp.1.0')])

    # for a, b in dummy_replace.items():
    #     for k in list(var_dict.keys()):
    #         if a in k:
    #             var_dict[k.replace(a,b)] = var_dict[k]
    #             del var_dict[k]


    # for k in list(var_dict.keys()):
    #     if k[-2:] == '.W':
    #         var_dict[k[:-2]+'.weight'] = var_dict[k]
    #         del var_dict[k]
    #     if k[-2:] == '.b':
    #         var_dict[k[:-2]+'.bias'] = var_dict[k]
    #         del var_dict[k]


    for k in list(var_dict.keys()):
        if 'weights' in k:
            m = re.search('gcn_dense_mse/graphconvolution_(\d+)_vars/weights_(\d+)', k)
            var_dict['layers.%d.weights_%d'%(int(m.group(1))-1,int(m.group(2)))] = var_dict[k]
            del var_dict[k]
        if 'bias' in k:
            m = re.search('gcn_dense_mse/graphconvolution_(\d+)_vars/bias', k)
            var_dict['layers.%d.bias'%(int(m.group(1))-1)] = var_dict[k]
            del var_dict[k]

    # for k in list(var_dict.keys()):
    #     if 'weights' in k:
    #         var_dict[k] = var_dict[k].transpose([1,0])
        # assert x[k].shape == var_dict[k].shape, k

    for k in list(var_dict.keys()):
        var_dict[k] = torch.from_numpy(var_dict[k]).float()

    return var_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tf-faster-rcnn model to pytorch-faster-rcnn model')
    parser.add_argument('--tensorflow_model',
                        help='the path of tensorflow_model',
                        default=None, type=str)

    args = parser.parse_args()
    torch.save(tf_to_pth(args.tensorflow_model), 'converted_params.pth')

