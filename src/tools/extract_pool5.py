import argparse
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet101, resnet152
import time


def extract_feature(image_list, model, preprocess, model_path, image_dir, feat_dir):

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), False)
    model.cuda()
    print('Done Init! ')
    net_time, cnt = 0, 0
    for i, index in enumerate(image_list):
        feat_name = os.path.join(feat_dir, index.split('.')[0] + '.npz')
        image_name = os.path.join(image_dir, index)
        lockname = feat_name + '.lock'
        if os.path.exists(feat_name):
            continue
        if os.path.exists(lockname):
            continue
        try:
            os.makedirs(lockname)
        except:
            continue
        t = time.time()
        cnt += 1
        image = preprocess(image_name)
        if image is None:
            print('no image')
            continue
        feat = run_feat(model, image)
        if not os.path.exists(os.path.dirname(feat_name)):
            try:
                os.makedirs(os.path.dirname(feat_name))
                print('## Make Directory: %s' % feat_name)
            except:
                pass
        np.savez_compressed(feat_name, feat=feat)
        net_time += time.time() - t
        if i % 1000 == 0:
            print('extracting feature [%d / %d] %s (%f sec)' % (i, len(image_list), feat_name, net_time / cnt * 1000),
                  feat.shape)
            net_time = 0
            cnt = 0
        cmd = 'rm -r %s' % lockname
        os.system(cmd)

def preprocess_resnet(image_name):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = cv2.imread(image_name)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = 256
    crop_size = 224
    im_size_min = np.min(image.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    height = image.shape[0]
    width = image.shape[1]
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    image = image[y: y + crop_size, x: x + crop_size]

    image = image.astype(np.float32)
    image[:, :, 0] -= _R_MEAN
    image[:, :, 1] -= _G_MEAN
    image[:, :, 2] -= _B_MEAN
    image = image[np.newaxis, :, :, :]
    return image


def preprocess_inception(image_name):
    image = cv2.imread(image_name)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = 256
    crop_size = 224
    im_size_min = np.min(image.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    height = image.shape[0]
    width = image.shape[1]
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    image = image[y: y + crop_size, x: x + crop_size]
    save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/debug/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_dir + '1.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = image.astype(np.float32)
    image /= 255
    image = 2 * image - 1
    image = image[np.newaxis, :, :, :]
    return image

def run_feat(model, image):
    feat = model(torch.from_numpy(image.transpose(0,3,1,2)).cuda()).squeeze().cpu().data.numpy()
    return feat

# def resnet_arg_scope(is_training=True,
#                      batch_norm_decay=0.997,
#                      batch_norm_epsilon=1e-5,
#                      batch_norm_scale=True):
#     batch_norm_params = {
#         'is_training': False,
#         'decay': batch_norm_decay,
#         'epsilon': batch_norm_epsilon,
#         'scale': batch_norm_scale,
#         'trainable': False,
#         'updates_collections': tf.GraphKeys.UPDATE_OPS
#     }
#     with arg_scope(
#             [slim.conv2d],
#             weights_initializer=slim.variance_scaling_initializer(),
#             trainable=is_training,
#             activation_fn=tf.nn.relu,
#             normalizer_fn=slim.batch_norm,
#             normalizer_params=batch_norm_params):
#         with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
#             return arg_sc


# def inception_arg_scope(is_training=True,
#                         batch_norm_decay=0.997,
#                         batch_norm_epsilon=1e-5,
#                         batch_norm_scale=True):
#     batch_norm_params = {
#         'is_training': False,
#         'decay': batch_norm_decay,
#         'epsilon': batch_norm_epsilon,
#         'trainable': False,
#         'updates_collections': tf.GraphKeys.UPDATE_OPS
#     }
#     with arg_scope(
#             [slim.conv2d],
#             weights_initializer=slim.variance_scaling_initializer(),
#             trainable=is_training,
#             activation_fn=tf.nn.relu,
#             normalizer_fn=slim.batch_norm,
#             normalizer_params=batch_norm_params):
#         with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
#             return arg_sc


def resnet(model_type):
    if model_type == 'res50':
        model = resnet50()
    elif model_type == 'res101':
        model = resnet101()
    elif model_type == 'res152':
        model = resnet152()
    model.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    for i in range(2, 5):
        getattr(model, 'layer%d'%i)[0].conv1.stride = (2,2)
        getattr(model, 'layer%d'%i)[0].conv2.stride = (1,1)
    del model.fc
    model.fc = lambda x:x

    return model



# def inception():
#     image = tf.placeholder(tf.float32, [None, 224, 224, 3], 'image')
#     with slim.arg_scope(inception_arg_scope(is_training=False)):
#         with variable_scope.variable_scope(
#                 'InceptionV1', 'InceptionV1', [image, 1000], reuse=None) as scope:
#             with arg_scope(
#                     [layers_lib.batch_norm, layers_lib.dropout], is_training=False):
#                 net, end_points = inception_v1_base(image, scope=scope)
#                 with variable_scope.variable_scope('Logits'):
#                     net_conv = layers_lib.avg_pool2d(
#                         net, [7, 7], stride=1, scope='MaxPool_0a_7x7')
#     print(net_conv.shape)

#     return net_conv, image


def parse_arg():
    parser = argparse.ArgumentParser(description='word embeddign type')
    parser.add_argument('--fc', type=str, default='res50',
                        help='word embedding type: [inception, res50]')
    parser.add_argument('--model_path', type=str, default='../pretrain_weights/resnet50-caffe.pth',
                        help='path to pretrained model')
    parser.add_argument('--image_file', type=str, default='../data/list/img-2-hops.txt',
                        help='list of image file')
    parser.add_argument('--image_dir', type=str, default='../images/',
                        help='directory to save features')
    parser.add_argument('--feat_dir', type=str, default='../feats/',
                        help='directory to save features')
    # parser.add_argument('--gpu', type=str, default='0',
    #                     help='gpu device')
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


args = parse_arg()

# if args.fc == 'inception':
#     pool5, image_holder = inception()
#     preprocess = preprocess_inception
if args.fc in ['res50', 'resnet101', 'resnet152']:
    model = resnet(args.fc)
    preprocess = preprocess_resnet
else:
    raise NotImplementedError
image_list, label_list = [], []
with open(args.image_file) as fp:
    for line in fp:
        index, l = line.split()
        image_list.append(index)
        label_list.append(int(l))

if __name__ == '__main__':
    extract_feature(image_list, model, preprocess, args.model_path, args.image_dir, args.feat_dir)
