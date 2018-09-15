from __future__ import division
from __future__ import print_function

import time

import os
import tensorflow as tf
import pickle as pkl
import numpy as np

from utils import *
from models import GCN_dense_mse, Pure_dense_mse

import torch
from torch import optim

import tensorboardX as tb

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '../../data/glove_res50/', 'Dataset string.')
flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_string('save_path', '../output/', 'save dir')
flags.DEFINE_integer('epochs', 350, 'Number of epochs to train.')
flags.DEFINE_string('hiddens', '2048,2048,1024,1024,512', 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('save_every', 50, 'Save model every x epochs.')
flags.DEFINE_float('lrelu_slope', 0.2, 'Leaky relu slope')
flags.DEFINE_string('adj_norm_type', 'sym', 'sym or in')
flags.DEFINE_string('feat_norm_type', 'dense2', 'dense2, none, l2')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

use_trainval = True
feat_suffix = 'allx_dense'

# Load data
adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask = \
        load_data_vis_multi(FLAGS.dataset, use_trainval, feat_suffix)

# Some preprocessing
if FLAGS.feat_norm_type == 'dense2':
    features, div_mat = preprocess_features_dense2(features)
elif FLAGS.feat_norm_type == 'l2':
    features = torch.nn.functional.normalize(torch.from_numpy(features)).numpy()

if FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse
elif FLAGS.model == 'pure_dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = Pure_dense_mse
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print(features.shape)

# Define placeholders
# placeholders = {
#     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'features': tf.placeholder(tf.float32, shape=(features.shape[0], features.shape[1])),  # sparse_placeholder
#     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#     'labels_mask': tf.placeholder(tf.int32),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
#     'learning_rate': tf.placeholder(tf.float32, shape=())
# }

device = 'cuda:0'

# Create model
model = model_func(input_dim=features.shape[1], output_dim =y_train.shape[1], support_num=num_supports, dropout=FLAGS.dropout, logging=True)
# tensorflow_model = '/home-nfs/rluo/rluo/zero-shot-gcn/src/init.ckpt'
# from convert import tf_to_pth
# model.load_state_dict(tf_to_pth(tensorflow_model))

model.to(device=device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)

cost_val = []

save_epochs = [300, 3000]

savepath = FLAGS.save_path
exp_name = os.path.basename(FLAGS.dataset)
savepath = os.path.join(savepath, exp_name)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('!!! Make directory %s' % savepath)
else:
    print('### save to: %s' % savepath)

summary_writer = tb.SummaryWriter(savepath)

features, y_train, train_mask = \
    torch.from_numpy(features).float().to(device=device),\
    torch.from_numpy(y_train).float().to(device=device),\
    torch.from_numpy(train_mask.astype(np.uint8)).to(device=device)

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    return torch.sparse.FloatTensor(torch.from_numpy(x[0]).long().t(), torch.from_numpy(x[1]).float(), x[2])

support = [to_sparse(_).to(device=device) for _ in support]

# Train model
now_lr = FLAGS.learning_rate
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    # feed_dict.update({placeholders['learning_rate']: now_lr})

    for group in optimizer.param_groups:
        group['lr'] = now_lr

    # Training step
    optimizer.zero_grad()
    loss = model(features, support, y_train, train_mask)
    loss.backward()
    optimizer.step()

    train_loss = model.losses
    for k,v in model.losses.items():
        summary_writer.add_scalar(k, v, epoch)

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            val_loss = model(features, support, y_train, train_mask)
            model.train()
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "val_loss=", "{:.5f}".format(val_loss.item()),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(now_lr))

    flag = 0
    if epoch % FLAGS.save_every == 0:
        flag = 1

    if flag == 1 or epoch % 500 == 0:
        model.eval()
        with torch.no_grad():
            outs = model(features, support).cpu().numpy()
        model.train()
        filename = savepath + '/feat_' + os.path.basename(FLAGS.dataset) + '_' + str(epoch)
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)

        filehandler = open(filename, 'wb')
        pkl.dump(outs, filehandler)
        filehandler.close()

print("Optimization Finished!")
