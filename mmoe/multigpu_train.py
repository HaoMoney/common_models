''' Multi-GPU Training Example.

Train a convolutional neural network on multiple GPU with TensorFlow.

This example is using TensorFlow layers, see 'convolutional_network_raw' example
for a raw TensorFlow implementation with variables.

This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
from utils import *
from siamese_network_multi import SiameseLSTMw2v
import get_pretrain_w2v
import os

flags = tf.app.flags
flags.DEFINE_string("train_data_dir", "./train_data/",
                    "Directory for storing train data")
flags.DEFINE_string("test_data_dir", "./test_data/",
                    "Directory for storing test data")
flags.DEFINE_string("model_dir", "./output/model",
                    "Directory for storing model")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 1024, "Training batch size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
FLAGS = flags.FLAGS


# Training Parameters
num_gpus = FLAGS.num_gpus
num_steps = FLAGS.train_steps
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
display_step = 10
train_data_path = FLAGS.train_data_dir 
test_data_path = FLAGS.test_data_dir
num_epochs = 10 # model easily overfits without pre-trained words embeddings, that's why train for a few epochs
model_path = FLAGS.model_dir

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
hidden_size = 200
sequence_length = 20
embedding_dim = 300
vocab_size = 636013 
embedding_ph = get_pretrain_w2v.loadWord2Vec('./thirdparty/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',vocab_size,embedding_dim)
# sequences pre-processing for words feature
vocabulary_size = embedding_ph.shape[0]


def inp_fn(file_dir):
    """Extract training data.
    @data   : line in training file.
    @return : training data in required format
    """
    x1 = list()
    x2 = list()
    y = list()
    files = os.listdir(file_dir)
    for fi in files:
        with open(file_dir + '/' + fi, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                if len(line) < 5:continue
                tmp_x1 = [] 
                tmp_x2 = []
                for one in line[1].split(" "):
                    tmp_x1.append(float(one))
                x1.append(tmp_x1)
                for one in line[3].split(" "):
                    tmp_x2.append(float(one))
                x2.append(tmp_x2)
                y.append(float(line[-1]))
    return zero_pad(x1,sequence_length),zero_pad(x2,sequence_length),y

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

train_x1,train_x2,train_y  = inp_fn(train_data_path)
test_x1,test_x2,test_y  = inp_fn(train_data_path)
train_freader = batch_generator_multi(train_x1, train_x2, train_y, batch_size*num_gpus)
test_freader = batch_generator_multi(test_x1, test_x2, test_y, batch_size*num_gpus)

if __name__ == "__main__":
    # Start Training
    with tf.device('/cpu:0'):
        tower_grads = []
        reuse_vars = False
        # Placeholders for input, output and dropout
        input_x1 = tf.placeholder(tf.int32, [None, 20], name="input_x1")
        input_x2 = tf.placeholder(tf.int32, [None, 20], name="input_x2")
        input_y = tf.placeholder(tf.float32, [None], name="input_y")
        for i in range(num_gpus):
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
    
                # Split data between GPUs
                _x1 = input_x1[i * batch_size: (i+1) * batch_size]
                _x2 = input_x2[i * batch_size: (i+1) * batch_size]
                _y = input_y[i * batch_size: (i+1) * batch_size]
    
                # Because Dropout have different behavior at training and prediction time, we
                # need to create 2 distinct computation graphs that share the same weights.
    
                # Create a graph for training
                siameseModel = SiameseLSTMw2v(
                    x1=_x1,
                    x2=_x2,
                    y=_y,
                    reuse=reuse_vars,
                    sequence_length=sequence_length,
                    vocab_size=vocabulary_size,
                    embedding_size=embedding_dim,
                    hidden_units=hidden_size,
                    batch_size=batch_size,
                    trainableEmbeddings=True
                )
    
                # Define loss and optimizer (with train logits, for dropout to take effect)
                optimizer = tf.train.AdamOptimizer(1e-3)
                print("initialized siameseModel object")
        
                grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    
    
                reuse_vars = True
                tower_grads.append(grads)
        tower_grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads)
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
    
            sess.run(init)
            sess.run(siameseModel.embedding_init, feed_dict={siameseModel.embedding_placeholder: embedding_ph})
            print("Start learning...")
            def train_step(x1_batch, x2_batch, y_batch, seq_len):
                """
                A single training step
                """
                feed_dict = {
                    input_x1: x1_batch,
                    input_x2: x2_batch,
                    input_y: y_batch,
                }
                _, loss = sess.run([train_op,siameseModel.loss],  feed_dict)
                return loss
            def dev_step(x1_batch, x2_batch, y_batch, seq_len):
                """
                A single training step
                """ 
                feed_dict = {
                    input_x1: x1_batch,
                    input_x2: x2_batch,
                    input_y: y_batch,
                }
                loss = sess.run(siameseModel.loss,  feed_dict)
                return loss
            
            saver = tf.train.Saver()
            for epoch in range(num_epochs):
                loss_train = 0
                loss_test = 0
    
                print("epoch: {}\t".format(epoch), end="")
    
                # Training
                cnt = 0
                for train_x1,train_x2,train_y in train_freader:
                    seq_len = np.array([list(x).index(0) + 1 for x in train_x1])
                    loss_tr = train_step(train_x1,train_x2,train_y,seq_len)
                    loss_train += loss_tr
                    cnt += 1
                loss_train /= cnt 
    
                # Testing
                cnt = 0
                for test_x1,test_x2,test_y in test_freader:
                    seq_len = np.array([list(x).index(0) + 1 for x in test_x1])
                    loss_te = dev_step(test_x1,test_x2,test_y,seq_len) 
                    #accuracy_test += acc
                    loss_test += loss_te
                    cnt += 1
                loss_test /= cnt 
    
                print("loss: {:.5f}, val_loss: {:.5f}".format(
                    loss_train, loss_test
                ))
                saver.save(sess, model_path+"_"+str(epoch)+'/model.ckpt')
                print("Epoch %d saved !" % (epoch))
