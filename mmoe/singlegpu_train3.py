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
from siamese_network import SiameseLSTMw2v
import get_pretrain_w2v
import os

flags = tf.app.flags
flags.DEFINE_string("train_data_dir", "./train_data/",
                    "Directory for storing train data")
flags.DEFINE_string("test_data_dir", "./test_data/",
                    "Directory for storing test data")
flags.DEFINE_string("model_dir", "./output/",
                    "Directory for storing model")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 8, "Training batch size")
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
embedding_ph = get_pretrain_w2v.loadWord2Vec('../sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',vocab_size,embedding_dim)
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

train_x1,train_x2,train_y  = inp_fn(train_data_path)
test_x1,test_x2,test_y  = inp_fn(train_data_path)
train_freader = batch_generator_multi(train_x1, train_x2, train_y, batch_size)
test_freader = batch_generator_multi(test_x1, test_x2, test_y, batch_size)

if __name__ == "__main__":
    # Start Training
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        siameseModel = SiameseLSTMw2v(
            sequence_length=sequence_length,
            vocab_size=vocabulary_size,
            embedding_size=embedding_dim,
            hidden_units=hidden_size,
            batch_size=batch_size,
            trainableEmbeddings=True
        )
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")
    
        grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
        train_op_set = optimizer.apply_gradients(grads_and_vars)

        sess.run(tf.global_variables_initializer())
        sess.run(siameseModel.embedding_init, feed_dict={siameseModel.embedding_placeholder: embedding_ph})
        print("Start learning...")
        def train_step(x1_batch, x2_batch, y_batch, seq_len):
            """
            A single training step
            """
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.seq_len: seq_len,
                siameseModel.dropout_keep_prob: 1.0,
            }
            _, dist, loss = sess.run([train_op_set,siameseModel.euci_distance,siameseModel.loss],  feed_dict)
            return dist,loss
        def dev_step(x1_batch, x2_batch, y_batch, seq_len):
            """
            A single training step
            """ 
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.seq_len: seq_len,
                siameseModel.dropout_keep_prob: 1.0,
            }
            loss = sess.run(siameseModel.loss,  feed_dict)
            return loss
        
        saver = tf.train.Saver()
        last_loss = 999999
        for epoch in range(num_epochs):
            loss_train = 0
            loss_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = int(410/float(batch_size))
            for b in range(num_batches):
                train_x1,train_x2,train_y = next(train_freader)
                seq_len = np.array([list(x).index(0) + 1 for x in train_x1])
                dist,loss_tr = train_step(train_x1,train_x2,train_y,seq_len)
                print(dist)
                loss_train += loss_tr
            loss_train /= num_batches

            # Testing
            for b in range(num_batches):
                test_x1,test_x2,test_y = next(test_freader)
                seq_len = np.array([list(x).index(0) + 1 for x in test_x1])
                loss_te = dev_step(test_x1,test_x2,test_y,seq_len) 
                #accuracy_test += acc
                loss_test += loss_te
            loss_test /= num_batches

            print("loss: {:.5f}, val_loss: {:.5f}".format(
                loss_train, loss_test
            ))
            if loss_test < last_loss:
                last_loss = loss_test
                saver.save(sess, model_path+'/model.ckpt')
                print("Epoch %d saved !" % (epoch))
            else:
                print("early stoped !")
                break
