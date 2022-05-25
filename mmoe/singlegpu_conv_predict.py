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
from siamese_network_conv import SiameseLSTMw2v
import get_pretrain_w2v
import os
from cal_gauc import *

flags = tf.app.flags
flags.DEFINE_string("train_data_dir", "./train_data/",
                    "Directory for storing train data")
flags.DEFINE_string("test_data_dir", "./test_data/",
                    "Directory for storing test data")
flags.DEFINE_string("model_dir", "./output",
                    "Directory for storing model")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
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
vocab_size = 636014 


def inp_fn(file_dir):
    """Extract training data.
    @data   : line in training file.
    @return : training data in required format
    """
    uid1 = list()
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
                uid1.append(line[0])
                y.append(float(line[-1]))
    return uid1,zero_pad(x1,sequence_length),zero_pad(x2,sequence_length),y

uid1,test_x1,test_x2,test_y  = inp_fn(test_data_path)

if __name__ == "__main__":
    # Start Training
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        siameseModel = SiameseLSTMw2v(
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embedding_size=embedding_dim,
            hidden_units=hidden_size,
            batch_size=batch_size,
            trainableEmbeddings=True
        )
        siameseModel.build_model()
        print("Start predicting...")
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
            vec1,vec2,sim = sess.run([siameseModel.vec1,siameseModel.vec2,siameseModel.sim],feed_dict)
            return vec1,vec2,sim 
        
        saver = tf.train.Saver()
        saver.restore(sess,model_path)
        seq_len = np.array([list(x).index(0) + 1 for x in test_x1])
        vec1,vec2,sim = dev_step(test_x1,test_x2,test_y,seq_len) 
        gauc,auc = gauc(map(int,test_y),sim,uid1)
        print(gauc,auc)
        #for i in range(len(test_y)):
        #    print(uid1[i]+"\t"+str(test_y[i])+"\t"+str(sim[i])+"\t"+" ".join(map(str,vec1[i]))+"\t"+" ".join(map(str,vec2[i])))
