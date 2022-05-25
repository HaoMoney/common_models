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
from mmoe_etq import MMoE
import get_pretrain_w2v
import os
from extract_feature import *
import dataproc

flags = tf.app.flags
flags.DEFINE_string("train_data_dir", "./train_data/",
                    "Directory for storing train data")
flags.DEFINE_string("test_data_dir", "./test_data/",
                    "Directory for storing test data")
flags.DEFINE_string("w2v_data_dir", "./thirdparty/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5",
                    "Directory for storing test data")
flags.DEFINE_string("model_dir", "./output",
                    "Directory for storing model")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
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
num_epochs = 5 # model easily overfits without pre-trained words embeddings, that's why train for a few epochs
model_path = FLAGS.model_dir
w2v_data_path = FLAGS.w2v_data_dir

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
hidden_size = 200
sequence_length = 20
embedding_dim = 300
vocab_size = 636013 
#embedding_ph = get_pretrain_w2v.loadWord2Vec('./thirdparty/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',vocab_size,embedding_dim)
#embedding_ph = get_pretrain_w2v.loadWord2Vec(w2v_data_path,vocab_size,embedding_dim)
# sequences pre-processing for words feature
#vocabulary_size = embedding_ph.shape[0]
#print(vocabulary_size)

def inp_fn(data):
    """Extract training data.
    @data   : line in training file.
    @return : training data in required format
    """
    x1 = list()
    x2 = list()
    y1 = list()
    y2 = list()
    for line in data:
        left_zero_flag = True
        right_zero_flag = True
        line = line.strip().split("\t")
        if len(line) < 9:continue
        query = line[0]
        query_fea = line[1]
        pid = line[2]
        prod_fea = line[3]
        prod_name = line[4]
        show = line[5]
        click = line[6]
        ctr = line[7]
        relevance = line[8]
        tmp_x1 = [] 
        tmp_x2 = []
        for one in query_fea.split(" "):
            if float(one) != 0:
                left_zero_flag = False
            tmp_x1.append(float(one))
        for one in prod_fea.split(" "):
            if float(one) != 0:
                right_zero_flag = False
            tmp_x2.append(float(one))
        if left_zero_flag and right_zero_flag:
            continue
        if float(ctr) > 0:
            label = 0
        else:
            label = 1
        x1.append(tmp_x1)
        x2.append(tmp_x2)
        y1.append(label)
        y2.append(relevance)
    return zero_pad(x1,sequence_length), zero_pad(x2,sequence_length), y1, y2

train_freader = dataproc.BatchReader(train_data_path, num_epochs)
#train_x1,train_x2,train_y  = inp_fn(train_data_path)
#train_freader = batch_generator2(train_x1, train_x2, train_y, batch_size)
#train_size = train_x1.shape[0]
#rint("train data size is %d" % (train_size))
#test_x1,test_x2,test_y  = inp_fn(train_data_path)
#test_freader = batch_generator_multi(test_x1, test_x2, test_y, batch_size)

if __name__ == "__main__":
    # Start Training
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        mmoe = MMoE(
            input_dimension=128,
            num_units=64,
            num_experts=3,
            num_tasks=2,
            batch_size=batch_size,
            a=0.1,
            b=0.2,
            c=0.3
        )
        mmoe.build_model()
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")
    
        grads_and_vars=optimizer.compute_gradients(mmoe.loss)
        train_op_set = optimizer.apply_gradients(grads_and_vars)

        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        def train_step(x1_batch, x2_batch, y1_batch, y2_batch, y3_batch):
            """
            A single training step
            """
            feed_dict = {
                mmoe.input_x1: x1_batch,
                mmoe.input_x2: x2_batch,
                mmoe.input_y1: y1_batch,
                mmoe.input_y2: y2_batch

            }
            _, final_outputs, loss = sess.run([train_op_set, mmoe.final_outputs, mmoe.loss],  feed_dict)
            return final_outputs,loss
        """
        def dev_step(x1_batch, x2_batch, y_batch, seq_len):
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.seq_len: seq_len,
                siameseModel.dropout_keep_prob: 1.0,
            }
            loss = sess.run(siameseModel.loss,  feed_dict)
            return loss
        """
        #x1 = np.random.rand(5,100)
        #x2 = np.random.rand(5,100)
        #y1 = np.random.randint(low=0,high=1,size=5)
        #y2 = np.random.randint(low=0,high=1,size=5)
        #y3 = np.random.randint(low=0,high=1,size=5)
        #print(train_step(x1,x2,y1,y2,make_one_hot(y3,3))[1])
        print(train_step(x1,x2,y1,y2,y3)[0])
        for epoch in range(num_epochs):
            loss_train = 0
            loss_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            #num_batches = int(train_size/float(batch_size))
            #num_batches = int(round(train_size/float(batch_size)))
            #for b in range(num_batches):
            num_batches = train_samples // batch_size
            for b in range(num_batches):
                batch_data = train_freader.get_batch(batch_size)
                if not batch_data:
                    break
                batch_train_x1, batch_train_x2, batch_train_y1, batch_train_y2  = inp_fn(batch_data)
                seq_len = np.array([list(x).index(0) + 1 for x in batch_train_x1])
                loss_tr = train_step(batch_train_x1,batch_train_x2,batch_train_y,seq_len)
                #print("batch_train_loss: {:.5f}".format(
                #    loss_tr
                #))
                loss_train += loss_tr
            loss_train /= num_batches
            print("loss: {:.5f}".format(
                loss_train
            ))
        """
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            train_freader = batch_generator_multi(train_x1, train_x2, train_y, batch_size)
            loss_train = 0
            loss_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            #num_batches = int(train_size/float(batch_size))
            num_batches = int(round(train_size/float(batch_size)))
            #for b in range(num_batches):
            for b in range(num_batches):
                batch_train_x1,batch_train_x2,batch_train_y = next(train_freader)
                seq_len = np.array([list(x).index(0) + 1 for x in batch_train_x1])
                loss_tr = train_step(batch_train_x1,batch_train_x2,batch_train_y,seq_len)
                #print("batch_train_loss: {:.5f}".format(
                #    loss_tr
                #))
                loss_train += loss_tr
            loss_train /= num_batches
            print("loss: {:.5f}".format(
                loss_train
            ))

            # Testing
            #for test_x1,test_x2,test_y in test_freader:
            #    seq_len = np.array([list(x).index(0) + 1 for x in test_x1])
            #    loss_te = dev_step(test_x1,test_x2,test_y,seq_len) 
            #    #accuracy_test += acc
            #    loss_test += loss_te

            #print("loss: {:.5f}, val_loss: {:.5f}".format(
            #    loss_train, loss_test
            #))
            saver.save(sess, model_path+'/model'+'_'+str(epoch)+'.ckpt')
        """
