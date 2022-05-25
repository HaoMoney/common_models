'''
Author: Qian Hao
To do: Train MMoE
'''

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
from utils import *
from mmoe_etq_test import MMoE
import os
from extract_feature import *
from batch_process import *

flags = tf.app.flags
flags.DEFINE_string("train_data_dir", "./train_data",
                    "Directory for storing train data")
flags.DEFINE_string("test_data_dir", "./test_data",
                    "Directory for storing test data")
flags.DEFINE_string("model_dir", "./output",
                    "Directory for storing model")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_integer("num_epochs", 10, "Training epoch")
flags.DEFINE_integer("num_batches", 64000, "Training batches")
flags.DEFINE_integer("num_units", 64, "Hidden units")
flags.DEFINE_integer("input_dimension", 128, "Input dimension")
flags.DEFINE_integer("num_experts", 3, "Num experts")
flags.DEFINE_integer("num_tasks", 2, "Num tasks")
flags.DEFINE_float("loss_weight1", 0.8, "Loss weight1")
flags.DEFINE_float("loss_weight2", 0.2, "Loss weight2")
flags.DEFINE_float("loss_weight3", 0.3, "Loss weight3")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
FLAGS = flags.FLAGS

#Parameters
TRAIN_PATH = FLAGS.train_data_dir
NUM_BATCHES = FLAGS.num_batches
BATCH_SIZE = FLAGS.batch_size
NUM_EPOCHS = FLAGS.num_epochs
MODEL_PATH = FLAGS.model_dir
NUM_UNITS = FLAGS.num_units
LOSS_WEIGHT1 = FLAGS.loss_weight1
LOSS_WEIGHT2 = FLAGS.loss_weight2
LOSS_WEIGHT3 = FLAGS.loss_weight3
INPUT_DIMENSION = FLAGS.input_dimension
NUM_EXPERTS = FLAGS.num_experts
NUM_TASKS =  FLAGS.num_tasks






if __name__ == "__main__":
    # Start Training
    gpu_options = tf.GPUOptions(allow_growth=True)
    dataset = BatchClickMasked(
            infiles=[TRAIN_PATH],
            batch_size=BATCH_SIZE,
            is_speech=False,
            num_epoch=NUM_EPOCHS,
            )
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        mmoe = MMoE(
            input_dimension=INPUT_DIMENSION,
            num_units=NUM_UNITS,
            num_experts=NUM_EXPERTS,
            num_tasks=NUM_TASKS,
            batch_size=BATCH_SIZE,
            a=LOSS_WEIGHT1,
            b=LOSS_WEIGHT2,
            c=LOSS_WEIGHT3
        )
        mmoe.build_model()
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized mmoe object")
    
        grads_and_vars=optimizer.compute_gradients(mmoe.loss)
        train_op_set = optimizer.apply_gradients(grads_and_vars)

        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        def train_step(x1_batch, x2_batch, y1_batch, y2_batch):
            """
            A single training step
            """
            feed_dict = {
                mmoe.input_x1: x1_batch,
                mmoe.input_x2: x2_batch,
                mmoe.input_y1: y1_batch,
                mmoe.input_y2: y2_batch

            }
            _, final_output0, loss = sess.run([train_op_set, mmoe.final_output0, mmoe.loss],  feed_dict)
            return final_output0,loss
        
        saver = tf.train.Saver()
        for step in range(NUM_EPOCHS * NUM_BATCHES):

            batch_train_x1, batch_train_x2, batch_train_y1, batch_train_y2  = sess.run(dataset.next_element)
            _, loss_tr = train_step(batch_train_x1, batch_train_x2, batch_train_y1, batch_train_y2)
            print("loss: {:.5f}".format(
                loss_tr
            ))

            saver.save(sess, MODEL_PATH + '/model.ckpt')
        print("Training process has been finished !!!")
