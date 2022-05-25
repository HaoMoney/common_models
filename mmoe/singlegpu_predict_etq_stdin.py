'''
Author: Qian Hao
To do: Train MMoE
'''

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
from utils import *
from mmoe_etq import MMoE
import os
from extract_feature import *
from batch_process import *
from cal_gauc import *

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
flags.DEFINE_integer("seq_len", 128, "Sequence length")
flags.DEFINE_float("loss_weight1", 0.8, "Loss weight1")
flags.DEFINE_float("loss_weight2", 0.2, "Loss weight2")
flags.DEFINE_float("loss_weight3", 0.3, "Loss weight3")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
FLAGS = flags.FLAGS

#Parameters
TRAIN_PATH = FLAGS.train_data_dir
TEST_PATH = FLAGS.test_data_dir
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
SEQ_LEN = FLAGS.seq_len

def inp_fn(file_path):
    """Extract testing data.
    @file_path   : test file.
    @return : testing data in required format
    """
    src = list()
    x1 = list()
    x2 = list()
    y1 = list()
    y2 = list()
    with open(file_path) as f:
        for line in f:
            src_line = line.strip()
            line = line.strip().split("\t")
            if len(line) < 10:continue
            tmp_x1 = [] 
            tmp_x2 = []
            for one in line[1].split(" "):
                tmp_x1.append(float(one))
            x1.append(tmp_x1)
            for one in line[4].split(" "):
                tmp_x2.append(float(one))
            x2.append(tmp_x2)
            y1.append(float(line[8]))
            y2.append(float(line[9]))
            src.append(src_line)
    return src, x1, x2, y1, y2


#src, test_x1, test_x2, test_y1, test_y2 = inp_fn(TEST_PATH)



if __name__ == "__main__":
    # Start Predicting
    gpu_options = tf.GPUOptions(allow_growth=True)
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

        print("Start predicting...")
        def dev_step(x1_batch, x2_batch):
            """
            A single training step
            """
            feed_dict = {
                mmoe.input_x1: x1_batch,
                mmoe.input_x2: x2_batch,
            }
            sim1, sim2 = sess.run([mmoe.sim1, mmoe.sim2],  feed_dict)
            return sim1, sim2
        
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
        for line in sys.stdin:
            src_line = line.strip()
            line = line.strip().split("\t")
            x1 = list()
            x2 = list()
            if len(line) < 6:continue
            tmp_x1 = [] 
            tmp_x2 = []
            for one in line[1].split(" "):
                tmp_x1.append(float(one))
            x1.append(tmp_x1)
            for one in line[3].split(" "):
                tmp_x2.append(float(one))
            x2.append(tmp_x2)
            sim1, sim2 = dev_step(x1, x2)
            print(src_line + "\t" + str(sim2[0]))

        #for i in range(len(sim2)):
        #    print(src[i] + "\t" + str(sim2[i]))
        #auc = cal_auc(map(int, test_y2), sim2)
        #print(auc)
