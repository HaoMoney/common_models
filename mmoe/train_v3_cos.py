#!/usr/bin/python
"""
toy example of attention layer use

train rnn (gru) on imdb dataset (binary classification)
learning and hyper-parameters were not tuned; script serves as an example 
"""
from __future__ import print_function, division

#from get_pretrain_w2v import loadword2vec as load_w2v
import get_pretrain_w2v
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.rnn import grucell
from tensorflow.contrib import rnn
#from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.python.ops.rnn import static_bidirectional_rnn as bi_rnn
#from tqdm import tqdm
#from extract_feature import load_data
import os
from utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator
from siamese_network_v4 import SiameseLSTMw2v
import sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ['cuda_visible_devices']='0'
import random
import datetime
import dataproc
sys.path.append('.')
sys.path.append('..')
sequence_length = 20
embedding_dim = 200
HIDDEN_SIZE = 200
KEEP_PROB = 0.6
BATCH_SIZE = 256
batch_size = 256
VOCAB_SIZE = 1963642
num_epochs = 10 # model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
model_path = './dssm_mtl_model_v4/'
embedding_ph = get_pretrain_w2v.loadWord2Vec('/data/ceph/10454/qianhao/tf-rnn-attention/word2vec/train.w2v.model.1',VOCAB_SIZE,embedding_dim)

# load the data set
#x_train, y_train, z_train, z_test, x_test, y_test = load_data("train_data/test.f") 

# sequences pre-processing for words feature
vocabulary_size = embedding_ph.shape[0]
print(vocabulary_size)
#x_test = fit_in_vocabulary(x_test, vocabulary_size)
#x_train = zero_pad(x_train, sequence_length)
#x_test = zero_pad(x_test, sequence_length)

# sequences pre-processing for pos feature
#z_test = fit_in_vocabulary(z_test, vocabulary_size)
#z_train = zero_pad(z_train, sequence_length_z)
#z_test = zero_pad(z_test, sequence_length_z)
# Batch generators
#train_batch_generator = batch_generator(x_train, z_train, y_train, BATCH_SIZE)
#test_batch_generator = batch_generator(x_test, z_test, y_test, BATCH_SIZE)
#session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
def inp_fn(data):
	"""Extract training data.
	@data	: line in training file.
	@return : training data in required format
	"""
	X = list()
	X_Field = list()
	X_sex = list()
	X_age = list()
	y1 = list()
	y2 = list()
	Z = list()
	Z_Field = list()
	for i, inst in enumerate(data):
		splits = inst.strip("\n").split('\t')
		if len(splits) != 9:continue
		interest = splits[0].split()+splits[1].split()
		user_field = splits[2].split()
		user_all_zero_flag = True
		#user feature
		tmp_x = []
		for j in interest[:40]:
			tmp_x.append(int(j))
		if len(tmp_x) < 40:
			for k in range(len(tmp_x),40):
				tmp_x.append(0)
		tmp_field_x = []
		for j in splits[2].split():
			tmp_field_x.append(int(j))
		if len(tmp_field_x) < 3:
			for k in range(len(tmp_field_x),3):
				tmp_field_x.append(0)
		tmp_age_x = []
		for j in splits[3].split():
			tmp_age_x.append(int(j))
		if len(tmp_age_x) < 1:
			for k in range(len(tmp_age_x),1):
				tmp_age_x.append(0)
		tmp_sex_x = []
		for j in splits[4].split():
			tmp_sex_x.append(int(j))
		if len(tmp_sex_x) < 1:
			for k in range(len(tmp_sex_x),1):
				tmp_sex_x.append(0)
		#query feature
		tmp_z = []
		for j in splits[5].split():
			tmp_z.append(int(j))
		tmp_field_z = []
		for j in splits[6].split():
			tmp_field_z.append(int(j))
		if len(tmp_field_z) < 1:
			for k in range(len(tmp_field_z),1):
				tmp_field_z.append(0)
		X.append(tmp_x)
		X_Field.append(tmp_field_x)
		X_age.append(tmp_age_x)
		X_sex.append(tmp_sex_x)
		Z.append(tmp_z)
		Z_Field.append(tmp_field_z)
		y1.append(int(splits[-2]))
		y2.append(int(splits[-1]))
	return X, X_Field, X_age, X_sex, y1, y2, zero_pad(Z, sequence_length), Z_Field
train_file = '../train_data/train.v3.f'
valid_file = "../train_data/valid.v3.f"
train_freader = dataproc.BatchReader(train_file, num_epochs)
valid_freader = dataproc.BatchReader(valid_file, num_epochs)

if __name__ == "__main__":
	gpu_options = tf.GPUOptions(allow_growth=True)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		siameseModel = SiameseLSTMw2v(
			sequence_length=sequence_length,
			vocab_size=vocabulary_size,
			field_size=1092,
			age_size=5,
			embedding_size=embedding_dim,
			hidden_units=HIDDEN_SIZE,
			batch_size=batch_size,
			trainableEmbeddings=True
		)
		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		print("initialized siameseModel object")
	
		grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
		tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		print("defined training_ops")
		
		sess.run(tf.global_variables_initializer())
		sess.run(siameseModel.embedding_init, feed_dict={siameseModel.embedding_placeholder: embedding_ph})
		print("Start learning...")
		def train_step(x1_batch, x1_field_batch, x1_input_age, x1_input_sex, x2_batch, x2_field_batch, y1_batch, y2_batch, z_len):
			"""
			A single training step
			"""
			feed_dict = {
				siameseModel.input_x1: x1_batch,
				siameseModel.input_topic_x1: x1_field_batch,
				siameseModel.input_age: x1_input_age,
				siameseModel.input_sex: x1_input_sex,
				siameseModel.input_x2: x2_batch,
				siameseModel.input_topic_x2: x2_field_batch,
				siameseModel.input_y1: y1_batch,
				siameseModel.input_y2: y2_batch,
				siameseModel.seq_len_r: z_len,
				siameseModel.dropout_keep_prob: KEEP_PROB,
			}
			_, step, loss = sess.run([tr_op_set, global_step, siameseModel.loss],  feed_dict)
			time_str = datetime.datetime.now().isoformat()
			return loss
		def dev_step(x1_batch, x1_field_batch, x1_input_age, x1_input_sex, x2_batch, x2_field_batch, y1_batch, y2_batch, z_len):
			"""
			A single training step
			""" 
			feed_dict = {
				siameseModel.input_x1: x1_batch,
				siameseModel.input_topic_x1: x1_field_batch,
				siameseModel.input_age: x1_input_age,
				siameseModel.input_sex: x1_input_sex,
				siameseModel.input_x2: x2_batch,
				siameseModel.input_topic_x2: x2_field_batch,
				siameseModel.input_y1: y1_batch,
				siameseModel.input_y2: y2_batch,
				siameseModel.seq_len_r: z_len,
				siameseModel.dropout_keep_prob: 1.0,
			}
			step, loss = sess.run([global_step, siameseModel.loss],  feed_dict)
			time_str = datetime.datetime.now().isoformat()
			return loss
		
		saver = tf.train.Saver()
		last_loss = 999999
		for epoch in range(num_epochs):
			loss_train = 0
			loss_test = 0

			print("epoch: {}\t".format(epoch), end="")

			# Training
			num_batches = 8295914 // BATCH_SIZE
			for b in range(num_batches):
				batch_data = train_freader.get_batch(batch_size)
				if not batch_data:
					break
				x_batch, x_field_batch, x_age_batch, x_sex_batch, y1_batch, y2_batch, z_batch, z_field_batch = inp_fn(batch_data)
				seq_len_z = np.array([list(x).index(0) + 1 for x in z_batch])  # actual lengths of sequences
				loss_tr = train_step(x_batch,x_field_batch,x_age_batch,x_sex_batch,z_batch,z_field_batch,y1_batch,y2_batch,seq_len_z)
				current_step = tf.train.global_step(sess, global_step)
				#accuracy_train += acc
				#loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
				loss_train += loss_tr
			loss_train /= num_batches
			#accuracy_train /= num_batches

			# Testing
			num_batches = 829592 // BATCH_SIZE
			for b in range(num_batches):
				batch_valid_data = valid_freader.get_batch(batch_size)
				if not batch_valid_data:
					break
				x_batch, x_field_batch, x_age_batch, x_sex_batch, y1_batch, y2_batch, z_batch, z_field_batch = inp_fn(batch_valid_data)
				#valid_q, valid_pt, valid_nt = inp_fn(batch_valid_data)
				#x_batch, z_batch, y_batch = next(test_batch_generator)
				seq_len_z = np.array([list(x).index(0) + 1 for x in z_batch])  # actual lengths of sequences
				loss_test_batch = dev_step(x_batch,x_field_batch,x_age_batch,x_sex_batch,z_batch,z_field_batch,y1_batch,y2_batch,seq_len_z) 
				#accuracy_test += acc
				loss_test += loss_test_batch
			#accuracy_test /= num_batches
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
		#print("Epoch %d saved !" % (epoch))
			#tf.train.write_graph(sess.graph_def, "./model_bak", "test2.pb", False)
		print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
