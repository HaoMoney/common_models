"""
author : qianhao
desc : model structure
to do : add user age sex short tag feature && use cos loss && tagid embedding
"""
import tensorflow as tf
import numpy as np
from attention import multihead_attention
from attention import attention
from tensorflow.contrib.rnn import GRUCell
class SiameseLSTMw2v(object):
	"""
	A LSTM based deep Siamese network for text similarity.
	Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
	"""
	
	def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
		n_hidden=hidden_units
		n_layers=2
		# Prepare data shape to match `static_rnn` function requirements
		#x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
		# print(x)
		# Define lstm cells with tensorflow
		# Forward direction cell

		with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
			stacked_rnn_fw = []
			for _ in range(n_layers):
				fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
				lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
				stacked_rnn_fw.append(lstm_fw_cell)
			lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
			outputs, _ = tf.nn.dynamic_rnn(cell=lstm_fw_cell_m, inputs= x, dtype=tf.float32,sequence_length=sequence_length)
		return outputs

	def normalRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
		n_hidden=hidden_units
		with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
			outputs, _ = tf.nn.dynamic_rnn(cell=GRUCell(n_hidden), inputs= x, dtype=tf.float32,sequence_length=sequence_length)
		return outputs

	def fully_connected_bn(prev_layer, num_units, is_training):
		"""
		Create a fully connectd layer with the given layer as input and the given number of neurons.
		
		:param prev_layer: Tensor
			The Tensor that acts as input into this layer
		:param num_units: int
			The size of the layer. That is, the number of units, nodes, or neurons.
		:param is_training: bool or Tensor
			Indicates whether or not the network is currently training, which tells the batch normalization
			layer whether or not it should update or use its population statistics.
		:returns Tensor
			A new fully connected layer
		"""
		layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
		layer = tf.layers.batch_normalization(layer, training=is_training)
		layer = tf.nn.relu(layer)
		return layer
	def fully_connected(prev_layer, num_units):
		"""
		Create a fully connectd layer with the given layer as input and the given number of neurons.
		
		:param prev_layer: Tensor
			The Tensor that acts as input into this layer
		:param num_units: int
			The size of the layer. That is, the number of units, nodes, or neurons.
		:returns Tensor
			A new fully connected layer
		"""
		layer = tf.layers.dense(prev_layer, num_units, activation=tf.nn.relu)
		return layer

	def contrastive_loss(self, y,d,batch_size):
		tmp= y *tf.square(d)
		#tmp= tf.mul(y,tf.square(d))
		tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
		return tf.reduce_sum(tmp +tmp2)/batch_size/2
	
	def __init__(
		self, sequence_length, vocab_size, field_size, age_size, embedding_size, hidden_units, batch_size, trainableEmbeddings):

		# Placeholders for input, output and dropout
		self.input_x1 = tf.placeholder(tf.int32, [None, 19], name="input_x1")
		self.input_topic_x1 = tf.placeholder(tf.int32, [None, 3], name="input_topic_x1")
		self.input_age = tf.placeholder(tf.int32, [None,1], name="input_age")
		print(self.input_age.name)
		self.input_sex = tf.placeholder(tf.int32, [None,1], name="input_sex")
		print(self.input_sex.name)
		self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
		self.input_topic_x2 = tf.placeholder(tf.int32, [None, 1], name="input_topic_x2")
		self.input_y1 = tf.placeholder(tf.float32, [None], name="input_y1")
		self.input_y2 = tf.placeholder(tf.float32, [None], name="input_y2")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		#self.seq_len_l = tf.placeholder(tf.int32,[None],name="seq_len_l")
		self.seq_len_r = tf.placeholder(tf.int32,[None],name="seq_len_r")
		# Keeping track of l2 regularization loss (optional)
		  
		# Embedding layer
		with tf.name_scope("tag_embedding"):
			self.W = tf.Variable(
				tf.random_uniform([tag_size, embedding_size], -1.0, 1.0),
				trainable=trainableEmbeddings,name="W")
			self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
			self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)
		# field embedding
		with tf.name_scope("field_embedding"):
			self.FW = tf.Variable(
				tf.random_uniform([field_size, 50], -1.0, 1.0),
				trainable=trainableEmbeddings,name="FW")
			embedded_tmp_fields1 = tf.nn.embedding_lookup(self.FW, self.input_topic_x1)
			print(embedded_tmp_fields1.shape)
			self.embedded_fields1 = tf.reshape(embedded_tmp_fields1,[tf.shape(embedded_tmp_fields1)[0],150])
			print(self.embedded_fields1.shape)
			embedded_tmp_fields2 = tf.nn.embedding_lookup(self.FW, self.input_topic_x2)
			print(embedded_tmp_fields2.shape)
			self.embedded_fields2 = tf.reshape(embedded_tmp_fields2,[tf.shape(embedded_tmp_fields2)[0],50])
		with tf.name_scope("age_embedding"):
			self.AW = tf.Variable(
				tf.random_uniform([age_size, 50], -1.0, 1.0),
				trainable=trainableEmbeddings,name="AW")
			embedded_tmp_age = tf.nn.embedding_lookup(self.AW, self.input_age)
			self.embedded_age = tf.reshape(embedded_tmp_age,[tf.shape(embedded_tmp_age)[0],50])

		with tf.name_scope("sex_embedding"):
			self.SW = tf.Variable(
				tf.random_uniform([3, 50], -1.0, 1.0),
				trainable=trainableEmbeddings,name="SW")
			embedded_tmp_sex = tf.nn.embedding_lookup(self.SW, self.input_sex)
			self.embedded_sex = tf.reshape(embedded_tmp_sex,[tf.shape(embedded_tmp_sex)[0],50])

		# concat
		#with tf.name_scope("concat"):
		#	self.embedded_fea1 = tf.concat([self.embedded_words1,self.embedded_fields1],axis=2)
		#	self.embedded_fea2 = tf.concat([self.embedded_words2,self.embedded_fields2],axis=2)

		# Create a convolution + maxpool layer for each filter size
		with tf.name_scope("output"):
			self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size, self.seq_len_r, hidden_units)
			self.out_field1=tf.layers.dense(self.embedded_fields1, 50)
			self.out_field2=tf.layers.dense(self.embedded_fields2, 50)
			#print(self.out1.shape)
			#print(self.out2.shape)
			attention_01 = attention(self.embedded_words1,50) 
			attention_02 = attention(self.out2,50)
			express_01 = tf.reduce_mean(attention_01,1)
			self.fea1 = tf.concat([express_01,self.out_field1,self.embedded_age,self.embedded_sex],axis=1)
			express_02 = tf.reduce_mean(attention_02,1)
			self.fea2 = tf.concat([express_02,self.out_field2],axis=1)
			#print(express_02.shape)
			self.user_fc1 = tf.layers.dense(self.fea1, 64, activation=tf.nn.tanh)
			self.user_fc = tf.layers.dense(self.user_fc1, 32, activation=tf.nn.tanh)
			#print(self.user_fc.shape)
			#print(self.user_fc.name)
			self.rela_fc1 = tf.layers.dense(self.fea2, 64, activation=tf.nn.tanh)
			self.rela_fc = tf.layers.dense(self.rela_fc1, 32, activation=tf.nn.tanh)
			self.ctr_fc1 = tf.layers.dense(self.fea2, 64, activation=tf.nn.tanh)
			self.ctr_fc = tf.layers.dense(self.ctr_fc1, 32, activation=tf.nn.tanh)
			self.user_vec = tf.nn.l2_normalize(self.user_fc, dim=1, name="user_vec")
			print(self.user_vec.name)
			self.rela_vec = tf.nn.l2_normalize(self.rela_fc, dim=1, name="rela_vec")
			self.ctr_vec = tf.nn.l2_normalize(self.ctr_fc, dim=1, name="ctr_vec")
		with tf.name_scope("cos_sim"):
			self.rela_sim = tf.reduce_sum(
				tf.multiply(self.user_vec, self.rela_vec), axis=1)
			self.ctr_sim = tf.reduce_sum(
				tf.multiply(self.user_vec, self.ctr_vec), axis=1)
		with tf.name_scope("loss"):
			self.rela_loss = tf.losses.mean_squared_error(self.input_y1,self.rela_sim)
			self.ctr_loss = tf.losses.mean_squared_error(self.input_y2,self.ctr_sim)
			self.loss = 2*self.rela_loss + self.ctr_loss
		#### Accuracy computation is outside of this class.
		#with tf.name_scope("accuracy"):
		#	 self.real_sim = tf.subtract(tf.ones_like(self.distance),self.distance, name="temp_sim") #auto threshold 0.5
		#	 self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
		#	 correct_predictions = tf.equal(self.temp_sim, self.input_y)
		#	 self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
