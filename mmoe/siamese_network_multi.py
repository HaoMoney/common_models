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

    def contrastive_loss(self,y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
        self, x1, x2, y, reuse, sequence_length, vocab_size, embedding_size, hidden_units, batch_size, trainableEmbeddings):

        # Placeholders for input, output and dropout
        #self.input_x1 = tf.placeholder(tf.int32, [None, 20], name="input_x1")
        #self.input_x2 = tf.placeholder(tf.int32, [None, 20], name="input_x2")
        #self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #self.seq_len = tf.placeholder(tf.int32,[None],name="seq_len")
          
        # Embedding layer
        with tf.name_scope("word_embedding",reuse=reuse):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings,name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, x1)
            self.attention_words1 = tf.reduce_mean(self.embedded_words1,1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, x2)
            self.attention_words2 = tf.reduce_mean(self.embedded_words2,1)

        with tf.name_scope('shared_fc_layer',reuse=reuse):
            W = tf.Variable(tf.truncated_normal([embedding_size, hidden_units], stddev=0.1),name='wvalue')  # Hidden size is multiplied by 2 for Bi-RNN
            b = tf.Variable(tf.constant(0., shape=[hidden_units]))
            self.out1 = tf.nn.xw_plus_b(self.attention_words1, W, b)
            #self.out1 = tf.nn.l2_normalize(self.out1, dim=1, name="u1_vec")
            
            self.out2 = tf.nn.xw_plus_b(self.attention_words2, W, b)
            #self.out2 = tf.nn.l2_normalize(self.out2, dim=1, name="u2_vec")

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("distance"):
            self.euci_distance = tf.sqrt(tf.reduce_sum(tf.square(self.out1-self.out2),axis=1))
            self.euci_distance = 1.0 / (1.0 + self.euci_distance)
            print(self.euci_distance.shape)
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(y,self.euci_distance,batch_size) 
        #### Accuracy computation is outside of this class.
        #with tf.name_scope("accuracy"):
        #    self.real_sim = tf.subtract(tf.ones_like(self.distance),self.distance, name="temp_sim") #auto threshold 0.5
        #    self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
        #    correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #    self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
