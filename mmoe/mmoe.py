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
import tensorflow.keras.backend as K
class MMoE(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    def __init__(self, input_dimension, num_units, num_experts, num_tasks, batch_size, a, b, c):
        self.input_dimension = input_dimension
        self.num_units = num_units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.a = a
        self.b = b
        self.c = c
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

    def fully_connected_bn(self,prev_layer, num_units, is_training):
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

    def fully_connected(self,prev_layer, num_units):
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
    
    def expert_kernel(self,inputs,input_dimension,num_units,num_experts,name):
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", [input_dimension,num_units,num_experts],
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [num_units,num_experts],
                initializer=tf.constant_initializer(0.0))
            expert_layer = tf.tensordot(a=inputs, b=weights, axes=1) + biases 
            return tf.nn.relu(expert_layer)

    def gate_kernel(self,inputs,input_dimension,num_experts,taskid):
        with tf.variable_scope("task_"+taskid):
            weights = tf.get_variable("weights", [input_dimension,num_experts],
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [num_experts],
                initializer=tf.constant_initializer(0.0))
            gate_layer = tf.nn.softmax(
                tf.tensordot(a=inputs,b=weights,axes=1) + biases)
            return gate_layer

    def multi_fc(self,input_layer):
        with tf.variable_scope("fc1"):
            # Variables created here will be named "conv1/weights", "conv1/biases".
            relu1 = tf.layers.dense(input_layer, 128, activation=tf.nn.relu)
        with tf.variable_scope("fc2"):
            relu2 = tf.layers.dense(relu1, 64, activation=tf.nn.relu)
            return relu2

    def conv_relu(self, input, kernel_shape, bias_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv1d(input, weights,
            stride=1, padding='SAME')
        return tf.nn.relu(conv + biases)

    def multi_conv(self,input_layer,name):
        with tf.variable_scope(name+"conv1"):
            # Variables created here will be named "conv1/weights", "conv1/biases".
            relu1 = self.conv_relu(input_layer, [2,300,128], [128])
            print(relu1.shape)
        with tf.variable_scope(name+"conv2"):
            # Variables created here will be named "conv2/weights", "conv2/biases".
            return self.conv_relu(relu1, [2,128,64], [64])

    def contrastive_loss(self,y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def build_model(self):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, self.input_dimension], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, self.input_dimension], name="input_x2")
        self.input_y1 = tf.placeholder(tf.float32, [None], name="input_y1")
        self.input_y2 = tf.placeholder(tf.float32, [None], name="input_y2")
        self.input_y3 = tf.placeholder(tf.float32, [None], name="input_y3")
          
        # Expert Layer 
        with tf.variable_scope("expert_layer"):
            self.expert_outputs1 = self.expert_kernel(self.input_x1,self.input_dimension,self.num_units,self.num_experts,"x1")
            self.expert_outputs2 = self.expert_kernel(self.input_x2,self.input_dimension,self.num_units,self.num_experts,"x2")

        # Gate Layer 
        with tf.variable_scope("gate_layer"):
            self.gate_outputs = []
            for i in range(self.num_tasks):
                self.gate_outputs.append(self.gate_kernel(self.input_x2,self.input_dimension,self.num_experts,str(i)))                
            #print(self.gate_outputs[0].shape)

        with tf.variable_scope('final_layer') as scope:
            self.final_outputs = []
            for gate_output in self.gate_outputs:
                expanded_gate_output = tf.expand_dims(gate_output, axis=1)
                #print(expanded_gate_output.shape)
                repeated_tmp = K.repeat_elements(expanded_gate_output, self.num_units, axis=1)
                #print(repeated_tmp.shape)
                weighted_expert_output = self.expert_outputs2 * repeated_tmp
                #print(weighted_expert_output.shape)
                self.final_outputs.append(tf.nn.l2_normalize(tf.reduce_sum(weighted_expert_output, axis=2),dim=1))
            self.x1_output = tf.nn.l2_normalize(tf.reduce_sum(self.expert_outputs1, axis=2),dim=1)
            #print(self.final_outputs[0].shape)


        
        # Create a convolution + maxpool layer for each filter size
        with tf.variable_scope("sim"):
            self.sim1 = tf.reduce_sum(
				tf.multiply(self.final_outputs[0], self.x1_output), axis=1)
            self.sim2 = tf.reduce_sum(
				tf.multiply(self.final_outputs[1], self.x1_output), axis=1)
            self.sim3 = tf.reduce_sum(
				tf.multiply(self.final_outputs[2], self.x1_output), axis=1)
            
            

        with tf.variable_scope("loss"):
            self.loss1 = tf.losses.mean_squared_error(self.input_y1,self.sim1)
            self.loss2 = tf.losses.mean_squared_error(self.input_y2,self.sim2)
            self.loss3 = tf.losses.mean_squared_error(self.input_y3,self.sim3) 
            self.loss =  self.a * self.loss1 + self.b * self.loss2 + self.c * self.loss3
            #self.loss = self.contrastive_loss(self.input_y,self.euci_distance,self.batch_size) 
        
        #### Accuracy computation is outside of this class.
        #with tf.name_scope("accuracy"):
        #    self.real_sim = tf.subtract(tf.ones_like(self.distance),self.distance, name="temp_sim") #auto threshold 0.5
        #    self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
        #    correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #    self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



