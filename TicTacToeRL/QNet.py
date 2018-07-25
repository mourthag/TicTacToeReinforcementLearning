import tensorflow as tf
import tensorflow.contrib.slim as slim

class QNetwork():
    def __init__(self, h_size, a_size, lr):
        #input and hidden layer
        self.input = tf.placeholder(shape=[None, 9],dtype=tf.float32)
        self.hiddena = slim.fully_connected(inputs=self.input, num_outputs = h_size, activation_fn=tf.nn.relu, biases_initializer=None)
        self.hidden = slim.fully_connected(inputs=self.hiddena, num_outputs = h_size, activation_fn=tf.nn.relu, biases_initializer=None)
        
        
        #split value and advantage streams
        self.streamAC,self.streamVC = tf.split(self.hidden,2,1)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, a_size]))
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))

        #calculate adv and val
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        #calculate final Q-Values
        self.QOut = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1,keep_dims=True))
        print(self.QOut.shape)
        _,self.predictions = tf.nn.top_k(self.QOut[1,:], 9, True)
        self.predict = tf.argmax(self.QOut, 1)

        #Obtain loss
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.QOut, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.updateModel = self.trainer.minimize(self.loss)
