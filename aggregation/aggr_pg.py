# class of the deep neural network with policy gradient method

# hold data and computation for the neural network and policy gradient method

# input observation, output the action from the neural net
# input a batch of data set(one set includes observation, action, and reward), and training
# the deep neural network

import tensorflow as tf

class PolicyGradient:
    MULTIPLIER = 4  # control the magnitude of the nn layers
    def __init__(self, n_div, learning_rate):
        self.n_div = n_div  # number of neurons for both inputs and outputs
            # observation as inputs, action as outputs
        self.n_full = self.n_div * self.MULTIPLIER
            # number of neurons for all hidden layers
        self.lr = learning_rate
        # build the network
        self.ep_obs, self.ep_acts, self.ep_rews = [], [], []
        self.build_net()
        self.sess = tf.Session()

    def build_net(self):
        with tf.name_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.n_div], name='observations')
            self.acts_ = tf.placeholder(tf.float32, [None, ], name='actions')  # labels
            self.rews = tf.placeholder(tf.float32, [None, ], name='rewards')
        # fc1
        dense1 = tf.layers.dense(
            inputs=x,
            units=self.n_full,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='dense1')
        # fc2
        dense2 = tf.layers.dense(
            inputs=dense1,
            units=self.n_full,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='dense2')
        # fc3
        dense3 = tf.layers.dense(
            inputs=dense2,
            units=self.n_full,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='dense3')
        # fc4
        dense4 = tf.layers.dense(
            inputs=dense3,
            units=self.n_div,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='dense4')
        # softmax layer
        self.acts = tf.nn.softmax(dense4, name='acts_softmax')
        # loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=dense4, labels=self.acts_)
            loss = tf.reduce_mean(cross_entropy * self.rews)  # reward guided loss
        # train step
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self):
        pass

    def store_transition(self):
        pass




