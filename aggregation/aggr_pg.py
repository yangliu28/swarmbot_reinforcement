# class of the deep neural network with policy gradient method

# hold data and computation for the neural network and policy gradient method

# input observation, output the action from the neural net
# input a batch of data set(one set includes observation, action, and reward), and training
# the deep neural network

import tensorflow as tf
import numpy as np

class PolicyGradient:
    MULTIPLIER = 4  # control the size of the hidden layers
    training_count = 0  # record times of training
    def __init__(self, n_div, learning_rate, training_repeats):
        self.n_div = n_div  # number of neurons for both inputs and outputs
            # observation as inputs, action as outputs
        self.n_full = self.n_div * self.MULTIPLIER
            # number of neurons for all hidden layers
        self.lr = learning_rate
        self.training_repeats = training_repeats
        # build the network
        self.ep_obs, self.ep_acts, self.ep_rews = [], [], []  # one episode of data
        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        with tf.name_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.n_div], name='observations')
            self.acts_ = tf.placeholder(tf.int32, [None, ], name='actions')  # labels
            self.rews = tf.placeholder(tf.float32, [None, ], name='rewards')
            self.keep_prob = tf.placeholder(tf.float32)
        # rotate and split the input vector
        with tf.name_scope('rot-split'):
            W_rot_split = tf.convert_to_tensor(
                self.create_rot_split_mat(), dtype=tf.float32)
            rs_vec = tf.matmul(self.obs, W_rot_split)
        # reshape the rot-split vector to matrix
        with tf.name_scope('reshape'):
            rs_mat = tf.reshape(rs_vec, [-1, self.n_div, self.n_div, 1])
        # convolution layer
        with tf.name_scope('conv1'):
            W_conv1 = tf.truncated_normal(shape=[1, self.n_div, 1, 32], stddev=0.3)
            b_conv1 = tf.constant(0.3, shape=[32])
            h_conv1 = tf.nn.relu(
                tf.nn.conv2d(rs_mat, W_conv1, strides=[1,1,1,1], padding='VALID')
                + b_conv1)
        # fully connected layer
        with tf.name_scope('fc1'):
            h_conv1_flat = tf.reshape(h_conv1, [-1, self.n_div * 1 * 32])  # reshape again
            W_fc1 = tf.truncated_normal(shape=[self.n_div * 1 * 32, 1024], stddev=0.3)
            b_fc1 = tf.constant(0.3, shape=[1024])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

        # dense1 = tf.layers.dense(
        #     inputs=self.obs,
        #     units=self.n_full,
        #     # activation=tf.nn.relu,
        #     activation=tf.nn.sigmoid,
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='dense1')

        # dropout
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # fully connected layer, map to output
        with tf.name_scope('fc2'):
            W_fc2 = tf.truncated_normal(shape=[1024, self.n_div], stddev=0.3)
            b_fc2 = tf.constant(0.3, shape=[self.n_div])
            acts = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # softmax layer
        self.acts_softmax = tf.nn.softmax(acts, name='acts_softmax')
        # loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=acts, labels=self.acts_)
            loss = tf.reduce_mean(cross_entropy * self.rews)  # reward guided loss
        # train step
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

    # create the rotate-split matrix for building the network
    def create_rot_split_mat(self):
        n = self.n_div  # get size of rotating vector
        mat = np.zeros((n, n*n))
        mat_rot = np.identity(n)  # the identity matrix that will be rotated
        mat[:, 0:n] = mat_rot  # top one being the original identity mat
        for i in range(1, n):
            # rotate the mat_rot on the column once
            col_temp = mat_rot[:,0]
            mat_rot = np.hstack((
                np.delete(mat_rot, (0), axis=1), col_temp[:, np.newaxis]))
            mat[:, i*n:(i+1)*n] = mat_rot
        return mat

    # randomly choice an action based on action probabilities from nn
    def choose_action(self, observation):
        print(observation)
        acts_prob = self.sess.run(self.acts_softmax, feed_dict={
            self.obs: observation[np.newaxis, :], self.keep_prob: 1.0})
        print(acts_prob)
        action = np.random.choice(range(self.n_div), p=acts_prob.ravel())
        return action

    # store one data sample including observation, action, and its reward
    def store_transition(self, observation, action, reward):
        self.ep_obs.append(observation)
        self.ep_acts.append(action)
        self.ep_rews.append(reward)

    def learn(self):
        # print the training count
        self.training_count = self.training_count + 1
        print("training count " + str(self.training_count))
        # normalize episode rewards
        rewards_norm = self.norm_rewards()
        # print(rewards_norm)
        # train on one episode of data, for multiple times
        for _ in range(self.training_repeats):
            self.sess.run(self.train_step, feed_dict={
                self.obs: np.vstack(self.ep_obs),
                self.acts_: np.array(self.ep_acts),
                self.rews: rewards_norm,
                self.keep_prob: 0.5
                })
        # empty episode data after each training
        self.ep_obs, self.ep_acts, self.ep_rews = [], [], []
        return rewards_norm

    # normalize episode rewards
    def norm_rewards(self):
        norm_ep_rews = np.array(self.ep_rews)
        norm_ep_rews = norm_ep_rews - np.mean(norm_ep_rews)
        norm_ep_rews = norm_ep_rews / np.std(norm_ep_rews)
        return norm_ep_rews


