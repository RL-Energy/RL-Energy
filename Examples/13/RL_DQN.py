"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

Using:
Tensorflow: 2.0
gym: 0.7.3
# Modified by Qizhi He (qizhi.he@pnnl.gov)
# Ref to https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
"""

import numpy as np
# import pandas as pd
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)
# tf.random.set_seed(1) # for tf2

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            n_inlets,
            n_outlets,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy_max=0.8,
            e_greedy_min=0.0,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
            randomness=3,
            CNN_enable=False,
            N_hidden=100,
            increment_degradation = [1, 2, 5, 10]
    ):
        self.n_actions = n_actions #n_outlets+1
        self.n_features = n_features #n_inlets*(n_outlets+2)
        self.n_inlets = n_inlets
        self.n_outlets = n_outlets
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy_max
        
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_min if e_greedy_increment is not None else self.epsilon_max
        self.randomness = randomness
        self.N_hidden = N_hidden
        self.increment_degradation = increment_degradation
        
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, d, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        # consist of [target_net, evaluate_net]
        if CNN_enable:
            self._build_net_CNN()
        else:
            self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_hidden, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
            #print("n_l1=========",n_l1)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def _build_net_CNN(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 150, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)   # config of layers
            out = tf.reshape(self.s,[-1,self.n_inlets,self.n_outlets+2,1])
            # CNN layers
            with tf.variable_scope("convnet"):
                out=tf.compat.v1.layers.conv2d(out, filters=3, kernel_size=3, strides=(2, 2), padding='valid', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
                out=tf.compat.v1.layers.conv2d(out, filters=12, kernel_size=3, strides=(2, 2), padding='valid', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
                out=tf.compat.v1.layers.conv2d(out, filters=20, kernel_size=3, strides=(1, 1), padding='valid', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
                conv_out = tf.compat.v1.layers.flatten(out)
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [conv_out.shape[1], n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(conv_out, w1) + b1)
                
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
            #print("n_l1=========",n_l1)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            out = tf.reshape(self.s_,[-1,self.n_inlets,self.n_outlets+2,1])
            # CNN layers
            with tf.variable_scope("convnet"):
            # original architecture
                #wc1 = tf.get_variable('wc1', [3, 3, 1, 16], initializer=w_initializer, collections=c_names)
                #bc1 = tf.get_variable('bc1', [16], initializer=b_initializer, collections=c_names)
                #out = tf.nn.relu(tf.compat.v1.nn.conv2d(out, wc1, strides=[1, 1, 1, 1], padding="VALID")+bc1)
                #wc2 = tf.get_variable('wc2', [3, 3, 16, 32], initializer=w_initializer, collections=c_names)
                #bc2 = tf.get_variable('bc2', [32], initializer=b_initializer, collections=c_names)
                #out = tf.nn.relu(tf.compat.v1.nn.conv2d(out, wc2, strides=[1, 1, 1, 1], padding="VALID")+bc2)
                #conv_out = tf.compat.v1.layers.flatten(out)
                out=tf.compat.v1.layers.conv2d(out, filters=3, kernel_size=3, strides=(2, 2), padding='valid', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
                out=tf.compat.v1.layers.conv2d(out, filters=12, kernel_size=3, strides=(2, 2), padding='valid', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
                out=tf.compat.v1.layers.conv2d(out, filters=20, kernel_size=3, strides=(1, 1), padding='valid', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
                conv_out = tf.compat.v1.layers.flatten(out)
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [conv_out.shape[1], n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(conv_out, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, ddone, s_): #Jie...      add "done"
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r, ddone], s_)) #Jie...      add "done"
        #print("transition")
        #print(transition)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, action_option = None): # action_option should be a list
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            #print(actions_value)
            #print("action ",action,actions_value.shape)
        else:
            # self.n_actions
            # actions_value = np.array([[0,0,0]])
            actions_value = np.zeros((1, self.n_actions))
            if type(action_option) is list:
                ind = np.random.randint(0, len(action_option))
                action = action_option[ind]
            else:
                action = np.random.randint(0, self.n_actions)
        return action, actions_value

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        ddone = batch_memory[:, self.n_features + 2] #Jie...      load information "done"
        idt = np.where(ddone)
        idf = np.where(np.logical_not(ddone))
        q_target[idt,eval_act_index[idt]] = reward[idt]
        q_target[idf,eval_act_index[idf]] = reward[idf] + self.gamma*np.max(q_next[idf[0],:],axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
        [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
        [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
        [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
        [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                    feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        if self.epsilon < 0.4:
            epsilon_increment = self.epsilon_increment / self.increment_degradation[0]
        elif self.epsilon < 0.6:
            epsilon_increment = self.epsilon_increment / self.increment_degradation[1]
        elif self.epsilon < 0.8:
            epsilon_increment = self.epsilon_increment / self.increment_degradation[2]
        else: # 0.8-1.0
            epsilon_increment = self.epsilon_increment / self.increment_degradation[3]

        self.epsilon = self.epsilon + epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        # print("\tepsilon=========   ",self.epsilon, " ", self.epsilon_increment, " ", self.epsilon_max)
        
        self.learn_step_counter += 1

    def plot_cost(self,dir_file,model_index=1,visualize=True):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig(dir_file+'Cost_'+str(model_index)+'.png')
        if visualize == True:
            plt.show()
        
        tmp_array = np.array(self.cost_his)
        np.savetxt(dir_file+'training_cost_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
        



