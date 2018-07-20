

import tensorflow as tf
import numpy as np
from pysc2.lib import actions
from utils import Constants, print_tensors


class StateNet:
    def __init__(self, scope, action_size=len(Constants.DEFAULT_ACTIONS),
                 max_multi_select=Constants.MAXIMUM_MULTI_SELECT,
                 max_cargo=Constants.MAXIMUM_CARGO,
                 max_build_queue=Constants.MAXIMUM_BUILD_QUEUE,
                 resolution=84, screen_channels=17, minimap_channels=7,
                 l2_scale=0.001, hidden_size=128, init=tf.contrib.layers.xavier_initializer()):
        self.resolution = resolution
        self.action_size = action_size
        self.variable_feature_sizes = {
            'multi_select': max_multi_select,
            'cargo': max_cargo,
            'build_queue': max_build_queue
        }
        # The following assumes that we will stack our minimap and screen features (and they will have the same size)
        with tf.variable_scope(scope):
            self.structured_observation = tf.placeholder(tf.float32, [None, 11], 'StructuredObservation')
            self.single_select = tf.placeholder(tf.float32, [None, 1, Constants.UNIT_ELEMENTS], 'SingleSelect')
            self.cargo = tf.placeholder(tf.float32, [None, max_cargo, Constants.UNIT_ELEMENTS], 'Cargo')
            self.multi_select = tf.placeholder(tf.float32,
                                               [None, max_multi_select, Constants.UNIT_ELEMENTS],
                                               'MultiSelect')
            self.build_queue = tf.placeholder(tf.float32,
                                              [None, max_build_queue, Constants.UNIT_ELEMENTS],
                                              'BuildQueue')
            self.units = tf.concat([self.single_select,
                                    self.multi_select,
                                    self.cargo,
                                    self.build_queue], axis=1,
                                   name='Units')
            self.control_groups = tf.placeholder(tf.float32, [None, 10, 2], 'ControlGroups')
            self.available_actions = tf.placeholder(tf.float32, [None, self.action_size], 'AvailableActions')
            self.used_actions = tf.placeholder(tf.float32, [None, self.action_size], 'UsedActions')
            self.actions = tf.concat([self.available_actions,
                                      self.used_actions], axis=1,
                                     name='Actions')
            self.nonspatial_features = tf.concat([
                self.structured_observation,
                tf.reshape(self.units, [-1, Constants.UNIT_ELEMENTS * (1 + sum(self.variable_feature_sizes.values()))]),
                tf.reshape(self.control_groups, [-1, 20]),
                tf.reshape(self.actions, [-1, 2 * self.action_size])
            ], axis=1, name='NonspatialFeatures')

            self.screen_features = tf.placeholder(tf.float32,
                                                  [None, screen_channels, resolution, resolution],
                                                  'ScreenFeatures')

            self.minimap_features = tf.placeholder(tf.float32,
                                                   [None, minimap_channels, resolution, resolution],
                                                   'MinimapFeatures')

            self.spatial_features = tf.concat([
                self.screen_features, self.minimap_features
            ], axis=1, name='SpatialFeatures')

            self.spatial_features = tf.transpose(self.spatial_features, [0, 2, 3, 1])

            self.conv1 = tf.layers.conv2d(inputs=self.spatial_features, filters=32,
                                          kernel_size=[5, 5],
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
                                          kernel_initializer=init,
                                          bias_initializer=init,
                                          activation=tf.nn.relu, name='Convolutional1')
            self.max_pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2],
                                                     strides=2, name='Pool1')
            self.conv2 = tf.layers.conv2d(inputs=self.max_pool1, filters=64,
                                          kernel_size=[5, 5],
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
                                          kernel_initializer=init,
                                          bias_initializer=init,
                                          activation=tf.nn.relu, name='Convolutional2')
            self.max_pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2],
                                                     strides=2, name='Pool2')
            self.max_pool2_flat = tf.reshape(self.max_pool2, [-1, 18 * 18 * 64], name='Pool2_Flattened')
            self.state_flattened = tf.concat([self.max_pool2_flat, self.nonspatial_features],
                                             1, name='StateFlattened')
            self.hidden_1 = tf.layers.dense(inputs=self.state_flattened,
                                            units=hidden_size,
                                            activation=tf.nn.relu,
                                            kernel_initializer=init,
                                            bias_initializer=init,
                                            name='Hidden1')
            self.output = self.hidden_1
            print_tensors(self)


class RecurrentNet:
    def __init__(self, scope, state_net, lstm_size=256):
        with tf.variable_scope(scope):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            current_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            hidden_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [current_init, hidden_init]
            current_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            hidden_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (current_in, hidden_in)
            rnn_in = tf.expand_dims(state_net.output, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(current_in, hidden_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in,
                time_major=False)
            lstm_current, lstm_hidden = lstm_state
            print('Shape of Mystery Error = {}'.format(tf.shape(tf.reshape(lstm_outputs, [-1, lstm_size]))))
            print('Shape of lstm_outputs = {}'.format(tf.shape(lstm_outputs)))
            self.state_out = (lstm_current[:1, :], lstm_hidden[:1, :])
            self.output = tf.reshape(lstm_outputs, [-1, lstm_size])
            print_tensors(self)


class A3CNet:
    def __init__(self, scope, state_net, rnn):
        with tf.variable_scope(scope):
            with tf.variable_scope('Critic-{}'.format(scope)):
                self.value = tf.layers.dense(rnn.output, 1, activation=None)
            with tf.variable_scope('Actor-{}'.format(scope)):
                self.policy = tf.layers.dense(inputs=rnn.output, units=state_net.action_size,
                                              activation=tf.nn.softmax, name='ActionPolicy')
                self.policy = tf.nn.softmax(tf.multiply(self.policy, state_net.available_actions))
                print('Policy: {}'.format(self.policy.shape))
                self.action = tf.argmax(self.policy, axis=1)
                self.arguments = {}
                self.argument_choices = {}
                for argument in actions.TYPES:
                    self.arguments[argument.name] = []
                    for dimension, size in enumerate(argument.sizes):
                        arg_layer = tf.layers.dense(
                          inputs=rnn.output, units=size if size != 0 else state_net.resolution,
                          activation=tf.nn.elu,
                          name='{}-{}'.format(argument.name, dimension))
                        soft_arg = tf.nn.softmax(arg_layer)
                        print(arg_layer)
                        self.arguments[argument.name].append(soft_arg)
                    
                    for arg in self.arguments[argument.name]:
                        print('Arg {} shape: {}'.format(argument.name, arg.shape))
                    self.argument_choices[argument.name] = [
                      tf.argmax(self.arguments[argument.name][dimension],axis=1)
                      for dimension, size in enumerate(argument.sizes)
                    ]
                    
        print_tensors(self)


class A3CWorker(A3CNet):
    def __init__(self, scope, state_net, rnn,
                 value_loss_weight=0.5, policy_loss_weight=1,
                 entropy_weight=0.01, gradient_norm=40,
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
        super().__init__(scope, state_net, rnn)
        with tf.variable_scope(scope):
            self.value_target = tf.placeholder(tf.float32, [None], name='ValueTarget')
            self.advantages = tf.placeholder(tf.float32, [None], name='Advantages')

            self.chosen_action = tf.placeholder(tf.int32, [None], name='ChosenAction')
            chosen_action_one_hot = tf.one_hot(self.chosen_action,
                                               state_net.action_size,
                                               dtype=tf.float32,
                                               name='ChosenActionOneHot')
            #
            #  onehot(3,5)  ->  [0,0,0,1,0]

            responsible_action = tf.reduce_sum(self.policy * chosen_action_one_hot,
                                               axis=[1], name='ResponsibleAction')
            self.action_loss = -tf.reduce_sum(tf.log(responsible_action) * self.advantages,
                                         name='ActionLoss')
            entropy_action = -tf.reduce_sum(self.policy * tf.log(self.policy), name='EntropyAction')
            self.chosen_arguments = {}
            self.used_arguments = {}
            zero = tf.constant(0, tf.float32, shape=[1], name='Zero')
            self.responsible_arguments = 1e-20                                    
            entropy_arguments = zero
            self.arguments_loss = zero
            for argument in actions.TYPES:
                for dimension, size in enumerate(argument.sizes):
                    chosen_argument = tf.placeholder_with_default(tf.cast(zero, tf.int32),
                                                                  shape=[None],
                                                                  name='Chosen{}-{}'.format(argument.name,
                                                                                            dimension))
                    chosen_argument_one_hot = tf.one_hot(chosen_argument,
                                                         size if size != 0 else state_net.resolution,
                                                         dtype=tf.float32,
                                                         name='ChosenOneHot{}-{}'.format(argument.name,
                                                                                         dimension))
                    used_argument = tf.placeholder_with_default(zero,
                                                                shape=[None],
                                                                name='Used{}-{}'.format(argument.name,
                                                                                        dimension))
                    self.responsible_arguments += tf.reduce_sum(#used_argument *
                                                           chosen_argument_one_hot *
                                                           self.arguments[argument.name][dimension],
                                                           axis=[1], name='Responsible{}-{}'.format(argument.name,
                                                                                                    dimension))
                    entropy_arguments -= tf.reduce_sum(#used_argument *
                                                       self.arguments[argument.name][dimension] *
                                                       tf.log(self.arguments[argument.name][dimension]),
                                                       axis=[1], name='Entropy{}-{}'.format(argument.name,
                                                                                            dimension))
                    self.chosen_arguments['{}-{}'.format(argument.name, dimension)] = chosen_argument
                    self.used_arguments['{}-{}'.format(argument.name, dimension)] = used_argument
            
            self.arguments_loss = tf.reduce_sum(#used_argument *
                                            tf.log(self.responsible_arguments) * self.advantages,
                                            name='ArgumentsLoss')
            # self.value_loss = tf.losses.mean_squared_error(self.value_target, tf.squeeze(self.value), scope=scope)
            self.value_loss = tf.reduce_sum(tf.square(self.value_target - tf.reshape(self.value, [-1])))
            self.entropy = entropy_action + entropy_arguments
            self.entropy = tf.reduce_sum(self.entropy)
            self.policy_loss = self.action_loss + self.arguments_loss
            self.loss = tf.reduce_mean(self.value_loss * value_loss_weight +
                                       self.policy_loss * policy_loss_weight -
                                       self.entropy * entropy_weight)
            local_variables = tf.trainable_variables(scope)
            self.gradients = tf.gradients(self.loss, local_variables)
            self.var_norms = tf.global_norm(local_variables)
            gradients, self.gradient_norms = tf.clip_by_global_norm(self.gradients, gradient_norm)
            global_variables = tf.trainable_variables(Constants.GLOBAL_SCOPE)
            self.apply_gradients = optimizer.apply_gradients(zip(gradients, global_variables))
            self.pull_global_variables = [local_variable.assign(global_variable)
                                          for local_variable, global_variable in
                                          zip(local_variables, global_variables)]

