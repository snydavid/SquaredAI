#https://github.com/greentfrapp/pysc2-RLagents/blob/master/Agents/PySC2_A3C_FullyConv.py

import tensorflow as tf
import numpy as np
import os
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from pysc2.env import environment
from utils import Constants, print_tensors, discount_rewards
from networks import StateNet, RecurrentNet, A3CNet, A3CWorker


# python3 -m pysc2.bin.agent --map MoveToBeacon --agent MyAgent.A3CAgent --feature_minimap_size 84 --nosave_replay


folder_path = './sc2-pg/'

class Brain:
    def __init__(self, race="T", action_set=Constants.DEFAULT_ACTIONS):
        self.race = race
        self.action_set = sorted(action_set)

    def reset(self):
        pass

    def step(self, obs):
        return 0, []


# This represents the actual agent which will play StarCraft II
class MyAgent(BaseAgent):
    def __init__(self, brain=Brain()):
        super().__init__()  # call parent constructor
        assert isinstance(brain, Brain)
        self.brain = brain

    def reset(self):
        self.brain.reset()

    def step(self, obs):  # This function is called once per frame to give the AI observation data and return its action
        super().step(obs)  # call parent base method
        action, params = self.brain.step(obs)
        return actions.FunctionCall(action, params)


class A3CBrain(Brain):
    global_brain = None
    sess = None


    def __init__(self, scope, race="T", action_set=Constants.DEFAULT_ACTIONS, step_buffer_size=79,
                 epsilon = 0, save_frequency = 20):
        super().__init__(race, action_set)
        if not A3CBrain.global_brain:

            A3CBrain.global_brain = 1
            A3CBrain.global_brain = A3CBrain(Constants.GLOBAL_SCOPE, race, action_set,step_buffer_size,epsilon)
        self.state_net = StateNet(scope, action_size=len(self.action_set))
        self.rnn = RecurrentNet(scope, self.state_net)
        self.a3c_net = A3CWorker(scope, self.state_net, self.rnn)
        if not A3CBrain.sess:
            A3CBrain.sess = tf.Session()
        self.sess = A3CBrain.sess
        self.sess.run(tf.global_variables_initializer())
        self.feature_placeholders = {
            'available_actions': self.state_net.available_actions,
            'last_actions': self.state_net.used_actions,
            'cargo': self.state_net.cargo,
            'multi_select': self.state_net.multi_select,
            'single_select': self.state_net.single_select,
            'build_queue': self.state_net.build_queue,
            'player': self.state_net.structured_observation,
            'control_groups': self.state_net.control_groups,
            'feature_screen': self.state_net.screen_features,
            'feature_minimap': self.state_net.minimap_features
        }
        self.step_buffer_size = step_buffer_size
        self.step_buffer = []
        self.episode_rewards = 0
        self.epsilon = epsilon
        self.rnn_state = self.rnn.state_init
        self.batch_rnn_state = self.rnn_state
        self.previous_state = None
        self.sess.run(self.a3c_net.pull_global_variables)
        self.episode = 1
        self.saver = tf.train.Saver(max_to_keep=5)
        checkpoint = tf.train.get_checkpoint_state(folder_path)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        self.save_frequency = save_frequency


    def reset(self):
        self.step_buffer.clear()
        self.episode_rewards = 0
        self.rnn_state = self.rnn.state_init
        self.batch_rnn_state = self.rnn_state

    def train(self, gamma=0.99, bootstrap_value = 0):
        # do some data formatting
        reward_array = [reward for state, reward, action, arguments, value_estimate in self.step_buffer]
        value_array = [value_estimate[0] for state, reward, action, arguments, value_estimate in self.step_buffer]
        rewards_plus = np.asarray(reward_array + [bootstrap_value])
        discounted_rewards = discount_rewards(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(value_array + [bootstrap_value])
        advantages = np.array(reward_array) + gamma * value_plus[1:] - value_plus[:-1]
        advantages = discount_rewards(advantages, gamma)
        feed_dict = {
            self.a3c_net.value_target: discounted_rewards,
            self.a3c_net.advantages: advantages,
            self.a3c_net.chosen_action: [action[0]
                                         for state, reward, action, arguments, value_estimate in self.step_buffer],
            self.rnn.state_in[0]: self.batch_rnn_state[0],
            self.rnn.state_in[1]: self.batch_rnn_state[1]
        }
        chosen_arguments = [arguments for state, reward, action, arguments, value_estimate in self.step_buffer]
        processed_args = {}
        for argument in actions.TYPES:
            for dimension in range(len(argument.sizes)):
                processed_args['{}-{}'.format(argument.name, dimension)] = [
                    chosen_argument[argument.name][dimension][0]
                    for chosen_argument in chosen_arguments
                ]
                feed_dict[self.a3c_net.chosen_arguments['{}-{}'.format(argument.name, dimension)]] = processed_args['{}-{}'.format(argument.name, dimension)]

        for text_label, feature_label in self.feature_placeholders.items():
            feed_dict[feature_label] = np.array(
                [state[feature_label][0] for state, reward, action, arguments, value_estimate in self.step_buffer])
        # feed into nn
        # run train op
        applied, pulled, self.batch_rnn_state, entropy, value_loss, policy_loss, action_loss, argument_loss, loss = self.sess.run([self.a3c_net.apply_gradients, self.a3c_net.pull_global_variables,
                                                                     self.rnn.state_out, self.a3c_net.entropy, self.a3c_net.value_loss, self.a3c_net.policy_loss, self.a3c_net.action_loss, self.a3c_net.arguments_loss, self.a3c_net.loss], feed_dict)
        #loss = self.sess.run(self.a3c_net.loss, feed_dict)
        print('Entropy:', entropy, 'Value Loss:', value_loss, 'Policy Loss:', policy_loss, 'Action Loss:', action_loss, 'Arg Loss:', argument_loss, 'Loss:', loss)
        
        self.step_buffer.clear()

    def step(self, obs):
        reward, feed_dict, episode_end = self.process_observations(obs)
        self.episode_rewards += reward

        action, arguments, self.rnn_state, value_estimate = self.sess.run([self.a3c_net.action,
                                                                           self.a3c_net.argument_choices,
                                                                           self.rnn.state_out,
                                                                           self.a3c_net.value],
                                                                          feed_dict)
        #print('Action: {}'.format(action))
        final_action = Constants.DEFAULT_ACTIONS[action[0]]
        #print('Final Action: {}'.format(final_action))
        if np.random.uniform()<self.epsilon:
            final_action =  int(np.random.uniform()*len(Constants.DEFAULT_ACTIONS))
        final_args = []
        for arg in actions.FUNCTIONS[final_action].args:
            if arg.name in ['screen', 'minimap', 'screen2']:
                final_args.append([arguments[arg.name][0][0], arguments[arg.name][1][0]])
            else:
                final_args.append(arguments[arg.name][0])
        #print(final_args)
        final_arguments = [arg[0] for arg in arguments['screen']]
        if(self.previous_state):
            self.step_buffer.append((self.previous_state[0], reward, self.previous_state[1],
                                     self.previous_state[2], self.previous_state[3]))
        if (len(self.step_buffer) >= self.step_buffer_size):
            if(episode_end):
                self.train()
            else:
                self.train(bootstrap_value=value_estimate)
            #pass
        #print('Value estimate:', value_estimate[0])
        self.previous_state = (feed_dict, action, arguments, value_estimate[0])
        if episode_end:
            self.episode+=1
            if self.episode%self.save_frequency == 0:
                pass
                self.saver.save(self.sess, folder_path + 'pg-checkpoint', self.episode)

        return final_action, final_args

    def process_observations(self, observation):
        # is episode over?
        episode_end = (observation.step_type == environment.StepType.LAST)
        # reward
        reward = observation.reward  # scalar?
        # features
        features = observation.observation
        # the shapes of some features depend on the state (eg. shape of multi_select depends on number of units)
        # since tf requires fixed input shapes, we set a maximum size then pad the input if it falls short
        processed_features = {
            self.rnn.state_in[0]: self.rnn_state[0],
            self.rnn.state_in[1]: self.rnn_state[1]
        }

        for feature_label in self.feature_placeholders:
            feature = features[feature_label]
            if feature_label in ['available_actions', 'last_actions']:
                action_inputs = np.zeros(len(self.action_set))
                for i, action in enumerate(self.action_set):
                    if action in feature:
                        action_inputs[i] = 1
                feature = action_inputs
            elif feature_label in ['single_select', 'multi_select', 'cargo', 'build_queue']:
                if feature_label in self.state_net.variable_feature_sizes:
                    padding = np.zeros(
                        (self.state_net.variable_feature_sizes[feature_label] - len(feature), Constants.UNIT_ELEMENTS))
                    feature = np.concatenate((feature, padding))
                feature = feature.reshape(-1, Constants.UNIT_ELEMENTS)
            placeholder = self.feature_placeholders[feature_label]
            processed_features[placeholder] = np.expand_dims(feature, axis=0)
        return reward, processed_features, episode_end


class A3CAgent(MyAgent):
    number_agents = 1
    def __init__(self, name='test'):  ## was "test"
        super().__init__(A3CBrain(name+str(A3CAgent.number_agents)))
        A3CAgent.number_agents += 1
      
