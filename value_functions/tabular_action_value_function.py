from value_functions.action_value_function import AbstractActionValueFunction

import random
import numpy as np


class TabularActionValueFunction(AbstractActionValueFunction):
    def __init__(self, num_states, num_actions):
        self.theta = [0.0001 * random.uniform(-1, 1) for _ in range(num_states*num_actions)]
        self.theta = np.asarray(self.theta)
        # for i in range(self.num_actions):
        #     self.theta[terminal_state+(i*num_states)] = 0
        self.num_actions = num_actions
        self.num_states = num_states

    def action_values(self, S):
        values = np.ndarray([self.num_actions])
        for i in range(self.num_actions):
            values[i] = self.theta[S+(i*self.num_states)]
        return values

    def value(self, S, A):
        return self.theta[S+(A*self.num_states)]

    def update(self, S, A, new_value):
        self.theta[S+(A*self.num_states)] += new_value

    def feature_vector(self, S, A):
        features = np.zeros([self.num_actions*self.num_states])
        features[S+(A*self.num_states)] = 1
        return features
