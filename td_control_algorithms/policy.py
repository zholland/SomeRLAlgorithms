from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    def __init__(self, env, action_value_function):
        self.env = env
        self.action_value_function = action_value_function

    @abstractmethod
    def select_action(self, S):
        """"""

    @abstractmethod
    def action_probability(self, A, S):
        """"""

    @abstractmethod
    def adjust_policy(self):
        """"""


class GreedyPolicy(Policy):
    def adjust_policy(self):
        pass

    def action_probability(self, A, S):
        return 1.0 if np.argmax(self.action_value_function.action_values(S)) == A else 0.0

    def select_action(self, S):
        return np.argmax(self.action_value_function.action_values(S))


class RandomPolicy(Policy):
    def adjust_policy(self):
        pass

    def action_probability(self, A, S):
        return 1 / self.env.action_space.n

    def select_action(self, S):
        return self.env.action_space.sample()


class EpsilonGreedyPolicy(Policy):
    def __init__(self, env, action_value_function, epsilon, epsilon_decay_factor):
        super().__init__(env, action_value_function)
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor

    def select_action(self, S):
        return self.epsilon_greedy_action(S)

    def adjust_policy(self):
        self.epsilon *= self.epsilon_decay_factor

    def action_probability(self, A, S):
        return self.epsilon_greedy_probability(A, S)

    def epsilon_greedy_probability(self, A, S):
        """
        Returns the probability of choosing action A in state S under the epsilon greedy policy.
        """
        greedy_action = np.argmax(self.action_value_function.action_values(S))
        epsilon_fraction = self.epsilon / self.env.action_space.n
        return 1 - self.epsilon + epsilon_fraction if A == greedy_action else epsilon_fraction

    def epsilon_greedy_action(self, S):
        """
        Returns an action selected by the epsilon greedy policy.
        """
        if self.epsilon > np.random.random():
            # Get action randomly
            return self.env.action_space.sample()
        else:
            # Get action greedily according to the action value function
            return np.argmax(self.action_value_function.action_values(S))
