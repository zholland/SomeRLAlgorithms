from abc import ABC, abstractmethod
import numpy as np
import gym.spaces
import value_functions
import td_control_algorithms


class RLSolutionMethod(ABC):
    """
    Base class for creating TD control methods.
    """

    @staticmethod
    def factory(method_type, env, learning_rate, epsilon=0.0, epsilon_decay_factor=1.0, gamma=1.0, n=1, lambda_=0.0,
                scale_inputs=False, num_tiles=2 ** 11, num_tilings=None, sigma=0.0):
        if type(env.action_space) is gym.spaces.Discrete:
            if type(env.observation_space) is gym.spaces.Discrete:
                action_value_function = \
                    value_functions.TabularActionValueFunction(env.observation_space.n,
                                                               env.action_space.n)
            else:
                if num_tilings is None:
                    num_tilings = RLSolutionMethod.compute_num_tilings(env)
                    learning_rate /= num_tilings
                action_value_function = \
                    value_functions.TileCodingActionValueFunction(env.observation_space.shape[0],
                                                                       RLSolutionMethod.get_input_ranges(env),
                                                                       num_actions=env.action_space.n,
                                                                       num_tiles=num_tiles,
                                                                       num_tilings=num_tilings,
                                                                       scale_inputs=scale_inputs)
            if method_type == "TrueOnlineSarsaLambda":
                return td_control_algorithms.TrueOnlineSarsaLambda(env,
                                                                   action_value_function,
                                                                   learning_rate,
                                                                   epsilon,
                                                                   epsilon_decay_factor,
                                                                   gamma,
                                                                   lambda_)
            if method_type == "Qlearning":
                return td_control_algorithms.Qlearning(env,
                                                       action_value_function,
                                                       learning_rate,
                                                       epsilon,
                                                       epsilon_decay_factor,
                                                       gamma)
            if method_type == "QSigma":
                return td_control_algorithms.QSigma(env,
                                                    action_value_function,
                                                    learning_rate,
                                                    epsilon,
                                                    epsilon_decay_factor,
                                                    gamma,
                                                    n,
                                                    sigma)
            if method_type == "Sarsa":
                return td_control_algorithms.Sarsa(env,
                                                   action_value_function,
                                                   learning_rate,
                                                   epsilon,
                                                   epsilon_decay_factor,
                                                   gamma)
            if method_type == "nStepSarsa":
                return td_control_algorithms.nStepSarsa(env,
                                                        action_value_function,
                                                        learning_rate,
                                                        epsilon,
                                                        epsilon_decay_factor,
                                                        gamma,
                                                        n)
            if method_type == "TreeBackup":
                return td_control_algorithms.TreeBackup(env,
                                                        action_value_function,
                                                        learning_rate,
                                                        epsilon,
                                                        epsilon_decay_factor,
                                                        gamma,
                                                        n)
            if method_type == "ExpectedSarsa":
                return td_control_algorithms.ExpectedSarsa(env,
                                                           action_value_function,
                                                           learning_rate,
                                                           epsilon,
                                                           epsilon_decay_factor,
                                                           gamma)

    @staticmethod
    def compute_num_tilings(env):
        i = 2
        while 2 ** i < 4 * env.observation_space.shape[0]:
            i += 1
        return 2 ** i

    @staticmethod
    def get_input_ranges(env):
        dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                      range(0, env.observation_space.high.size)]
        return dim_ranges

    def __init__(self, env, action_value_function, alpha, epsilon=0.0, epsilon_decay_factor=1.0, gamma=1.0):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.gamma = gamma
        self.action_value_function = action_value_function
        self.episode_return = []

    @abstractmethod
    def do_learning(self, num_episodes, target_return, target_window, show_env=False):
        """"""

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

    def random_action(self):
        return self.env.action_space.sample()

    def random_action_probability(self):
        return 1 / self.env.action_space.n

    def value(self, S):
        """
        Returns the value of state S
        """
        value = 0
        for a in range(0, self.env.action_space):
            value += self.epsilon_greedy_probability(a, S) * self.action_value_function.value(S, a)
        return value / self.env.action_space

    def max_action_value(self, S):
        return np.max(self.action_value_function.action_values(S))


class CompoundMultiStepMethod(RLSolutionMethod):
    """
    Base class for creating TD control methods using compound backups with eligibility traces.
    """

    def __init__(self, env, action_value_function, alpha, epsilon=0.0, epsilon_decay_factor=1.0, gamma=1.0,
                 lambda_=0.0):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma)
        self.lambda_ = lambda_

    @abstractmethod
    def do_learning(self, num_episodes, target_return, target_window, show_env=False):
        """"""


class AtomicMultiStepMethod(RLSolutionMethod):
    """
    Base class for creating n-step TD control methods.
    """

    def __init__(self, env, action_value_function, alpha, epsilon=0.0, epsilon_decay_factor=1.0, gamma=1.0, n=1):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma)
        self.n = n

    @abstractmethod
    def do_learning(self, num_episodes, target_return, target_window, show_env=False):
        """"""
