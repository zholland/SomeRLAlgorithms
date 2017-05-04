import numpy as np

import td_control_algorithms


class TrueOnlineSarsaLambda(td_control_algorithms.CompoundMultiStepMethod):
    def __init__(self, env, action_value_function, alpha, epsilon=0.0, epsilon_decay_factor=1.0, gamma=1.0,
                 lambda_=0.0):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, lambda_)

    def do_learning(self, num_episodes, target_return, target_window, show_env=False):
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            A = self.epsilon_greedy_action(S)
            done = False
            Rsum = 0
            psi = self.action_value_function.feature_vector(S, A)
            e = np.zeros(self.action_value_function.theta.size)
            Q_old = 0
            while not done:
                if show_env:
                    self.env.render()
                Snext, R, done, info = self.env.step(A)
                Rsum += R
                Anext = self.epsilon_greedy_action(Snext)

                psi_prime = self.action_value_function.feature_vector(Snext, Anext)

                Q = np.dot(psi, self.action_value_function.theta)
                Q_prime = np.dot(psi_prime, self.action_value_function.theta)
                delta = R + self.gamma * Q_prime - Q
                e = self.gamma * self.lambda_ * e + psi - self.alpha * self.gamma * self.lambda_ * np.dot(e, psi) * psi

                self.action_value_function.theta = self.action_value_function.theta + self.alpha * (
                    delta + Q - Q_old) * e - self.alpha * (Q - Q_old) * psi
                Q_old = Q_prime
                psi = psi_prime
                A = Anext

            self.epsilon = self.epsilon * self.epsilon_decay_factor
            self.episode_return.append(Rsum)
            if episodeNum >= target_window and np.mean(
                    self.episode_return[episodeNum - target_window:episodeNum]) > target_return:
                break
