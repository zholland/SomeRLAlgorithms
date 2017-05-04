import td_control_algorithms
import numpy as np


class QSigma(td_control_algorithms.AtomicMultiStepMethod):
    """
    Implementation of n-step Q(Sigma) with epsilon greedy behaviour and target policies
    """
    def __init__(self, env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, n, sigma):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, n)
        self.sigma = sigma

    def do_learning(self, num_episodes, target_return, target_window,show_env=False):
        for episodeNum in range(num_episodes):
            S = [0]*(self.n+1)
            S[0] = self.env.reset()
            A = [0]*(self.n+1)
            A[0] = self.epsilon_greedy_action(S[0])
            Q = [0]*(self.n+1)
            Q[0] = self.action_value_function.value(S[0], A[0])
            Rsum = 0
            T = float("inf")
            t = 0
            tau = 0
            delta = [0]*self.n
            # rho = [0]*self.n
            pi = [0]*self.n
            while tau != T - 1:
                if t < T:
                    if show_env:
                        self.env.render()
                    Snext, Rnext, done, info = self.env.step(A[t % (self.n+1)])
                    S[(t+1) % (self.n+1)] = Snext
                    Rsum += Rnext
                    if done:  # If we are in the terminal state
                        T = t + 1
                        delta[t % self.n] = Rnext - self.action_value_function.value(S[t % (self.n+1)], A[t % (self.n+1)])
                    else:
                        A[(t+1) % (self.n + 1)] = self.epsilon_greedy_action(Snext)
                        Q[(t+1) % (self.n + 1)] = self.action_value_function.value(S[(t+1) % (self.n + 1)], A[(t+1) % (self.n + 1)])
                        temp_sum = 0
                        for a in range(0, self.env.action_space.n):
                            temp_sum += self.epsilon_greedy_probability(a, S[(t+1) % (self.n + 1)]) * self.action_value_function.value(S[(t+1) % (self.n + 1)], a)
                        # delta[t % self.n] = Rnext + self.gamma * self.sigma.value(episodeNum) * Q[(t+1) % (self.n + 1)] + self.gamma * (1 - self.sigma.value(episodeNum)) * temp_sum - Q[t % (self.n + 1)]
                        delta[t % self.n] = Rnext + self.gamma * self.sigma * Q[(t+1) % (self.n + 1)] + self.gamma * (1 - self.sigma) * temp_sum - Q[t % (self.n + 1)]
                        pi[(t + 1) % self.n] = self.epsilon_greedy_probability(A[(t + 1) % (self.n + 1)], S[(t + 1) % (self.n + 1)])
                        # rho[(t + 1) % self.n] = pi[(t+1) % self.n]/self.epsilon_greedy_probability(A[(t + 1) % (self.n + 1)], S[(t + 1) % (self.n + 1)])
                tau = t - self.n + 1
                if tau >= 0:
                    # rho_var = 1
                    E = 1
                    G = Q[tau % (self.n + 1)]
                    for k in range(tau, min([tau + self.n, T])):
                        G += E * delta[k % self.n]
                        # E = self.gamma * E * ((1-self.sigma.value(episodeNum))*pi[(k + 1) % self.n]+self.sigma.value(episodeNum))
                        E = self.gamma * E * ((1-self.sigma)*pi[(k + 1) % self.n]+self.sigma)
                        # rho_var = rho_var * (1-self.sigma.value(episodeNum)+self.sigma.value(episodeNum)*rho[k % self.n])
                    self.action_value_function.update(S[tau % (self.n + 1)], A[tau % (self.n + 1)], self.alpha * (G - self.action_value_function.value(S[tau % (self.n + 1)], A[tau % (self.n + 1)])))
                t += 1
            self.episode_return.append(Rsum)
            self.epsilon *= self.epsilon_decay_factor
            if episodeNum >= target_window and np.mean(self.episode_return[episodeNum - target_window:episodeNum]) > target_return:
                break
