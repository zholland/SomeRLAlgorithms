import td_control_algorithms
import numpy as np


class Qlearning(td_control_algorithms.AtomicMultiStepMethod):
    def __init__(self, env, action_value_function, alpha, behaviour_policy, gamma):
        super().__init__(env, action_value_function, alpha, None, behaviour_policy, gamma)

    def do_learning(self, num_episodes, target_return, target_window, show_env=False):
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            A = self.behaviour_policy.select_action(S)
            done = False
            Rsum = 0
            while not done:
                if show_env:
                    self.env.render()
                Snext, R, done, info = self.env.step(A)
                Rsum += R
                Anext = self.behaviour_policy.select_action(Snext)
                self.action_value_function.update(S, A, self.alpha * (
                R + self.gamma * np.max(self.action_value_function.action_values(Snext)) - self.action_value_function.value(S,A)))
                S = Snext
                A = Anext
            self.behaviour_policy.adjust_policy()
            self.episode_return.append(Rsum)
            if episodeNum >= target_window and np.mean(
                    self.episode_return[episodeNum - target_window:episodeNum]) > target_return:
                break
