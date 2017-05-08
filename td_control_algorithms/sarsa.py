import td_control_algorithms


class Sarsa(td_control_algorithms.QSigma):
    def __init__(self, env, action_value_function, alpha, target_policy, behaviour_policy, gamma):
        super().__init__(env, action_value_function, alpha, target_policy, behaviour_policy, gamma, n=1, sigma=td_control_algorithms.Sigma(1))
