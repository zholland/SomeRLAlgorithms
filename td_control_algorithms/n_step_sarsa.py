import td_control_algorithms


class nStepSarsa(td_control_algorithms.QSigma):
    def __init__(self, env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, n):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, n, sigma=1)
