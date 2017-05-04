import td_control_algorithms


class ExpectedSarsa(td_control_algorithms.QSigma):
    def __init__(self, env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, n=1, sigma=0)
