import td_control_algorithms as td


class ExpectedSarsa(td.QSigma):
    def __init__(self, env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma):
        super().__init__(env, action_value_function, alpha, epsilon, epsilon_decay_factor, gamma, n=1, sigma=td.Sigma(0))
