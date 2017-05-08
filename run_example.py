import td_control_algorithms as td
import gym
import numpy as np


def main():
    env = gym.make('WindyGridworld-v0')

    agent = td.RLSolutionMethod.factory(
        method_type=td.ExpectedSarsa,
        env=env,
        learning_rate=0.5,
        epsilon=0.1,
        lambda_=0.5)

    agent.do_learning(num_episodes=100,
                      target_return=-10.0,
                      target_window=10)
    print(np.mean(agent.episode_return))


if __name__ == "__main__":
    main()
