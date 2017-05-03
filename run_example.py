import td_control_algorithms as td
import gym
import numpy as np


def main():
    env = gym.make('MountainCar-v0')
    agent = td.RLSolutionMethod.factory(
        method_type=td.QSigma.__name__,
        env=env,
        learning_rate=0.9,
        scale_inputs=True)
    agent.do_learning(num_episodes=500,
                      target_return=-110.0,
                      target_window=10)
    print(np.mean(agent.episode_return))


if __name__ == "__main__":
    main()
