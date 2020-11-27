import gym
from Tensorflow.A2C.a2c_agent import A2Cagnet


def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = A2Cagnet(env)

    agent.train(max_episode_num)

    agent.plot_Result()


if __name__ == "__main__":
    main()