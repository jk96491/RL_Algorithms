import gym
from Pytorch.PPO.ppo_agent import PPOAgent


def main():

    max_episode_num = 1000
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = PPOAgent(env)

    agent.train(max_episode_num)

    agent.plot_Result()


if __name__=="__main__":
    main()