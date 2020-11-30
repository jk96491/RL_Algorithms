from time import time
import gym
from Pytorch.SAC.Agent import SAC

ENV_ID = 'Pendulum-v0'
env = gym.make(ENV_ID)
seed = 0
env.seed(seed)

agent = SAC(env, entropy_tune=True, reward_scale=5.0, writer=None)

max_episode = 1000

# Train Step
cumulative_reward = 0


for episode in range(max_episode):
    done = False
    time = 0
    state = env.reset()

    while not done:
        action, _ = agent.get_action(state)
        next_state, reward, done, _ = env.step(action * env.action_space.high[0])
        cumulative_reward += reward

        count = agent.store_transition(state, action, next_state, reward, False)

        if count % 16 == 15:
            loss_critic, loss_actor, loss_entropy = agent.update()

        state = next_state
        time += 1

    print('episode:{:<3}     Time : {}     score : {}'.format(episode + 1, time, cumulative_reward))
    cumulative_reward = 0

