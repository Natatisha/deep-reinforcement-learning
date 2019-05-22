from agent import Agent
from monitor import interact
import gym
from plot import plot_rewards

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

plot_rewards(avg_rewards)
