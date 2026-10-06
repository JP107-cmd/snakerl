import gymnasium as gym
import snake_env  # registers Snake-v0

from gymnasium.utils.env_checker import check_env

env = gym.make("Snake-v0", max_steps=100)

check_env(env.unwrapped)

print("Environment spec:", env.spec)

env.close()