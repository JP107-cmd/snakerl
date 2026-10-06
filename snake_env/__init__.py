from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="snake_env.snake:SnakeEnv",
)