import gymnasium
import numpy as np
import pygame
from gymnasium import spaces


class SnakeEnv(gymnasium.Env):
    """A small Snake environment with relative left/straight/right actions."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    # 0 = up, 1 = right, 2 = down, 3 = left.
    DIRECTION_VECTORS = (
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    )

    # 0 = turn left, 1 = go straight, 2 = turn right.
    TURN_OFFSETS = (-1, 0, 1)

    def __init__(self, size=15, render_mode=None, max_steps=10000, expanded_obs_space=False):
        if size < 5:
            raise ValueError("size must be at least 5")
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.size = size
        self.render_mode = render_mode
        self.max_steps = max_steps or size * size * 4

        self.action_space = spaces.Discrete(3)
        if expanded_obs_space:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * 7 * 7 + 4 + 4 + 2,),
                dtype=np.float32,
            )
        else:    
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([3, 3, 1, 1, 1], dtype=np.float32),
                dtype=np.float32,
            )

        self.expanded_obs_space = expanded_obs_space
        self.window_size = 720
        self.window = None
        self.clock = None
        self.state = None
        self.body = None
        self.agent_head_x = None
        self.agent_head_y = None
        self.food_x = None
        self.food_y = None
        self.direction = None
        self.score = None
        self.steps = None
        self.steps_since_last_food = None
        self.won = False

    def _rebuild_state(self):
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
        for x, y in self.body[1:]:
            self.state[x, y] = 1
        self.state[self.agent_head_x, self.agent_head_y] = 2
        if self.food_x is not None and self.food_y is not None:
            self.state[self.food_x, self.food_y] = 3

    def _empty_cells(self):
        occupied = self.body
        return [
            [x, y]
            for x in range(self.size)
            for y in range(self.size)
            if [x, y] not in occupied
        ]

    def _spawn_food(self):
        empty_cells = self._empty_cells()
        if not empty_cells:
            self.food_x = None
            self.food_y = None
            self.won = True
            return False

        index = self.np_random.integers(len(empty_cells))
        self.food_x, self.food_y = empty_cells[index]
        return True

    def _food_direction(self):
        if self.food_x is None:
            return 0

        dx = self.food_x - self.agent_head_x
        dy = self.food_y - self.agent_head_y

        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 0
        if dy != 0:
            return 1 if dy > 0 else 3
        return 0

    def _food_vector(self):

        if self.food_x is None:
            return np.zeros(2, dtype=np.float32)

        dx = (self.food_x-self.agent_head_x)/(self.size-1)
        dy = (self.food_y-self.agent_head_y)/(self.size-1)
        return np.array([dx, dy], dtype=np.float32)

    def _danger(self):
        relative_directions = (
            (self.direction - 1) % 4,  # left
            self.direction,             # straight
            (self.direction + 1) % 4,  # right
        )

        danger = []
        body_without_tail = {tuple(segment) for segment in self.body[:-1]}

        for direction in relative_directions:
            dx, dy = self.DIRECTION_VECTORS[direction]
            x = self.agent_head_x + dx
            y = self.agent_head_y + dy

            outside = x < 0 or x >= self.size or y < 0 or y >= self.size
            occupied = (x, y) in body_without_tail
            danger.append(float(outside or occupied))

        return danger

    def _get_obs(self):
        if not self.expanded_obs_space:
            return np.array(
                [self.direction, self._food_direction(), *self._danger()],
                dtype=np.float32,
            )
        else:
            grid = self._get7x7()
            direction_one_hot = np.eye(4, dtype=np.float32)[self.direction]
            food_direction_one_hot = np.eye(
                4, dtype=np.float32
            )[self._food_direction()]
            normalized_food_direction = self._food_vector()
            return np.concatenate([
                grid.flatten(),
                direction_one_hot,
                food_direction_one_hot,
                normalized_food_direction
            ]).astype(np.float32)

    def _get7x7(self):
        observation = np.zeros((3, 7, 7), dtype=np.float32)

        for local_x in range(7):
            for local_y in range(7):

                x = self.agent_head_x + local_x - 3
                y = self.agent_head_y + local_y - 3
                outside = (
                    x < 0 or x >= self.size or 
                    y < 0 or y >= self.size
                )

                if outside:
                    observation[2, local_x, local_y] = 1.0
                    continue

                if [x, y] in self.body:
                    observation[0, local_x, local_y] = 1.0

                if x == self.food_x and y == self.food_y:
                    observation[1, local_x, local_y] = 1.0

        return observation

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "won": self.won,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        center = self.size // 2
        self.agent_head_x = center
        self.agent_head_y = center
        self.body = [
            [center, center],
            [center - 1, center],
            [center - 2, center],
        ]
        self.direction = 1
        self.score = 0
        self.steps = 0
        self.steps_since_last_food = 0
        self.won = False
        self.state = np.zeros((self.size, self.size), dtype=np.int8)

        self.food_x = min(center + 6, self.size - 2)
        self.food_y = center
        self._rebuild_state()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        self.direction = (
            self.direction + self.TURN_OFFSETS[int(action)]
        ) % 4

        dx, dy = self.DIRECTION_VECTORS[self.direction]
        next_x = self.agent_head_x + dx
        next_y = self.agent_head_y + dy

        outside = (
            next_x < 0 or next_x >= self.size or
            next_y < 0 or next_y >= self.size
        )
        if outside:
            return self._get_obs(), -1.0, True, False, self._get_info()

        eating = next_x == self.food_x and next_y == self.food_y
        body_to_check = self.body if eating else self.body[:-1]
        if [next_x, next_y] in body_to_check:
            return self._get_obs(), -1.0, True, False, self._get_info()

        self.agent_head_x = next_x
        self.agent_head_y = next_y
        self.body.insert(0, [next_x, next_y])

        if eating:
            self.steps_since_last_food = 0
            self.score += 1
            reward = 1.0
            has_empty_cell = self._spawn_food()
        else:
            self.steps_since_last_food += 1
            reward = -0.01
            self.body.pop()
            has_empty_cell = True

        self._rebuild_state()

        terminated = self.won or not has_empty_cell
        truncated = (self.steps >= self.max_steps or self.steps_since_last_food > 200) and not terminated

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode != "human":
            return None

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

        cell_size = self.window_size / self.size
        colors = {
            0: (0, 200, 0),
            1: (0, 0, 0),
            2: (30, 30, 255),
            3: (200, 0, 0),
        }

        self.window.fill(colors[0])
        for x in range(self.size):
            for y in range(self.size):
                value = int(self.state[x, y])
                rect = pygame.Rect(
                    y * cell_size,
                    x * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(self.window, colors[value], rect)

        for coordinate in range(self.size + 1):
            position = coordinate * cell_size
            pygame.draw.line(
                self.window,
                (0, 250, 0),
                (0, position),
                (self.window_size, position),
                width=1,
            )
            pygame.draw.line(
                self.window,
                (0, 250, 0),
                (position, 0),
                (position, self.window_size),
                width=1,
            )

        font = pygame.font.SysFont("Arial", 24)
        score_surface = font.render(f"Score: {self.score}", True, "white")
        self.window.blit(score_surface, score_surface.get_rect())

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None


if __name__ == "__main__":
    env = SnakeEnv(render_mode="human", expanded_obs_space=True)
    observation, _ = env.reset()
    print(env._get_obs())
