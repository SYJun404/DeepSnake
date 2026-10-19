import gym
import numpy as np
import pygame

from src.config import BOARD_SIZE, GAMMA, SEED
from src.snake_game import SnakeGame


class GameToCNN(gym.Env):
    ACTION_DIRECTION = {  # noqa: RUF012
        0: "UP",
        1: "LEFT",
        2: "RIGHT",
        3: "DOWN",
    }

    OPPOSITE_DIRECTION = {  # noqa: RUF012
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
    }

    def __init__(
        self, seed=SEED, board_size=BOARD_SIZE, silent_mode=True, limit_step=True
    ):
        super().__init__()

        self.game = SnakeGame(
            seed=seed, board_size=board_size, silent_mode=silent_mode, grid_lines=False
        )
        assert self.game.snake is not None

        self.silent_mode = silent_mode

        # Action space: 4 discrete actions (UP, LEFT, RIGHT, DOWN)
        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size**2  # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)  # 3

        self.done = False

        if limit_step:
            self.step_limit = (
                self.grid_size * 3
            )  # Enough exploration time without allowing endless loops.
        else:
            self.step_limit = 1_000_000_000  # Basically no limit.

        self.reward_step_counter = 0
        self._resize_indices = np.floor(np.arange(84) * board_size / 84).astype(int)
        self.seed(seed)

    def seed(self, seed=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Gym 0.21 / SB3 1.8 call this through wrappers before resetting.
        self.game.seed_value = seed
        self.game.rng.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        self.done, info = self.game.step(action)
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if info["is_success"]:
            reward = 10.0
            self.done = True
            if not self.silent_mode and self.game.sound_victory != None:
                self.game.sound_victory.play()
            return obs, reward, self.done, info
        if info["food_obtained"]:
            self.reward_step_counter = 0
        if not self.done and self.reward_step_counter >= self.step_limit:
            self.done = True
            info["termination_reason"] = "timeout"
            info["TimeLimit.truncated"] = True

        # Snake bumps into wall or itself. Episode is over
        if self.done:
            # Keep terminal feedback on the same scale as food rewards.
            # The former exponentially large penalty overwhelmed every other
            # learning signal on a small board.
            reward = -1.0
            return obs, reward, self.done, info

        # Food should dominate the small distance-shaping reward.
        elif info["food_obtained"]:
            reward = 1.0
            self.reward_step_counter = 0

        # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not
        # Not competing with game over penalty or the food eaten reward
        else:
            # Normalize by board diameter and charge more than the maximum
            # discount offset, so cycling cannot generate positive rewards.
            distance = np.abs(info["snake_head_pos"] - info["food_pos"]).sum()
            previous = np.abs(info["prev_snake_head_pos"] - info["food_pos"]).sum()
            diameter = 2 * (self.board_size - 1)
            reward = 0.2 * (previous - GAMMA * distance) / diameter - 0.003

        return obs, reward, self.done, info

    def render(self, mode="human"):
        if self.silent_mode:
            return
        self.game.render()

        # SnakeGame.render() only draws to the back buffer; the caller is
        # responsible for presenting the frame. In the training path there
        # is no such caller, so the frame must be flipped here or the
        # window stays black.
        #
        # pygame.event.pump() keeps the window responsive to the OS
        # (without it macOS reports the app as "Not Responding").
        pygame.event.pump()
        pygame.display.flip()

    def close(self):
        if not self.silent_mode:
            pygame.quit()

    def get_action_mask(self):
        return np.array(
            [self._check_action_validity(a) for a in range(self.action_space.n)],
            dtype=bool,
        )

    def _check_action_validity(self, action):
        assert self.game.direction is not None and self.game.snake is not None

        # Check if the action is opposite to the current direction
        next_direction = self.ACTION_DIRECTION[action]
        current_direction = self.game.direction

        if next_direction == self.OPPOSITE_DIRECTION[current_direction]:
            return False

        snake_list = self.game.snake

        row, col = snake_list[0]
        dr, dc = self.game.DIRECTION_OFFSET[next_direction]

        # Calculate the new position
        row, col = row + dr, col + dc

        # Check if snake collided with itself or the wall
        # Note that the tail of the snake would be poped if the snake did not eat food in the current step
        if (row, col) == self.game.food:
            game_over = (
                (row, col)
                in snake_list  # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col)
                in snake_list[
                    :-1
                ]  # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        return not game_over

    def _generate_observation(self):
        assert self.game.snake is not None

        obs = np.zeros((self.board_size, self.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        rows, cols = zip(*self.game.snake)
        obs[rows, cols] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)

        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)

        # Preserve the original model encoding: green head, red tail, blue food.
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Do not paint a food pixel after the board has been filled.
        if self.game.food is not None:
            obs[self.game.food] = [0, 0, 255]

        # Resize to the fixed input shape expected by the CNN.  Index-based
        # nearest-neighbour sampling keeps this valid for the easier 10x10
        # board as well as the original 12x12 board.
        indices = self._resize_indices
        obs = obs[indices][:, indices]

        return obs


def test_game_to_cnn():
    import matplotlib.pyplot as plt

    env = GameToCNN(
        seed=0,
        board_size=12,
        silent_mode=False,
        limit_step=True,
    )

    # 获取初始 observation
    obs = env.reset()

    print("========== Observation ==========")
    print("shape:", obs.shape)
    print("dtype:", obs.dtype)
    print("min:", obs.min())
    print("max:", obs.max())

    # 获取 action mask
    action_mask = env.get_action_mask()

    print("\n========== Action Mask ==========")

    action_names = ["UP", "LEFT", "RIGHT", "DOWN"]

    for action, valid in zip(action_names, action_mask):
        print(f"{action:>5}: {'✓ 可以走' if valid else '✗ 不能走'}")

    print("\n原始 mask:", action_mask)

    # ==============================
    # Matplotlib 可视化
    # ==============================

    plt.figure(figsize=(8, 8))

    # obs 是 RGB 图像，因此直接显示
    plt.imshow(obs)

    # 棋盘是 12x12，每个格子放大成 7x7
    plt.xticks(
        np.arange(3.5, 84, 7),
        np.arange(12),
    )
    plt.yticks(
        np.arange(3.5, 84, 7),
        np.arange(12),
    )

    plt.grid(True)

    plt.title(f"Snake Observation\nshape={obs.shape}")

    plt.show()


if __name__ == "__main__":
    test_game_to_cnn()
