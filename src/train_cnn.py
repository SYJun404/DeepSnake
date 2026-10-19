import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecTransposeImage,
)

from src.config import BOARD_SIZE, GAMMA, NUM_ENV, SAVE_ROOT, SEED, resolve_device
from src.game_to_cnn import GameToCNN


def make_env(seed, board_size=BOARD_SIZE):
    def _init():
        env = GameToCNN(seed=seed, board_size=board_size)
        env = ActionMasker(env, GameToCNN.get_action_mask)
        return Monitor(env)

    return _init


def linear_schedule(initial_value, final_value=0.0):
    initial_value, final_value = float(initial_value), float(final_value)

    def schedule(progress):
        return final_value + progress * (initial_value - final_value)

    return schedule


class ReproducibleEvalCallback(MaskableEvalCallback):
    """Restart the held-out RNG stream for each evaluation batch."""

    def __init__(self, *args, evaluation_seed=SEED + 10_000, **kwargs):
        self.evaluation_seed = evaluation_seed
        super().__init__(*args, **kwargs)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.evaluation_seed)
        return super()._on_step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=NUM_ENV)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--save-root", type=Path, default=Path(SAVE_ROOT))
    args = parser.parse_args()
    if min(args.timesteps, args.num_envs, args.eval_every, args.eval_episodes) < 1:
        parser.error("timesteps, num-envs and evaluation settings must be positive")
    if args.board_size < 4 or args.n_steps < 2:
        parser.error("board-size must be >= 4 and n-steps must be >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device == "mps":
        torch.mps.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Small CNN batches suffer from excessive CPU thread contention.
    torch.set_num_threads(1)

    run_dir = args.save_root / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_board{args.board_size}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(vars(args), device=device, gamma=GAMMA, observation_version=2)
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    factories = [make_env(args.seed + i, args.board_size) for i in range(args.num_envs)]
    env = eval_env = None
    try:
        vector = (
            DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
        )
        env = VecTransposeImage(vector)
        eval_env = VecTransposeImage(
            DummyVecEnv([make_env(args.seed + 10_000, args.board_size)])
        )
        rollout_size = args.n_steps * args.num_envs
        batch_size = min(256, rollout_size)
        # Avoid incomplete minibatches for user-selected rollout lengths.
        while rollout_size % batch_size:
            batch_size -= 1
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device=device,
            seed=args.seed,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=4,
            gamma=GAMMA,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=linear_schedule(3e-4, 3e-5),
            clip_range=linear_schedule(0.20, 0.05),
        )
        # CSV and console output need no optional TensorBoard installation.
        from stable_baselines3.common.logger import configure

        model.set_logger(configure(str(run_dir), ["stdout", "csv"]))
        evaluation = ReproducibleEvalCallback(
            eval_env,
            evaluation_seed=args.seed + 10_000,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=max(args.eval_every // args.num_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        checkpoint = CheckpointCallback(
            save_freq=max(500_000 // args.num_envs, 1),
            save_path=str(run_dir),
            name_prefix="ppo_snake",
        )
        print(f"Run: {run_dir.resolve()} | board={args.board_size} | seed={args.seed}")
        try:
            model.learn(
                total_timesteps=args.timesteps, callback=[checkpoint, evaluation]
            )
        except KeyboardInterrupt:
            model.save(str(run_dir / "interrupted_model"))
            print("Interrupted model saved.")
        else:
            model.save(str(run_dir / "ppo_snake_final"))
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
