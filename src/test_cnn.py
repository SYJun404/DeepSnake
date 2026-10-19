import argparse
import json
import time
from pathlib import Path

import pygame
from sb3_contrib import MaskablePPO

from src.config import SAVE_ROOT, SEED, resolve_device
from src.game_to_cnn import GameToCNN

STEP_INTERVAL = 0.03


def find_model():
    root = Path(SAVE_ROOT)
    for directory in sorted(root.glob("*"), reverse=True):
        for name in ("best_model.zip", "ppo_snake_final.zip", "interrupted_model.zip"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("No new model found; train first or pass --model PATH.")


def wait_after_victory(env, last_episode=False):
    """Keep the final board visible without advancing or resetting the game."""
    clock = pygame.time.Clock()
    label = "SPACE: CLOSE" if last_episode else "SPACE: NEXT"
    while True:
        env.game.draw_game_over_screen(button_text=label)
        pygame.display.flip()
        events = pygame.event.get()
        if any(
            event.type == pygame.QUIT
            or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
            for event in events
        ):
            return False
        if any(
            event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE
            for event in events
        ):
            return True
        clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, help="Default: saved training seed")
    parser.add_argument(
        "--board-size", type=int, help="Required for old models without config.json"
    )
    parser.add_argument(
        "--device", default="mps", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        help="Default: 20 headless episodes, unlimited in window",
    )
    parser.add_argument(
        "--repeat-seed",
        action="store_true",
        help="Restart the same food RNG every episode",
    )
    parser.add_argument("--interval", type=float, default=STEP_INTERVAL)
    args = parser.parse_args()
    if args.interval < 0 or (args.episodes is not None and args.episodes < 1):
        parser.error("interval must be >= 0 and episodes must be positive")

    try:
        model_path = args.model or find_model()
    except FileNotFoundError as error:
        parser.error(str(error))
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    config_file = model_path.parent / "config.json"
    metadata = json.loads(config_file.read_text()) if config_file.exists() else {}
    if args.seed is None:
        args.seed = metadata.get("seed", SEED)
    board_size = args.board_size or metadata.get("board_size")
    if board_size is None:
        parser.error(
            "Pass --board-size 12 (or the original board size) for a legacy model"
        )
    if (
        args.board_size is not None
        and metadata.get("board_size", args.board_size) != args.board_size
    ):
        parser.error("board-size differs from the model's training configuration")

    device = resolve_device(args.device)
    # One CPU thread also speeds up single-frame CNN inference.
    import torch

    torch.set_num_threads(1)
    model = MaskablePPO.load(str(model_path), device=device)
    limit = args.episodes or (20 if args.headless else None)
    env = GameToCNN(seed=args.seed, board_size=board_size, silent_mode=args.headless)
    print(
        f"Model: {model_path} | board={board_size} | seed={args.seed} | device={device}"
    )
    lengths, scores, rewards, wins = [], [], [], 0
    reasons = {}
    episode = 0
    try:
        while limit is None or episode < limit:
            # The first episode explicitly resets the seed: model loading,
            # wrapper construction and earlier resets cannot shift its RNG.
            obs = env.reset(
                seed=args.seed if episode == 0 or args.repeat_seed else None
            )
            total_reward, steps, done = 0.0, 0, False
            while not done:
                if not args.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            return
                action, _ = model.predict(
                    obs,
                    action_masks=env.get_action_mask(),
                    deterministic=True,
                )
                obs, reward, done, info = env.step(int(action))
                total_reward += reward
                steps += 1
                if not args.headless:
                    env.render()
                    time.sleep(args.interval)
            episode += 1

            score = info["snake_size"] - env.init_snake_size
            wins += int(info["is_success"])
            reasons[info["termination_reason"]] = (
                reasons.get(info["termination_reason"], 0) + 1
            )
            lengths.append(steps)
            scores.append(score)
            rewards.append(total_reward)
            env.game.high_score = max(env.game.high_score, env.game.score)
            print(
                f"Episode {episode}: food={score}, length={info['snake_size']}, "
                f"steps={steps}, result={info['termination_reason']}"
            )
            if info["is_success"] and not args.headless:
                last_episode = limit is not None and episode >= limit
                if not wait_after_victory(env, last_episode=last_episode):
                    return
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()
        if scores:
            print(
                f"Win rate: {wins}/{len(scores)} ({wins / len(scores):.1%}); "
                f"mean food: {sum(scores) / len(scores):.2f}; "
                f"mean steps: {sum(lengths) / len(lengths):.1f}; "
                f"mean reward: {sum(rewards) / len(rewards):.3f}; results: {reasons}"
            )


if __name__ == "__main__":
    main()
