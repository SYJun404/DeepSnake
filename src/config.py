"""Shared defaults for new, easier CNN experiments (legacy Gym/SB3 1.8)."""

SEED = 20260927
BOARD_SIZE = 8
GAMMA = 0.99
NUM_ENV = 8
SAVE_ROOT = "trained_models_cnn_easy"


def resolve_device(requested="auto"):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
