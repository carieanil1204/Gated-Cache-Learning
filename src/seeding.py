"""Seed every RNG source a run touches. Call set_all_seeds() first thing
in any entry script, before data loading or model construction."""

import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seeded_generator(seed: int):
    """A torch Generator for data-loader shuffling, so shuffling doesn't
    depend on unseeded global state."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
