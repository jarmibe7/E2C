"""
Utility functions for data4wm.

Authors: Jared Berry, Ayush Gaggar
"""
import math
import random
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


# Maps config-level policy tags to the objective label used in run/save paths.
# Keep these strings stable; downstream analysis scripts parse them.
_POLICY_TO_OBJECTIVE = {'eig': 'pixel', 'maxdyn': 'dynamics', 'random': 'random'}


def build_run_path(config, config_name, runs_root):
    """
    Compute the per-run save directory from ``config`` + ``config_name``.

    Layout: ``<runs_root>/<env>/<env>_<objective>_<seed>`` where ``<env>`` is
    the leading token of ``config['train']['dataset']`` and ``<objective>`` is
    ``pixel`` / ``dynamics`` / ``random`` for the three core policies, or the
    raw policy tag otherwise (e.g. ``hardware``, ``informative``).
    """
    env = config['train']['dataset'].split('_')[0]
    policy = config_name.split('/')[-1].split('_')[1]
    objective = _POLICY_TO_OBJECTIVE.get(policy, policy)
    seed = str(config.get('seed', 0))
    save_name = f"{env}_{objective}_{seed}"
    return Path(runs_root) / env / save_name

def anim_frames(frames):
    """
    Animation function to play an animation from an image tensor of shape
    (num_frames, H, W, C).
    """
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0].numpy())

    def update(frame):
        im.set_array(frame.numpy())
        return [im]

    # Animate and display
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    plt.show()
    plt.close(fig)

def set_seed(seed: int = 42):
    """
    Set random seed for all packages
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Torch gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def format_time(seconds: float) -> str:
    """Format a float duration in seconds as ``HH:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def wrapped_angle_error(pred, true):
    """Signed angular error in ``(-pi, pi]`` (radians)."""
    diff = pred - true
    return (diff + math.pi) % (2 * math.pi) - math.pi


#
# ----- Latent-space distribution metrics -----
# Each operates on a tensor of per-dim latent means ``mu_vec`` with the batch
# axis given by ``dim`` (default 0) and returns a per-dim scalar.
#
def central_mass_ratio(mu_vec: torch.Tensor, r: float = 1.0, dim: int = 0) -> torch.Tensor:
    """Fraction of samples within ``r`` std-devs of the mean (per latent dim)."""
    mu = mu_vec.mean(dim=dim, keepdim=True)
    std = mu_vec.std(dim=dim, keepdim=True)
    return (torch.abs(mu_vec - mu) <= r * std).float().mean(dim=dim)


def excess_kurtosis(mu_vec: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Sample excess kurtosis (fourth-moment - 3) per latent dim."""
    mean = mu_vec.mean(dim=dim, keepdim=True)
    var = ((mu_vec - mean) ** 2).mean(dim=dim, keepdim=True)
    fourth = ((mu_vec - mean) ** 4).mean(dim=dim, keepdim=True)
    return fourth / (var ** 2 + eps) - 3.0


def shoulder_mass(
    mu_vec: torch.Tensor, r1: float = 0.5, r2: float = 2.0, dim: int = 0
) -> torch.Tensor:
    """Fraction of samples with distance-to-mean in ``(r1, r2]`` std-devs."""
    mean = mu_vec.mean(dim=dim, keepdim=True)
    std = mu_vec.std(dim=dim, keepdim=True)
    d = torch.abs(mu_vec - mean)
    return ((d > r1 * std) & (d <= r2 * std)).float().mean(dim=dim)
