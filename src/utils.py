"""
Utility functions for E2C

Authors: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np
import torch

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Torch gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    