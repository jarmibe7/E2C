"""
Generate a dummy dataset for testing dimensions of E2C pipeline.

Author: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pathlib import Path

from src.utils import anim_frames

# Get data directory - use Path relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

def gen_dummy_dataset(N=100, T=50, scale=(64,64)):
    """
    Function to generate frames of particle in gravity

    Args:
        T: length/seconds of animation duration
        scale: Desired dimenions of final image formatted as (H,W)

    Returns: Tensor of shape (N, H, W, 3)
    """
    # Specify axis limits
    xm = -2
    xM = 2
    ym = -2
    yM = 2

    # Create figure and static background
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_xlim(xm, xM)
    ax.set_ylim(ym, yM)
    ax.axis('off')

    # Force a draw to access canvas
    fig.canvas.draw()

    dataset = []
    for n in range(N):
        frames = []
        color = np.random.uniform(low=0.0, high=1.0, size=(3,))
        for t in range(T):
            # Next color frame is close to previous frame
            frame = torch.zeros((scale[0], scale[1], 3), dtype=torch.float32)
            color = color + np.random.uniform(low=0.0, high=0.05, size=(3,))
            color = np.clip(color, 0.0, 1.0)
            frame[:, :, 0] = color[0]
            frame[:, :, 1] = color[1]
            frame[:, :, 2] = color[2]
            ax.imshow(frame)

            # Redraw updated only
            fig.canvas.draw()

            # Convert to tensor
            image = torch.frombuffer(fig.canvas.buffer_rgba(), dtype=torch.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :-1].permute(2, 0, 1)  # Get image tensor into (C, H, W)

            # Image processing
            normalized = image.unsqueeze(0).float() / 255.0                       # Normalize to [0,1]
            image_resized = torchvision.transforms.functional.resize(normalized, scale, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)   # Downscaling
            frames.append(image_resized.permute(0, 2, 3, 1))

        dataset.append(torch.concat(frames, dim=0).unsqueeze(0))

    dataset_tensor = torch.concat(dataset, dim=0)

    plt.close(fig)
    return dataset_tensor

def main():
    print('*** STARTING ***\n')
    # Parameters
    N = 10              # Number of samples
    T = 5               # Sequence length
    shape = (64,64)     # Image shape
    control_dim = 2     # Control input dimension

    # Generate dataset with N sample trajectories, each of length T
    img = gen_dummy_dataset(N=N, T=T, scale=shape)
    control = torch.zeros((N, T, control_dim))
    print(f'\nDataset shape: {img.shape}')
    anim_frames(img[0])

    # Saving dataset
    dataset_dir = DATA_PATH / 'dummy'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    img_filepath = dataset_dir / 'img.pt'
    control_filepath = dataset_dir / 'control.pt'
    print(f'\nSaving dataset to {dataset_dir}')
    torch.save(img, img_filepath)
    torch.save(control, control_filepath)

    print('\n*** DONE ***')
    return


if __name__ == '__main__':
    main()