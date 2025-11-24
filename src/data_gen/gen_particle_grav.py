"""
Generate a dataset for a particle falling in gravity.
"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils import set_seed, rk4_sim

set_seed(42)

# Get data directory - use Path relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# ---------------------------------
# Parameters for simulation
m = 1.0         # Particle mass
g = -9.8        # Gravity
dt = 0.01       # Timestep
t0 = 0.0        # Initial simulation time
tf = 0.5        # Final simulation time

# Parameters for dataset
n_samples = 100 # Number of total trajectories
# ---------------------------------

def traj_to_frames(traj, T, scale=(64,64)):
    """
    Function to generate frames of particle in gravity

    Args:
      traj:
          trajectory of theta1 and theta2, should be a NumPy array with
          shape of (N, 2) for N timesteps
      T:
          length/seconds of animation duration
      scale:
          Desired dimenions of final image formatted as (H,W)

    Returns: Tensor of shape (N, H, W, 3)
    """
    # Specify axis limits
    xm = -2
    xM = 2
    ym = -2
    yM = 2
    N = traj.shape[0]

    # Create figure and static background
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ax.set_xlim(xm, xM)
    ax.set_ylim(ym, yM)
    ax.axis('off')
    particle = ax.plot([], [], 'ks', markersize=50)[0]   # Moving particle

    # Force a draw so we can access the canvas
    fig.canvas.draw()

    frames = []
    for k in range(N):
      # Update particle position
      particle.set_data([traj[k, 0]], [traj[k, 1]])

      # Redraw only the updated elements
      fig.canvas.draw()

      # Convert to tensor
      image = torch.frombuffer(fig.canvas.buffer_rgba(), dtype=torch.uint8)
      image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :-1].permute(2, 0, 1)  # Get image tensor into (C, H, W)

      # Image processing
      normalized = image.unsqueeze(0).float() / 255.0                       # Normalize to [0,1]
      image_resized = torchvision.transforms.functional.resize(normalized, scale, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)   # Downscaling
      frames.append(image_resized.permute(0, 2, 3, 1))

    plt.close(fig)
    frames_tensor = torch.concat(frames, dim=0)
    return frames_tensor

def dynamics(x):
  """
  Particle in gravity system dynamics function
  """
  return np.array([x[2], x[3], 0, g])

def main():
    print('*** STARTING ***\n')
    # Simulate samples and create
    x0_vec = np.zeros((n_samples,))
    y0_vec = np.linspace(-0.5, 1.5, n_samples)
    samples = []
    for s in range(n_samples):
        if s % 100 == 0: print(f'On sample {s} out of {n_samples}')
        x0, y0 = x0_vec[s], y0_vec[s]
        x0 = [x0, y0, 0.0, 0.0]
        traj = rk4_sim(dynamics, x0, [t0, tf], dt)
        frames = traj_to_frames(traj[:, :2], T=(tf - t0))
        samples.append(frames.unsqueeze(0))

    img = torch.concat(samples, axis=0)  # Dataset shape is [num_samples, frames_per_sample, H, W, C]
    control = torch.zeros((n_samples, img.shape[1], 2))
    print(f'\nDataset shape: {img.shape}')

    # Saving dataset
    dataset_dir = DATA_PATH / 'particle_grav'
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