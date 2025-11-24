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

def rk4_sim(f, x0, tspan, dt):
  """
  RK4 Integration given a dynamics function, initial conditions,
  timespan, and timestep.
  """
  def rk4_int(xt, dt):
    k1 = dt * f(xt)
    k2 = dt * f(xt+k1/2.)
    k3 = dt * f(xt+k2/2.)
    k4 = dt * f(xt+k3)
    new_xt = xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
    return new_xt

  # Run simulation
  N = int((max(tspan) - min(tspan))/dt)
  x = np.copy(x0)
  tvec = np.linspace(min(tspan), max(tspan),N)
  xtraj = np.zeros((N, len(x0)))
  for i in range(N):
      xtraj[i,:] = rk4_int(x, dt)
      x = np.copy(xtraj[i,:])
  return xtraj

    