"""
Classes for evaluating E2C model performance during and after training.

Authors: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch
import yaml
import time
from datetime import datetime
from pathlib import Path

from src.e2c import E2CDataset, E2CLoss, E2C
from src.utils import set_seed, anim_frames

PROJECT_ROOT = Path(__file__).parent.parent
FIG_PATH = PROJECT_ROOT / "figures"
VID_PATH = PROJECT_ROOT / "videos"

class Plotter():
    """
    Simple class for visualizing training progress
    """
    def __init__(self, render, plot_freq):
        """
        Initalize figure for live plotting
        """
        self.num_steps = 0
        self.render = render
        self.plot_freq = plot_freq
        self.plot_history = {"recon": [], "recon_next": [], "kld": [], "kld_contract": []}
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10))
        self.titles = [
            r"$x$ Reconstruction Loss",
            r"$x_{next}$ Reconstruction Loss",
            "KLD",
            "KLD Contraction"
        ]
        self.colors = ['blue', 'orange', 'green', 'red']
        for ax, title in zip(self.axs, self.titles):
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True)
        plt.tight_layout()
        if self.render: 
            plt.ion()
            plt.show()
        else:
            plt.ioff()

    def log(self, recon_loss, recon_next_loss, kld_loss, kld_contract_loss):
        """
        Update live training plot logs, and plot at frequency self.plot_freq
        """
        # Update plot history arrays
        self.num_steps += 1
        self.plot_history["recon"].append(recon_loss.detach().cpu().item())
        self.plot_history["recon_next"].append(recon_next_loss.detach().cpu().item())
        self.plot_history["kld"].append(kld_loss.detach().cpu().item())
        self.plot_history["kld_contract"].append(kld_contract_loss.detach().cpu().item())

        # Replot
        if self.num_steps % self.plot_freq == 0: self.plot()

    def plot(self):
        """
        Update live plot
        """
        for i, key in enumerate(self.plot_history.keys()):
            self.axs[i].cla() 
            self.axs[i].plot(self.plot_history[key], label=self.titles[i], color=self.colors[i])
            self.axs[i].legend()
            self.axs[i].grid(True)

        plt.tight_layout()
        if self.render: plt.pause(0.001)

    def save(self, name_stem, timestamp=None):
        """
        Save live training plot
        """
        self.plot()
        if timestamp is None: timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        fig_name = f'{name_stem}_fig_{timestamp}.png'
        try:
            FIG_PATH.mkdir(parents=True, exist_ok=True)
            filepath = FIG_PATH / fig_name
            print(f'\nSaving figure to {filepath}')
            self.fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saving to current directory')
            self.fig.savefig(fig_name)
        self.close()
        return
    
    def close(self):
        plt.close(self.fig)
        return
    
class Evaluator():
    """
    Class for evaluating model performance on a test set.
    """
    def __init__(self, model, test_dataset, batch_size, device):
        # Set model to eval mode
        self.model = model
        self.model.eval()
        self.test_dataset = test_dataset
        # self.img_shape = test_dataset.img_shape.permute(1, 2, 0)

        # Params
        self.batch_size = batch_size
        self.device = device

    def eval_all(self):
        self.eval_traj()
        
    def eval_traj(self, name_stem, timestamp=None):
        # Create figure
        fig, ax = plt.subplots(2, 2, figsize=(8, 10))
        ax[0, 0].set_title("Predicted Current Image")
        ax[1, 0].set_title("True Current Image")
        ax[0, 1].set_title("Predicted Future Image")
        ax[1, 1].set_title("True Future Image")

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True
        )

        def update_plot(frame):
            add_some = 15
            for i, (x, x_next, u) in enumerate(test_loader):
                if i == add_some + frame:
                    break
                x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
                x_recon, x_pred = self.model.sample(x, u)
            
                for i in range(self.model.pred_length):
                    ax[0, 0].imshow(x_recon[i].permute(1, 2, 0).detach().cpu().numpy())
                    ax[1, 0].imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
                    ax[0, 1].imshow(x_pred[i].permute(1, 2, 0).detach().cpu().numpy())
                    ax[1, 1].imshow(x_next[i].permute(1, 2, 0).detach().cpu().numpy())

            plt.show()

        # Create and save animation
        ani = FuncAnimation(fig, update_plot, frames=50, interval=5.)
        writer = FFMpegWriter(fps=2)
        if timestamp is None: timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        vid_name = f'{name_stem}_eval_vid_{timestamp}.mp4'
        try:
            VID_PATH.mkdir(parents=True, exist_ok=True)
            filepath = VID_PATH / vid_name
            print(f'\nSaving figure to {filepath}')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('\nException occured, saving to current directory')
            ani.save(vid_name, writer=writer)
        plt.close(fig)
        return
        