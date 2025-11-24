"""
Classes for evaluating E2C model performance during and after training.

Authors: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
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

    def save(self, config_name):
        """
        Save live training plot
        """
        self.plot()
        timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        fig_name = f'{config_name}_fig_{timestamp}.png'
        try:
            FIG_PATH.mkdir(parents=True, exist_ok=True)
            filepath = FIG_PATH / fig_name
            print(f'\nSaving figure to {filepath}')
            self.fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saving to current directory')
            self.fig.savefig(fig_name)
        return
    
class Evaluator():
    """
    Class for evaluating model performance on a test set.
    """
    def __init__(self, model, X_test, U_test):
        self.model = model
        self.model.eval()
        self.X_test = X_test
        self.U_test = U_test

    def evaluate():
        pass