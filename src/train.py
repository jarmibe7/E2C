"""
train.py

Main training script for E2C

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

# Set random seed globally
set_seed(42)

# Paths - relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
MODEL_PATH = PROJECT_ROOT / "models"
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

def train(dataset, config):
    """
    Function for training an E2C model
    """
    num_epochs = config['train']['num_epochs']
    device = config['train']['device']

    # Create autoencoder model and optimizer
    model = E2C(
        enc_latent_size=config['vae']['enc_latent_size'],
        latent_size=config['trans']['latent_size'],
        control_size=config['trans']['control_size'],
        conv_params=config['vae'],
        device=device
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['trans']['alpha'], weight_decay=config['trans']['weight_decay'])

    # Create Dataset and DataLoader to handle batching of training data
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['train']['batch_size'], shuffle=True
    )

    # Create loss criterion
    criterion = E2CLoss(config['train']['num_epochs'], config['loss'])

    # Create visualizer
    plotter = Plotter(config['train']['render'], config['train']['plot_freq'])

    # Training loop
    print('\nBeginning Training:')
    for epoch in range(num_epochs):
        total_loss = 0.0

        for x, x_next, u in train_loader:
            # Send training data to GPU
            x, x_next, u = x.to(device), x_next.to(device), u.to(device)

            # Forward pass
            train_return = model(x, x_next, u)
            train_return['x'] = x
            train_return['x_next'] = x_next

            # Compute loss and backprop
            loss, recon, recon_next, kld, kld_contract = criterion(train_return, epoch)
            plotter.log(recon, recon_next, kld, kld_contract)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                print("NaN loss encountered, stopping training.")
                break

            total_loss += loss.item() * x.size(0)   # Aggregate total epoch loss

        # Display average loss for the epoch
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f'\n--------------------------------------------------')
        print(f'EPOCH {epoch+1}/{num_epochs}')
        print(f"Average Epoch Loss: {epoch_loss:.4f}")
        print(f'--------------------------------------------------\n')

    plotter.save(config['config_name'])
    return model

def main():
    print('*** STARTING ***\n')
    # Load config and choose torch device
    config_name = 'e2c_config0'
    with open(CONFIG_PATH / f'{config_name}.yaml', "r") as f:
        config = yaml.safe_load(f)
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])
    config['train']['device'] = device   # Replace device string with device object in config
    config['config_name'] = config_name

    # Make E2CDataset object
    print(f"Loading dataset: {config['train']['dataset']}")
    dataset = E2CDataset(config)
    config['vae']['in_image_shape'] = dataset.X.shape[1:]   # Shape is [num_traj*(seq_len - 1), C, H, H]
    config['trans']['control_size'] = dataset.U.shape[-1]
    model = train(dataset, config)

    # Save model
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'{config_name}_model_{timestamp}.pt'
    try:
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        filepath = MODEL_PATH / model_name
        print(f'\nSaving model to {filepath}')
        torch.save(model.state_dict(), filepath)
    except Exception as e:
        print(e)
        print('\nException occured, saving to current directory')
        torch.save(model.state_dict(), model_name)

    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()