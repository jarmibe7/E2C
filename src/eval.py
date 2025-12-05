"""
Classes for evaluating E2C model performance during and after training.

Authors: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch
import numpy as np
import itertools
from tqdm import tqdm

from src.e2c import E2CDataset, E2CLoss, E2C
from src.utils import set_seed, anim_frames

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
        self.plot_history = {"recon": [], "recon_next": [], "kld": [], "kld_trans": []}
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10))
        self.titles = [
            r"$x$ Reconstruction Loss",
            r"$x_{next}$ Reconstruction Loss",
            "KLD",
            "Transition KLD"
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

    def log(self, recon_loss, recon_next_loss, kld_loss, kld_trans_loss):
        """
        Update live training plot logs, and plot at frequency self.plot_freq
        """
        # Update plot history arrays
        self.num_steps += 1
        self.plot_history["recon"].append(recon_loss.detach().cpu().item())
        self.plot_history["recon_next"].append(recon_next_loss.detach().cpu().item())
        self.plot_history["kld"].append(kld_loss.detach().cpu().item())
        self.plot_history["kld_trans"].append(kld_trans_loss.detach().cpu().item())

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

    def save(self, run_path, timestamp=None):
        """
        Save live training plot
        """
        self.plot()
        # if timestamp is None: timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        fig_name = f'loss_fig.png'
        try:
            filepath = run_path / fig_name
            print(f'\nSaved loss figure to {filepath}')
            self.fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saved loss figure to current directory')
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
    def __init__(self, model, test_dataset, batch_size, device, dataset_name):
        # Set model to eval mode
        self.model = model
        self.model.eval()
        self.test_dataset = test_dataset

        # Params
        self.batch_size = batch_size
        self.device = device

        if dataset_name == 'particle_grav': self.dataset_latent_func = self.eval_four_var_latent
        elif dataset_name == 'cartpole': self.dataset_latent_func = self.eval_four_var_latent
        elif dataset_name == 'reacher': self.dataset_latent_func = self.eval_four_var_latent
        self.dataset_name = dataset_name

    def eval(self, run_path, vid_max_frames=50):
        print("generating latent space figure...")
        self.dataset_latent_func(run_path)
        print("generating trajectory video...")
        self.eval_traj(run_path, max_frames=vid_max_frames)
        # self.eval_latent(run_path)
        
    def eval_traj(self, run_path, max_frames=50):
        # Create figure
        fig, ax = plt.subplots(2, 2, figsize=(8, 10))
        ax[0, 0].set_title("Predicted Current Image")
        ax[1, 0].set_title("True Current Image")
        ax[0, 1].set_title("Predicted Future Image")
        ax[1, 1].set_title("True Future Image")

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=True
        )

        # Precompute frames
        assert self.model.pred_length == 1, 'Pred length >1 not supported for eval video'
        x_list, x_next_list, x_recon_list, x_pred_list = [], [], [], []
        for i, (x, x_next, u) in enumerate(test_loader):
            if i >= max_frames:
                break
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)
            x_recon, x_pred = self.model.sample(x, u)
            x_list.append(x[0]); x_next_list.append(x_next[0])
            x_recon_list.append(x_recon); x_pred_list.append(x_pred)

        # Initialize axes
        ims = []
        img_pred = x_recon_list[0][:3]
        img_pred_next = x_pred_list[0][:3]
        img = x_list[0][:3]
        img_next = x_next_list[0][:3]
        ims.append(ax[0, 0].imshow(img_pred.permute(1, 2, 0).detach().cpu().numpy()))
        ims.append(ax[1, 0].imshow(img.permute(1, 2, 0).detach().cpu().numpy()))
        ims.append(ax[0, 1].imshow(img_pred_next.permute(1, 2, 0).detach().cpu().numpy()))
        ims.append(ax[1, 1].imshow(img_next.permute(1, 2, 0).detach().cpu().numpy()))

        def update_plot(frame):
            x, x_next = x_list[frame], x_next_list[frame]
            x_recon, x_pred = x_recon_list[frame], x_pred_list[frame]
            ims[0].set_data(x_recon[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[1].set_data(x[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[2].set_data(x_pred[:3].permute(2, 1, 0).detach().cpu().numpy()) # for some reason, need to transpose these differently?
            ims[3].set_data(x_next[:3].permute(2, 1, 0).detach().cpu().numpy()) # for some reason, need to transpose these differently?

            # plt.show()

        # Create and save animation
        ani = FuncAnimation(fig, update_plot, frames=50, interval=5.)
        writer = FFMpegWriter(fps=2)
        # if timestamp is None: timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        vid_name = f'eval_vid.mp4'
        try:
            filepath = run_path / vid_name
            print(f'\nSaved eval video to {filepath}')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('\nException occured, saved eval video to current directory')
            ani.save(vid_name, writer=writer)
        plt.close(fig)
        return
    
    def eval_four_var_latent(self, run_path):
        """
        Visalize all variable combos of a four variable E2C latent space on the test dataset

        Credit: Jueun Kwon, Northwestern University
        """
        # Visualize latent space considering mean and variance
        fig, axes = plt.subplots(3, 2, figsize=(16, 16), dpi=200, tight_layout=True)

        # Initialize axes
        combo_array = list(itertools.combinations([0, 1, 2, 3], r=2))
        for ax, combo in zip(axes.flatten(), combo_array):
            ax.set_aspect('equal')
            ax.set_title(f'Latent Space from Test Dataset (Vars {combo[0]+1} and {combo[1]+1})')

        latent_mean = []
        latent_var = []

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=128, shuffle=True
        )

        # Iterate over DataLoader
        colors = ['blue', 'red']
        max_val = 0.0
        for x, x_next, u in tqdm(test_loader):
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)
            # Encode current and next state
            enc_out = self.model.encoder(x)

            # Get record latent space
            mu = self.model.mu(enc_out)
            log_var = self.model.log_var(enc_out)
            latent_mean.append(mu)
            latent_var.append(torch.exp(log_var))

            # Convert to numpy for scatter plot
            z_mean_np = mu.cpu().detach().numpy()
            z_var_np = torch.exp(log_var).cpu().detach().numpy()
            max_val = max(max_val, z_mean_np.max())

            # Represent uncertainty by point size
            point_sizes = np.mean(z_var_np, axis=1) * 1000  # Adjust scaling as needed

            # Choose colors based on configuration
            if self.dataset_name in ['particle_grav', 'cartpole']: 
                color = colors[round(u.cpu().detach().numpy().flatten()[0])]
            elif self.dataset_name == 'reacher':
                u = u.cpu().detach().numpy().flatten()
                if u[0] > 0.0 and u[1] > 0.0: color = 'blue'
                elif u[0] > 0.0 and u[1] < 0.0: color = 'green'
                elif u[0] < 0.0 and u[1] > 0.0: color = 'yellow'
                else: color = 'red'

            # Plotting all variable combos
            for ax, combo in zip(axes.flatten(), combo_array):
                sc = ax.scatter(z_mean_np[:, combo[0]], z_mean_np[:, combo[1]], s=point_sizes, alpha=0.1, label=None, color=color)

        # Combine all latent means and variances
        latent_mean = torch.cat(latent_mean).cpu().detach().numpy()
        latent_var = torch.cat(latent_var).cpu().detach().numpy()

        # Adjust plot limits
        for ax, combo in zip(axes.flatten(), combo_array):
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            a_min = np.minimum(x_min, y_min)
            a_max = np.maximum(x_max, y_max)
            ax.set_xlim(a_min, a_max)
            ax.set_ylim(a_min, a_max)
            # ax.set_xlim(-max_val, max_val)
            # ax.set_ylim(-max_val, max_val)

        fig_name = f'latent_fig.png'
        try:
            filepath = run_path / fig_name
            print(f'\nSaved all variable latent space figure to {filepath}')
            fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saved all variable latent space figure to current directory')
            fig.savefig(fig_name)
        plt.close(fig)
        return
    
    def eval_latent(self, run_path):
        """
        Visalize the E2C latent space on the test dataset

        Credit: Jueun Kwon, Northwestern University
        """
        # Visualize latent space considering mean and variance
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150, tight_layout=True)
        ax.set_aspect('equal')
        ax.set_title('Latent Space from Test Dataset (Mean and Var)')

        latent_mean = []
        latent_var = []

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=True
        )

        # Iterate over DataLoader
        # TODO: Visualize based on configuration in latent space
        # Cartpole Example: Left control left move is blue
        #                   Right control left move is red ...
        for x, x_next, u in test_loader:
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)
            # Encode current and next state
            enc_out = self.model.encoder(x)

            # Get record latent space
            mu = self.model.mu(enc_out)
            log_var = self.model.log_var(enc_out)
            latent_mean.append(mu)
            latent_var.append(torch.exp(log_var))

            # Convert to numpy for scatter plot
            z_mean_np = mu.cpu().detach().numpy()
            z_var_np = torch.exp(log_var).cpu().detach().numpy()

            # Represent uncertainty by point size
            point_sizes = np.mean(z_var_np, axis=1) * 100  # Adjust scaling as needed
            sc = ax.scatter(z_mean_np[:, 0], z_mean_np[:, 1], s=point_sizes, alpha=0.1, label=None)

        # Combine all latent means and variances
        latent_mean = torch.cat(latent_mean).cpu().detach().numpy()
        latent_var = torch.cat(latent_var).cpu().detach().numpy()

        # Adjust plot limits
        # x_min, x_max = ax.get_xlim()
        # y_min, y_max = ax.get_ylim()
        # a_min = np.minimum(x_min, y_min)
        # a_max = np.maximum(x_max, y_max)
        # ax.set_xlim(a_min, a_max)
        # ax.set_ylim(a_min, a_max)

        fig_name = f'latent_fig.png'
        try:
            filepath = run_path / fig_name
            print(f'\nSaved latent space figure to {filepath}')
            fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saved latent space figure to current directory')
            fig.savefig(fig_name)
        plt.close(fig)
        return  