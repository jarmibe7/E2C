"""
Classes for evaluating E2C model performance during and after training.

Authors: Jared Berry, Ayush Gaggar
"""
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch
import numpy as np
import itertools
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio as psnr
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
import lpips

from src.conv_e2c import ConvE2C
from src.dataset import E2CDataset
from src.utils import set_seed, anim_frames, shoulder_mass, excess_kurtosis, central_mass_ratio

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
        self.plot_history = None
        self.fig = None
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

    def log(self, lr):
        """
        Update live training plot logs, and plot at frequency self.plot_freq
        """
        # Create plot history dictionary if none exists
        self.num_steps += 1
        if self.plot_history is None:
            self.plot_history = {}
            for key in lr.keys():
                self.plot_history[key] = []

        # Update plot history arrays
        for key in lr.keys():
            self.plot_history[key].append(lr[key])

        # Replot
        if self.num_steps % self.plot_freq == 0: self.plot()

    def plot(self):
        """
        Update live plot
        """
        # Initialize figure if first plot
        if self.fig is None:
            self.fig, self.axs = plt.subplots(len(self.plot_history), 1, figsize=(8, len(self.plot_history)*3))
            for ax, key in zip(self.axs, self.plot_history.keys()):
                ax.set_title(key)
                ax.set_xlabel("Step")
                ax.grid(True)
            plt.tight_layout()
            if self.render: 
                plt.ion()
                plt.show()
            else:
                plt.ioff()

        for i, key in enumerate(self.plot_history.keys()):
            self.axs[i].cla() 
            self.axs[i].plot(self.plot_history[key], label=key, color=self.colors[i])
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
        self.test_dataset = test_dataset

        # Params
        self.batch_size = batch_size
        self.device = device

        if 'particle_grav' in dataset_name: self.dataset_latent_func = self.eval_four_var_latent
        elif 'cartpole' in dataset_name: self.dataset_latent_func = self.eval_four_var_latent
        elif 'reacher' in dataset_name: self.dataset_latent_func = self.eval_four_var_latent
            
        self.dataset_name = dataset_name

    def eval(self, run_path, vid_max_frames=50):
        self.model.eval()
        print("\Calculating eval metrics...")
        self.eval_metrics(run_path=run_path)
        print("Generating latent space figure...")
        self.dataset_latent_func(run_path)
        print("\nGenerating trajectory video...")
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
            self.test_dataset, batch_size=1, shuffle=False
        )

        # Precompute frames
        assert self.model.pred_length == 1, 'Pred length >1 not supported for eval video'
        x_list, x_next_list, x_recon_list, x_pred_list = [], [], [], []
        if self.model.output_uncertainty: x_pred_uncertainty_list = []
        for i, (x, x_next, u) in enumerate(test_loader):
            if i >= max_frames:
                break
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)
            x_recon, x_pred, sample_return = self.model.sample(x, u, return_all=True)
            x_list.append(x[0]); x_next_list.append(x_next[0])
            x_recon_list.append(x_recon); x_pred_list.append(x_pred)
            if self.model.output_uncertainty: x_pred_uncertainty_list.append(sample_return['x_pred_recon_uncertainty'].mean().item())

        # Initialize axes
        ims = []
        img_pred = x_recon_list[0][:3]
        img_pred_next = x_pred_list[0][:3]
        img = x_list[-1][:3]
        img_next = x_next_list[-1][:3]
        ims.append(ax[0, 0].imshow(img_pred.permute(1, 2, 0).detach().cpu().numpy()))
        ims.append(ax[1, 0].imshow(img.permute(1, 2, 0).detach().cpu().numpy()))
        ims.append(ax[0, 1].imshow(img_pred_next.permute(1, 2, 0).detach().cpu().numpy()))
        ims.append(ax[1, 1].imshow(img_next.permute(1, 2, 0).detach().cpu().numpy()))

        def update_plot(frame):
            x, x_next = x_list[frame], x_next_list[frame]
            x_recon, x_pred = x_recon_list[frame], x_pred_list[frame]
            x_pred_uncertainty = x_pred_uncertainty_list[frame]
            ims[0].set_data(x_recon[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[1].set_data(x[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[2].set_data(x_pred[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[3].set_data(x_next[:3].permute(1, 2, 0).detach().cpu().numpy())
            if self.model.output_uncertainty: 
                ax[0, 1].set_title(f"Pred: Uncertainty={x_pred_uncertainty:.4f}")

            # plt.show()

        # Create and save animation
        ani = FuncAnimation(fig, update_plot, frames=50, interval=5.)
        writer = FFMpegWriter(fps=2)
        vid_name = f'eval_vid.mp4'
        try:
            filepath = run_path / vid_name
            print(f'Saved eval video to {filepath}')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('Exception occured, saved eval video to current directory')
            ani.save(vid_name, writer=writer)
        plt.close(fig)
        return
    
    def eval_metrics(self, max_samples=None, run_path=None):
        """
        Calculate metrics for test set
        """
        # Image metrics
        psnr_fn = psnr(data_range=1.0).to(self.device)
        ssim_fn = ssim(data_range=1.0).to(self.device)
        # TODO: have to double check this is correct for lpips - check nerfstudio code
        lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        recon_psnr, recon_ssim, recon_lpips, recon_mse = [], [], [], []
        pred_psnr, pred_ssim, pred_lpips, pred_mse = [], [], [], []
        pred_cmr, pred_kurt, pred_shoulder = [], [], []

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False)

        for i, (x, x_next, u) in tqdm(enumerate(test_loader)):
            if max_samples:
                if i >= max_samples:
                    break
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            x_next = torch.hstack([x_next for _ in range(self.model.past_length)]).to(self.device)

            x_recon, x_pred, sample_return = self.model.sample(x, u, return_all=True)
            mu_pred = sample_return['mu_pred']

            img_true = x[0][:3].unsqueeze(0)
            img_true_next = x_next[0][:3].unsqueeze(0)
            img_recon = x_recon[0][:3].unsqueeze(0)
            img_pred = x_pred[0][:3].unsqueeze(0)

            # PSNR / SSIM
            # recon_psnr.append(psnr_fn(img_recon, img_true).item())
            # recon_ssim.append(ssim_fn(img_recon, img_true).item())
            pred_psnr.append(psnr_fn(img_pred, img_true_next).item())
            pred_ssim.append(ssim_fn(img_pred, img_true_next).item())

            # MSE
            # recon_mse.append(F.mse_loss(img_recon, img_true).item())
            # pred_mse.append(F.mse_loss(img_pred, img_true_next).item())

            # LPIPS (scale to [-1,1])
            # recon_lpips.append(lpips_fn(img_recon*2-1, img_true*2-1).item())
            pred_lpips.append(lpips_fn(img_pred*2-1, img_true_next*2-1).item())

            # CMR, Kurtosis, Shoulder Mass
            pred_cmr.append(central_mass_ratio(mu_pred).mean().item())
            pred_kurt.append(excess_kurtosis(mu_pred).mean().item())
            pred_shoulder.append(shoulder_mass(mu_pred).mean().item())

        results = {
            # "recon_psnr": np.mean(recon_psnr),
            # "recon_ssim": np.mean(recon_ssim),
            # "recon_lpips": np.mean(recon_lpips),
            # "recon_mse": np.mean(recon_mse),
            "pred_psnr": np.mean(pred_psnr),
            "pred_ssim": np.mean(pred_ssim),
            "pred_lpips": np.mean(pred_lpips),
            # "pred_mse": np.mean(pred_mse),
            "pred_cmr": np.mean(pred_cmr),
            "pred_kurt": np.mean(pred_kurt),
            "pred_shoulder": np.mean(pred_shoulder)
        }

        # Save metrics as JSON
        if run_path is not None:
            metrics_path = run_path / "metrics.json"
            try:
                with open(metrics_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Saved evaluation metrics to {metrics_path}")
            except Exception as e:
                print(f"Failed to save metrics.json: {e}")

        return results
    
    def eval_four_var_latent(self, run_path, lv=4):
        """
        Visalize all variable combos of a four variable E2C latent space on the test dataset

        Credit: Jueun Kwon, Northwestern University
        """
        # Visualize latent space considering mean and variance
        fig, axes = plt.subplots(lv-1, lv-1, figsize=(16, 16), dpi=200, tight_layout=True)

        # Initialize axes
        combo_array = pairs = [(i, j) for i in range(lv-1) for j in range(i + 1)]
        lv_array = list(itertools.combinations([0, 1, 2, 3], r=2))
        for combo, lv in zip(combo_array, lv_array):
            ax = axes[combo]
            ax.set_aspect('equal')
            ax.set_title(f'Latent Space from Test Dataset (Vars {lv[0]+1} and {lv[1]+1})')

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
            if 'cartpole' in self.dataset_name or 'particle_grav' in self.dataset_name: 
                color = colors[round(u.cpu().detach().numpy().flatten()[0])]
            elif 'reacher' in self.dataset_name:
                u = u.cpu().detach().numpy().flatten()
                if u[0] > 0.0 and u[1] > 0.0: color = 'blue'
                elif u[0] > 0.0 and u[1] < 0.0: color = 'green'
                elif u[0] < 0.0 and u[1] > 0.0: color = 'yellow'
                else: color = 'red'

            # Plotting all variable combos
            for combo, lv in zip(combo_array, lv_array):
                sc = axes[combo].scatter(z_mean_np[:, lv[0]], z_mean_np[:, lv[1]], s=point_sizes, alpha=0.1, label=None, color=color)

        # Combine all latent means and variances
        latent_mean = torch.cat(latent_mean).cpu().detach().numpy()
        latent_var = torch.cat(latent_var).cpu().detach().numpy()

        # Adjust plot limits
        for combo in combo_array:
            ax = axes[combo]
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            a_min = np.minimum(x_min, y_min)
            a_max = np.maximum(x_max, y_max)
            ax.set_xlim(a_min, a_max)
            ax.set_ylim(a_min, a_max)
            # ax.set_xlim(-max_val, max_val)
            # ax.set_ylim(-max_val, max_val)
        
        for row in range(3):
            for col in range(3):
                if row < col:
                    axes[row, col].set_visible(False)

        fig_name = f'latent_fig.png'
        try:
            filepath = run_path / fig_name
            print(f'Saved all variable latent space figure to {filepath}')
            fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('Exception occured, saved all variable latent space figure to current directory')
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
    

if __name__ == "__main__":
    from pathlib import Path
    import yaml, json
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data"
    CONFIG_PATH = PROJECT_ROOT / "config"
    RUNS_PATH = PROJECT_ROOT / "runs"

    config_name = 'e2c_reacher_500k'
    with open(CONFIG_PATH / f'{config_name}.yaml', "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config['train']['device'])
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
        
    dataset = E2CDataset(config)
    train_size = int(len(dataset) * config['train']['train_ratio'])
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    config['vae']['out_image_shape'] = dataset.img_shape
    config['trans']['control_size'] = dataset.U.shape[-1]
    model = ConvE2C(
        enc_latent_size=config['vae']['enc_latent_size'],
        latent_size=config['trans']['latent_size'],
        control_size=config['trans']['control_size'],
        past_length=config['trans']['past_length'],
        pred_length=config['trans']['pred_length'],
        conv_params=config['vae'],
        device=device
    )
    model.to(device)
    model_path = config['train']['load_path'] + '/model.pt'
    model.load_state_dict(torch.load(model_path))
    config['run_path'] = PROJECT_ROOT / Path(config['train']['load_path'])

    # Evaluate model performance
    print('*** EVAL ***\n')
    if config['train']['eval']:
        evaluator = Evaluator(
            model, 
            test_dataset,
            batch_size=config['train']['batch_size'], 
            device=config['train']['device'],
            dataset_name=config['train']['dataset']
        )
        results = evaluator.eval_metrics()
        save_results = config['run_path'] / 'eval_metrics.json'
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nSaved eval metrics to {save_results}')

    print('\n*** DONE ***')