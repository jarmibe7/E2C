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
from pathlib import Path
import yaml
from torchmetrics import PeakSignalNoiseRatio as psnr
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
import lpips
import mujoco

from src.model.e2c import ConvE2C
from src.model.rssm import RSSME2C
from src.dataset import E2CDataset
from src.utils import set_seed, anim_frames, shoulder_mass, excess_kurtosis, central_mass_ratio
from src.data_gen.gen_fetch import process_image
from src.data_gen.gen_state_rep import get_env_state
from src.model.state_rep import StateRepresentationModel

PROJECT_ROOT = Path(__file__).parent.parent

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

            # Ensure self.axs is a list
            if len(self.plot_history) == 1:
                self.axs = [self.axs]

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
        else: self.dataset_latent_func = self.eval_four_var_latent
            
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

    def eval_state_rep(self, trainer, run_path, max_steps=25, closed_loop=True):
        """
        Use a pretrained state representation model to evaluate prediction errors

        Args:
            trainer: Trainer instance (must expose env, model, device, past_length, pred_length/config).
            run_path: Path to save the resulting video.
            max_steps: Number of environment steps to visualize.
            closed_loop: If True, feed ground-truth observations back into the model window.
                         If False, feed the model's predicted next frame (open-loop rollout).
        """
        # Load state rep model
        sr_run_path = PROJECT_ROOT / 'runs' / self.dataset_name.split('_')[0] / 'state_rep' / 'trained'
        device = trainer.device
        with open(sr_run_path / f'config.yaml', "r") as f:
            sr_config = yaml.safe_load(f)
        sr_model = StateRepresentationModel(
            state_size=sr_config['train']['state_size'],
            conv_params=sr_config['conv'],
            device=device
        )
        sr_model_path = sr_run_path / 'model.pt'
        sr_model.load_state_dict(torch.load(sr_model_path))
        sr_model.to(device)

        model = trainer.model
        env = trainer.env
        past_len = model.past_length if hasattr(model, 'past_length') else 3
        pred_len = model.pred_length if hasattr(model, 'pred_length') else 3
        try:
            closed_loop_policy = trainer.config['closed_loop']['policy']
        except:
            print('No closed loop policy found in trainer config, defaulting to random')
            closed_loop_policy = 'random'

        model.eval()
        sr_model.eval()
        obs, _ = env.reset()

        frame_buffer = []   # frames used as model input window
        immediate_state_errors = []         # Prediction error at timestep t+1
        cumulative_state_errors = []        # cumulative pred error for entire pred_length

        # Prime buffer with the first observation
        first_img = process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1)
        for _ in range(past_len):
            frame_buffer.append(first_img)

        step_idx = 0
        for step_idx in tqdm(range(max_steps), desc="Eval with state representation model"):
            # Current frame (ground truth)
            curr_img = process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1)

            # Action: reuse trainer.collect_rollouts logic (random sample)
            if closed_loop_policy == 'informative':
                trainer._init_cem_mu_sig()
                action_seq = trainer._sample_cem(frame_buffer[-past_len:])[0] # pred_len, act_size
            else:
                # repeat pred len number of times for action horizon
                act = [env.action_space.sample() for _ in range(pred_len)]
                action_seq = torch.from_numpy(np.array(act)).to(device)
            env_act = action_seq.cpu().detach().numpy()[0]

            # Step env
            obs, _, terminated, truncated, _ = env.step(env_act)
            next_img_true = process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1)
            state_true = get_env_state(env, obs, self.dataset_name)

            # Model forward pass
            with torch.no_grad():
                # frames = [f.to(trainer.device) for f in frame_buffer[-past_len:]]
                # window = trainer._frames_to_tensor(frames)
                window = torch.stack(frame_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                mu_q, log_var_q, z_q = trainer.model.encode_posterior(window)
                x_recon = trainer._decode_latent(z_q[:, -1])
                x_recon_next = []
                h = torch.zeros(model.num_layers, 1, model.deterministic_size, device=trainer.device)
                for act in action_seq:
                    act_batch = act.view(1, -1).to(trainer.device)
                    if len(z_q.shape) == 2:
                        h, z_q, mu_p, log_var_p = trainer.model.rssm_step(h, z_q.unsqueeze(1), act_batch)
                    else:
                        h, z_q, mu_p, log_var_p = trainer.model.rssm_step(h, z_q, act_batch)
                    # z_q = torch.normal(0.5*torch.ones_like(mu_q), 0.1*torch.ones_like(log_var_q))
                    x_recon_next.append(trainer._decode_latent(z_q).detach().cpu())

            # State rep eval for error per dimension of state vec
            with torch.no_grad():
                # Immediate error
                s_true_t1 = torch.as_tensor(state_true)
                s_pred_t1 = sr_model(x_recon_next[0].unsqueeze(0).to(device))['state_pred'].squeeze(0).detach().cpu()
                immediate_err = (s_pred_t1 - s_true_t1).abs().squeeze(0).numpy()
                immediate_state_errors.append(immediate_err)

                # Cumulative error
                np.zeros_like(immediate_err)

                # # Save env state for temporary rollout
                cum_err = np.zeros_like(immediate_err)

                mj_model = env.unwrapped.model
                data  = env.unwrapped.data
                spec = (
                    mujoco.mjtState.mjSTATE_INTEGRATION |
                    mujoco.mjtState.mjSTATE_PHYSICS |
                    mujoco.mjtState.mjSTATE_CTRL
                )

                state_dim = mujoco.mj_stateSize(mj_model, spec)
                mj_state = np.zeros(state_dim, dtype=np.float64)
                mujoco.mj_getState(mj_model, data, mj_state, spec)
                obs_k = obs
                for k in range(pred_len):
                    # Rollout predicted actions to get ground truth next states
                    if k > 0:
                        obs_k, _, _, _, _ = env.step(action_seq[k].cpu().numpy())

                    s_true_k = torch.as_tensor(get_env_state(env, obs_k, self.dataset_name))
                    s_pred_k = sr_model(x_recon_next[k].unsqueeze(0).to(device))['state_pred'].squeeze(0).detach().cpu()
                    cum_err += (s_pred_k - s_true_k).abs().squeeze(0).numpy()

                cum_err = (cum_err / pred_len)
                cumulative_state_errors.append(cum_err)

                # Restore env state
                mujoco.mj_setState(mj_model, data, mj_state, spec)
                mujoco.mj_forward(mj_model, data)

            if step_idx == 0:
                print("Reconstruction shape:", x_recon.shape, "Next Reconstruction shape:", len(x_recon_next), x_recon_next[0].shape)

            # Feed next frame based on loop type
            next_for_buffer = next_img_true if closed_loop else x_recon_next.detach().cpu()

            # Update buffers/logs
            frame_buffer.append(next_for_buffer)

            if terminated or truncated:
                env.reset()
                done = False
                frame_buffer = [process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1) for _ in range(past_len)]


        ##### DEBUG #####
        num_debug_frames = 5
        debug_frames = x_recon_next[:num_debug_frames]  # list of [C,H,W] tensors

        fig, axes = plt.subplots(1, len(debug_frames), figsize=(15, 3), dpi=150)
        for i, frame in enumerate(debug_frames):
            img = frame.permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize to [0,1]
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Step {i}")

        plt.tight_layout()
        fig.savefig("debug.png")
        ##### DEBUG #####


        # Convert to arrays for boxplot
        immediate_arr = np.stack(immediate_state_errors, axis=0)   # [num_steps, state_dim]
        cumulative_arr = np.stack(cumulative_state_errors, axis=0) # [num_steps, state_dim]
        state_dim = immediate_arr.shape[1]

        # Boxplot per state dimension
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, tight_layout=True)

        axes[0].boxplot([immediate_arr[:, i] for i in range(state_dim)], patch_artist=True)
        axes[0].set_xticklabels([f"Dim {i}" for i in range(state_dim)], rotation=45)
        axes[0].set_title("Immediate (1-step) Prediction Error per Dimension")
        axes[0].set_ylabel("Absolute Error")
        axes[0].grid(True)

        axes[1].boxplot([cumulative_arr[:, i] for i in range(state_dim)], patch_artist=True)
        axes[1].set_xticklabels([f"Dim {i}" for i in range(state_dim)], rotation=45)
        axes[1].set_title(f"Cumulative ({pred_len}-step) Prediction Error per Dimension")
        axes[1].set_ylabel("Absolute Error")
        axes[1].grid(True)

        fig_name = f'sr_error_fig.png'
        try:
            filepath = run_path / fig_name
            print(f'\nSaved state rep error figure to {filepath}')
            fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saved state rep error figure to current directory')
            fig.savefig(fig_name)
        plt.close(fig)
        return

    def visualize_planner(self, trainer, run_path, max_steps=25, closed_loop=True):
        """
        Visualize trajectories using the trainer's planner/rollout logic.

        Args:
            trainer: Trainer instance (must expose env, model, device, past_length, pred_length/config).
            run_path: Path to save the resulting video.
            max_steps: Number of environment steps to visualize.
            closed_loop: If True, feed ground-truth observations back into the model window.
                         If False, feed the model's predicted next frame (open-loop rollout).
        """
        model = trainer.model
        device = trainer.device
        env = trainer.env
        past_len = model.past_length if hasattr(model, 'past_length') else 3
        pred_len = model.pred_length if hasattr(model, 'pred_length') else 3
        try:
            closed_loop_policy = trainer.config['closed_loop']['policy']
        except:
            print('No closed loop policy found in trainer config, defaulting to random')
            closed_loop_policy = 'random'

        model.eval()
        obs, _ = env.reset()
        frame_buffer = []   # frames used as model input window
        true_frames = []    # ground-truth frames for visualization
        recon_frames = []   # model reconstructions
        pred_sequences = [] # list of predicted sequences per step
        plan_obj_vals = []  # objective function values per step
        env_rew = []        # env rewards [blind during training]

        # Prime buffer with the first observation
        first_img = process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1)
        for _ in range(past_len):
            frame_buffer.append(first_img)

        step_idx = 0
        for step_idx in tqdm(range(max_steps), desc="Visualizing Planner timesteps"):
            # Current frame (ground truth)
            curr_img = process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1)

            # Action: reuse trainer.collect_rollouts logic (random sample)
            if closed_loop_policy == 'informative':
                trainer._init_cem_mu_sig()
                mu, costs = trainer._sample_cem(frame_buffer[-past_len:]) # pred_len, act_size
                action_seq = mu.clone()
                plan_obj_vals.append(costs[0].clone().cpu().item())
            else:
                # repeat pred len number of times for action horizon
                act = [env.action_space.sample() for _ in range(pred_len)]
                action_seq = torch.from_numpy(np.array(act)).to(device)
                plan_obj_vals.append(0.0)   # no cost info for random policy
            env_act = action_seq.cpu().detach().numpy()[0]
            # if not getattr(trainer, 'env_continuous', True):
            #     env_act = int(env_act.item())

            # Step env
            _, rew, done, _, _ = env.step(env_act)
            env_rew.append(rew)
            next_img_true = process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1)

            # Model inputs
            with torch.no_grad():
                # frames = [f.to(trainer.device) for f in frame_buffer[-past_len:]]
                # window = trainer._frames_to_tensor(frames)
                window = torch.stack(frame_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                mu_q, log_var_q, z_q = trainer.model.encode_posterior(window)
                x_recon = trainer._decode_latent(z_q[:, -1]) # recon current frame
                x_recon_next = []
                h = torch.zeros(model.num_layers, 1, model.deterministic_size, device=trainer.device)
                for act in action_seq[:pred_len]:
                    act_batch = act.view(1, -1).to(trainer.device)
                    if len(z_q.shape) == 2:
                        h, z_q, mu_p, log_var_p = trainer.model.rssm_step(h, z_q.unsqueeze(1), act_batch)
                    else:
                        h, z_q, mu_p, log_var_p = trainer.model.rssm_step(h, z_q, act_batch)
                    # z_q = torch.normal(0.5*torch.ones_like(mu_q), 0.1*torch.ones_like(log_var_q))
                    x_recon_next.append(trainer._decode_latent(z_q).detach().cpu())

            # Feed next frame based on loop type
            next_for_buffer = next_img_true if closed_loop else x_recon_next[-1].detach().cpu()

            # Update buffers/logs
            frame_buffer.append(next_for_buffer)
            # if len(frame_buffer) > past_len:
            #     frame_buffer = frame_buffer[-past_len:]

            true_frames.append(curr_img)
            recon_frames.append(x_recon.detach().cpu())
            pred_sequences.append(x_recon_next)

            if (step_idx + 1) % trainer.config['closed_loop']['num_rollout_steps'] == 0:
                env.reset()
                done = False
                frame_buffer = [process_image(env.render(), self.dataset_name).squeeze(0).permute(2, 0, 1) for _ in range(past_len)]

        # Build visualization grid: 2 rows, (pred_len + 1) columns
        cols = pred_len + 1
        fig, ax = plt.subplots(2, cols, figsize=(3 * cols, 10))
        ax = np.atleast_2d(ax)
        ax[0, 0].set_title(f"Pred t=0; {plan_obj_vals[0]:.2f}")
        ax[1, 0].set_title(f"True t=0; {env_rew[0]:.2f}")
        for j in range(1, cols):
            ax[0, j].set_title(f"Pred t={j}")
            ax[1, j].set_title(f"True t={j}")

        ims = []
        # Initialize cells
        ims.clear()
        for j in range(cols):
            ims.append(ax[0, j].imshow(np.zeros_like(true_frames[0].permute(1, 2, 0))))
            ims.append(ax[1, j].imshow(np.zeros_like(true_frames[0].permute(1, 2, 0))))
        for a in ax.flatten():
            a.axis('off')

        def update(frame_idx):
            # TODO: Plot KLD value
            # TODO: somehow show "reward"/number of times interacted with object?
            true_curr = true_frames[frame_idx]
            true_next = true_frames[frame_idx + 1] if frame_idx + 1 < len(true_frames) else true_curr
            recon = recon_frames[frame_idx]

            # Pred current recon
            ax[0, 0].set_title(f"Pred t=0; {plan_obj_vals[frame_idx]:.2f}")
            ax[1, 0].set_title(f"True t=0; {env_rew[frame_idx]:.2f}")
            ims[0].set_data(recon[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[1].set_data(true_curr[:3].permute(1, 2, 0).detach().cpu().numpy())

            # Predictions across horizon
            for j in range(1, cols):
                pred_frame = pred_sequences[frame_idx][j-1]
                ims[2 * j].set_data(pred_frame.permute(1, 2, 0).detach().cpu().numpy())
                if j == 1:
                    true_frame = true_next  # reuse true_next for all future slots
                    ims[2 * j + 1].set_data(true_frame[:3].permute(1, 2, 0).detach().cpu().numpy())

        ani = FuncAnimation(fig, update, frames=len(true_frames), interval=5.)
        writer = FFMpegWriter(fps=2)
        vid_name = 'planner_CL_' + closed_loop_policy + f'_{trainer.curr_epoch}.mp4' if closed_loop else 'planner_vis_OL_' + closed_loop_policy + f'_{trainer.curr_epoch}.mp4'
        try:
            filepath = run_path / vid_name
            print(f'Saved planner visualization to {filepath}\n')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('Exception occurred, saved planner visualization to current directory')
            ani.save(vid_name, writer=writer)
        plt.close(fig)
        return
        
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
        # assert self.model.pred_length == 1, 'Pred length >1 not supported for eval video'
        x_list, x_next_list, x_recon_list, x_pred_list = [], [], [], []
        if self.model.output_uncertainty: x_pred_uncertainty_list = []
        for i, (x, x_next, u) in enumerate(test_loader):
            if i >= max_frames:
                break
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            # x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            # x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)
            # x_next = x_next[:, 0]       # TODO: Only eval on first transition?
            # u = u[:, 0]
            # x_recon, x_pred, sample_return = self.model.sample(x, u, return_all=True)
            sample_return = self.model(x, x_next, u)
            x_list.append(x[0]); x_next_list.append(x_next[0])
            x_recon_list.append(sample_return['x_recon']); x_pred_list.append(sample_return['x_preds'][0])
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
            ims[0].set_data(x_recon[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[1].set_data(x[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[2].set_data(x_pred[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[3].set_data(x_next[:3].permute(1, 2, 0).detach().cpu().numpy())
            if self.model.output_uncertainty: 
                x_pred_uncertainty = x_pred_uncertainty_list[frame]
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
            # x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            # x_next = torch.hstack([x_next for _ in range(self.model.past_length)]).to(self.device)
            # x_next = x_next[:, 0]       # TODO: Only eval on first transition?
            # u = u[:, 0]

            # x_recon, x_pred, sample_return = self.model.sample(x, u, return_all=True)
            sample_return = self.model(x, x_next, u)
            mu_pred = sample_return['mu_priors']
            x_recon = sample_return['x_recon']
            x_pred = sample_return['x_preds'][0]

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
            # x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)
            # # Encode current and next state
            # enc_out = self.model.encoder(x)

            # # Get record latent space
            # mu = self.model.mu(enc_out)
            # log_var = self.model.log_var(enc_out)
            x_next = x_next[:, 0]       # TODO: Only eval on first transition?
            u = u[:, 0]
            x_recon, x_pred, sample_return = self.model.sample(x, u, return_all=True)
            mu, log_var = sample_return['mu'], sample_return['log_var']

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
            else:
                color = 'blue'

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
    config['vae']['in_image_shape'] = dataset.in_img_shape
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