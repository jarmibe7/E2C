"""
Training diagnostics for data4wm: live loss plotting and closed-loop planner
visualization/video rendering.

Authors: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm import tqdm

from src.data_gen.gen_fetch import (
    get_mujoco_geom_keys_index,
    is_robot_contact_geometry,
    process_image,
)


class Plotter():
    """Simple class for visualizing training progress."""
    def __init__(self, render, plot_freq):
        """Initalize figure for live plotting."""
        self.num_steps = 0
        self.render = render
        self.plot_freq = plot_freq
        self.plot_history = None
        self.fig = None
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'pink']
        self.twentyfifth = []
        self.seventyfifth = []

    def log(self, lr):
        """Update live training plot logs, and plot at frequency ``self.plot_freq``."""
        self.num_steps += 1
        if self.plot_history is None:
            self.plot_history = {}
            for key in lr.keys():
                self.plot_history[key] = []
            self.plot_history['obj'] = []

        for key in lr.keys():
            if key == 'obj':
                # 'obj' is the full (steps, samples) CEM cost tensor; summarise
                # with median + 25/75 quantiles for a compact training plot.
                data = lr[key].cpu()
                med_vals = torch.median(data, axis=1).values
                twofive = torch.quantile(data, 0.25, axis=1)
                sevenfive = torch.quantile(data, 0.75, axis=1)
                self.plot_history[key] = []
                self.twentyfifth = []
                self.seventyfifth = []
                for val in range(len(med_vals)):
                    self.plot_history[key].append(med_vals[val].item())
                    self.twentyfifth.append(twofive[val].item())
                    self.seventyfifth.append(sevenfive[val].item())
            else:
                self.plot_history[key].append(lr[key])

        if self.num_steps % self.plot_freq == 0:
            self.plot()

    def plot(self):
        """Update live plot."""
        if self.fig is None:
            self.fig, self.axs = plt.subplots(len(self.plot_history), 1, figsize=(8, len(self.plot_history)*3))
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
            if key == 'obj':
                self.axs[i].plot(self.plot_history[key], label='median obj', color='darkorchid')
                self.axs[i].fill_between(range(len(self.plot_history[key])), self.twentyfifth, self.seventyfifth, color='darkorchid', alpha=0.1)
                self.axs[i].set_ylim([0., 1e4])
            else:
                self.axs[i].plot(self.plot_history[key], label=key, color=self.colors[i])
            self.axs[i].legend()

        plt.tight_layout()
        if self.render:
            plt.pause(0.001)

    def save(self, run_path, timestamp=None):
        """Save live training plot."""
        self.plot()
        fig_name = 'loss_fig.png'
        try:
            filepath = run_path / fig_name
            print(f'\nSaved loss figure to {filepath}')
            self.fig.savefig(filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saved loss figure to current directory')
            self.fig.savefig(fig_name)
        self.close()

    def close(self):
        plt.close(self.fig)


class Evaluator():
    """Evaluator that renders closed-loop planner rollouts as videos."""
    def __init__(self, model, test_dataset, batch_size, device, dataset_name):
        self.model = model
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.device = device
        self.dataset_name = dataset_name

    # ------------------------------------------------------------------
    # Shared rollout loop used by both visualize_planner and render_video.
    # ------------------------------------------------------------------
    def _collect_planner_rollout(self, trainer, max_steps, closed_loop, env_reset_seed):
        """Step the env under the trainer's planner and return all the per-step
        buffers both visualizers need.

        Returns a dict with keys: ``true_frames``, ``recon_frames``,
        ``pred_sequences``, ``plan_obj_vals``, ``env_rew``, ``saved_state``,
        ``closed_loop_policy``.
        """
        model = trainer.model
        device = trainer.device
        env = trainer.env
        past_len = model.past_length if hasattr(model, 'past_length') else 3
        pred_len = model.pred_length if hasattr(model, 'pred_length') else 3
        try:
            closed_loop_policy = trainer.config['closed_loop']['policy']
        except KeyError:
            print('No closed loop policy found in trainer config, defaulting to random')
            closed_loop_policy = 'random'

        saved_state = torch.zeros((max_steps, 13), device='cpu')
        if env_reset_seed is None:
            env_reset_seed = np.random.randint(0, 1e2)

        model.eval()
        obs, _ = env.reset(seed=env_reset_seed)
        frame_buffer = []
        true_frames = []
        recon_frames = []
        pred_sequences = []
        plan_obj_vals = []
        env_rew = []

        mj_data = env.unwrapped.data
        robot_geom, obj_geom = get_mujoco_geom_keys_index(self.dataset_name)

        # Prime buffer with the first observation
        first_render_raw = env.render()
        for _ in range(past_len):
            frame_buffer.append(process_image(first_render_raw, self.dataset_name, downscale=True).permute(2, 0, 1))

        if closed_loop_policy in ['informative', 'maxdyn']:
            trainer._init_cem_mu_sig()
            mu = trainer.init_control.clone()
            sigma = trainer.sigma.clone()

        for step_idx in tqdm(range(max_steps), desc="Visualizing Planner timesteps"):
            curr_render_raw = env.render()
            curr_img = process_image(curr_render_raw, self.dataset_name, downscale=False).permute(2, 0, 1)

            # Plan next action sequence
            if closed_loop_policy in ['informative', 'maxdyn']:
                mu, costs, sigma = trainer._sample_cem(frame_buffer[-past_len:], mu=mu, sigma=sigma)
                action_seq = mu.clone()
                plan_obj_vals.append(costs[0].clone().cpu().item())
            else:
                act = [env.action_space.sample() for _ in range(pred_len)]
                action_seq = torch.from_numpy(np.array(act)).to(device)
                plan_obj_vals.append(0.0)
            env_act = action_seq.cpu().detach().numpy()[0]

            # Step env (possibly with meta_ts sub-steps for Meta-World)
            for _ in range(trainer.meta_ts):
                obs, rew, done, trunc, _ = env.step(env_act)
                if trunc:
                    print("forced to reset", step_idx)
                    _, _ = env.reset(seed=env_reset_seed + step_idx + 1)
            contact = int(is_robot_contact_geometry(mj_data, robot_geom, obj_geom))
            saved_state[step_idx] = torch.as_tensor([*obs[0:7], rew, *env_act, contact], device='cpu')
            env_rew.append(rew)
            next_render_raw = env.render()

            # Model rollout over the planned action sequence
            with torch.no_grad():
                window = torch.stack(frame_buffer[-past_len:], dim=0).unsqueeze(0).to(device)
                _, _, zs = model.encode_posterior(window)
                h = torch.zeros(model.num_layers, 1, model.deterministic_size, device=device)
                z = zs[:, -1]
                if model.output_uncertainty:
                    x_recon, _ = model.decoder(z)
                else:
                    x_recon = model.decoder(z)
                x_recon_next = []
                for act in action_seq:
                    act_batch = act.view(1, -1).to(device)
                    h, z_prior, _, _ = model.rssm_step(h, z.unsqueeze(1), act_batch)
                    if model.output_uncertainty:
                        x_pred, _ = model.decoder(z_prior)
                    else:
                        x_pred = model.decoder(z_prior)
                    x_recon_next.append(x_pred.detach().cpu())

                    if trainer.past_length > 1:
                        window_frames = window[:, 1:]
                        window = torch.cat([window_frames, x_pred.unsqueeze(1).detach()], dim=1)
                    else:
                        window = x_pred.detach()
                    _, _, zs = model.encode_posterior(window)
                    z = zs[:, -1]

            # Decide what to feed back into the model window
            if closed_loop:
                next_for_buffer = process_image(next_render_raw, self.dataset_name, downscale=True).permute(2, 0, 1)
            else:
                next_for_buffer = x_recon_next[-1].detach().cpu()
            frame_buffer.append(next_for_buffer)

            true_frames.append(curr_img)
            recon_frames.append(x_recon.detach().cpu())
            pred_sequences.append(x_recon_next)

        return {
            'true_frames': true_frames,
            'recon_frames': recon_frames,
            'pred_sequences': pred_sequences,
            'plan_obj_vals': plan_obj_vals,
            'env_rew': env_rew,
            'saved_state': saved_state,
            'closed_loop_policy': closed_loop_policy,
            'pred_len': pred_len,
        }

    @staticmethod
    def _save_animation(ani, writer, run_path, vid_name):
        try:
            filepath = run_path / vid_name
            print(f'Saved planner visualization to {filepath}\n')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('Exception occurred, saved planner visualization to current directory')
            ani.save(vid_name, writer=writer)

    # ------------------------------------------------------------------
    # Training-time visualization: predicted vs. true image across horizon.
    # ------------------------------------------------------------------
    def visualize_planner(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None):
        """Render a pred-vs-true panel video for the current planner rollout.

        Args:
            trainer: Trainer instance (must expose env, model, device, past_length, pred_length/config).
            run_path: Path to save the resulting video.
            max_steps: Number of environment steps to visualize.
            closed_loop: If True, feed ground-truth observations back into the
                model window. If False, feed the model's predicted next frame
                (open-loop rollout).
        """
        data = self._collect_planner_rollout(trainer, max_steps, closed_loop, env_reset_seed)
        true_frames = data['true_frames']
        recon_frames = data['recon_frames']
        pred_sequences = data['pred_sequences']
        plan_obj_vals = data['plan_obj_vals']
        env_rew = data['env_rew']
        pred_len = data['pred_len']
        closed_loop_policy = data['closed_loop_policy']

        cols = pred_len + 1
        fig, ax = plt.subplots(2, cols, figsize=(3 * cols, 8))
        ax = np.atleast_2d(ax)
        ax[0, 0].set_title(f"Pred t=0; {plan_obj_vals[0]:.2f}")
        ax[1, 0].set_title(f"True t=0; {env_rew[0]:.2f}")
        for j in range(1, cols):
            ax[0, j].set_title(f"Pred t={j}")
            ax[1, j].set_title(f"True t={j}")

        ims = []
        for j in range(cols):
            ims.append(ax[0, j].imshow(np.zeros_like(true_frames[0].permute(1, 2, 0))))
            ims.append(ax[1, j].imshow(np.zeros_like(true_frames[0].permute(1, 2, 0))))
        for a in ax.flatten():
            a.axis('off')

        def update(frame_idx):
            # TODO: Plot KLD value / interaction count
            true_curr = true_frames[frame_idx].squeeze(0)
            true_next = true_frames[frame_idx + 1].squeeze(0) if frame_idx + 1 < len(true_frames) else true_curr
            recon = recon_frames[frame_idx].squeeze(0)

            ax[0, 0].set_title(f"Pred t=0; {plan_obj_vals[frame_idx]:.2f}")
            ax[1, 0].set_title(f"True t=0; {env_rew[frame_idx]:.2f}")
            ims[0].set_data(recon[:3].permute(1, 2, 0).detach().cpu().numpy())
            ims[1].set_data(true_curr[:3].permute(1, 2, 0).detach().cpu().numpy())

            for j in range(1, cols):
                pred_frame = pred_sequences[frame_idx][j-1].squeeze(0)
                ims[2 * j].set_data(pred_frame.permute(1, 2, 0).detach().cpu().numpy())
                if j == 1:
                    true_frame = true_next
                    ims[2 * j + 1].set_data(true_frame[:3].permute(1, 2, 0).detach().cpu().numpy())

        ani = FuncAnimation(fig, update, frames=len(true_frames), interval=5.)
        writer = FFMpegWriter(fps=20)
        vid_name = ('CL_' if closed_loop else 'vis_OL_') + closed_loop_policy + f'_{trainer.curr_epoch}.mp4'
        self._save_animation(ani, writer, run_path, vid_name)
        plt.close(fig)
        return data['saved_state']

    # ------------------------------------------------------------------
    # Final paper-quality single-panel video of the true rollout only.
    # ------------------------------------------------------------------
    def render_video(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None):
        """Single-panel video of the ground-truth rollout (no prediction tiles)."""
        trainer.env.unwrapped.max_path_length = 5000
        data = self._collect_planner_rollout(trainer, max_steps, closed_loop, env_reset_seed)
        true_frames = data['true_frames']
        env_rew = data['env_rew']

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ims = [ax.imshow(np.zeros_like(true_frames[0].permute(1, 2, 0)))]
        ax.axis('off')

        def update(frame_idx):
            true_curr = true_frames[frame_idx].squeeze(0)
            ax.set_title(f"t={frame_idx}; {env_rew[frame_idx]:.2f}")
            ims[0].set_data(true_curr[:3].permute(1, 2, 0).detach().cpu().numpy())

        ani = FuncAnimation(fig, update, frames=len(true_frames), interval=5.)
        writer = FFMpegWriter(fps=20)
        vid_name = f'{trainer.env_name}_{env_reset_seed}.mp4'
        self._save_animation(ani, writer, run_path, vid_name)
        plt.close(fig)
        return data['saved_state']
