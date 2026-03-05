import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import gymnasium as gym
import random
from pathlib import Path
from tqdm import tqdm
from pyvirtualdisplay import Display
import time
import os
# os.environ['MUJOCO_GL'] = 'egl'
from src.utils import set_seed, format_time

from src.model.loss import E2CLoss, UncertaintyE2CLoss, RSSMLoss
from src.eval import Plotter, Evaluator
from src.replay_buffer import RLReplayBuffer
from src.tactile.gen_tactile import get_env_modes, process_tactile, process_feature, tactile_envs
from src.trainer import BaseTrainer, ClosedLoopInformativeTrainer
from src.tactile.tactile_utils import TactileReplayBuffer
from src.model.loss import RSSMLoss
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)

class ClosedLoopTactileTrainer(BaseTrainer):
    """
    Use CEM pixel space exploration to learn a world model in tactile space.

    Args:
        dataset: Torch dataset object
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
        prints: Whether to display debug prints
    """
    def __init__(self, dataset, model, config, device, prints=True):
        super().__init__(dataset, model, config, device, prints=prints)
        # Closed loop training parameters
        self.num_features = dataset.num_features
        self.closed_cfg = config.get('closed_loop', {})
        self.num_rollout_steps = self.closed_cfg['num_rollout_steps']
        self.num_batches = self.closed_cfg['num_batches']
        self.pred_length = config['trans'].get('pred_length', 3)                          # How long to predict ahead in network
        self.plan_horizon = self.closed_cfg.get('plan_horizon', self.pred_length)   # How long to imagine rollouts
        self.past_length = config['trans']['past_length']
        self.obj_fun = self.closed_cfg.get('policy', None)
        self.uses_rssm = hasattr(self.model, 'rssm_step') and hasattr(self.model, 'post')
        self.epochs_warmup = self.closed_cfg.get('epochs_warmup', 3)

        # CEM parameters
        self.num_action_samples = self.closed_cfg.get('samples', 100)
        self.elite_frac = self.closed_cfg.get('elite_frac', 0.1)
        self.cem_iters = self.closed_cfg.get('iters', 3)
        self._init_cem_mu_sig()
        self.alpha = self.closed_cfg.get('alpha', 0.7)
        self.plan_horizon = self.closed_cfg.get('plan_horizon', self.pred_length)

        self.model_criterion = RSSMLoss(config['train']['num_epochs'], config['loss'])

        # Initialize environment
        # disp = Display(visible=0, size=(480, 480))
        # disp.start()
        self.env_name = config['train']['dataset'].split('_')[0]
        self.env = tactile_envs[self.env_name](
            max_steps=config['tactile']['episode_length'],
            env_modes=get_env_modes(),
            show_gui=False,             # Don't render in training env
            show_tactile=False,
            image_size=self.in_image_shape[1:],
        )

        # Overwrite super() evaluator with tactile evaluator
        # Initialize replay buffer
        self.replay_buffer = TactileReplayBuffer(
            self.in_image_shape, 
            self.num_features,
            model.control_size, 
            self.closed_cfg['buffer_capacity'],
            device,
            config
        )
        assert self.num_rollout_steps <= self.replay_buffer.capacity, 'Steps per rollout should be <= buffer_capacity!'

        num_states_to_save = 6      # EE position/orientation
        self.saved_state = torch.zeros((self.num_epochs * self.num_rollout_steps, num_states_to_save), device='cpu')

        # TODO: Create new Evaluator

    #
    # ---------- Saving and Eval ----------
    # 
    def save(self, config_save, run_path):
        """
        Save model, policy, and config to run directory
        """
        super().save(config_save, run_path)
        if hasattr(self, 'saved_state'):
            # Save logged states
            # take velocity as well?
            # torch.diff(self.saved_state[:, :3], dim=0, prepend=self.saved_state[0:1, :3])
            try:
                filepath = run_path / 'training_states.pt'
                print(f'Saved state tensor to {filepath}')
                torch.save(self.saved_state, filepath)
            except Exception as e:
                print(e)
                print('Exception occured, saved state tensor to current directory')
                torch.save(self.saved_state, 'training_states.pt')

    #
    # ---------- Utils ----------
    # 
    def _frames_to_tensor(self, frames):
        """
        Stack a window of observation frames to tensor of shape [1, past_length*C, H, W]
        """
        stacked = torch.stack(frames[-self.past_length:], dim=0).unsqueeze(0)
        _, _, C, H, W = stacked.shape
        return stacked.view(1, -1, H, W)

    def _update_window(self, frames, new_frame):
        """
        Shift a frame to include a new frame and remove the oldest
        """
        window = list(frames[-self.past_length+1:]) if len(frames) >= self.past_length else list(frames)
        window.append(new_frame)
        return window

    def _decode_latent(self, z):
        decoded = self.model.decoder(z)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded.detach().squeeze(0)

    def _encode_posterior_e2c(self, obs_tensor):
        enc = self.model.encoder(obs_tensor)
        mu = self.model.mu(enc)
        log_var = self.model.log_var(enc)
        z = self.model.reparameterize(mu, log_var)
        return mu, log_var, z

    def _encode_posterior_rssm(self, obs_tensor, h):
        enc = self.model.encoder(obs_tensor)
        post_in = torch.cat([enc, h], dim=-1)
        post_stats = self.model.post(post_in)
        mu, log_var = post_stats.chunk(2, dim=-1)
        z = self.model.reparameterize(mu, log_var)
        return mu, log_var, z
    
    #
    # ---------- CEM Functionality ----------
    # 
    def _init_cem_mu_sig(self):
        """
        Initialize a Gaussian distribution over action sequences
        """
        # Initialize mean
        cfg_val = self.closed_cfg.get('init_control', 'zero')
        if cfg_val == 'from_txt':
            print("Only 'zero' and 'random' init_control methods are currently supported. Defaulting to 'random'.")
            self.init_control = torch.stack([torch.from_numpy(self.env.action_space.sample()).to(self.device) for _ in range(self.plan_horizon)], dim=0)
        elif cfg_val == 'random':
            # self.init_control = torch.stack([torch.from_numpy(self.env.action_space.sample()).to(self.device) for _ in range(self.plan_horizon)], dim=0)
            self.init_control = torch.stack([torch.from_numpy(np.array([0.04, 0.04, 0.0, 0.0])).to(self.device, torch.float32) for _ in range(self.plan_horizon)], dim=0)
            # if self.env_name in meta_world_envs:
            #     # Scale to reasonable range for Meta-World envs
            #     self.init_control *= 2.0
        else:
            self.init_control = torch.zeros(self.plan_horizon, len(self.env.action_space.sample()), device=self.device)

        # Initialize variance
        self.sigma_init = self.closed_cfg.get('sigma_init', 0.5)    
        self.sigma_min = self.closed_cfg.get('sigma_min', 0.05)     # Min value for clamping variance
        self.sigma = torch.ones_like(self.init_control, device=self.device) * self.sigma_init
        # self.sigma[:, -2] = self.sigma_init * 0.5   # Less variance on z-axis actions
        # self.sigma[:, -1] = self.sigma_init * 0.25   # Less variance on gripper actions

    @staticmethod
    def _kl_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p):
        # Same exact function as RSSMLoss.kl_divergence
        return 0.5 * (
            log_var_p - log_var_q
            + (torch.exp(log_var_q) + (mu_q - mu_p) ** 2) / torch.exp(log_var_p)
            - 1
        ).sum(dim=-1)

    def _rollout_info_gain(self, window, action_seq, t0_dist=None):
        """
        KL Divergence across _latent states_ --> drives agent to dynamic regions
        “go where stuff changes”
        latent novelty
        """
        with torch.no_grad():
            total_kl = 0.0

            if self.uses_rssm:
                h = torch.zeros(self.model.num_layers, 1, self.model.deterministic_size, device=self.device)
                # TODO: can re-use t0 encoding during MPC instead of re-encoding for every action_seq
                if t0_dist is None:
                    mu_q, log_var_q, z_q = self.model.encode_posterior(window)
                else:
                    mu_q, log_var_q = t0_dist
                    z_q = self.model.reparameterize(mu_q, log_var_q)
                for act in action_seq:
                    act_batch = act.view(1, -1).to(self.device)
                    if len(z_q.shape) == 2:
                        h, z_q, mu_p, log_var_p = self.model.rssm_step(h, z_q.unsqueeze(1), act_batch)
                    else:
                        h, z_q, mu_p, log_var_p = self.model.rssm_step(h, z_q, act_batch)
                    
                    # current vs t0
                    # total_kl += self._kl_diag_gaussian(mu_p, log_var_p, mu_q, log_var_q).mean().item()

                    # current vs t_prev
                    total_kl += self._kl_diag_gaussian(mu_p, log_var_p, mu_q, log_var_q).mean().item()
                    mu_q = mu_p
                    log_var_q = log_var_p
                    z_q = self.model.reparameterize(mu_q, log_var_q)
            else:
                # TODO: e2c-specific rollout function is outdated
                mu_q, log_var_q, z_q = self._encode_posterior_e2c(window)
                for act in action_seq:
                    act_batch = act.view(1, -1).to(self.device)
                    mu_p, log_var_p, z_prior = self.model.transition(z_q, mu_q, log_var_q, act_batch)
                    x_pred = self._decode_latent(z_prior)
                    frames = self._update_window(frames, x_pred)
                    window = self._frames_to_tensor(frames)
                    mu_q, log_var_q, z_q = self._encode_posterior_e2c(window)
                    total_kl += self._kl_diag_gaussian(mu_p, log_var_p, mu_q, log_var_q).mean().item()

            return total_kl / self.plan_horizon    # Average info gain per step

    def rollout_info_gain_decoded_batch(self, window, action_samples, t0_dist):
        with torch.no_grad(), torch.cuda.amp.autocast():
            # TODO: Compare how learning architecture has changed, since mcar performace has decreased.
            B, T, A = action_samples.shape
            if t0_dist is None:
                mu_q, log_var_q, zs = self.model.encode_posterior(window)
            else:
                mu_q, log_var_q, zs = t0_dist
            total_kl = torch.zeros(B, device=self.device)

            h = torch.zeros(self.model.num_layers, B, self.model.deterministic_size, device=self.device)
            # only need to repeat if don't broadcast window across batch
            # z = zs[:, -1].repeat(B, 1)
            # mu_q = mu_q[:, -1].repeat(B, 1)
            # log_var_q = log_var_q[:, -1].repeat(B, 1)
            # old way, preserves past_length of zs?
            # z = zs.repeat(B, 1, 1)
            # mu_q = mu_q.repeat(B, 1, 1)
            # log_var_q = log_var_q.repeat(B, 1, 1)
            z = zs[:, -1] # take last timestep
            mu_q = mu_q[:, -1] # take last timestep
            log_var_q = log_var_q[:, -1] # take last timestep
            # mu_t0 = mu_q[:, -1] # take last timestep
            # log_var_t0 = log_var_q[:, -1] # take last timestep

            for t in range(T):
                action = action_samples[:, t]
                # Prior from dynamics
                h, z_prior, mu_p, log_var_p = self.model.rssm_step(
                    h, z.unsqueeze(1), action
                    # h, zs, action
                )
                if self.obj_fun == 'maxdyn':
                    # Encourage dynamics change
                    # total_kl += self._kl_diag_gaussian(mu_p, log_var_p, mu_t0, log_var_t0).mean().item()
                    total_kl += self._kl_diag_gaussian(mu_p, log_var_p, mu_q, log_var_q) # .mean().item()
                    mu_q = mu_p
                    log_var_q = log_var_p
                    # compare with t0
                    # z_prior = self.model.reparameterize(mu_q, log_var_q)
                    # compare with previous --> would expect more dramatic results
                    z = self.model.reparameterize(mu_q, log_var_q)
                else:
                    # Decode prior to observation space
                    if self.model.output_uncertainty:
                        x_pred, x_pred_uncertainty = self.model.decoder(z_prior)
                    else:
                        x_pred = self.model.decoder(z_prior)

                    if self.past_length > 1:
                        window_frames = window[:, 1:]   # drop first frame
                        window = torch.cat([window_frames, x_pred.unsqueeze(1).detach()], dim=1)
                    else:
                        window = x_pred.detach()  # past_length==1, just use pred image
                    mu_post, log_var_post, zs = self.model.encode_posterior(window)
                    z = zs[:, -1]
                    mu_post = mu_post[:, -1]
                    log_var_post = log_var_post[:, -1]

                    # # Posterior from updated observation window
                    # enc = self.model.encoder(x_pred)
                    # stats = self.model.post(enc)
                    # mu_post, log_var_post = stats.chunk(2, dim=-1)

                    # expected information gain - KL(q || p)
                    total_kl += self._kl_diag_gaussian(mu_post, log_var_post, mu_p, log_var_p) # .mean(dim=-1)
                    # TODO: try doing KLD over t0 z, instead of t_prev
                    # total_kl += self._kl_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p).mean(dim=-1)
                    # total_kl += self._kl_diag_gaussian(mu_post, log_var_post, mu_q, log_var_q).mean(dim=-1)
                    
                    # to compare over t0, comment this out
                    mu_q = mu_post
                    log_var_q = log_var_post

                    # update belief <-- happens inherently with model.encode_posterior
                    # zs = self.model.reparameterize(mu_post, log_var_post)

            return total_kl # / T # Average info gain per step
    
    def _sample_cem(self, frame_buffer, mu=None, sigma=None):
        # Initialize CEM distribution
        if mu is None:
            mu = self.init_control.clone() # (plan_horizon, action_size)
        if sigma is None:
            sigma = self.sigma.clone() # (plan_horizon, action_size)
        act_low = torch.as_tensor(self.env.action_space.low, device=self.device, dtype=torch.float32)
        act_high = torch.as_tensor(self.env.action_space.high, device=self.device, dtype=torch.float32)

        # TODO: Need to scale action bounds?
        if self.evaluator.dataset_name == 'pointmaze':
            # scale action bounds for pointmaze
            act_low *= 2.5
            act_high *= 2.5
        for _ in range(self.cem_iters):
            costs = torch.zeros(self.num_action_samples, device=self.device)
            # Sample action sequences from current distribution
            action_samples = torch.normal(mu.unsqueeze(0), sigma.unsqueeze(0)).expand(self.num_action_samples, -1, -1)
            # Clip to action space bounds
            action_samples = torch.clamp(action_samples, act_low, act_high)
            
            # Evaluate information gain for each sequence
            # window = torch.stack(frame_buffer[-self.past_length:], dim=0).unsqueeze(0).to(self.device)
            window = torch.stack(frame_buffer[-self.past_length:], dim=0).unsqueeze(0).to(self.device).repeat(self.num_action_samples, 1, 1, 1, 1)
            mu_prior, log_var_prior, z_prior = self.model.encode_posterior(window)
            costs = self.rollout_info_gain_decoded_batch(window, action_samples, t0_dist=(mu_prior, log_var_prior, z_prior))
            
            # Select elite sequences
            num_elites = max(1, int(self.elite_frac * self.num_action_samples))
            elite_idxs = costs.argsort()[-num_elites:]      # TODO: Is sorting afterwards the most optimal way?
            elite_seqs = action_samples[elite_idxs]
            
            # Update CEM distribution parameters
            # take mean and stddev across elite sequences at each time step
            new_mu = torch.stack([torch.mean(torch.stack([seq[t] for seq in elite_seqs], dim=0), dim=0) for t in range(self.plan_horizon)], dim=0)
            # new_sigma = torch.stack([torch.std(torch.stack([seq[t] for seq in elite_seqs], dim=0), dim=0) for t in range(self.plan_horizon)], dim=0)
            new_sigma = 1 / len(new_mu - 1) * torch.stack([torch.sum(torch.stack([torch.sqrt((seq[t] - new_mu[t])**2) for seq in elite_seqs], dim=0), dim=0) for t in range(self.plan_horizon)], dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            mu = torch.clamp(mu, act_low, act_high)
            sigma = torch.nan_to_num(self.alpha * sigma + (1 - self.alpha) * new_sigma, nan=self.sigma_min)
            sigma = torch.clamp(sigma, min=self.sigma_min)
        
        mu[:-1] = mu[1:].clone()  # shift mean sequence left
        mu[-1] = self.init_control[0]   # last action reverts to initial mean
        sigma[:-1] = sigma[1:].clone()
        sigma[-1] = self.sigma_init     # last action reverts to initial std

        return mu, costs, sigma
    
    def _select_information_action(self, frame_buffer, mu=None, sigma=None):
        mu, costs, sigma = self._sample_cem(frame_buffer, mu=mu, sigma=sigma)
        return mu, costs, sigma

    #
    # ---------- Training Functionality ----------
    # 
    def collect_rollouts(self, epoch):
        """
        Collect observations and save them to replay buffer
        """
        # Initialize env and buffers
        self.model.eval()
        obs, _ = self.env.reset()
        tactile_buffer = []
        feature_buffer = []
        act_buffer = []
        idx = 0

        # Seed initial observation into buffer
        tactile_buffer.append(process_tactile(obs["tactile"]))
        feature_buffer.append(process_feature(obs["extended_feature"]))
        
        self._init_cem_mu_sig()
        mu = self.init_control.clone()
        sigma = self.sigma.clone()
        while idx < self.num_rollout_steps:
            # Select action using random or informative policy based on epoch and buffer length
            if epoch >= self.epochs_warmup and len(tactile_buffer) >= self.past_length:
                if epoch == self.epochs_warmup and len(tactile_buffer) == self.past_length:
                    print(f'Switching to informative action selection using CEM over {self.num_action_samples} samples and {self.cem_iters} iterations. \n')
                tic = time.time()
                mu, costs, sigma = self._select_information_action(frame_buffer=tactile_buffer, mu=mu, sigma=sigma)
                act = mu[0]
                toc = time.time()
                if idx == 0 and epoch == self.epochs_warmup:
                    print(f"CEM planning will take ~{((toc - tic)*self.num_rollout_steps/60):.3f} minutes.")
            else:
                if epoch == 0 and len(tactile_buffer) == 1: 
                    print(f'Initializing data from random actions. \n')
                act = torch.from_numpy(self.env.action_space.sample()).to(self.device)

            # Take action and save state
            env_act = act.cpu().detach().numpy()
            next_obs, rew, done, _, _ = self.env.step(env_act)

            tactile = process_tactile(next_obs['tactile'])
            feature = process_feature(next_obs['extended_feature'])

            tactile_buffer.append(tactile)
            feature_buffer.append(feature)
            act_buffer.append(act)
            if epoch >= 0: self.saved_state[epoch*self.num_rollout_steps + idx] = torch.as_tensor([*next_obs[0:7], rew, *env_act], device='cpu')

            # Maintain sliding window size
            total_len = self.past_length + self.pred_length
            if len(tactile_buffer) > total_len:
                tactile_buffer.pop(0)
                feature_buffer.pop(0)
                act_buffer.pop(0)

            # Save sample when buffer full
            if len(tactile_buffer) == total_len:

                tactile_add = torch.stack(tactile_buffer[:self.past_length], dim=0)
                feature_add = torch.stack(feature_buffer[:self.past_length], dim=0)

                act_add = torch.stack(
                    [
                        a.float()
                        for a in act_buffer[self.past_length - 1 : self.past_length - 1 + self.pred_length]
                    ]
                )

                next_tactile_add = torch.stack(tactile_buffer[self.past_length:], dim=0)
                next_feature_add = torch.stack(feature_buffer[self.past_length:], dim=0)


                self.replay_buffer.add(
                    tactile=tactile_add,
                    feature=feature_add,
                    action=act_add,
                    reward = rew,
                    next_tactile=next_tactile_add,
                    next_feature=next_feature_add,
                    done = done
                )
                idx += 1
        self.model.train()

    def train(self, epoch):
        """
        Gradient update for world model
        """
        total_model_loss = 0.0
        for i in range(self.num_batches):
            # Unload batch
            sample  = self.replay_buffer.sample(self.batch_size)
            tactile, tactile_next = sample.tactile.to(self.device), sample.tactile_next.to(self.device)
            feature, feature_next = sample.feature.to(self.device), sample.feature_next.to(self.device)
            u = sample.u.to(self.device)

            # Forward pass
            train_return = self.model(tactile, feature, tactile_next, feature_next, u)
            train_return['x'] = tactile
            train_return['x_next'] = tactile_next

            # Model loss and backprop
            model_loss, loss_return = self.model_criterion(train_return, epoch)
            self.plotter.log(loss_return)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()

            # TODO: Raise exception
            if torch.isnan(model_loss):
                print("NaN loss encountered, stopping training.")
                break

            total_model_loss += model_loss.item() * tactile.size(0)   # Aggregate total epoch loss

        # Compute average model loss
        avg_model_loss = total_model_loss / (self.batch_size*self.num_batches)

        return avg_model_loss
        
    
    def learn(self):
        model_loss = 0.0
        pbar = tqdm(range(self.curr_epoch, self.num_epochs), desc="Training")

        print("Initializing buffer with random rollouts...")
        for _ in range(max(self.num_batches // self.num_rollout_steps, self.closed_cfg['buffer_capacity'] // self.num_rollout_steps)):
            self.collect_rollouts(-1)

        start_epoch = self.curr_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.curr_epoch = epoch
            model_loss = self.train(epoch)

            pbar.set_postfix({
                'Epoch': epoch+1,
                'Model Loss': f"{model_loss:.4f}"})
            pbar.update(1)

            self.collect_rollouts(epoch)
            if (epoch + 1) % 50 == 0:
                # show model video every 50 epochs
                self.model.eval()
                if self.config['train']['eval']: self.evaluate(self.config['run_path'])
                self.model.train()
            if (epoch + 1) % 100 == 0:
                if self.config['train']['save']: super().save(self.config, self.config['run_path'], model_name=f'model_epoch{epoch+1}.pt')
            
        pbar.close()