"""
Trainers for data4wm world models.

This module contains:

- :class:`BaseTrainer` -- shared optimizer/plotter/evaluator wiring.
- :class:`RSSMPretrainer` -- offline (open-loop) pretraining of the RSSM on a
  fixed dataset produced by ``src/data_gen/gen_fetch.py``.
- :class:`ClosedLoopRandomTrainer` -- online closed-loop baseline that collects
  rollouts with uniform random actions.
- :class:`ClosedLoopInformativeTrainer` -- online closed-loop trainer that
  picks actions via CEM over the learned latent belief model. Implements the
  two objectives from the paper (see :meth:`rollout_info_gain_decoded_batch`):
  pixel information gain (``policy='informative'``, equiv. ``eig``) and
  dynamics information gain (``policy='maxdyn'``).
- :class:`ClosedLoopHardwareTrainer` -- stub preserved so ``main.py`` can route
  ``policy='hardware'``; the actual hardware implementation lives outside this
  repo.

Authors: Jared Berry
"""
import os
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import yaml
from tqdm import tqdm

gym.register_envs(gymnasium_robotics)
os.environ.setdefault('MUJOCO_GL', 'egl')

from src.data_gen.gen_fetch import (
    env_to_aspace,
    meta_world_envs,
    name_to_env,
    process_image,
)
from src.eval import Evaluator, Plotter
from src.model.loss import RSSMLoss
from src.replay_buffer import ReplayBuffer
from src.utils import set_seed


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"


class BaseTrainer():
    """
    Base trainer wiring common to all training loops.

    Args:
        dataset: Torch dataset object that can be split into train/test sets.
        model: World model instance (currently :class:`RSSME2C`).
        config: Loaded yaml config dict with ``train`` / ``vae`` / ``trans`` /
            ``loss`` / ``closed_loop`` sections.
        device: Torch device object.
        prints: Whether to print the closed-loop training stats banner.
    """
    def __init__(self, dataset, model, config, device, prints=True):
        # Split into training and test sets
        train_size = int(len(dataset) * config['train']['train_ratio'])
        test_size = len(dataset) - train_size
        self.dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        self.num_epochs = config['train']['num_epochs']
        self.batch_size = config['train']['batch_size']
        self.in_image_shape = config['vae']['in_image_shape']
        self.out_image_shape = config['vae']['out_image_shape']
        self.device = device
        self.config = config
        if config['closed_loop']['closed_loop']:
            # Derived training-volume stats -- kept as attributes so they can
            # be logged back into the saved config at the end of training.
            self.num_updates = self.num_epochs * config['closed_loop']['num_batches']
            self.num_train_inters = self.batch_size * self.num_updates
            self.num_env_inters = self.num_epochs * config['closed_loop']['num_rollout_steps']
            # num_batches * num_epochs * batch_size        -- gradient-step training interactions
            # num_rollout_steps * num_epochs               -- real environment interactions
            # num_rollout_steps * num_epochs * cem_iters * num_action_samples -- total (real+imagined)
            if prints:
                print(
                    f"Closed-loop training: \n"
                    f"- Total final training interactions: {self.num_train_inters}\n"
                    f"- Total environment interactions collected: {self.num_env_inters}\n"
                    f"- Gradient updates (training iters / batch size): {self.num_updates}\n"
                    f"- Replay buffer capacity: {config['closed_loop']['buffer_capacity']}\n"
                    f"- Rollout steps per epoch: {config['closed_loop']['num_rollout_steps']}\n"
                )
        else:
            print(f"Train size: {train_size}        Test size: {test_size}\n")

        set_seed(config.get('seed', 0))
        self.model = model
        model.to(device)
        self.model_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['trans']['alpha'],
            weight_decay=config['trans']['weight_decay'],
        )
        self.policy = None  # kept for forward-compat with hardware trainer

        # Loss criterion. Only RSSMLoss survives after the legacy E2C cleanup;
        # the ``rssm`` and ``uncertainty`` tags are both routed here because
        # their distinction is baked into the RSSM decoder, not the loss.
        loss_type = config['loss'].get('loss_type', None)
        if loss_type in ('rssm', 'uncertainty'):
            self.model_criterion = RSSMLoss(config['train']['num_epochs'], config['loss'])
        else:
            raise NotImplementedError(f'Loss type "{loss_type}" not supported!')

        # Live plotting + optional test-set evaluator
        self.plotter = Plotter(config['train']['render'], config['train']['plot_freq'])
        if config['train']['eval']:
            self.evaluator = Evaluator(
                self.model,
                test_dataset,
                batch_size=config['train']['batch_size'],
                device=config['train']['device'],
                dataset_name=config['train']['dataset'],
            )
        self.curr_epoch = 0

    def collect_rollouts(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def learn(self, *args, **kwargs):
        pass

    def evaluate(self, run_path):
        """Evaluate model and output figures to run directory."""
        print('\n*** EVAL ***\n')
        self.model.eval()
        self.evaluator.visualize_planner(self, run_path, max_steps=50, closed_loop=True)

    def save(self, config_save, run_path, model_name='model.pt'):
        """Save model + config (and policy, if any) to ``run_path``."""
        self.plotter.save(run_path)

        try:
            filepath = run_path / model_name
            print(f'Saved model to {filepath}')
            torch.save(self.model.state_dict(), filepath)
        except Exception as e:
            print(e)
            print('Exception occured, saved model to current directory')
            torch.save(self.model.state_dict(), model_name)

        if self.policy is not None:
            policy_name = 'policy.pt'
            try:
                filepath = run_path / policy_name
                print(f'Saved policy to {filepath}')
                torch.save(self.policy.state_dict(), filepath)
            except Exception as e:
                print(e)
                print('Exception occured, saved policy to current directory')
                torch.save(self.policy.state_dict(), policy_name)

        try:
            yaml_name = 'config.yaml'
            yaml_path = run_path / yaml_name
            print(f'Saved config to {yaml_path}')
            with open(yaml_path, 'w') as f:
                yaml.dump(config_save, f, sort_keys=False, default_flow_style=False)
        except Exception as e:
            print(e)
            print('Exception occurred, saved config to current directory')
            with open(yaml_name, 'w') as f:
                yaml.dump(config_save, f, sort_keys=False, default_flow_style=False)


class RSSMPretrainer(BaseTrainer):
    """
    Offline pretraining of the RSSM on a fixed ``E2CDataset`` (no environment
    interaction, no replay buffer). Intended as a warm start before closed-loop
    training; see the ``closed_loop: False`` configs in ``config/configs_*``.
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)

    def train(self):
        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        print('\nBeginning Training:')
        epoch_loss = 0.0
        pbar = tqdm(range(self.num_epochs), desc="Training")
        for epoch in pbar:
            total_loss = 0.0

            for x, x_next, u in train_loader:
                x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
                x = x.reshape(x.shape[0], -1, *self.in_image_shape[1:])    # Stack obs history in channel dim

                # Forward pass
                train_return = self.model(x, x_next, u)
                train_return['x'] = x[:, -self.out_image_shape[0]:] # Only compute loss with current frame, not history
                train_return['x_next'] = x_next

                # Compute loss and backprop
                loss, loss_return = self.model_criterion(train_return, epoch)
                self.plotter.log(loss_return)
                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()

                if torch.isnan(loss):
                    print("NaN loss encountered, stopping training.")
                    break

                total_loss += loss.item() * x.size(0)

            epoch_loss = total_loss / len(train_loader.dataset)
            pbar.set_postfix({
                'Epoch': epoch+1,
                'Epoch Loss': f"{epoch_loss:.4f}"})

        pbar.close()

    def learn(self):
        self.train()


class ClosedLoopRandomTrainer(BaseTrainer):
    """
    Online closed-loop trainer that collects rollouts with uniform random
    actions. Serves as the random-policy baseline and as the base class for
    the CEM-based informative trainer below.
    """
    def __init__(self, dataset, model, config, device, prints=True):
        super().__init__(dataset, model, config, device, prints=prints)
        self.num_rollout_steps = config['closed_loop']['num_rollout_steps']
        self.num_batches = config['closed_loop']['num_batches']
        self.pred_length = config['trans'].get('pred_length', 3)   # How long to predict ahead in network
        self.plan_horizon = config['closed_loop'].get('plan_horizon', self.pred_length)   # How long to imagine rollouts
        self.hardware = config['closed_loop'].get('hardware', False)
        self.meta_ts = 1
        assert self.batch_size < config['closed_loop']['buffer_capacity'], \
            'Batch size must be < buffer capacity!'
        if self.num_rollout_steps <= self.num_batches * self.batch_size:
            print(
                f"Warning: num_rollout_steps ({self.num_rollout_steps}) <= "
                f"num_batches * batch_size ({self.num_batches * self.batch_size}).\n"
                f"This may lead to inefficient training, as the probability of "
                f"'double counting' samples in a single epoch is high."
            )

        # Initialize environment
        env_name = config['train']['dataset'].split('_')[0]
        self.env_name = env_name
        if env_name in meta_world_envs:
            # Camera name is baked into the dataset filename: "<env>_gripper_*"
            # uses gripperPOV, "<env>_isom_*" uses corner3. Default to corner3
            # so the original legacy ``<env>_<N>k`` datasets remain loadable.
            camera_name = 'corner3'
            if 'isom' in config['train']['dataset']: camera_name = 'corner3'
            if 'gripper' in config['train']['dataset']: camera_name = 'gripperPOV'
            # NOTE: Set line 153 of metaworld/sawyer_xyz_env.py max_path_length to 5000
            # to run the long horizons we use during data collection.
            self.env = gym.make(
                'Meta-World/MT1', max_episode_steps=5000,
                env_name=name_to_env[env_name], render_mode='rgb_array',
                camera_name=camera_name,
            )
            if config['train']['dataset'].split('_')[-1][:-1] == '2':
                self.meta_ts = 4
        else:
            self.env = gym.make(name_to_env[env_name], render_mode="rgb_array") if not self.hardware else None
            self.meta_ts = 1
        self.env_continuous = (env_to_aspace.get(env_name, 'continuous') == 'continuous')
        self.past_length = config['trans']['past_length']
        self.obj_fun = config['closed_loop'].get('policy', None)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.in_image_shape,
            model.control_size,
            config['closed_loop']['buffer_capacity'],
            device,
            config,
        )
        assert self.num_rollout_steps <= self.replay_buffer.capacity, \
            'Steps per rollout should be <= buffer_capacity!'

        # [x, y, z, gripper, x_obj, y_obj, z_obj, reward, xdot, ydot, zdot, gripper_vel]
        num_states_to_save = 12
        self.saved_state = torch.zeros(
            (self.num_epochs * self.num_rollout_steps, num_states_to_save), device='cpu'
        )

    def save(self, config_save, run_path):
        """Save model + config and any extra state tensors collected online."""
        super().save(config_save, run_path)
        if hasattr(self, 'saved_state'):
            try:
                filepath = run_path / 'training_states.pt'
                print(f'Saved state tensor to {filepath}')
                torch.save(self.saved_state, filepath)
            except Exception as e:
                print(e)
                print('Exception occured, saved state tensor to current directory')
                torch.save(self.saved_state, 'training_states.pt')
        if hasattr(self, 'costs_rollout'):
            try:
                filepath = run_path / 'costs_rollout_500.pt'
                print(f'Saved costs rollout tensor to {filepath}')
                torch.save(self.costs_rollout, filepath)
            except Exception as e:
                print(e)
                print('Exception occured, saved costs rollout tensor to current directory')
                torch.save(self.costs_rollout, 'costs_rollout.pt')

    def collect_rollouts(self, epoch):
        """Collect ``num_rollout_steps`` (image, action, next_image) transitions
        with random actions and push them into ``self.replay_buffer``.
        """
        self.model.eval()
        obs, _ = self.env.reset()
        frame_buffer = []
        act_buffer = []
        idx = 0

        while idx < self.num_rollout_steps:
            # Render current frame
            curr_img = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            if len(frame_buffer) == 0:
                frame_buffer.append(curr_img)

            # Sample and take action
            if self.env_name in meta_world_envs:
                act = np.random.uniform(
                    low=5.0 * self.env.action_space.low,
                    high=5.0 * self.env.action_space.high,
                    size=self.env.action_space.shape,
                ).astype(self.env.action_space.dtype)
            else:
                act = self.env.action_space.sample()
            env_act = act
            if not self.env_continuous:
                env_act = int(env_act.item())
            act_buffer.append(torch.from_numpy(act).to(self.device))
            for _ in range(self.meta_ts):
                next_obs, rew, done, _, _ = self.env.step(env_act)
            self.saved_state[epoch*self.num_rollout_steps + idx] = torch.as_tensor(
                [*next_obs[0:7], rew, *env_act], device='cpu'
            )

            # Slide frame history forward by one
            if len(frame_buffer) == self.past_length + self.pred_length:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            frame_buffer.append(next_image)

            # Add sample to replay buffer once we have enough context
            if len(frame_buffer) == self.past_length + self.pred_length:
                if self.env_continuous:
                    act_add = torch.stack(
                        [a for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                    )
                else:
                    act_add = act_buffer[self.past_length-1:self.past_length-1+self.pred_length].unsqueeze(-1)

                self.replay_buffer.add(
                    img=torch.stack(frame_buffer[0:self.past_length], dim=0),
                    action=act_add,
                    reward=rew,
                    next_img=torch.stack(frame_buffer[self.past_length:(self.past_length+self.pred_length)], dim=0),
                    done=done,
                )
                idx += 1
        self.model.train()

    def train(self, epoch):
        """One epoch worth of gradient updates over samples from the replay buffer."""
        total_model_loss = 0.0
        for i in range(self.num_batches):
            x, u, r, x_next, done = self.replay_buffer.sample(self.batch_size)
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)

            train_return = self.model(x, x_next, u)
            train_return['x'] = x
            train_return['x_next'] = x_next

            model_loss, loss_return = self.model_criterion(train_return, epoch)
            self.plotter.log(loss_return)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()

            # TODO: Raise exception
            if torch.isnan(model_loss):
                print("NaN loss encountered, stopping training.")
                break

            total_model_loss += model_loss.item() * x.size(0)

        avg_model_loss = total_model_loss / (self.batch_size*self.num_batches)
        return avg_model_loss

    def learn(self):
        model_loss = 0.0
        pbar = tqdm(range(self.curr_epoch, self.num_epochs), desc="Training")
        print("Initializing buffer with random rollouts...")
        warmup_rollouts = max(
            self.num_batches // self.num_rollout_steps,
            self.config['closed_loop']['buffer_capacity'] // self.num_rollout_steps,
        )
        for _ in range(warmup_rollouts):
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
            if hasattr(self, 'costs_rollout'):
                self.plotter.log({
                    'obj': self.costs_rollout[:(epoch+1)*self.num_rollout_steps]})
            if (epoch + 1) % 50 == 0:
                self.model.eval()
                if self.config['train']['eval']:
                    self.evaluate(self.config['run_path'])
                self.model.train()
            if (epoch + 1) % 100 == 0:
                if self.config['train']['save']:
                    super().save(self.config, self.config['run_path'], model_name=f'model_epoch{epoch+1}.pt')

        pbar.close()


class ClosedLoopInformativeTrainer(ClosedLoopRandomTrainer):
    """
    Closed-loop trainer that selects actions by maximizing an information-gain
    objective over CEM-sampled action sequences. See
    :meth:`rollout_info_gain_decoded_batch` for the two objective variants.
    """
    def __init__(self, dataset, model, config, device, prints=True):
        super().__init__(dataset, model, config, device, prints=prints)
        self.closed_cfg = config.get('closed_loop', {})
        self.uses_rssm = hasattr(self.model, 'rssm_step') and hasattr(self.model, 'post')
        self.epochs_warmup = config['closed_loop'].get('epochs_warmup', 3)

        # CEM params
        self.num_action_samples = self.closed_cfg.get('samples', 100)
        self.elite_frac = self.closed_cfg.get('elite_frac', 0.1)
        self.cem_iters = self.closed_cfg.get('iters', 3)
        self._init_cem_mu_sig()
        self.alpha = self.closed_cfg.get('alpha', 0.7)
        self.plan_horizon = self.closed_cfg.get('plan_horizon', self.pred_length)
        self.costs_rollout = torch.zeros(
            (self.num_epochs * self.num_rollout_steps, self.num_action_samples),
            device=self.device,
        )

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
        # self.sigma[:, -1] = self.sigma_init * 0.5   # No variance on gripper actions

    @staticmethod
    def _kl_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p):
        """Analytic KL(q || p) for independent diagonal Gaussians.

        Intentionally duplicated with :meth:`src.model.loss.RSSMLoss.kl_divergence`
        so the two callers (training loss vs. online planning objective) stay
        decoupled; the user has explicitly asked that both implementations be
        preserved as-is.
        """
        return 0.5 * (
            log_var_p - log_var_q
            + (torch.exp(log_var_q) + (mu_q - mu_p) ** 2) / torch.exp(log_var_p)
            - 1
        ).sum(dim=-1)

    def rollout_info_gain_decoded_batch(self, window, action_samples, t0_dist):
        """Evaluate the per-trajectory information-gain objective under each
        candidate action sequence in ``action_samples``.

        Two variants (selected by ``self.obj_fun``):

        - ``'maxdyn'`` -- dynamics information gain; objective J_D is the KL
          between successive predictive priors under the RSSM transition
          model. Paper Eq. for J_D:
          ``KL( p(z_{t+1} | g(h_t, z_t, u_t))  ||  q(z_t | o_t) )``.
        - otherwise (``'informative' / 'eig'``) -- pixel information gain
          objective J_P, KL between the posterior on the next *decoded*
          observation and the predictive prior:
          ``KL( q(z_{t+1} | o_hat_{t+1})  ||  p(z_{t+1} | g(h_t, z_t, u_t)) )``.
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            # TODO: Compare how learning architecture has changed, since mcar performace has decreased.
            B, T, A = action_samples.shape
            if t0_dist is None:
                mu_q, log_var_q, zs = self.model.encode_posterior(window)
            else:
                mu_q, log_var_q, zs = t0_dist
            total_kl = torch.zeros(B, device=self.device)

            h = torch.zeros(self.model.num_layers, B, self.model.deterministic_size, device=self.device)
            z = zs[:, -1] # take last timestep
            mu_q = mu_q[:, -1] # take last timestep
            log_var_q = log_var_q[:, -1] # take last timestep
            # old way, preserves past_length of zs?
            # z = zs.repeat(B, 1, 1)
            # mu_q = mu_q.repeat(B, 1, 1)
            # log_var_q = log_var_q.repeat(B, 1, 1)
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
                    # compare with t0
                    mu_q = mu_p
                    log_var_q = log_var_p
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
                    # to compare over t0, comment this out
                    mu_q = mu_post
                    log_var_q = log_var_post

                    # expected information gain - KL(q || p)
                    total_kl += self._kl_diag_gaussian(mu_post, log_var_post, mu_p, log_var_p) # .mean(dim=-1)
                    # KLD over t0 z, instead of t_prev
                    # total_kl += self._kl_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p).mean(dim=-1)
                    # total_kl += self._kl_diag_gaussian(mu_post, log_var_post, mu_q, log_var_q).mean(dim=-1)

            return total_kl / T # Average info gain per step

    def _sample_cem(self, frame_buffer, mu=None, sigma=None):
        # Initialize CEM distribution
        if mu is None:
            mu = self.init_control.clone() # (plan_horizon, action_size)
        if sigma is None:
            sigma = self.sigma.clone() # (plan_horizon, action_size)
        act_low = torch.as_tensor(self.env.action_space.low, device=self.device, dtype=torch.float32)
        act_high = torch.as_tensor(self.env.action_space.high, device=self.device, dtype=torch.float32)
        if self.evaluator.dataset_name == 'pointmaze':
            # scale action bounds for pointmaze
            act_low *= 2.5
            act_high *= 2.5
        elif self.evaluator.dataset_name == 'mountaincar':
            act_low *= 1.5
            act_high *= 1.5
        elif self.env_name in meta_world_envs:
            act_low *= 5.0
            act_high *= 5.0
        for _ in range(self.cem_iters):
            costs = torch.zeros(self.num_action_samples, device=self.device)
            # Sample action sequences from current distribution
            action_samples = torch.normal(
                mu.unsqueeze(0).repeat(self.num_action_samples, 1, 1),
                sigma.unsqueeze(0).repeat(self.num_action_samples, 1, 1)
            )
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

    def collect_rollouts(self, epoch):
        """
        Collect observations using an information-gain MPC objective and save them to replay buffer.
        """
        self.model.eval()
        obs, _ = self.env.reset()
        frame_buffer = []
        act_buffer = []
        idx = 0

        self._init_cem_mu_sig()
        mu = self.init_control.clone()
        sigma = self.sigma.clone()
        while idx < self.num_rollout_steps:
            if self.hardware:
                curr_image = self.env.process_image(self.env.render()).permute(2, 0, 1)
            else:
                curr_img = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            if len(frame_buffer) == 0:
                frame_buffer.append(curr_img)

            # Select action using random or informative policy based on epoch and buffer length
            if epoch >= self.epochs_warmup and len(frame_buffer) >= self.past_length:
                if epoch == self.epochs_warmup and len(frame_buffer) == self.past_length:
                    print(f'Switching to informative action selection using CEM over {self.num_action_samples} samples and {self.cem_iters} iterations. \n')
                tic = time.time()
                mu, costs, sigma = self._select_information_action(frame_buffer=frame_buffer, mu=mu, sigma=sigma)
                self.costs_rollout[epoch*self.num_rollout_steps + idx] = costs
                act = mu[0]
                toc = time.time()
                if idx == 0 and epoch == self.epochs_warmup:
                    print(f"CEM planning will take ~{((toc - tic)*self.num_rollout_steps/60):.3f} minutes.")
            else:
                if epoch == 0 and len(frame_buffer) == 1:
                    print('Initializing data from random actions. \n')
                act = torch.from_numpy(self.env.action_space.sample()).to(self.device)
            env_act = act.cpu().detach().numpy()
            if not self.env_continuous:
                # TODO: round to closest valid discrete action?
                env_act = torch.round(act)
                env_act = env_act.item()
            act_buffer.append(act)
            for _ in range(self.meta_ts):
                next_obs, rew, done, _, _ = self.env.step(env_act)
            self.saved_state[epoch*self.num_rollout_steps + idx] = torch.as_tensor(
                [*next_obs[0:7], rew, *env_act], device='cpu'
            )

            # Slide frame history forward by one
            if len(frame_buffer) == self.past_length + self.pred_length:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            frame_buffer.append(next_image)

            # Add sample to replay buffer once we have enough context
            if len(frame_buffer) == self.past_length + self.pred_length:
                if self.env_continuous:
                    act_add = torch.stack(
                        [a for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                    )
                else:
                    act_add = act_buffer[self.past_length-1:self.past_length-1+self.pred_length].unsqueeze(-1)

                self.replay_buffer.add(
                    img=torch.stack(frame_buffer[0:self.past_length], dim=0),
                    action=act_add,
                    reward=rew,
                    next_img=torch.stack(frame_buffer[self.past_length:(self.past_length+self.pred_length)], dim=0),
                    done=done,
                )
                idx += 1
        self.model.train()


class ClosedLoopHardwareTrainer(ClosedLoopInformativeTrainer):
    """
    Placeholder routed to by ``policy: 'hardware'`` in the yaml configs. The
    actual physical-robot rollout loop lives outside this repository; this stub
    exists so ``src.main._build_trainer`` can keep a single dispatch table.
    """
    def collect_rollouts(self, epoch):
        raise NotImplementedError(
            "ClosedLoopHardwareTrainer.collect_rollouts is provided by the "
            "private hardware repository; this stub is intentional."
        )
