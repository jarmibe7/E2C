"""
Contains Trainer classes for open loop and closed loop E2C/policy training
and active learning.

Authors: Jared Berry
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from pathlib import Path
from tqdm import tqdm
from pyvirtualdisplay import Display
import time
import os
os.environ['MUJOCO_GL'] = 'egl'

from src.model.loss import E2CLoss, UncertaintyE2CLoss, RSSMLoss
from src.eval import Plotter, Evaluator
from src.replay_buffer import ReplayBuffer
from src.data_gen.gen_fetch import name_to_env, env_to_aspace, process_image, meta_world_envs, metaworld_cam_name

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"

class BaseTrainer():
    """
    Base Trainer class for training world models/policies

    Args:
        dataset: Torch dataset object that can be split into train/test sets
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
        policy (optional): A policy class instance for closed-loop training
    """
    def __init__(self, dataset, model, config, device, policy=None):
        # Split into training and test sets
        train_size = int(len(dataset) * config['train']['train_ratio'])
        test_size = len(dataset) - train_size
        self.dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Save training params
        self.num_epochs = config['train']['num_epochs']
        self.batch_size = config['train']['batch_size']
        self.in_image_shape = config['vae']['in_image_shape']
        self.out_image_shape = config['vae']['out_image_shape']
        self.device = device
        self.config = config
        if config['closed_loop']['closed_loop']:
            # Compute training stats
            self.num_updates = self.num_epochs * config['closed_loop']['num_batches']
            self.num_train_inters = self.batch_size * self.num_updates
            self.num_env_inters = self.num_epochs * config['closed_loop']['num_rollout_steps']
            # num_batches * num_epochs * batch_size - # interactions agents trained on
            # num_rollout_steps * num_epochs - # interactions collected
            # num_rollout_steps * num_epochs * cem_iters * num_action_samples - # iteractions collected and imagined

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

        # Save model and policy
        self.model = model
        model.to(device)
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config['trans']['alpha'], weight_decay=config['trans']['weight_decay'])
        if policy is not None:
            self.policy = policy
            policy.to(device)
        else:
            self.policy = None

        # Create loss criterion
        loss_type = config['loss'].get('loss_type', None)
        if loss_type == 'uncertainty':
            self.model_criterion = UncertaintyE2CLoss(config['train']['num_epochs'], config['loss'])
        elif loss_type == 'rssm':
            self.model_criterion = RSSMLoss(config['train']['num_epochs'], config['loss'])
        elif loss_type == 'mse':
            self.model_criterion = E2CLoss(config['train']['num_epochs'], config['loss'])
        else:
            raise NotImplementedError(f'Loss type "{loss_type}" not supported!')

        # Create visualizer and evaluator
        self.plotter = Plotter(config['train']['render'], config['train']['plot_freq'])
        if config['train']['eval']:
            self.evaluator = Evaluator(
                self.model, 
                test_dataset,
                batch_size=config['train']['batch_size'], 
                device=config['train']['device'],
                dataset_name=config['train']['dataset']
            )
        self.curr_epoch = 0
    
    def collect_rollouts(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def learn(self, *args, **kwargs):
        pass

    def evaluate(self, run_path):
        """
        Evaluate model and output figures to run directory
        """
        print('\n*** EVAL ***\n')
        self.model.eval()
        # self.evaluator.eval(run_path)
        self.evaluator.visualize_planner(self, run_path, max_steps=50, closed_loop=True)

    def save(self, config_save, run_path, model_name='model.pt'):
        """
        Save model, policy, and config to run directory
        """
        self.plotter.save(run_path)

        # Save model
        try:
            filepath = run_path / model_name
            print(f'Saved model to {filepath}')
            torch.save(self.model.state_dict(), filepath)
        except Exception as e:
            print(e)
            print('Exception occured, saved model to current directory')
            torch.save(self.model.state_dict(), model_name)

        # Save policy
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

        # Save config dictionary
        try:
            yaml_name = 'config.yaml'
            yaml_path = run_path / yaml_name
            print(f'Saved config to {yaml_path}')
            with open(yaml_path, 'w') as f:
                yaml.dump(config_save, f, sort_keys=False, default_flow_style=False) # Save original config so model can be loaded later
        except Exception as e:
            print(e)
            print('Exception occurred, saved config to current directory')
            with open(yaml_name, 'w') as f:
                yaml.dump(config_save, f, sort_keys=False, default_flow_style=False)

class E2CPretrainer(BaseTrainer):
    """
    Trainer for pretraining E2C without active learning

    Args:
        dataset: Torch dataset object that can be split into train/test sets
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)

    def train(self):
        """
        Train an E2C model
        """
        # Create Dataset and DataLoader to handle batching of training data
        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        print('\nBeginning Training:')
        epoch_loss = 0.0
        pbar = tqdm(range(self.num_epochs), desc="Training")
        for epoch in pbar:
            total_loss = 0.0

            for x, x_next, u in train_loader:
                # Send training data to GPU
                x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
                x = x.reshape(x.shape[0], -1, *self.in_image_shape[1:])    # Stack obs history in channel dim
                x_next = x_next[:, 0]       # pred_length == 1 only for E2C
                u = u[:, 0]
                x_next_enc = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)  # Need to stack for encoding

                # Forward pass
                train_return = self.model(x, x_next_enc, u)
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

                total_loss += loss.item() * x.size(0)   # Aggregate total epoch loss

            # Display average loss for the epoch
            epoch_loss = total_loss / len(train_loader.dataset)
            pbar.set_postfix({
                'Epoch': epoch+1,
                'Epoch Loss': f"{epoch_loss:.4f}"})

        pbar.close()

    def learn(self):
        self.train()

class RSSMPretrainer(BaseTrainer):
    """
    Trainer for pretraining world model with RSSM without active learning

    Args:
        dataset: Torch dataset object that can be split into train/test sets
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)

    def train(self):
        # Create Dataset and DataLoader to handle batching of training data
        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        print('\nBeginning Training:')
        epoch_loss = 0.0
        pbar = tqdm(range(self.num_epochs), desc="Training")
        for epoch in pbar:
            total_loss = 0.0

            for x, x_next, u in train_loader:
                # Send training data to GPU
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

                total_loss += loss.item() * x.size(0)   # Aggregate total epoch loss

            # Display average loss for the epoch
            epoch_loss = total_loss / len(train_loader.dataset)
            pbar.set_postfix({
                'Epoch': epoch+1,
                'Epoch Loss': f"{epoch_loss:.4f}"})

        pbar.close()

    def learn(self):
        self.train()

class ClosedLoopRandomTrainer(BaseTrainer):
    """
    Train a world model in closed loop, where actions are chosen randomly.
    Args:
        dataset: Torch dataset object, from which test set is used during evaluation.
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)
        self.num_rollout_steps = config['closed_loop']['num_rollout_steps']
        self.num_batches = config['closed_loop']['num_batches']
        self.pred_length = config['trans'].get('pred_length', 3)   # How long to predict ahead in network
        self.plan_horizon = config['closed_loop'].get('plan_horizon', self.pred_length)   # How long to imagine rollouts
        self.hardware = config['closed_loop'].get('hardware', False)
        self.meta_ts = 1
        # assert self.num_rollout_steps > self.batch_size, 'Steps per rollout must be > batch_size!'
        assert self.batch_size < config['closed_loop']['buffer_capacity'], 'Batch size must be < buffer capacity!'
        if self.num_rollout_steps <= self.num_batches * self.batch_size:
            print(
                f"Warning: num_rollout_steps ({self.num_rollout_steps}) > "
                f"num_batches * batch_size ({self.num_batches * self.batch_size}).\n"
                f"This may lead to inefficient training, as the probability of "
                f"'double counting' samples in a single epoch is high."
            )

        # Initialize environment
        # disp = Display(visible=0, size=(480, 480))
        # disp.start()
        env_name = config['train']['dataset'].split('_')[0]
        self.env_name = env_name
        if env_name in meta_world_envs:
            self.env = gym.make('Meta-World/MT1', env_name=name_to_env[env_name], render_mode='rgb_array', camera_name='corner', max_episode_steps=2000)
            print("made steps 2000?")
            if config['train']['dataset'].split('_')[1][:-1] == '2':
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
            config
        )
        assert self.num_rollout_steps <= self.replay_buffer.capacity, 'Steps per rollout should be <= buffer_capacity!'

        num_states_to_save = 12 # [x, y, z, gripper, x_obj, y_obj, z_obj, reward, xdot, ydot, zdot, gripper_vel]
        # self.saved_state = torch.zeros((self.num_epochs * self.num_rollout_steps, num_states_to_save), device=self.device)
        self.saved_state = torch.zeros((self.num_epochs * self.num_rollout_steps, num_states_to_save), device='cpu')

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

    def collect_rollouts(self, epoch):
        """
        Collect observations and save them to replay buffer
        """
        # Initialize env and buffers
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
                            size=self.env.action_space.shape).astype(self.env.action_space.dtype)
            else:
                act = self.env.action_space.sample()
            env_act = act
            if not self.env_continuous:
                env_act = int(env_act.item())
            act_buffer.append(torch.from_numpy(act).to(self.device))
            for _ in range(self.meta_ts):
                next_obs, rew, done, _, _ = self.env.step(env_act)
            self.saved_state[idx] = torch.as_tensor([*next_obs[0:7], rew, *env_act], device='cpu')

            # If done reset env, otherwise add sample to dataset
            # if done:
            #     obs, _ = self.env.reset()
            #     done = False
            #     frame_buffer = []
            #     act_buffer = []
            #     continue
            # else:
            # Slide frame obs history frame buffer to next image
            if len(frame_buffer) == self.past_length + self.pred_length:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            frame_buffer.append(next_image)

            # Add sample to replay buffer
            if len(frame_buffer) == self.past_length + self.pred_length:
                # Compute action and img windows
                if self.env_continuous:
                    act_add = torch.stack(
                        [a for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                    )
                else:
                    act_add = act_buffer[self.past_length-1:self.past_length-1+self.pred_length].unsqueeze(-1)

                self.replay_buffer.add(
                    img = torch.stack(frame_buffer[0:self.past_length], dim=0),
                    action = act_add,
                    reward = rew,
                    next_img = torch.stack(frame_buffer[self.past_length:(self.past_length+self.pred_length)], dim=0),
                    done = done
                )
                idx += 1
        self.model.train()

    def train(self, epoch):
        """
        Gradient update for model and asychronous policy
        """
        # World model gradient update
        total_model_loss, total_policy_loss = 0.0, 0.0
        for i in range(self.num_batches):
            # Unload batch
            x, u, r, x_next, done  = self.replay_buffer.sample(self.batch_size)
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)

            # Forward pass
            train_return = self.model(x, x_next, u)
            train_return['x'] = x
            train_return['x_next'] = x_next

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

            total_model_loss += model_loss.item() * x.size(0)   # Aggregate total epoch loss

        # Compute average model loss
        avg_model_loss = total_model_loss / (self.batch_size*self.num_batches)

        return avg_model_loss
    
    def learn(self):
        model_loss = 0.0
        pbar = tqdm(range(self.num_epochs), desc="Training")
        print("Initializing buffer with random rollouts...")
        for _ in range(max(self.num_batches // self.num_rollout_steps, self.config['closed_loop']['buffer_capacity'] // self.num_rollout_steps)):
            self.collect_rollouts(-1)
        for epoch in range(self.num_epochs):
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
                if self.config['train']['save']: super().save(self.config['run_path'], model_name=f'model_epoch{epoch+1}.pt')
                self.model.train()
            
        pbar.close()


class ClosedLoopInformativeTrainer(ClosedLoopRandomTrainer):
    """
    Closed-loop trainer that selects actions by maximizing expected information gain
    (posterior vs prior KL) over candidate action sequences.
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)
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
            self.init_control = torch.stack([torch.from_numpy(self.env.action_space.sample()).to(self.device) for _ in range(self.plan_horizon)], dim=0)
            if self.env_name in meta_world_envs:
                # Scale to reasonable range for Meta-World envs
                self.init_control *= 2.5
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
                    total_kl += self._kl_diag_gaussian(mu_p, log_var_p, mu_q, log_var_q).mean().item()
                    mu_q = mu_p
                    log_var_q = log_var_p
                    z_prior = self.model.reparameterize(mu_q, log_var_q)
                    # compare with previous --> would expect more dramatic results
                    # z = self.model.reparameterize(mu_q, log_var_q)
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
                    total_kl += self._kl_diag_gaussian(mu_post, log_var_post, mu_p, log_var_p).mean(dim=-1)
                    # TODO: try doing KLD over t0 z, instead of t_prev
                    # total_kl += self._kl_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p).mean(dim=-1)
                    # total_kl += self._kl_diag_gaussian(mu_post, log_var_post, mu_q, log_var_q).mean(dim=-1)
                    
                    # to compare over t0, comment this out
                    mu_q = mu_post
                    log_var_q = log_var_post

                    # update belief <-- happens inherently with model.encode_posterior
                    # zs = self.model.reparameterize(mu_post, log_var_post)

            return total_kl # / T # Average info gain per step
    
    def _sample_cem(self, frame_buffer):
        # Initialize CEM distribution
        mu = self.init_control  # (plan_horizon, action_size)
        sigma = self.sigma      # (plan_horizon, action_size)
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
            new_sigma = torch.stack([torch.std(torch.stack([seq[t] for seq in elite_seqs], dim=0), dim=0) for t in range(self.plan_horizon)], dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            sigma = torch.nan_to_num(self.alpha * sigma + (1 - self.alpha) * new_sigma, nan=self.sigma_min)
            sigma = torch.clamp(sigma, min=self.sigma_min)

        return mu, costs
    
    def _select_information_action(self, frame_buffer):
        mu, costs = self._sample_cem(frame_buffer)
        return mu[0].clone(), costs[0].clone()

    def collect_rollouts(self, epoch):
        """
        Collect observations using an information-gain MPC objective and save them to replay buffer.
        """
        self.model.eval()
        obs, _ = self.env.reset()
        frame_buffer = []
        act_buffer = []
        idx = 0
        
        while idx < self.num_rollout_steps:
            self._init_cem_mu_sig()
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
                act, cost = self._select_information_action(frame_buffer)
                toc = time.time()
                if idx == 0 and epoch == self.epochs_warmup:
                    print(f"CEM planning will take ~{((toc - tic)*self.num_rollout_steps/60):.3f} minutes.")
            else:
                if epoch == 0 and len(frame_buffer) == 1: 
                    print(f'Initializing data from random actions. \n')
                act = torch.from_numpy(self.env.action_space.sample()).to(self.device)
            env_act = act.cpu().detach().numpy()
            if not self.env_continuous:
                # TODO: round to closest valid discrete action?
                env_act = torch.round(act)
                env_act = env_act.item()
            act_buffer.append(act)
            for _ in range(self.meta_ts):
                next_obs, rew, done, _, _ = self.env.step(env_act)
            self.saved_state[idx] = torch.as_tensor([*next_obs[0:7], rew, *env_act], device='cpu')

            # all tasks are continuous now, no need to reset
            # if done:
            #     obs, _ = self.env.reset()
            #     done = False
            #     frame_buffer = []
            #     act_buffer = []
            #     continue
            # else:
            # Slide frame obs history frame buffer to next image
            if len(frame_buffer) == self.past_length + self.pred_length:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            frame_buffer.append(next_image)

            # Add sample to replay buffer
            if len(frame_buffer) == self.past_length + self.pred_length:
                # Compute action and img windows
                if self.env_continuous:
                    act_add = torch.stack(
                        [a for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                    )
                else:
                    act_add = act_buffer[self.past_length-1:self.past_length-1+self.pred_length].unsqueeze(-1)

                self.replay_buffer.add(
                    img = torch.stack(frame_buffer[0:self.past_length], dim=0),
                    action = act_add,
                    reward = rew,
                    next_img = torch.stack(frame_buffer[self.past_length:(self.past_length+self.pred_length)], dim=0),
                    done = done
                )
                idx += 1
        self.model.train()

class ClosedLoopHardwareTrainer(ClosedLoopInformativeTrainer):
    """
    Closed-loop trainer for real-world hardware experiments that selects actions
    by maximizing expected information gain over candidate action sequences.
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)
    
    def reward_planner(self, frame_buffer):
        pass
    
    def collect_rollouts(self, epoch):
        """
        Collect observations and save them to replay buffer
        """
        # TODO: do some version of this: 
        # self.saved_state[idx] = torch.as_tensor([*next_obs[0:7], rew, *env_act], device='cpu')
        pass
