import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import gymnasium as gym
import gymnasium_robotics
import random
gym.register_envs(gymnasium_robotics)
from pathlib import Path
from tqdm import tqdm
from pyvirtualdisplay import Display
import time
import os
os.environ['MUJOCO_GL'] = 'egl'
from src.utils import set_seed, format_time

from src.model.loss import E2CLoss, UncertaintyE2CLoss, RSSMLoss
from src.eval import Plotter, Evaluator
from src.replay_buffer import ReplayBuffer
from src.data_gen.gen_fetch import name_to_env, env_to_aspace, process_image, meta_world_envs, metaworld_cam_name
from src.trainer import ClosedLoopInformativeTrainer

class PixExpRL(ClosedLoopInformativeTrainer):
    """
    Train an RL policy to complete a task, using task-agnostic affordance discovery from pixels as exploration.
    Args:
        dataset: Torch dataset object, from which test set is used during evaluation.
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
        policy: Asynchronous policy
    """
    def __init__(self, dataset, model, config, device, policy):
        super().__init__(dataset, model, config, device, policy)
        self.num_rollout_steps = config['closed_loop']['num_rollout_steps']
        self.num_batches = config['closed_loop']['num_batches']
        self.pre_explore = config['closed_loop'].get('pre_explore', 0.1)
        assert self.num_rollout_steps > self.batch_size, 'Steps per rollout must be > batch_size!'

        # Initialize environment
        disp = Display(visible=0, size=(480, 480))
        disp.start()
        env_name = config['train']['dataset'].split('_')[0]
        self.env = gym.make(name_to_env[env_name], render_mode="rgb_array")
        self.env_continuous = (env_to_aspace[env_name] == 'continuous')
        self.past_length = config['trans']['past_length']

        # Policy optimizer and loss (TODO)
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), 
            lr=config['closed_loop']['alpha'], 
            weight_decay=config['closed_loop']['weight_decay']
        )
        self.policy_criterion = torch.nn.functional.mse_loss # TODO: Placeholder loss

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.in_image_shape, 
            model.control_size, 
            config['closed_loop']['buffer_capacity'],
            device,
            config
        )
        assert self.num_rollout_steps <= self.replay_buffer.capacity, 'Steps per rollout should be <= buffer_capacity!'
        self.curr_epoch = 0

    def take_action(self, frame_buffer, epoch, mu, sigma, eval=False):
        if epoch < self.epochs_warmup:
            # Warmup, random actions
            return torch.from_numpy(self.env.action_space.sample()).to(self.device), mu
        elif epoch < self.pre_explore*self.num_epochs:
            # First epochs are always explore
            mu, costs, sigma = self._select_information_action(frame_buffer=frame_buffer, mu=mu, sigma=sigma)
            act = mu[0]
            return act, mu
        else:
            # Annealed Greedy explore/exploit
            warmup = self.pre_explore*self.num_epochs
            epsilon = 1.0 - (epoch - warmup) / (self.num_epochs - warmup)
            epsilon = torch.clip(epsilon, 0.0, 1.0)

            # Mak uniform and assign
            u = random.random()

            if u < epsilon:
                mu, costs, sigma = self._select_information_action(frame_buffer=frame_buffer, mu=mu, sigma=sigma)
                act = mu[0]
                return act, mu
            else:
                act = self.policy(frame_buffer)
                return act, mu



    def collect_rollouts(self, epoch):
        """
        Collect observations and save them to replay buffer
        """
        # Initialize env and buffers
        self.model.eval()
        self.policy.eval()
        obs, _ = self.env.reset()
        frame_buffer = []
        act_buffer = []
        idx = 0

        self._init_cem_mu_sig()
        mu = self.init_control.clone()
        sigma = self.sigma.clone()
        while idx < self.num_rollout_steps:
            # Render current frame
            curr_img = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            if len(frame_buffer) == 0:
                frame_buffer.append(curr_img)

            # Sample and take action
            # act = self.policy(curr_img.unsqueeze(0).to(self.device)).flatten()
            # act = torch.from_numpy(self.env.action_space.sample()).to(self.device)
            act = self.take_action(curr_img.unsqueeze(0).to(self.device))
            act_buffer.append(act)
            next_obs, rew, done, _, _ = self.env.step(act.cpu().detach().numpy())

            # If done reset env, otherwise add sample to dataset
            if done:
                obs, _ = self.env.reset()
                done = False
                frame_buffer = []
                act_buffer = []
                continue
            else:
                # Slide frame obs history frame buffer to next image
                if len(frame_buffer) == self.past_length + 1:
                    frame_buffer.pop(0)
                    act_buffer.pop(0)
                next_image = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
                frame_buffer.append(next_image)

                # Add sample to replay buffer
                if len(frame_buffer) == self.past_length + 1:
                    self.replay_buffer.add(
                        img = torch.stack(frame_buffer[0:self.past_length], dim=0),
                        action = act_buffer[-1],
                        reward = rew,
                        next_img = frame_buffer[self.past_length],
                        done = done
                    )
                    idx += 1

    def train(self, epoch):
        """
        Gradient update for model and asychronous policy
        """
        # Sample batches
        batches = []
        for i in range(self.num_batches):
            # Sampling
            batches.append(self.replay_buffer.sample(self.batch_size))

        # World model and policy update
        total_model_loss, total_policy_loss = 0.0, 0.0
        for i, batch in enumerate(batches):
            # Unload batch
            x, u, r, x_next, done  = batch
            x, x_next, u = x.to(self.device), x_next.to(self.device), u.to(self.device)
            x = x.reshape(x.shape[0], -1, *self.in_image_shape[1:])    # Stack obs history in channel dim
            x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)

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

            # Forward pass through policy
            C = self.out_image_shape[0]
            current_frame = x[:, -C:, :, :]  # Take only current frame
            pred_action = self.policy(current_frame)

            # Policy loss and backprop
            policy_loss = self.policy_criterion(u, pred_action)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # TODO: Raise exception
            if torch.isnan(policy_loss):
                print("NaN loss encountered, stopping training.")
                break

            total_policy_loss += policy_loss.item() * x.size(0)

        # Compute average model loss
        avg_model_loss = total_model_loss / (self.batch_size*self.num_batches)

        # # Policy update on same samples
        # total_policy_loss = 0.0
        # for i, batch in enumerate(batches):
        #     x, u, r, x_next, done = batch
        #     x = x.to(self.device).reshape(x.shape[0], -1, *self.in_image_shape[1:])

        #     # Forward pass through policy
        #     pred_action = self.policy(x)  

        #     # Example policy loss: maximize expected reward in replay buffer
        #     # You can replace this with SAC-style loss or any other RL objective
        #     policy_loss = -torch.mean(pred_action)  

        #     self.policy_optimizer.zero_grad()
        #     policy_loss.backward()
        #     self.policy_optimizer.step()

        #     total_policy_loss += policy_loss.item() * x.size(0)

        avg_policy_loss = total_policy_loss / (self.num_batches * self.batch_size)
        return avg_model_loss, avg_policy_loss

    def learn(self):
        model_loss, policy_loss = 0.0, 0.0
        pbar = tqdm(range(self.curr_epoch, self.num_epochs), desc="Training")
        for _ in range(max(self.num_batches // self.num_rollout_steps, self.config['closed_loop']['buffer_capacity'] // self.num_rollout_steps)):
            self.collect_rollouts(-1)
        start_epoch = self.curr_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.curr_epoch = epoch
            model_loss, policy_loss = self.train(epoch)

            pbar.set_postfix({
                'Epoch': epoch+1,
                'Model Loss': f"{model_loss:.4f}",
                'Policy Loss': f"{policy_loss:.4f}"})
            pbar.update(1)

            self.collect_rollouts(epoch)
            if (epoch + 1) % 50 == 0:
                # Show model video every 50 epochs
                self.model.eval()
                if self.config['train']['eval']: self.evaluate(self.config['run_path'])
                self.model.train()
            if (epoch + 1) % 100 == 0:
                if self.config['train']['save']: super().save(self.config, self.config['run_path'], model_name=f'model_epoch{epoch+1}.pt')

            
        pbar.close()
