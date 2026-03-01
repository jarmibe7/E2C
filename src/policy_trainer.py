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
from src.replay_buffer import RLReplayBuffer
from src.data_gen.gen_fetch import name_to_env, env_to_aspace, process_image, meta_world_envs, metaworld_cam_name
from src.trainer import BaseTrainer, ClosedLoopInformativeTrainer
from src.data_gen.gen_fetch import get_mujoco_geom_keys_index, is_robot_contact_geometry
from src.model.policy_loss import OffPolicyStochasticActorLoss

class ContactRewardActorCritic(ClosedLoopInformativeTrainer):
    """
    Train an actor-critic RL policy to complete a task, using task-agnostic affordance discovery from pixels as exploration.
    Instead of using the objective directly in the policy loss, this method adds reward for contacts.

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
        # assert self.num_rollout_steps > self.batch_size, 'Steps per rollout must be > batch_size!'
        self.obj_fun = 'informative'

        # Policy optimizer and loss (TODO)
        self.policy = policy
        self.policy.to(device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config['closed_loop']['alpha'], 
            weight_decay=config['trans']['weight_decay']
        )
        self.actor_criterion = OffPolicyStochasticActorLoss()
        self.critic_criterion = torch.nn.functional.mse_loss

        # TODO: Add discount factor hparam
        self.discount = 0.95

        # Overwrite super() replay buffer
        self.replay_buffer = RLReplayBuffer(
            self.in_image_shape, 
            model.control_size, 
            config['closed_loop']['buffer_capacity'],
            device,
            config
        )
        assert self.num_rollout_steps <= self.replay_buffer.capacity, 'Steps per rollout should be <= buffer_capacity!'
        self.curr_epoch = 0

    def take_action(self, frame_buffer, epoch, mu, sigma, eval=False):
        policy_return = {
            'dist': torch.tensor(torch.nan),
            'log_probs': torch.full((1, ), torch.nan),
            'values': torch.full((1, ), torch.nan)
        }
        if epoch < self.epochs_warmup:
            # Warmup, random actions
            act = torch.from_numpy(self.env.action_space.sample()).to(self.device)
        elif epoch < self.pre_explore*self.num_epochs:
            # First epochs are always explore
            # Use CEM distribution to calculate action probs
            mu, costs, sigma = self._select_information_action(frame_buffer=frame_buffer, mu=mu, sigma=sigma)
            policy_return['dist'] = torch.distributions.Normal(mu, sigma)
            action_full = policy_return['dist'].sample()
            act = action_full[0]
            policy_return['log_probs'] = policy_return['dist'].log_prob(action_full)[0].sum(-1).unsqueeze(-1)
        else:
            # Annealed Greedy explore/exploit
            warmup = max(self.epochs_warmup, self.pre_explore*self.num_epochs)
            epsilon = 1.0 - (epoch - warmup) / (self.num_epochs - warmup)
            epsilon = max(0.0, min(epsilon, 1.0))

            u = random.random()
            if u < epsilon:
                mu, costs, sigma = self._select_information_action(frame_buffer=frame_buffer, mu=mu, sigma=sigma)
                policy_return['dist'] = torch.distributions.Normal(mu, sigma)
                action_full = policy_return['dist'].sample()
                act = action_full[0]
                policy_return['log_probs'] = policy_return['dist'].log_prob(action_full)[0].sum(-1).unsqueeze(-1)
            else:
                policy_return = self.policy(frame_buffer[-1].unsqueeze(0).to(self.device))
                act = policy_return['dist'].sample().squeeze(0)
                policy_return['log_probs'] = policy_return['dist'].log_prob(act).sum(-1)   # Sum over action dims
                policy_return['values'] = torch.full((1, ), torch.nan)
        return act, mu, policy_return
            
    def calc_reward(self, raw_reward, contact):
        # Metaworld env reward is clipped at 10
        # https://www.emergentmind.com/papers/2505.11289
        return raw_reward + contact


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

        mj_data = self.env.unwrapped.data
        robot_geom, obj_geom = get_mujoco_geom_keys_index(self.env_name)

        self._init_cem_mu_sig()
        mu = self.init_control.clone()
        sigma = self.sigma.clone()
        cum_reward = 0.0
        while idx < self.num_rollout_steps:
            # Render current frame
            curr_img = process_image(self.env.render(), self.evaluator.dataset_name).permute(2, 0, 1)
            if len(frame_buffer) == 0:
                frame_buffer.append(curr_img)

            # Sample and take action
            # act = self.policy(curr_img.unsqueeze(0).to(self.device)).flatten()
            # act = torch.from_numpy(self.env.action_space.sample()).to(self.device)
            act, mu, policy_return = self.take_action(frame_buffer, epoch, mu, sigma)
            next_obs, raw_rew, done, _, _ = self.env.step(act.cpu().detach().numpy())
            act_buffer.append((act, policy_return['log_probs'], policy_return['values']))

            # Calculate reward based on contact
            contact = int(is_robot_contact_geometry(mj_data, robot_geom, obj_geom))
            rew = self.calc_reward(raw_rew, contact)
            cum_reward += rew

            # If done reset env, otherwise add sample to dataset
            if done:
                obs, _ = self.env.reset()
                done = False
                frame_buffer = []
                act_buffer = []
                continue
            else:
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
                            [a[0] for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                        )
                        action_prob_add = torch.stack(
                            [a[1] for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                        )
                        values_add = torch.stack(
                            [a[2] for a in act_buffer[self.past_length-1:self.past_length-1+self.pred_length]]
                        )
                    else:
                        act_add = act_buffer[self.past_length-1:self.past_length-1+self.pred_length].unsqueeze(-1)

                    self.replay_buffer.add(
                        img = torch.stack(frame_buffer[0:self.past_length], dim=0),
                        action = act_add,
                        reward = rew,
                        next_img = torch.stack(frame_buffer[self.past_length:(self.past_length+self.pred_length)], dim=0),
                        done = done,
                        action_probs=action_prob_add,
                        value=values_add
                    )
                    idx += 1

        self.plotter.log_value("Avg Train Rollout Rew", cum_reward / self.num_rollout_steps)
        self.model.train()
        self.policy.train()

    def train(self, epoch):
        """
        Gradient update for model and asychronous policy
        """
        # World model and policy update
        total_model_loss, total_actor_loss, total_critic_loss = 0.0, 0.0, 0.0
        for i in range(self.num_batches):
            # Unload batch
            sample  = self.replay_buffer.sample(self.batch_size)
            x, x_next, u = sample.x.to(self.device), sample.x_next.to(self.device), sample.u.to(self.device)

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
            current_frame = x[:, -1]  # Take only current frame
            policy_return = self.policy(current_frame)

            # Compute targetes, critic loss
            values = self.policy.get_value(current_frame)
            with torch.no_grad():
                next_values = self.policy.get_value(x_next[:, 0])
                targets = sample.rewards + (self.discount*next_values)*(1 - (sample.dones))
            critic_loss = self.critic_criterion(values, targets)

            # TODO: Raise exception
            if torch.isnan(critic_loss):
                print("NaN critic loss encountered!")
                break

            # Compute policy loss and backprop
            log_probs = policy_return['dist'].log_prob(u[:, 0]).sum(-1)
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            actor_loss = self.actor_criterion(log_probs, advantages)
            
            # TODO: Raise exception
            if torch.isnan(actor_loss):
                print("NaN actor loss encountered!")
                break

            self.policy_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.policy_optimizer.step()

            total_actor_loss += actor_loss.item() * x.size(0)
            total_critic_loss += critic_loss.item() * x.size(0)

        # Compute average model loss
        avg_model_loss = total_model_loss / (self.batch_size*self.num_batches)

        avg_critic_loss = total_critic_loss / (self.num_batches * self.batch_size)
        self.plotter.log_value('Avg Critic Loss', min(avg_critic_loss, 100.0))
        self.plotter.log_value('Avg Actor Loss', min(100, max(actor_loss, -100.0)))
        avg_actor_loss = total_actor_loss / (self.num_batches * self.batch_size)
        return avg_model_loss, avg_critic_loss, avg_actor_loss

    def learn(self):
        model_loss, critic_loss, actor_loss = 0.0, 0.0, 0.0
        pbar = tqdm(range(self.curr_epoch, self.num_epochs), desc="Training")
        for _ in range(max(self.num_batches // self.num_rollout_steps, self.config['closed_loop']['buffer_capacity'] // self.num_rollout_steps)):
            self.collect_rollouts(-1)
        start_epoch = self.curr_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.curr_epoch = epoch
            model_loss, critic_loss, actor_loss = self.train(epoch)

            pbar.set_postfix({
                'Epoch': epoch+1,
                'Model Loss': f"{model_loss:.4f}",
                'Critic Loss': f"{critic_loss:.4f}",
                'Actor Loss': f"{actor_loss:.4f}"})
            pbar.update(1)

            self.collect_rollouts(epoch)
            if (epoch + 1) % 50 == 0:
                # Show model video every 50 epochs
                self.model.eval()
                if self.config['train']['eval']: self.evaluate(self.config['run_path'])
                self.model.train()
            if (epoch + 1) % 100 == 0:
                if self.config['train']['save']: super(BaseTrainer, self).save(self.config, self.config['run_path'], model_name=f'model_epoch{epoch+1}.pt')

        pbar.close()
