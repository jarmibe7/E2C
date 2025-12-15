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
from pathlib import Path
from tqdm import tqdm
from pyvirtualdisplay import Display

from src.e2c import E2CLoss, ConvE2C
from src.eval import Plotter, Evaluator
from src.replay_buffer import ReplayBuffer
from src.data_gen.gen_gym import name_to_env, env_to_aspace, process_image

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
        self.out_image_shape = config['vae']['out_image_shape']
        self.device = device
        self.config = config
        if config['closed_loop']['closed_loop']:
            print(f"Model will be trained on {self.batch_size*self.num_epochs*config['closed_loop']['num_batches']} samples in {config['closed_loop']['num_batches']*self.num_epochs} gradient updates\n")
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
        if isinstance(model, ConvE2C):
            self.model_criterion = E2CLoss(config['train']['num_epochs'], config['loss'])

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
        self.evaluator.eval(run_path)

    def save(self, config_save, run_path):
        """
        Save model, policy, and config to run directory
        """
        self.plotter.save(run_path)

        # Save model
        model_name = 'model.pt'
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
                yaml.dump(config_save, f, default_flow_style=False) # Save original config so model can be loaded later
        except Exception as e:
            print(e)
            print('Exception occurred, saved config to current directory')
            with open(yaml_name, 'w') as f:
                yaml.dump(config_save, f, default_flow_style=False)

class WorldModelPretrainer(BaseTrainer):
    """
    Trainer for pretraining world model without active learning

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
                x = x.reshape(x.shape[0], -1, *self.out_image_shape[1:])    # Stack obs history in channel dim
                x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)

                # Forward pass
                train_return = self.model(x, x_next, u)
                train_return['x'] = x
                train_return['x_next'] = x_next

                # Compute loss and backprop
                loss, recon, recon_next, kld, kld_trans = self.model_criterion(train_return, epoch)
                self.plotter.log(recon, recon_next, kld, kld_trans)
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

class ClosedLoopPolicyTrainer(BaseTrainer):
    """
    Train a world model in closed loop, where actions are chosen based an asychronous
    policy that is being trained concurrently.

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
            self.out_image_shape, 
            model.control_size, 
            config['closed_loop']['buffer_capacity'],
            device,
            config
        )
        assert self.num_rollout_steps <= self.replay_buffer.capacity, 'Steps per rollout should be <= buffer_capacity!'

    def collect_rollouts(self):
        """
        Collect observations and save them to replay buffer
        """
        # Initialize env and buffers
        obs, _ = self.env.reset()
        frame_buffer = []
        act_buffer = []
        idx = 0
        while idx < self.num_rollout_steps:
            # Render current frame
            curr_img = process_image(self.env.render()).squeeze(0).permute(2, 0, 1)
            if len(frame_buffer) == 0:
                frame_buffer.append(curr_img)

            # Sample and take action
            # act = self.policy(curr_img.unsqueeze(0).to(self.device)).flatten()
            act = torch.from_numpy(self.env.action_space.sample()).to(self.device)
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
                next_image = process_image(self.env.render()).squeeze(0).permute(2, 0, 1)
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
        # TODO: Where does policy update go, should be latent rollouts right?
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
            x = x.reshape(x.shape[0], -1, *self.out_image_shape[1:])    # Stack obs history in channel dim
            x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)

            # Forward pass
            train_return = self.model(x, x_next, u)
            train_return['x'] = x
            train_return['x_next'] = x_next

            # Model loss and backprop
            model_loss, recon, recon_next, kld, kld_trans = self.model_criterion(train_return, epoch)
            self.plotter.log(recon, recon_next, kld, kld_trans)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()

            # TODO: Raise exception
            if torch.isnan(model_loss):
                print("NaN loss encountered, stopping training.")
                break

            total_model_loss += model_loss.item() * x.size(0)   # Aggregate total epoch loss

            # Forward pass through policy
            C = self.out_image_shape[0] // self.past_length
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
        #     x = x.to(self.device).reshape(x.shape[0], -1, *self.out_image_shape[1:])

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
        pbar = tqdm(range(self.num_epochs), desc="Training")
        for epoch in range(self.num_epochs):
            self.collect_rollouts()
            model_loss, policy_loss = self.train(epoch)

            pbar.set_postfix({
                'Epoch': epoch+1,
                'Model Loss': f"{model_loss:.4f}",
                'Policy Loss': f"{policy_loss:.4f}"})
            pbar.update(1)
            
        pbar.close()

class ClosedLoopUncertaintyTrainer(BaseTrainer):
    """
    Train a world model in closed loop, where actions are chosen based on prediction variance.

    Args:
        dataset: Torch dataset object, from which test set is used during evaluation.
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
    """
    def __init__(self, dataset, model, config, device):
        super().__init__(dataset, model, config, device)
        self.num_rollout_steps = config['closed_loop']['num_rollout_steps']

    def collect_rollouts(self):
        """
        Collect observations and save them to replay buffer
        """
        for i in range(self.num_rollout_steps):
            pass

    def train(self):
        """
        Gradient update for model and asychronous policy
        """
        pass

    def learn(self):
        for i in range(self.num_epochs):
            self.collect_rollouts()
            self.train()