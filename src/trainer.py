"""
Contains Trainer classes for open loop and closed loop E2C/policy training
and active learning.

Authors: Jared Berry
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.e2c import E2CLoss, ConvE2C
from src.eval import Plotter, Evaluator

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
        print(f"Train size: {train_size}        Test size: {test_size}\n")

        # Save training params
        self.num_epochs = config['train']['num_epochs']
        self.batch_size = config['train']['batch_size']
        self.out_image_shape = config['vae']['out_image_shape'][1:]
        self.device = device
        self.config = config

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
            self.criterion = E2CLoss(config['train']['num_epochs'], config['loss'])

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
    
    def collect_rollouts(self):
        pass

    def train(self):
        pass

    def learn(self):
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
                x = x.reshape(x.shape[0], -1, *self.out_image_shape)    # Stack obs history in channel dim
                x_next = torch.hstack([x_next for i in range(self.model.past_length)]).to(self.device)

                # Forward pass
                train_return = self.model(x, x_next, u)
                train_return['x'] = x
                train_return['x_next'] = x_next

                # Compute loss and backprop
                loss, recon, recon_next, kld, kld_trans = self.criterion(train_return, epoch)
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
        dataset: Torch dataset object that can be split into train/test sets
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
        policy: Asynchronous policy
        num_rollout_steps: Number of timesteps to collect in a rollout
    """
    def __init__(self, dataset, model, config, device, policy, num_rollout_steps):
        super().__init__(dataset, model, config, device, policy)
        self.num_rollout_steps = num_rollout_steps

    def collect_rollouts(self):
        """
        Collect observations and save them to replay buffer
        """
        for i in self.num_rollout_steps():
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

class ClosedLoopUncertaintyTrainer(BaseTrainer):
    """
    Train a world model in closed loop, where actions are chosen based on prediction variance.

    Args:
        dataset: Torch dataset object that can be split into train/test sets
        model: A world model class instance for training
        config: Config dictionary with required params
        device: Torch device object
        policy: Asynchronous policy
        num_rollout_steps: Number of timesteps to collect in a rollout
    """
    def __init__(self, dataset, model, config, device, policy):
        super().__init__(dataset, model, config, device, policy)
        self.num_rollout_steps = config['closed_loop']['num_rollout_steps']

    def collect_rollouts(self):
        """
        Collect observations and save them to replay buffer
        """
        for i in self.num_rollout_steps():
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