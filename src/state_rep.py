"""
A pretrained state estimation model that infers states from gym.env images

Authors: Jared Berry, Ayush Gaggar
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import time
from datetime import datetime
import copy
import traceback
import json

from src.eval import Plotter
from src.utils import set_seed, wrapped_angle_error, format_time

set_seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"


class StateRepresentationModel(nn.Module):
    """
    A state representation model infers robot states from images of a Gym environment. 

    Args:
        state_size: Dimension of control vector
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
    """
    def __init__(self, state_size, conv_params, device):
        super().__init__()
        self.device = device
        self.state_size = state_size    # Output size
        
        # CNN parameters
        in_channels = conv_params['in_image_shape'][0]
        k = conv_params['kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']

        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=k+2, stride=s-1, padding=p+1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            # nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.zeros(1, in_channels, conv_params['in_image_shape'][1], conv_params['in_image_shape'][2])
            enc_out = self.cnn(x)
            self.out_dim_flat = enc_out.view(enc_out.size(0), -1).shape[1] # Keep batch dim, determine number of elements
            self.out_shape = enc_out.shape

        # Linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.out_dim_flat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_size),
            # nn.ReLU(),
            # nn.Linear(64, self.state_size),
        )

    def forward(self, x):
        conv_out = self.cnn(x)
        flattened = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(flattened)

        tr = {
            'state_pred': out
        }
        return tr
    
class StateRepresentationLoss(nn.Module):
    """
    Loss for state representation model is difference between true and predicted states
    """
    def __init__(self, loss_params):
        super().__init__()
        self.recon_mult = loss_params['recon_mult']

    def forward(self, tr):
        state_pred = tr['state_pred']
        state_true = tr['state_true']

        # Wrap first angle
        pred_angle = state_pred[:, 0]
        true_angle = state_true[:, 0]
        wrapped = wrapped_angle_error(pred_angle, true_angle)

        # Replace raw difference with wrapped for loss
        loss_angle = wrapped.pow(2).mean()

        # If you also regress other dimensions
        loss_rest  = F.mse_loss(state_pred[:, 1:], state_true[:, 1:])

        loss = loss_angle + loss_rest

        # Make return dictionary for loss values
        loss_return = {
            r"MSE Loss": loss.detach().cpu().item(),
        }
        return loss, loss_return

class StateRepesentationDataset():
    """
    An State Representation Dataset consists of an image, and a corresponding ground truth state.
    """
    def __init__(self, config):
        # Load raw dataset
        env_name = config['train']['dataset'].split('_')[0]
        dataset_dir = DATA_PATH / env_name / 'state_rep'
        data = torch.load(dataset_dir / f"{config['train']['dataset']}_state_from_image.pt")
        self.img = data["images"].permute(0, 3, 1, 2)  # Shape: [num_samples, C, H, W]
        self.img_shape = (self.img.shape[1:])
        self.state = data["states"]#[:, :2]      # TODO: Only joint angles for now
        self.state_size = self.state.shape[1]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.state[idx]

class StateRepresentationPretrainer():
    def __init__(self, dataset, model, config, device):
        # Split into training and test sets
        train_size = int(len(dataset) * config['train']['train_ratio'])
        test_size = len(dataset) - train_size
        self.dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Save training params
        self.num_epochs = config['train']['num_epochs']
        self.batch_size = config['train']['batch_size']
        self.device = device
        self.config = config

        # Initialize model
        self.model = model
        model.to(device)
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['alpha'], weight_decay=config['train']['weight_decay'])

        # Create loss criterion
        loss_type = config['loss'].get('loss_type', None)
        if loss_type == 'mse':
            self.model_criterion = StateRepresentationLoss(config['loss'])
        else:
            raise NotImplementedError(f'Loss type "{loss_type}" not supported!')

        # Create visualizer
        self.plotter = Plotter(config['train']['render'], config['train']['plot_freq'])
    
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

    def evaluate(self, run_path=None):
        """
        Evaluate state representation model on test set using
        state-space error metrics only.
        """
        # Create dataloader
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        mse_sum = 0.0
        mae_sum = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for img, state in tqdm(test_loader, desc="Evaluating"):
                img = img.to(self.device)
                state = state.to(self.device)

                out = self.model(img)
                state_pred = out["state_pred"]

                # Wrap first angle

                angle_diff = wrapped_angle_error(state_pred[:, 0], state[:, 0])

                # Replace first dim with wrapped difference for MSE/MAE calc
                diff = state_pred - state
                diff[:, 0] = angle_diff

                # Aggregate losses using wrapped difference
                mse = (diff ** 2).sum()
                mae = diff.abs().sum()

                mse_sum += mse.item()
                mae_sum += mae.item()
                num_samples += state.shape[0]

                all_preds.append(state_pred.cpu())
                all_targets.append(state.cpu())

        # Stack predictions
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Per-dimension MSE
        per_dim_mse = torch.cat([
            (wrapped_angle_error(preds[:, 0], targets[:, 0]).unsqueeze(1) ** 2),
            (preds[:, 1:] - targets[:, 1:]) ** 2
        ], dim=1).mean(dim=0)

        # Mean metrics
        mse_mean = mse_sum / (num_samples * targets.shape[1])
        rmse = mse_mean ** 0.5
        mae_mean = mae_sum / (num_samples * targets.shape[1])

        # Explained variance per dimension
        var_targets = targets.var(dim=0, unbiased=False)
        explained_variance = 1.0 - per_dim_mse / (var_targets + 1e-8)
        explained_variance_mean = explained_variance.mean().item()

        results = {
            "mse": mse_mean,
            "rmse": rmse,
            "mae": mae_mean,
            "explained_variance": explained_variance_mean,
            "per_dim_mse": per_dim_mse.tolist(),
        }

        # Save metrics
        if run_path is not None:
            metrics_path = run_path / "metrics.json"
            try:
                with open(metrics_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Saved evaluation metrics to {metrics_path}")
            except Exception as e:
                print(f"Failed to save metrics.json: {e}")

        self.model.train()
        return results

    def train(self):
        """
        Train a state representation model
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

            for img, state in train_loader:
                # Send training data to GPU
                img, state = img.to(self.device), state.to(self.device)

                # Forward pass
                train_return = self.model(img)
                train_return['state_true'] = state

                # Compute loss and backprop
                loss, loss_return = self.model_criterion(train_return)
                self.plotter.log(loss_return)
                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()

                if torch.isnan(loss):
                    print("NaN loss encountered, stopping training.")
                    break

                total_loss += loss.item() * img.size(0)   # Aggregate total epoch loss

            # Display average loss for the epoch
            epoch_loss = total_loss / len(train_loader.dataset)
            pbar.set_postfix({
                'Epoch': epoch+1,
                'Epoch Loss': f"{epoch_loss:.4f}"})

        pbar.close()

    def learn(self):
        self.train()

if __name__ == "__main__":
    start_time = time.perf_counter()
    print('*** STARTING ***\n')
    # Load config, make run path, and choose torch device
    # ---------- CONFIG HERE ----------
    config_name = 'state_rep_push_v0'
    # ---------- CONFIG HERE ----------
    if 'state_rep' not in config_name:
        raise ValueError('Must use state representation model config!')
    with open(CONFIG_PATH / f'{config_name}.yaml', "r") as f:
        config = yaml.safe_load(f)
    config['config_name'] = config_name
    config_save = copy.deepcopy(config)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    run_path = RUNS_PATH / Path(config['train']['dataset'].split('_')[0]) / 'state_rep' / timestamp
    config['run_path'] = run_path
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])

    # Make E2CDataset object
    print(f"Loading dataset: {config['train']['dataset']}\n")
    dataset = StateRepesentationDataset(config)
    config['conv']['in_image_shape'] = dataset.img_shape
    config['train']['state_size'] = dataset.state_size

    # Create or load model
    model = StateRepresentationModel(
        state_size=config['train']['state_size'],
        conv_params=config['conv'],
        device=device
    )
    load_path = config['train'].get('load_path', None)
    if load_path is None:
        print(f'Training model from scratch\n')
        config['run_path'].mkdir(parents=True, exist_ok=True)
        
    else:
        # Load existing model to train from checkpoint
        print(f'Loading model from checkpoint\n')
        model_path = load_path + '/model.pt'
        model.load_state_dict(torch.load(model_path))
        config['run_path'] = PROJECT_ROOT / Path(load_path)

    # Train, save, and evaluate
    trainer = StateRepresentationPretrainer(dataset, model, config, device)
    try:
        trainer.learn()

        # Save and evaluate
        config_save['runtime'] = format_time(time.perf_counter() - start_time)
        if config['train']['save']: trainer.save(config_save, config['run_path'])
        if config['train']['eval']: trainer.evaluate(config['run_path'])
    except Exception:
        print('\n\n'); traceback.print_exc(); print('\n\n')
        if config['train']['save']:
            config_save['runtime'] = format_time(time.perf_counter() - start_time)
            trainer.save(config_save, config['run_path'])
            if config['train']['eval']: trainer.evaluate(config['run_path'])    
            print(f'\nException occured, saving current checkpoint')
        else: 
            print('Exception occured, ending training')
    except KeyboardInterrupt:
        if config['train']['save']:
            config_save['runtime'] = format_time(time.perf_counter() - start_time)
            trainer.save(config_save, config['run_path'])
            if config['train']['eval']: trainer.evaluate(config['run_path'])    
            print(f'\nManual interrupt occured, saving current checkpoint')
        else: 
            print('Manual interrupt occured, ending training')


    print('\n*** DONE ***')