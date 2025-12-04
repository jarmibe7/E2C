"""
train.py

Main training script for E2C

Authors: Jared Berry, Ayush Gaggar
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import time
from datetime import datetime
from pathlib import Path

from src.e2c import E2CDataset, E2CLoss, E2C
from src.utils import set_seed, anim_frames
from src.eval import Plotter, Evaluator

# Set random seed globally
set_seed(42)

# Paths - relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"


def train(dataset, config):
    """
    Function for training an E2C model
    """
    num_epochs = config['train']['num_epochs']
    device = config['train']['device']

    # Create autoencoder model and optimizer
    model = E2C(
        enc_latent_size=config['vae']['enc_latent_size'],
        latent_size=config['trans']['latent_size'],
        control_size=config['trans']['control_size'],
        past_length=config['trans']['past_length'],
        pred_length=config['trans']['pred_length'],
        conv_params=config['vae'],
        device=device
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['trans']['alpha'], weight_decay=config['trans']['weight_decay'])

    # Create Dataset and DataLoader to handle batching of training data
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['train']['batch_size'], shuffle=True
    )

    # Create loss criterion
    criterion = E2CLoss(config['train']['num_epochs'], config['loss'])

    # Create visualizer
    plotter = Plotter(config['train']['render'], config['train']['plot_freq'])

    # Training loop
    print('\nBeginning Training:')
    for epoch in range(num_epochs):
        total_loss = 0.0

        for x, x_next, u in train_loader:
            # Send training data to GPU
            x, x_next, u = x.to(device), x_next.to(device), u.to(device)

            # Forward pass
            train_return = model(x, x_next, u)
            train_return['x'] = x
            train_return['x_next'] = x_next

            # Compute loss and backprop
            loss, recon, recon_next, kld, kld_trans = criterion(train_return, epoch)
            plotter.log(recon, recon_next, kld, kld_trans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                print("NaN loss encountered, stopping training.")
                break

            total_loss += loss.item() * x.size(0)   # Aggregate total epoch loss

        # Display average loss for the epoch
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f'\n--------------------------------------------------')
        print(f'EPOCH {epoch+1}/{num_epochs}')
        print(f"Average Epoch Loss: {epoch_loss:.4f}")
        print(f'--------------------------------------------------\n')

    if config['train']['save']: plotter.save(config['run_path'])
    else: plotter.close()
    return model

def main():
    print('*** STARTING ***\n')
    # Load config and choose torch device
    config_name = 'e2c_config1'
    with open(CONFIG_PATH / f'{config_name}.yaml', "r") as f:
        config = yaml.safe_load(f)
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])
    config['train']['device'] = device   # Replace device string with device object in config
    config['config_name'] = config_name
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    run_path = RUNS_PATH / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    config['run_path'] = run_path

    # Make E2CDataset object
    print(f"Loading dataset: {config['train']['dataset']}")
    dataset = E2CDataset(config)
    config['vae']['out_image_shape'] = dataset.img_shape
    config['trans']['control_size'] = dataset.U.shape[-1]

    # Split into training and test sets
    train_size = int(len(dataset) * config['train']['train_ratio'])
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Train on training dataet
    model = train(train_dataset, config)

    # Save model
    if config['train']['save']:
        model_name = 'model.pt'
        try:
            filepath = config['run_path'] / model_name
            print(f'\nSaved model to {filepath}')
            torch.save(model.state_dict(), filepath)
        except Exception as e:
            print(e)
            print('\nException occured, saved model to current directory')
            torch.save(model.state_dict(), model_name)

        # Save final config dictionary
        try:
            yaml_name = 'config.yaml'
            yaml_path = config['run_path'] / yaml_name
            print(f'\nSaved config to {yaml_path}')
            with open(yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(e)
            print('\nException occurred, saved config to current directory')
            with open(yaml_name, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    # Evaluate model performance
    if config['train']['eval']:
        evaluator = Evaluator(
            model, 
            test_dataset,
            batch_size=config['train']['batch_size'], 
            device=config['train']['device'],
            dataset_name=config['train']['dataset']
        )
        evaluator.eval(config['run_path'])
    

    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()