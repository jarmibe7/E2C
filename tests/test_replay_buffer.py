"""
Tests for replay buffer
"""
import torch
import pytest
from src.replay_buffer import ReplayBuffer
from pathlib import Path
import yaml
import copy
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"

def load_config():
    # ---------- CONFIG HERE ----------
    config_name = 'e2c_reacher_v4'
    # ---------- CONFIG HERE ----------
    with open(CONFIG_PATH / f'{config_name}.yaml', "r") as f:
        config = yaml.safe_load(f)
    config['config_name'] = config_name
    config_save = copy.deepcopy(config)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    run_path = RUNS_PATH / Path(config['train']['dataset'].split('_')[0]) / timestamp
    config['run_path'] = run_path
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])
    return config, device

def test_init():
    # Reacher past_length=3 setup
    config, device = load_config()
    buffer = ReplayBuffer(
        img_shape=(9, 64, 64),
        control_size=2,
        capacity=100,
        device=device,
        config=config
    )
    assert len(buffer) == 0
    assert buffer.capacity == 100

def test_push():
    # Reacher past_length=3 setup
    config, device = load_config()
    buffer = ReplayBuffer(
        img_shape=(9, 64, 64),
        control_size=2,
        capacity=100,
        device=device,
        config=config
    )

    img = torch.zeros(1, 3, 3, 64, 64)
    action = torch.zeros(1, 2)
    reward = 0.0
    next_img = torch.zeros(1, 3, 64, 64)
    done = 0
    
    assert len(buffer) == 0
    assert buffer.capacity == 100

    buffer.add(
        img,
        action,
        reward,
        next_img,
        done
    )

    assert len(buffer) == 1
    assert buffer.capacity == 100

def test_full():
    # Reacher past_length=3 setup
    config, device = load_config()
    buffer = ReplayBuffer(
        img_shape=(9, 64, 64),
        control_size=2,
        capacity=100,
        device=device,
        config=config
    )

    img = torch.zeros(1, 3, 3, 64, 64)
    action = torch.zeros(1, 2)
    next_img = torch.zeros(1, 3, 64, 64)
    done = 0
    
    assert len(buffer) == 0
    assert buffer.capacity == 100

    for i in range(buffer.capacity):
        buffer.add(
            img,
            action,
            i,
            next_img,
            done
        )

    assert len(buffer) == 100
    assert buffer.capacity == 100

    for i, r in enumerate(buffer.rewards):
        assert i == r

    # Test circle buffer pointer + replacement
    extra = 5
    for i in range(buffer.capacity, buffer.capacity + extra):
        buffer.add(
            img,
            action,
            i,
            next_img,
            done
        )

    assert buffer.ptr == extra

    for i in range(extra):
        assert int(buffer.rewards[i].item()) == buffer.capacity + i

def test_sample_shape():
    # Reacher past_length=3 setup
    config, device = load_config()
    buffer = ReplayBuffer(
        img_shape=(9, 64, 64),
        control_size=2,
        capacity=100,
        device=device,
        config=config
    )

    img = torch.zeros(1, 3, 3, 64, 64)
    action = torch.zeros(1, 2)
    next_img = torch.zeros(1, 3, 64, 64)
    done = 0
    
    assert len(buffer) == 0
    assert buffer.capacity == 100

    for i in range(buffer.capacity):
        buffer.add(
            img,
            action,
            i,
            next_img,
            done
        )

    # Test sample shapes
    batch_size = 32
    sample = buffer.sample(batch_size)
    for item in sample:
        assert item.shape[0] == 32

    imgs, actions, rewards, next_imgs, dones = sample
    assert imgs.shape == (batch_size, 3, 3, 64, 64)
    assert actions.shape == (batch_size, 2)
    assert rewards.shape == (batch_size, 1)
    assert next_imgs.shape == (batch_size, 3, 64, 64)
    assert dones.shape == (batch_size, 1)

def test_pred_length():
    # Reacher past_length=3 setup
    config, device = load_config()
    config['trans']['pred_length'] = 3
    buffer = ReplayBuffer(
        img_shape=(9, 64, 64),
        control_size=2,
        capacity=100,
        device=device,
        config=config
    )

    img = torch.zeros(1, 3, 3, 64, 64)
    action = torch.zeros(1, 3, 2)
    next_img = torch.zeros(1, 3, 3, 64, 64)
    done = 0
    
    assert len(buffer) == 0
    assert buffer.capacity == 100

    for i in range(buffer.capacity):
        buffer.add(
            img,
            action,
            i,
            next_img,
            done
        )

    # Test sample shapes
    batch_size = 32
    sample = buffer.sample(batch_size)
    for item in sample:
        assert item.shape[0] == 32

    imgs, actions, rewards, next_imgs, dones = sample
    assert imgs.shape == (batch_size, 3, 3, 64, 64)
    assert actions.shape == (batch_size, 3, 2)
    assert rewards.shape == (batch_size, 1)
    assert next_imgs.shape == (batch_size, 3, 3, 64, 64)
    assert dones.shape == (batch_size, 1)

