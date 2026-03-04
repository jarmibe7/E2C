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
from src.tactile.gen_tactile import get_env_modes, process_tactile, process_feature
from src.trainer import BaseTrainer, ClosedLoopInformativeTrainer

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
        self.num_rollout_steps = config['closed_loop']['num_rollout_steps']
        self.num_batches = config['closed_loop']['num_batches']
        self.pred_length = config['trans'].get('pred_length', 3)                          # How long to predict ahead in network
        self.plan_horizon = config['closed_loop'].get('plan_horizon', self.pred_length)   # How long to imagine rollouts