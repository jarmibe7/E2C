import random
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pyvirtualdisplay import Display
from matplotlib.animation import FuncAnimation, FFMpegWriter

from src.replay_buffer import ReplayBuffer
from src.tactile.gen_tactile import (
    build_observation_adapter,
    get_env_modes,
    resolve_env_name_from_dataset,
    tactile_envs,
)

# Get paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"

#
# ---------- Data Management and Sampling ----------
# 
class TactileDataset():
    """
    An tactile sample consists of a current tactile image + feature, a future tactile image + feature, and a control input.
    """
    def __init__(self, config):
        # Load raw dataset
        env_name = resolve_env_name_from_dataset(config['train']['dataset'])
        dataset_dir = DATA_PATH / env_name
        data = torch.load(dataset_dir / f"{config['train']['dataset']}.pt")

        obs_adapter = build_observation_adapter(config)
        self.image_source = data.get("image_source", obs_adapter.image_source)
        self.camera_mode = data.get("camera_mode", obs_adapter.camera_mode)
        self.feature_components = data.get("feature_components", obs_adapter.feature_components)

        # Observations
        if "prev_obs" in data and "next_obs" in data:
            self.obs = data["prev_obs"]
            self.next_obs = data["next_obs"]
        else:
            # Backward compatibility with older tactile-only datasets
            self.obs = data["prev_tactile"]
            self.next_obs = data["next_tactile"]

        self.tactile = self.obs
        self.next_tactile = self.next_obs
        self.in_img_shape = [
            self.obs[0, 0].shape[0] * config['trans']['past_length'],
            *self.obs[0, 0].shape[1:]
        ]

        self.feature = data["prev_feature"]
        self.next_feature = data["next_feature"]
        self.num_features = self.feature.shape[-1]

        if len(self.obs.shape) == 5:
            self.past_length = self.obs.shape[1]
            assert self.past_length == config['trans']['past_length'], f"past_length={config['trans']['past_length']} in config file, but dataset has past_length={self.past_length}"
        else:
            self.past_length = 1

        self.pred_length = config['trans']['pred_length']
        control = data["actions"]
        self.U = control

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.next_obs[idx], self.feature[idx], self.next_feature[idx], self.U[idx]

class TactileReplayBufferSample():
    def __init__(self, t, f, u, r, t_next, f_next, d):
        self.tactile = t
        self.feature = f
        self.u = u
        self.rewards = r
        self.tactile_next = t_next
        self.feature_next = f_next
        self.dones = d

class TactileReplayBuffer(ReplayBuffer):
    """
    Simple ring buffer ReplayBuffer class, with random sampling. Has tactile and feature data.

    Args:
        img_shape: Shape of tactile pixel obs is (past_length*1, H, W)
        num_features: Length of non-tactile feature vector
        control_size: Dimension of control vector
        capacity: How many samples to hold at once
        device: CPU or GPU
        config: Config dictionary
    """
    def __init__(self, img_shape, num_features, control_size, capacity, device, config):
        super().__init__(img_shape, control_size, capacity, device, config)

        # Feature buffers
        past_length = config['trans']['past_length']
        pred_length = config['trans']['pred_length']
        self.features = torch.zeros((capacity, past_length, num_features), device=device)      # TODO: Don't hardcode this
        self.features_next = torch.zeros((capacity, pred_length, num_features), device=device)

    @torch.no_grad()
    def add(self, tactile, feature, action, reward, next_tactile, next_feature, done):
        super().add(tactile, action, reward, next_tactile, done, update_ptr=False)
        self.features[self.ptr] = feature.to(self.device)
        self.features_next[self.ptr] = next_feature.to(self.device)

        # Update ring buffer params
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Generate batch_size random samples from the replay buffer
        # idx = torch.randint(0, self.size, (batch_size,), device=self.device)  # Samples with replacement
        idx = random.sample(range(self.size), min(batch_size, self.size))  # Samples without replacement

        x, u, r, x_next, d = super().sample(batch_size, idx=idx)
        f = self.features[idx]
        f_next = self.features_next[idx]

        return TactileReplayBufferSample(x, f, u, r, x_next, f_next, d)
    
#
# ---------- Evaluation ----------
# 
class TactileEvaluator():
    """
    Class for evaluating model performance on a test set.
    """
    def __init__(self, model, batch_size, device, env_name, tactile_image_size, config):
        # Params
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.obs_adapter = build_observation_adapter(config)

        self.disp = Display(visible=0, size=(480, 480))
        self.disp.start()
        self.env_name = env_name
        self.env = tactile_envs[self.env_name](
            max_steps=1000, # Will be limited in function
            env_modes=get_env_modes(
                config=config,
                observation_mode=self.obs_adapter.observation_mode,
                camera_mode=self.obs_adapter.camera_mode,
            ),
            show_gui=True,             # Render in eval env
            show_tactile=True,
            image_size=tactile_image_size
        )
            

    def eval(self, run_path, vid_max_frames=50):
        self.model.eval()
        print("\Calculating eval metrics...")
        self.eval_metrics(run_path=run_path)
        print("Generating latent space figure...")
        self.dataset_latent_func(run_path)
        print("\nGenerating trajectory video...")
        self.eval_traj(run_path, max_frames=vid_max_frames)
        # self.eval_latent(run_path)

    def render(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None):
        """
        Render wrapper to enable/disable virtual display
        """
        # with Display(visible=0, size=(480, 480)):
        self._render_video(trainer, run_path, max_steps, closed_loop, env_reset_seed)

    def _render_video(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None):
        """
        Visualize trajectories using the trainer's planner/rollout logic.

        Args:
            trainer: Trainer instance (must expose env, model, device, past_length, pred_length/config).
            run_path: Path to save the resulting video.
            max_steps: Number of environment steps to visualize.
            closed_loop: If True, feed ground-truth observations back into the model window.
                         If False, feed the model's predicted next frame (open-loop rollout).
        """
        model = trainer.model
        device = trainer.device
        past_len = model.past_length if hasattr(model, 'past_length') else 3
        pred_len = model.pred_length if hasattr(model, 'pred_length') else 3

        meta_ts = trainer.config['train'].get('meta_ts', 1)

        try:
            closed_loop_policy = trainer.config['closed_loop']['policy']
        except:
            print('No closed loop policy found in trainer config, defaulting to random')
            closed_loop_policy = 'random'

        if env_reset_seed is None:
            env_reset_seed = np.random.randint(0, 1e2)
        
        saved_state = torch.zeros(trainer.saved_state.shape, device='cpu')

        model.eval()
        obs, _ = self.env.reset()

        image_buffer = []
        feature_buffer = []
        act_buffer = []

        render_frames = []
        plan_obj_vals = []  # Objective function values per step
        env_rew = []        # Env rewards [blind during training]

        frame = self.env.render()
        if frame is not None:
            render_frames.append(frame)

        # Seed buffer with first observation
        for _ in range(past_len):
            image_buffer.append(self.obs_adapter.process_image(obs))
            feature_buffer.append(self.obs_adapter.process_feature(obs, self.env))

        step_idx = 0
        total_len = past_len + pred_len
        if closed_loop_policy in ['pixel', 'dynamics']:
            trainer._init_cem_mu_sig()
            mu = trainer.init_control.clone()
            sigma = trainer.sigma.clone()
        for step_idx in tqdm(range(max_steps), desc="Rendering Eval Video"): # range(max_steps): #
            # Take actions
            if closed_loop_policy in ['pixel', 'dynamics']:
                mu, costs, sigma = trainer._sample_cem(image_buffer[-past_len:], feature_buffer[-past_len:], mu=mu, sigma=sigma) # pred_len, act_size
                action_seq = mu.clone()
                plan_obj_vals.append(costs[0].clone().cpu().item())
            else:
                # repeat pred len number of times for action horizon
                act = [self.env.action_space.sample() for _ in range(pred_len)]
                action_seq = torch.from_numpy(np.array(act)).to(device)
                plan_obj_vals.append(0.0)   # no cost info for random policy
            env_act = action_seq.cpu().detach().numpy()[0]

            # Step env
            for _ in range(meta_ts):
                next_obs, rew, done, _, _ = self.env.step(env_act)

            frame = self.env.render()
            if frame is not None:
                render_frames.append(frame)

            image = self.obs_adapter.process_image(next_obs)
            feature = self.obs_adapter.process_feature(next_obs, self.env)

            image_buffer.append(image)
            feature_buffer.append(feature)
            act_buffer.append(torch.tensor(env_act))
            env_rew.append(rew)

            # Maintain sliding window
            if len(image_buffer) > total_len:
                image_buffer.pop(0)
                feature_buffer.pop(0)
                act_buffer.pop(0)

            saved_state[step_idx] = torch.as_tensor([*feature, rew, *env_act], device='cpu')

            # Model inputs
            with torch.no_grad():
                image_window = torch.stack(image_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                feature_window = torch.stack(feature_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                mu_prior, log_var_prior, zs = trainer.model.encode_posterior(image_window, feature_window)
                h = torch.zeros(model.num_layers, 1, model.deterministic_size, device=trainer.device)
                z = zs[:, -1]
                image_recon = trainer.model.decoder(z)
                feature_recon = trainer.model.feature_decoder(z)
                image_recon_next = []
                for act in action_seq:
                    act_batch = act.view(1, -1).to(trainer.device)
                    h, z_prior, mu_p, log_var_p = trainer.model.rssm_step(h, z.unsqueeze(1), act_batch)
                    image_pred = trainer.model.decoder(z_prior)
                    feature_pred = trainer.model.feature_decoder(z_prior)
                    image_recon_next.append(image_pred.detach().cpu())
                    
                    if trainer.past_length > 1:
                        image_frames = image_window[:, 1:]   # drop first frame
                        feature_frames = feature_window[:, 1:]
                        image_window = torch.cat([image_frames, image_pred.unsqueeze(1).detach()], dim=1)
                        feature_window = torch.cat([feature_frames, feature_pred.unsqueeze(1).detach()], dim=1)
                    else:
                        image_window = image_pred.detach()  # past_length==1, just use pred image
                        feature_window = feature_pred.detach()  # past_length==1, just use pred image
                    mu_q, log_var_q, zs = trainer.model.encode_posterior(image_window, feature_window)
                    z = zs[:, -1]

        # Build visualization grid: 2 rows, (pred_len + 1) columns
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # ax.set_title(f"Pred t=0; {plan_obj_vals[0]:.2f}")

        ims = []
        # Initialize cells
        ims.clear()
        ims.append(ax.imshow(render_frames[0]))
        ax.axis('off')

        def update(frame_idx):
            ims[0].set_data(render_frames[frame_idx])

        ani = FuncAnimation(fig, update, frames=len(render_frames), interval=5.)
        writer = FFMpegWriter(fps=20)
        vid_name = f'{trainer.env_name}_{env_reset_seed}.mp4'
        try:
            filepath = run_path / vid_name
            print(f'Saved planner visualization to {filepath}')
            ani.save(filepath, writer=writer)
        except Exception as e:
            print(e)
            print('Exception occurred, saved planner visualization to current directory')
            ani.save(vid_name, writer=writer)
        plt.close(fig)
        return saved_state