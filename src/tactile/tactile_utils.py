import random
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pyvirtualdisplay import Display
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch.nn.functional as F
import lpips
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim

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
        self.psnr_fn = psnr(data_range=1.0).to(self.device)
        self.ssim_fn = ssim(data_range=1.0).to(self.device)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)

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

    def _robot_object_contact_flag(self, env):
        """
        Return 1 if the robot and object are touching in the current PyBullet step.

        This mirrors the timestep-level contact indicator used in the Mujoco
        evaluator, but relies on PyBullet's contact query API.
        """
        sim_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if not hasattr(sim_env, "_pb") or not hasattr(sim_env, "robot") or not hasattr(sim_env, "obj_id"):
            return 0

        robot_id = getattr(sim_env.robot, "robot_id", None)
        if robot_id is None:
            return 0

        # Match the legacy evaluator first: any robot/object contact should count.
        if sim_env._pb.getContactPoints(bodyA=robot_id, bodyB=sim_env.obj_id):
            return 1

        # Also check explicit EE/tactile contacts so sensor-specific geometry is counted.
        ee_link_ids = self._get_ee_contact_link_ids(sim_env)
        for link_id in ee_link_ids:
            if sim_env._pb.getContactPoints(bodyA=robot_id, bodyB=sim_env.obj_id, linkIndexA=link_id):
                return 1
            if sim_env._pb.getContactPoints(bodyA=sim_env.obj_id, bodyB=robot_id, linkIndexB=link_id):
                return 1

        # Some EE/sensor links have collisions disabled in the robot URDF, so
        # getContactPoints can miss visible contact. Check closest points as a
        # conservative fallback for those links.
        proximity_epsilon = 1e-2
        for link_id in ee_link_ids:
            closest_points = sim_env._pb.getClosestPoints(
                bodyA=robot_id,
                bodyB=sim_env.obj_id,
                distance=proximity_epsilon,
                linkIndexA=link_id,
            )
            if any(point[8] <= proximity_epsilon for point in closest_points):
                return 1

            closest_points = sim_env._pb.getClosestPoints(
                bodyA=sim_env.obj_id,
                bodyB=robot_id,
                distance=proximity_epsilon,
                linkIndexB=link_id,
            )
            if any(point[8] <= proximity_epsilon for point in closest_points):
                return 1

        return 0

    def _get_ee_contact_link_ids(self, sim_env):
        """Collect link indices that belong to EE/tactile geometry on the robot."""
        ee_link_ids = set()

        robot = getattr(sim_env, "robot", None)
        if robot is None:
            return ee_link_ids

        arm = getattr(robot, "arm", None)
        if arm is not None:
            for attr in ("EE_link_id", "TCP_link_id"):
                link_id = getattr(arm, attr, None)
                if isinstance(link_id, (int, np.integer)):
                    ee_link_ids.add(int(link_id))

            # Capture additional EE/tactile links from naming conventions.
            link_name_to_index = getattr(arm, "link_name_to_index", None)
            tactile_name = str(getattr(robot, "t_s_name", "")).lower()
            if isinstance(link_name_to_index, dict):
                for link_name, link_id in link_name_to_index.items():
                    if not isinstance(link_id, (int, np.integer)):
                        continue
                    key = str(link_name).lower()
                    if (
                        "ee" in key
                        or "tcp" in key
                        or "tact" in key
                        or (tactile_name and tactile_name in key)
                    ):
                        ee_link_ids.add(int(link_id))

        t_s = getattr(robot, "t_s", None)
        tactile_link_ids = getattr(t_s, "tactile_link_ids", None)
        if isinstance(tactile_link_ids, dict):
            for link_id in tactile_link_ids.values():
                if isinstance(link_id, (int, np.integer)):
                    ee_link_ids.add(int(link_id))

        return ee_link_ids

    def _to_display_image(self, image):
        """Convert a tensor or array image into an 8-bit RGB array for video rendering."""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        image = np.squeeze(image)

        if image.ndim == 2:
            image = image[..., np.newaxis]
        elif image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)

        if image.dtype != np.uint8:
            if np.nanmax(image) <= 1.0:
                image = np.clip(image * 255.0, 0, 255)
            else:
                image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        return image

    def _compose_planner_frame(self, current_image, predicted_images):
        """Create a single visualization frame showing the current image and future predictions."""
        tiles = [self._to_display_image(current_image)]
        tiles.extend(self._to_display_image(image) for image in predicted_images)
        return np.concatenate(tiles, axis=1)

    def _prepare_metric_image(self, image):
        """Convert image tensors to [B, C, H, W] float tensors in [0, 1]."""
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)

        image = image.detach().float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() == 5:
            image = image[:, -1]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2)

        if image.max() > 1.0:
            image = image / 255.0

        return image.clamp(0.0, 1.0)

    def _prepare_lpips_image(self, image):
        image = self._prepare_metric_image(image)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image * 2.0 - 1.0

    def eval_metrics(
        self,
        gt_image_window,
        pred_image_window,
        gt_feature_window=None,
        pred_feature_window=None,
        gt_latent_window=None,
        pred_latent_window=None,
        env=None,
        step_idx=None,
    ):
        """Compute metric values for a single ground-truth/prediction pair."""
        metrics = {}

        gt_image = self._prepare_metric_image(gt_image_window[:, -1]) if torch.is_tensor(gt_image_window) and gt_image_window.dim() == 5 else self._prepare_metric_image(gt_image_window)
        pred_image = self._prepare_metric_image(pred_image_window[:, 0]) if torch.is_tensor(pred_image_window) and pred_image_window.dim() == 5 else self._prepare_metric_image(pred_image_window)
        gt_feature = torch.as_tensor(gt_feature_window).detach().float()
        pred_feature = torch.as_tensor(pred_feature_window).detach().float()
        gt_latent = torch.as_tensor(gt_latent_window).detach().float()
        pred_latent = torch.as_tensor(pred_latent_window).detach().float()

        if gt_image.shape != pred_image.shape:
            min_h = min(gt_image.shape[-2], pred_image.shape[-2])
            min_w = min(gt_image.shape[-1], pred_image.shape[-1])
            gt_image = gt_image[..., :min_h, :min_w]
            pred_image = pred_image[..., :min_h, :min_w]

        if gt_feature.dim() > 2:
            gt_feature = gt_feature[:, -1]
        if pred_feature.dim() > 2:
            pred_feature = pred_feature[:, 0]
        if gt_latent.dim() > 2:
            gt_latent = gt_latent[:, -1]
        if pred_latent.dim() > 2:
            pred_latent = pred_latent[:, 0]

        # Ensure all metric tensors are on the same device as metric modules.
        gt_image = gt_image.to(self.device)
        pred_image = pred_image.to(self.device)
        gt_feature = gt_feature.to(self.device)
        pred_feature = pred_feature.to(self.device)
        gt_latent = gt_latent.to(self.device)
        pred_latent = pred_latent.to(self.device)

        metrics['image_mse'] = F.mse_loss(pred_image, gt_image).item()
        metrics['image_psnr'] = self.psnr_fn(pred_image, gt_image).item()
        metrics['image_ssim'] = self.ssim_fn(pred_image, gt_image).item()
        metrics['image_lpips'] = self.lpips_fn(self._prepare_lpips_image(pred_image), self._prepare_lpips_image(gt_image)).item()
        metrics['latent_mse'] = F.mse_loss(pred_latent, gt_latent).item()
        metrics['feature_mse'] = F.mse_loss(pred_feature, gt_feature).item()

        return metrics

    def _average_metric_lists(self, metric_lists):
        """Average each metric list, returning NaN when no samples are available."""
        out = {}
        for key, values in metric_lists.items():
            out[key] = float(np.mean(values)) if values else float('nan')
        return out
            

    def eval(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None, count_contacts=True):
        """
        Eval-style rollout that can optionally record contact metrics.
        """
        self.model.eval()
        if env_reset_seed is None:
            env_reset_seed = np.random.randint(0, 1e2)

        # self._render_video(trainer, run_path, max_steps=max_steps, closed_loop=closed_loop, env_reset_seed=env_reset_seed)
        eval_return = self.eval_planner(
            trainer,
            run_path,
            max_steps=max_steps,
            closed_loop=closed_loop,
            env_reset_seed=env_reset_seed,
            count_contacts=count_contacts,
        )

        if isinstance(eval_return, tuple) and len(eval_return) == 2:
            _, contact_metrics = eval_return
            metrics_path = Path(run_path) / "eval_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(contact_metrics, f, indent=2)
            print(f"Saved eval metrics to {metrics_path}")

        return eval_return

    def render(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None):
        """
        Render wrapper to enable/disable virtual display
        """
        # with Display(visible=0, size=(480, 480)):
        self._render_video(trainer, run_path, max_steps, closed_loop, env_reset_seed)

    def eval_planner(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None, count_contacts=True):
        """
        Roll out the tactile planner, render predicted futures, and collect step metrics.
        """
        model = trainer.model
        device = trainer.device
        past_len = model.past_length if hasattr(model, 'past_length') else 3
        pred_len = model.pred_length if hasattr(model, 'pred_length') else 3

        meta_ts = trainer.config['train'].get('meta_ts', 1)

        try:
            closed_loop_policy = trainer.config['closed_loop']['policy']
        except Exception:
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

        true_frames = []
        true_feature_frames = []
        true_latent_frames = []
        recon_frames = []
        pred_sequences = []
        pred_feature_sequences = []
        pred_latent_sequences = []
        plan_obj_vals = []
        env_rew = []
        contact_flags = [] if count_contacts else None

        # Seed buffer with first observation
        for _ in range(past_len):
            image_buffer.append(self.obs_adapter.process_image(obs))
            feature_buffer.append(self.obs_adapter.process_feature(obs, self.env))

        total_len = past_len + pred_len
        if closed_loop_policy in ['pixel', 'dynamics']:
            trainer._init_cem_mu_sig()
            mu = trainer.init_control.clone()
            sigma = trainer.sigma.clone()

        for step_idx in tqdm(range(max_steps), desc='Rendering Planner Eval Video'):
            if closed_loop_policy in ['pixel', 'dynamics']:
                mu, costs, sigma = trainer._sample_cem(
                    image_buffer[-past_len:],
                    feature_buffer[-past_len:],
                    mu=mu,
                    sigma=sigma,
                )
                action_seq = mu.clone()
                plan_obj_vals.append(costs[0].clone().cpu().item())
            else:
                act = [self.env.action_space.sample() for _ in range(pred_len)]
                action_seq = torch.from_numpy(np.array(act)).to(device)
                plan_obj_vals.append(0.0)

            env_act = action_seq.cpu().detach().numpy()[0]
            step_true_image = image_buffer[-1].detach().cpu()
            step_true_feature = feature_buffer[-1].detach().cpu()

            for _ in range(meta_ts):
                next_obs, rew, done, _, _ = self.env.step(env_act)

            if count_contacts:
                contact_flags.append(self._robot_object_contact_flag(self.env))

            image = self.obs_adapter.process_image(next_obs)
            feature = self.obs_adapter.process_feature(next_obs, self.env)

            image_buffer.append(image)
            feature_buffer.append(feature)
            act_buffer.append(torch.tensor(env_act))
            env_rew.append(rew)

            if len(image_buffer) > total_len:
                image_buffer.pop(0)
                feature_buffer.pop(0)
                act_buffer.pop(0)

            saved_state[step_idx] = torch.as_tensor([*feature, rew, *env_act], device='cpu')

            with torch.no_grad():
                step_gt_image_window = torch.stack(image_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                step_gt_feature_window = torch.stack(feature_buffer[-trainer.model.past_length:], dim=0).unsqueeze(0).to(trainer.device)
                model_image_window = step_gt_image_window.clone()
                model_feature_window = step_gt_feature_window.clone()
                _, _, zs = trainer.model.encode_posterior(model_image_window, model_feature_window)
                h = torch.zeros(model.num_layers, 1, model.deterministic_size, device=trainer.device)
                z = zs[:, -1]
                step_gt_latent = z.detach().cpu()
                image_recon = trainer.model.decoder(z)

                predicted_images = []
                predicted_latents = []
                predicted_feature_window = []
                for act in action_seq:
                    act_batch = act.view(1, -1).to(trainer.device)
                    h, z_prior, _, _ = trainer.model.rssm_step(h, z.unsqueeze(1), act_batch)
                    image_pred = trainer.model.decoder(z_prior)
                    feature_pred = trainer.model.feature_decoder(z_prior)

                    predicted_images.append(image_pred.detach().cpu())
                    predicted_latents.append(z_prior.detach().cpu())
                    predicted_feature_window.append(feature_pred.detach().cpu())

                    if trainer.past_length > 1:
                        model_image_window = torch.cat([model_image_window[:, 1:], image_pred.unsqueeze(1).detach()], dim=1)
                        model_feature_window = torch.cat([model_feature_window[:, 1:], feature_pred.unsqueeze(1).detach()], dim=1)
                    else:
                        model_image_window = image_pred.detach()
                        model_feature_window = feature_pred.detach()

                    _, _, zs = trainer.model.encode_posterior(model_image_window, model_feature_window)
                    z = zs[:, -1]

            true_feature_frames.append(step_true_feature)
            true_latent_frames.append(step_gt_latent)

            true_frames.append(step_true_image)
            recon_frames.append(image_recon.detach().cpu())
            pred_sequences.append(predicted_images)
            pred_feature_sequences.append(predicted_feature_window)
            pred_latent_sequences.append(predicted_latents)

        metric_names = [
            'image_mse',
            'image_psnr',
            'image_ssim',
            'image_lpips',
            'latent_mse',
            'feature_mse',
        ]
        single_step_lists = {k: [] for k in metric_names}
        cumulative_lists = {k: [] for k in metric_names}

        for t in range(len(true_frames)):
            per_t_cumulative = {k: [] for k in metric_names}

            for j in range(1, pred_len + 1):
                future_idx = t + j
                if future_idx >= len(true_frames):
                    break

                pred_img = pred_sequences[t][j - 1]
                pred_feat = pred_feature_sequences[t][j - 1]
                pred_lat = pred_latent_sequences[t][j - 1]

                gt_img = true_frames[future_idx]
                gt_feat = true_feature_frames[future_idx]
                gt_lat = true_latent_frames[future_idx]

                pair_metrics = self.eval_metrics(
                    gt_image_window=gt_img,
                    pred_image_window=pred_img,
                    gt_feature_window=gt_feat,
                    pred_feature_window=pred_feat,
                    gt_latent_window=gt_lat,
                    pred_latent_window=pred_lat,
                    env=self.env,
                    step_idx=t,
                )

                for key in metric_names:
                    value = pair_metrics[key]
                    per_t_cumulative[key].append(value)
                    if j == 1:
                        single_step_lists[key].append(value)

            for key in metric_names:
                if per_t_cumulative[key]:
                    cumulative_lists[key].append(float(np.mean(per_t_cumulative[key])))

        cols = pred_len + 1
        fig, ax = plt.subplots(2, cols, figsize=(3 * cols, 8))
        ax = np.atleast_2d(ax)
        ax[0, 0].set_title(f"Pred t=0; {plan_obj_vals[0]:.2f}")
        ax[1, 0].set_title(f"True t=0; {env_rew[0]:.2f}")
        for j in range(1, cols):
            ax[0, j].set_title(f"Pred t={j}")
            ax[1, j].set_title(f"True t={j}")

        ims = []
        ims.clear()
        blank = self._to_display_image(true_frames[0])
        for j in range(cols):
            ims.append(ax[0, j].imshow(blank.copy()))
            ims.append(ax[1, j].imshow(blank.copy()))
        for axis in ax.flatten():
            axis.axis('off')

        def update(frame_idx):
            true_curr = true_frames[frame_idx]
            recon = recon_frames[frame_idx]

            ax[0, 0].set_title(f"Pred t=0; {plan_obj_vals[frame_idx]:.2f}")
            ax[1, 0].set_title(f"True t=0; {env_rew[frame_idx]:.2f}")
            ims[0].set_data(self._to_display_image(recon))
            ims[1].set_data(self._to_display_image(true_curr))

            for j in range(1, cols):
                pred_frame = pred_sequences[frame_idx][j - 1].squeeze(0)
                ims[2 * j].set_data(self._to_display_image(pred_frame))
                true_index = frame_idx + j
                if true_index < len(true_frames):
                    true_frame = true_frames[true_index]
                else:
                    true_frame = true_frames[-1]
                ims[2 * j + 1].set_data(self._to_display_image(true_frame))

        ani = FuncAnimation(fig, update, frames=len(true_frames), interval=5.)
        writer = FFMpegWriter(fps=20)
        vid_name = f'{trainer.env_name}_{env_reset_seed}.mp4'
        try:
            filepath = run_path / vid_name
            print(f'Saved planner visualization to {filepath}')
            ani.save(filepath, writer=writer)
            if count_contacts:
                torch.save(torch.tensor(contact_flags, dtype=torch.int8), run_path / f'{vid_name[:-4]}_contacts.pt')
        except Exception as e:
            print(e)
            print('Exception occurred, saved planner visualization to current directory')
            ani.save(vid_name, writer=writer)
            if count_contacts:
                torch.save(torch.tensor(contact_flags, dtype=torch.int8), f'{vid_name[:-4]}_contacts.pt')
        plt.close(fig)

        metrics = {
            'contact_timesteps': int(sum(contact_flags)) if count_contacts else 0,
            'total_timesteps': len(contact_flags) if count_contacts else max_steps,
            'planner_video_filename': vid_name,
        }
        single_step_avg = self._average_metric_lists(single_step_lists)
        cumulative_avg = self._average_metric_lists(cumulative_lists)

        for metric_block in (single_step_avg, cumulative_avg):
            if 'feature_mse' in metric_block and np.isnan(metric_block['feature_mse']):
                metric_block['feature_mse'] = 'N/A'

        metrics['planner_metrics_single_step'] = single_step_avg
        metrics['planner_metrics_cumulative'] = cumulative_avg

        self.last_contact_count = metrics['contact_timesteps']
        self.last_contact_total = metrics['total_timesteps']
        print(f'Contact timesteps: {self.last_contact_count}/{self.last_contact_total}')
        return saved_state, metrics

    def _render_video(self, trainer, run_path, max_steps=25, closed_loop=True, env_reset_seed=None, count_contacts=False):
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