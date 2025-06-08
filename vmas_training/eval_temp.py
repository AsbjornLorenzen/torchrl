# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import json
import pickle
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from datetime import datetime

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from utils.logging import init_logging, log_evaluation
from utils.utils import DoneTransform
from models.pgat_actor import PGATActor, ObservationConfig
from models.gnn_critic import GNNCritic


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


def extract_episode_metrics(rollouts, env):
    """Extract detailed episode metrics from evaluation rollouts"""
    
    # Get the final info from each episode
    # rollouts has shape [n_envs, max_steps, ...]
    batch_size = rollouts.batch_size[0]  # number of environments
    
    metrics = {
        'total_agent_collisions': 0,
        'total_obstacle_collisions': 0,
        'avg_time_to_formation': 0,
        'formation_success_rate': 0,
        'episodes_evaluated': batch_size,
        'avg_formation_error': 0,
        'avg_best_formation_error': 0,
        'formation_error_over_time': [],
        'agent_collisions_over_time': [],
        'obstacle_collisions_over_time': [],
        'formation_achieved_over_time': []
    }
    
    try:
        # Get episode metrics from the environment
        # get from agent 0 (all agents have identical data)
        episode_metrics = rollouts['agents']['info']
        
        # Aggregate metrics across all environments
        metrics['total_agent_collisions'] = episode_metrics['agent_collisions'][:,-1,0,0].sum().item()
        metrics['total_obstacle_collisions'] = episode_metrics['obstacle_collisions'][:,-1,0,0].sum().item()
        
        # Formation metrics
        formation_achieved = episode_metrics['formation_achieved'][:,-1,0,0]
        time_to_formation = episode_metrics['time_to_formation'][:,-1,0,0]
        
        # Only consider episodes where formation was achieved for average time
        achieved_mask = formation_achieved > 0
        if achieved_mask.sum() > 0:
            metrics['avg_time_to_formation'] = time_to_formation[achieved_mask].float().mean().item()
        else:
            metrics['avg_time_to_formation'] = -1  # Indicates no formation achieved
        
        metrics['formation_success_rate'] = formation_achieved.mean().item()
        
        # Additional metrics
        metrics['avg_agent_collisions_per_episode'] = metrics['total_agent_collisions'] / batch_size
        metrics['avg_obstacle_collisions_per_episode'] = metrics['total_obstacle_collisions'] / batch_size

        # Formation error metrics
        if 'formation_accuracy_per_step' in episode_metrics:
            formation_accuracy = episode_metrics['formation_accuracy_per_step']
            # Average formation error across all episodes and timesteps
            metrics['avg_formation_error'] = formation_accuracy.mean().item()
            
            # Best formation error (minimum error achieved per episode, then averaged)
            max_accuracies_per_batch = formation_accuracy.max(dim=1)[0]  # Max across time dimension
            metrics['avg_best_formation_error'] = max_accuracies_per_batch.mean().item()
            
            # Time series data - average across all environments at each timestep
            formation_error_series = formation_accuracy.mean(dim=[0, 2, 3]).cpu().numpy()
            metrics['formation_error_over_time'] = formation_error_series.tolist()
        
        # Collision time series - cumulative collisions at each timestep
        agent_collisions_series = episode_metrics['agent_collisions'].mean(dim=[2, 3]).cpu().numpy()  # [n_envs, max_steps]
        obstacle_collisions_series = episode_metrics['obstacle_collisions'].mean(dim=[2, 3]).cpu().numpy()  # [n_envs, max_steps]
        
        # Average across environments
        metrics['agent_collisions_over_time'] = agent_collisions_series.mean(axis=0).tolist()
        metrics['obstacle_collisions_over_time'] = obstacle_collisions_series.mean(axis=0).tolist()
        
        # Formation achievement over time
        if 'formation_achieved' in episode_metrics:
            formation_achieved_series = episode_metrics['formation_achieved'].mean(dim=[2, 3]).cpu().numpy()  # [n_envs, max_steps]
            metrics['formation_achieved_over_time'] = formation_achieved_series.mean(axis=0).tolist()
        
    except Exception as e:
        print(f"Warning: Could not extract episode metrics: {e}")
        # Return default metrics if extraction fails
        pass
    
    return metrics


def save_metrics(metrics, mode, model_name="", timestamp=None):
    """Save metrics to local files for later analysis"""
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory structure
    save_dir = os.path.join(os.getcwd(), 'evaluation_results', mode)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with timestamp
    base_filename = f"eval_metrics_{timestamp}"
    if model_name:
        base_filename = f"eval_metrics_{model_name}_{timestamp}"
    
    # Save as JSON (human readable)
    json_path = os.path.join(save_dir, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save as pickle (preserves numpy arrays if any)
    pickle_path = os.path.join(save_dir, f"{base_filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save time series data as separate numpy files for easy plotting
    np_dir = os.path.join(save_dir, 'numpy_data')
    os.makedirs(np_dir, exist_ok=True)
    
    # Save individual time series
    if metrics['formation_error_over_time']:
        np.save(os.path.join(np_dir, f"{base_filename}_formation_error.npy"), 
                np.array(metrics['formation_error_over_time']))
    
    if metrics['agent_collisions_over_time']:
        np.save(os.path.join(np_dir, f"{base_filename}_agent_collisions.npy"), 
                np.array(metrics['agent_collisions_over_time']))
    
    if metrics['obstacle_collisions_over_time']:
        np.save(os.path.join(np_dir, f"{base_filename}_obstacle_collisions.npy"), 
                np.array(metrics['obstacle_collisions_over_time']))
    
    if metrics['formation_achieved_over_time']:
        np.save(os.path.join(np_dir, f"{base_filename}_formation_achieved.npy"), 
                np.array(metrics['formation_achieved_over_time']))
    
    print(f"Metrics saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Pickle: {pickle_path}")
    print(f"  NumPy arrays: {np_dir}")
    
    return json_path, pickle_path


def create_summary_report(metrics, mode, save_dir):
    """Create a human-readable summary report"""
    
    report_path = os.path.join(save_dir, f"summary_report_{mode}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Summary Report - Mode: {mode}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Episodes Evaluated: {metrics['episodes_evaluated']}\n")
        f.write(f"Formation Success Rate: {metrics['formation_success_rate']:.3f}\n")
        f.write(f"Average Formation Error: {metrics['avg_formation_error']:.4f}\n")
        f.write(f"Average Best Formation Error: {metrics['avg_best_formation_error']:.4f}\n")
        f.write(f"Average Time to Formation: {metrics['avg_time_to_formation']:.1f} steps\n\n")
        
        f.write("Collision Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Agent Collisions: {metrics['total_agent_collisions']}\n")
        f.write(f"Total Obstacle Collisions: {metrics['total_obstacle_collisions']}\n")
        f.write(f"Avg Agent Collisions per Episode: {metrics['avg_agent_collisions_per_episode']:.2f}\n")
        f.write(f"Avg Obstacle Collisions per Episode: {metrics['avg_obstacle_collisions_per_episode']:.2f}\n\n")
        
        if 'mean_reward' in metrics:
            f.write("Reward Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean Reward: {metrics['mean_reward']:.4f}\n")
            f.write(f"Reward Std: {metrics['reward_std']:.4f}\n")
            f.write(f"Min Reward: {metrics['min_reward']:.4f}\n")
            f.write(f"Max Reward: {metrics['max_reward']:.4f}\n\n")
        
        f.write("Time Series Data Available:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Formation Error Over Time: {len(metrics['formation_error_over_time'])} timesteps\n")
        f.write(f"Agent Collisions Over Time: {len(metrics['agent_collisions_over_time'])} timesteps\n")
        f.write(f"Obstacle Collisions Over Time: {len(metrics['obstacle_collisions_over_time'])} timesteps\n")
        f.write(f"Formation Achievement Over Time: {len(metrics['formation_achieved_over_time'])} timesteps\n")
    
    print(f"Summary report saved to: {report_path}")
    return report_path


def load_model_checkpoint(checkpoint_path, policy, value_module=None, device='cpu'):
    """
    Load a model checkpoint for evaluation.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        policy: The policy network to load weights into
        value_module: Optional value network to load weights into
        device: Device to load the checkpoint on
        
    Returns:
        Dictionary containing loaded checkpoint info (iteration, metrics, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    torchrl_logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    policy.load_state_dict(checkpoint['policy_state_dict'])
    torchrl_logger.info("Loaded policy state")
    
    if value_module is not None and 'value_state_dict' in checkpoint:
        value_module.load_state_dict(checkpoint['value_state_dict'])
        torchrl_logger.info("Loaded value module state")
    
    # Extract checkpoint info
    iteration = checkpoint.get('iteration', 0)
    metrics = checkpoint.get('metrics', {})
    
    torchrl_logger.info(f"Loaded model from iteration {iteration}")
    if metrics:
        torchrl_logger.info(f"Previous training metrics: {metrics}")
    
    return {
        'iteration': iteration,
        'metrics': metrics,
        'checkpoint': checkpoint
    }


@hydra.main(version_base="1.1", config_path="", config_name="eval_mappo_gat")
def evaluate(cfg: "DictConfig"):  # noqa: F821
    # Add model_dir to config if not present
    if not hasattr(cfg, 'model_dir'):
        cfg.model_dir = ""  # Default empty string
    
    # Add checkpoint_path to config if not present
    checkpoint_path = cfg.get("checkpoint_path", None)
    
    mode = cfg.env.scenario.get("mode", "basic")
    
    # Handle run description for logging
    run_description = cfg.get("run_description") or os.getenv("RUN_DESCRIPTION") or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_tags = cfg.get("tags", [])
    
    # Add to config for complete tracking
    cfg.run_description = run_description
    cfg.tags = run_tags

    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Create test environment
    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env_test = TransformedEnv(
        env_test,
        RewardSum(in_keys=[env_test.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # Model configuration
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    n_attention_heads = cfg.model.get("n_attention_heads", 4)
    dropout = cfg.model.get("dropout", 0.0)
    k_neighbors = cfg.model.get("k_neighbors", 2)
    k_obstacles = cfg.model.get("k_obstacles", 2)

    # Position indices configuration
    agent_pos_indices_list = cfg.model.get("agent_pos_indices", [0, 2])
    agent_pos_indices = slice(agent_pos_indices_list[0], agent_pos_indices_list[1])

    observation_config = ObservationConfig(k_neighbors=k_neighbors, k_obstacles=k_obstacles)
    
    # Create the PGATActor
    actor_net = nn.Sequential(
        PGATActor(
            obs_config=observation_config,
            total_obs_dim=env_test.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env_test.action_spec.shape[-1],
            gnn_hidden_dim=gnn_hidden_dim,
            n_gnn_layers=gnn_layers,
            n_attention_heads=n_attention_heads,
            k_neighbors=k_neighbors,
            k_obstacles=k_obstacles,
            dropout=dropout,
            pos_indices=slice(0,2),
            device=cfg.train.device,
        ),
        NormalParamExtractor(),
    )
    actor_net = actor_net.to(cfg.train.device)

    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    lowest_action = env_test.full_action_spec_unbatched[("agents", "action")].space.low.to(cfg.train.device)
    highest_action = env_test.full_action_spec_unbatched[("agents", "action")].space.high.to(cfg.train.device)
    print(f"Got action spec lowest {lowest_action} highest {highest_action}")

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env_test.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env_test.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": highest_action
        },
        return_log_prob=True,
    )

    # Create critic (optional for evaluation)
    critic_module = GNNCritic(
        n_agent_inputs=env_test.observation_spec["agents", "observation"].shape[-1],
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=None,
        pos_indices=agent_pos_indices,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
    )
    value_module = ValueOperator(
        module=critic_module,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")]
    )

    # Load pre-trained model parameters
    model_name_for_saving = ""
    if checkpoint_path:
        try:
            checkpoint_info = load_model_checkpoint(
                checkpoint_path=checkpoint_path,
                policy=policy,
                value_module=value_module,
                device=cfg.train.device
            )
            model_name_for_saving = f"iter_{checkpoint_info['iteration']}"
            torchrl_logger.info(f"Successfully loaded checkpoint from iteration {checkpoint_info['iteration']}")
        except Exception as e:
            torchrl_logger.error(f"Failed to load checkpoint: {e}")
            torchrl_logger.info("Using untrained models for evaluation")
            model_name_for_saving = "untrained"
    elif cfg.model_dir:
        # Legacy support for separate actor/critic files
        actor_path = os.path.join(cfg.model_dir)
        if os.path.exists(actor_path):
            checkpoint = torch.load(actor_path, map_location=cfg.train.device)
            if 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                policy.load_state_dict(checkpoint)
            print(f"Loaded actor model from {actor_path}")
            model_name_for_saving = "legacy_model"
        else:
            print(f"Warning: Actor model not found at {actor_path}")
            model_name_for_saving = "untrained"
    else:
        print("Warning: No checkpoint_path or model_dir specified. Using untrained models.")
        model_name_for_saving = "untrained"

    # Logging setup
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO_Eval"
        )
        logger = init_logging(cfg, model_name, run_description, run_tags)

    # Evaluation
    print(f"Starting evaluation with {cfg.eval.evaluation_episodes} episodes in mode: {mode}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        env_test.frames = []
        rollouts = env_test.rollout(
            max_steps=cfg.env.max_steps,
            policy=policy,
            callback=rendering_callback if cfg.eval.get("render", False) else None,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )

        # Extract detailed episode metrics
        episode_metrics = extract_episode_metrics(rollouts, env_test)
        
        # Calculate basic evaluation metrics
        rewards = rollouts.get(("agents", "episode_reward"), None)
        if rewards is not None:
            episode_metrics['mean_reward'] = rewards.mean().item()
            episode_metrics['reward_std'] = rewards.std().item()
            episode_metrics['min_reward'] = rewards.min().item()
            episode_metrics['max_reward'] = rewards.max().item()
            
            print(f"Average reward: {episode_metrics['mean_reward']:.4f}")
            print(f"Reward std: {episode_metrics['reward_std']:.4f}")
            print(f"Min reward: {episode_metrics['min_reward']:.4f}")
            print(f"Max reward: {episode_metrics['max_reward']:.4f}")
        
        # Print detailed metrics
        print(f"\nDetailed Evaluation Metrics (Mode: {mode}):")
        print(f"  Formation Success Rate: {episode_metrics['formation_success_rate']:.3f}")
        print(f"  Avg Time to Formation: {episode_metrics['avg_time_to_formation']:.1f} steps")
        print(f"  Avg Formation Error: {episode_metrics['avg_formation_error']:.4f}")
        print(f"  Avg Best Formation Error: {episode_metrics['avg_best_formation_error']:.4f}")
        print(f"  Total Agent Collisions: {episode_metrics['total_agent_collisions']}")
        print(f"  Total Obstacle Collisions: {episode_metrics['total_obstacle_collisions']}")
        print(f"  Avg Collisions per Episode: Agent={episode_metrics['avg_agent_collisions_per_episode']:.2f}, "
              f"Obstacle={episode_metrics['avg_obstacle_collisions_per_episode']:.2f}")
        
        # Add evaluation metadata
        episode_metrics['evaluation_mode'] = mode
        episode_metrics['evaluation_timestamp'] = timestamp
        episode_metrics['model_name'] = model_name_for_saving
        episode_metrics['num_episodes'] = cfg.eval.evaluation_episodes
        episode_metrics['max_steps'] = cfg.env.max_steps
        episode_metrics['checkpoint_path'] = checkpoint_path
        
        # Save metrics locally
        json_path, pickle_path = save_metrics(episode_metrics, mode, model_name_for_saving, timestamp)
        
        # Create summary report
        save_dir = os.path.join(os.getcwd(), 'evaluation_results', mode)
        create_summary_report(episode_metrics, mode, save_dir)
        
        # Log evaluation results (if logger is configured)
        if cfg.logger.backend:
            log_evaluation(logger, rollouts, env_test, 0, step=0)
            
            # Log detailed episode metrics to wandb if using wandb
            if cfg.logger.backend == "wandb":
                logger.experiment.log({
                    "eval/total_agent_collisions": episode_metrics['total_agent_collisions'],
                    "eval/total_obstacle_collisions": episode_metrics['total_obstacle_collisions'],
                    "eval/avg_agent_collisions_per_episode": episode_metrics['avg_agent_collisions_per_episode'],
                    "eval/avg_obstacle_collisions_per_episode": episode_metrics['avg_obstacle_collisions_per_episode'],
                    "eval/formation_success_rate": episode_metrics['formation_success_rate'],
                    "eval/avg_time_to_formation": episode_metrics['avg_time_to_formation'],
                    "eval/avg_formation_error": episode_metrics['avg_formation_error'],
                    "eval/avg_best_formation_error": episode_metrics['avg_best_formation_error'],
                    "eval/episodes_evaluated": episode_metrics['episodes_evaluated'],
                }, step=0)
            
        # Save video if configured
        if cfg.eval.get('save_video', False) and hasattr(env_test, 'frames') and env_test.frames:
            try:
                from moviepy.editor import ImageSequenceClip
                import numpy as np
                
                video_dir = os.path.join(save_dir, 'videos')
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"eval_video_{mode}_{timestamp}.mp4")
                clip = ImageSequenceClip([np.array(frame) for frame in env_test.frames], fps=30)
                clip.write_videofile(video_path)
                print(f"Video saved to {video_path}")
            except ImportError:
                print("Could not save video. Make sure moviepy is installed.")

    if not env_test.is_closed:
        env_test.close()
    
    print(f"Evaluation complete for mode: {mode}!")
    print(f"Results saved in: {save_dir}")
    return episode_metrics


if __name__ == "__main__":
    evaluate()
