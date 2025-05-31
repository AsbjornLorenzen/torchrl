# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from utils.logging import init_logging, log_evaluation

from utils.logging import init_logging, log_evaluation
from utils.utils import DoneTransform
from models.gnn_actor import GNNActor
from models.pgat_actor import PGATActor, ObservationConfig
from models.gnn_critic import GNNCritic


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))
    

def load_model(model_path, model):
    """Load parameters from a saved model file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')['policy_state_dict']
    model.load_state_dict(state_dict)
    print(f"Successfully loaded model from {model_path}")
    return model

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
        'avg_best_formation_error': 0  # New metric
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

        metrics['avg_formation_error'] = episode_metrics['formation_accuracy_per_step'].mean().item()

        formation_accuracy = episode_metrics['formation_accuracy_per_step']  # Shape: [batch_size, max_steps, ...]
        # Take max across time steps for each batch index, then mean across batch
        max_accuracies_per_batch = formation_accuracy.max(dim=1)[0]  # Max across time dimension
        metrics['avg_best_formation_error'] = max_accuracies_per_batch.mean().item()

        
    except Exception as e:
        print(f"Warning: Could not extract episode metrics: {e}")
        # Return default metrics if extraction fails
        pass
    
    return metrics

# 3. ADD DETAILED LOGGING AFTER EVALUATION (replace existing logging section)
def add_detailed_logging_to_eval():
    """
    Add this section after the rollout generation in the evaluate() function:
    """
    
    # Extract detailed episode metrics
    episode_metrics = extract_episode_metrics(rollouts, env_test)
    
    # Log to wandb if backend is configured
    if cfg.logger.backend == "wandb":
        logger.experiment.log({
            # Episode-specific metrics
            "eval/total_agent_collisions": episode_metrics['total_agent_collisions'],
            "eval/total_obstacle_collisions": episode_metrics['total_obstacle_collisions'],
            "eval/avg_agent_collisions_per_episode": episode_metrics['avg_agent_collisions_per_episode'],
            "eval/avg_obstacle_collisions_per_episode": episode_metrics['avg_obstacle_collisions_per_episode'],
            "eval/formation_success_rate": episode_metrics['formation_success_rate'],
            "eval/avg_time_to_formation": episode_metrics['avg_time_to_formation'],
            "eval/episodes_evaluated": episode_metrics['episodes_evaluated'],
            "eval/avg_formation_error": episode_metrics['avg_formation_error'],
            "eval/avg_best_formation_error": episode_metrics['avg_best_formation_error'],
            # Basic reward metrics (existing)
            "eval/mean_reward": rewards.mean().item() if rewards is not None else 0,
            "eval/reward_std": rewards.std().item() if rewards is not None else 0,
            "eval/min_reward": rewards.min().item() if rewards is not None else 0,
            "eval/max_reward": rewards.max().item() if rewards is not None else 0,
        }, step=0)
    
    # Print detailed metrics to console
    print(f"\nDetailed Evaluation Metrics:")
    print(f"  Formation Success Rate: {episode_metrics['formation_success_rate']:.3f}")
    print(f"  Avg Time to Formation: {episode_metrics['avg_time_to_formation']:.1f} steps")
    print(f"  Total Agent Collisions: {episode_metrics['total_agent_collisions']}")
    print(f"  Total Obstacle Collisions: {episode_metrics['total_obstacle_collisions']}")
    print(f"  Avg Collisions per Episode: Agent={episode_metrics['avg_agent_collisions_per_episode']:.2f}, "
          f"Obstacle={episode_metrics['avg_obstacle_collisions_per_episode']:.2f}")
    print(f"  Avg Formation Error: {episode_metrics['avg_formation_error']:.4f}")
    print(f"  Avg Best Formation Error: {episode_metrics['avg_best_formation_error']:.4f}")


# 4. ADD CONFIGURATION HANDLING FOR RUN DESCRIPTION AND TAGS
def add_run_config_handling():
    """
    Add this at the beginning of the evaluate() function after hydra.main decorator:
    """
    pass

# 5. MODIFY LOGGER INITIALIZATION
def modify_logger_init():
    """
    Replace the existing logger initialization section with:
    """
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO_Eval"
        )
        logger = init_logging(cfg, model_name, cfg.run_description, cfg.tags)  # Add run_description and tags


@hydra.main(version_base="1.1", config_path="", config_name="eval_mappo_gat")
def evaluate(cfg: DictConfig):
    # Add model_dir to config if not present
    if not hasattr(cfg, 'model_dir'):
        cfg.model_dir = ""  # Default empty string
        
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Handle run description - priority: CLI override > env var > default
    from datetime import datetime
    
    run_description = cfg.get("run_description") or os.getenv("RUN_DESCRIPTION") or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_tags = cfg.get("tags", [])
    
    # Add to config for complete tracking
    cfg.run_description = run_description
    cfg.tags = run_tags

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

    print(f"In torchrl, the given env action spec is {env_test.action_spec}")
    
    # GNN POLICY
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    n_attention_heads = cfg.model.get("n_attention_heads", 4)  # New parameter for GAT
    dropout = cfg.model.get("dropout", 0.0)  # Optional dropout for attention
    k_neighbors = cfg.model.get("k_neighbors", 2)
    k_obstacles = cfg.model.get("k_obstacles", 2)
    agent_pos_indices_list = cfg.model.get("agent_pos_indices", [0, 2])
    agent_pos_indices = slice(agent_pos_indices_list[0], agent_pos_indices_list[1])


    observation_config = ObservationConfig(k_neighbors=k_neighbors, k_obstacles=k_obstacles)
    
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
    ).to(cfg.train.device)
    
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    
    lowest_action = torch.zeros_like(
        env_test.full_action_spec_unbatched[("agents", "action")].space.low, 
        device=cfg.train.device
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env_test.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env_test.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": env_test.full_action_spec_unbatched[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

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
    if cfg.model_dir:
        actor_path = os.path.join(cfg.model_dir, "model_snapshot_iter_180.pt")
        critic_path = os.path.join(cfg.model_dir, "critic.pth")
        
        if os.path.exists(actor_path):
            policy = load_model(actor_path, policy)
            print(f"Loaded actor model at {actor_path}")
        else:
            print(f"Warning: Actor model not found at {actor_path}")
            
        if os.path.exists(critic_path):
            value_module = load_model(critic_path, value_module)
        else:
            print(f"Warning: Critic model not found at {critic_path}")
    else:
        print("Warning: No model_dir specified. Using untrained models.")

    # Logging setup
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO_Eval"
        )
        logger = init_logging(cfg, model_name, cfg.run_description, cfg.tags)  # Add run_description and tags



    # Evaluation
    print(f"Starting evaluation with {cfg.eval.evaluation_episodes} episodes")
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        env_test.frames = []
        rollouts = env_test.rollout(
            max_steps=cfg.env.max_steps,
            policy=policy,
            callback=rendering_callback if cfg.eval.render else None,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )

        # Calculate evaluation metrics
        rewards = rollouts.get(("agents", "episode_reward"), None)
        if rewards is not None:
            print(f"Average reward: {rewards.mean().item():.4f}")
            print(f"Reward std: {rewards.std().item():.4f}")
            print(f"Min reward: {rewards.min().item():.4f}")
            print(f"Max reward: {rewards.max().item():.4f}")
        
        # Extract episode metrics
        with torch.no_grad():
            training_episode_metrics = extract_episode_metrics(rollouts, env_test)

        add_detailed_logging_to_eval()

        # Log evaluation results
        if cfg.logger.backend:
            log_evaluation(logger, rollouts, env_test, 0, step=0)

            
        # Save video if configured
        if hasattr(cfg.eval, 'save_video') and cfg.eval.save_video and hasattr(env_test, 'frames') and env_test.frames:
            try:
                from moviepy.editor import ImageSequenceClip
                import numpy as np
                
                video_path = os.path.join(os.getcwd(), "eval_video.mp4")
                clip = ImageSequenceClip([np.array(frame) for frame in env_test.frames], fps=30)
                clip.write_videofile(video_path)
                print(f"Video saved to {video_path}")
            except ImportError:
                print("Could not save video. Make sure moviepy is installed.")

    if cfg.logger.backend == "wandb":
        logger.experiment.log({}, commit=True)
    if not env_test.is_closed:
        env_test.close()
    
    print("Evaluation complete!")


if __name__ == "__main__":
    evaluate()
