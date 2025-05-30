# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
import os
import hydra
import torch
from datetime import datetime

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform
# from models.gat_actor import GATActor, ObservationConfig
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

def save_model_snapshot(policy, value_module, iteration, metrics=None):
    """
    Save a snapshot of the model parameters.
    
    Args:
        policy: The policy network
        value_module: The value network
        save_dir: Directory to save the snapshot
        iteration: Current training iteration
        metrics: Optional evaluation metrics to include in the filename
    """
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'snapshots')
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a checkpoint dictionary with both policy and value module states
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_module.state_dict(),
        'iteration': iteration,
    }
    
    # Add metrics if provided
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Generate filename with metrics if available
    if metrics is not None and 'mean_reward' in metrics:
        mean_reward = metrics['mean_reward']
        filename = f"model_snapshot_iter_{iteration}_reward_{mean_reward:.2f}.pt"
    else:
        filename = f"model_snapshot_iter_{iteration}.pt"
    
    # Save the model
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    
    torchrl_logger.info(f"Model snapshot saved to {save_path}")
    
    return save_path

def monitor_action_distribution(tensordict_data, iteration):
    """Monitor action distribution to ensure proper exploration"""
    actions = tensordict_data["agents"]["action"]
    
    # Flatten actions across all agents and environments
    actions_flat = actions.reshape(-1, actions.shape[-1])
    
    action_stats = {
        'action_mean': actions_flat.mean(dim=0),
        'action_std': actions_flat.std(dim=0),
        'action_min': actions_flat.min(dim=0)[0],
        'action_max': actions_flat.max(dim=0)[0],
        'action_range': actions_flat.max(dim=0)[0] - actions_flat.min(dim=0)[0]
    }
    
    print(f"\nIteration {iteration} Action Statistics:")
    print(f"  Mean: {action_stats['action_mean']}")
    print(f"  Std:  {action_stats['action_std']}")
    print(f"  Min:  {action_stats['action_min']}")
    print(f"  Max:  {action_stats['action_max']}")
    print(f"  Range: {action_stats['action_range']}")
    
    # Check if actions are stuck at boundaries
    boundary_threshold = 0.95
    high_boundary_ratio = (actions_flat > boundary_threshold).float().mean()
    low_boundary_ratio = (actions_flat < -boundary_threshold).float().mean()
    
    print(f"  Actions near +1: {high_boundary_ratio:.3f}")
    print(f"  Actions near -1: {low_boundary_ratio:.3f}")

@hydra.main(version_base="1.1", config_path="", config_name="mappo_gat")
def train(cfg: "DictConfig"):  # noqa: F821
    # Handle run description - priority: CLI override > env var > default
    run_description = cfg.get("run_description") or os.getenv("RUN_DESCRIPTION") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_tags = cfg.get("tags", [])
    
    # Add to config for complete tracking
    cfg.run_description = run_description
    cfg.tags = run_tags



    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    cfg.snapshots = cfg.get("snapshots", {})
    cfg.snapshots.save_dir = cfg.snapshots.get("save_dir", "model_snapshots")
    cfg.snapshots.save_best_only = cfg.snapshots.get("save_best_only", False)

    # Create env and env_test
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

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

    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    n_attention_heads = cfg.model.get("n_attention_heads", 4)  # New parameter for GAT
    dropout = cfg.model.get("dropout", 0.0)  # Optional dropout for attention
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
            total_obs_dim=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
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

    lowest_action = env.full_action_spec_unbatched[("agents", "action")].space.low.to(cfg.train.device)
    highest_action = env.full_action_spec_unbatched[("agents", "action")].space.high.to(cfg.train.device)
    print(f"Got action spec lowest {lowest_action} highest {highest_action}")

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": highest_action
        },
        return_log_prob=True,
    )

    critic_module = GNNCritic(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        gnn_hidden_dim=gnn_hidden_dim, # Use same GNN params as actor (can be configured separately if needed)
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=None,
        pos_indices=agent_pos_indices,
        share_params=cfg.model.shared_parameters, # Kept for consistency, GNN shares anyway
        device=cfg.train.device, # Pass device object
    )
    value_module = ValueOperator(
        module=critic_module,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")]
    )

    collector = SyncDataCollector(
        env,
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        policy_device=cfg.env.device,
        env_device=cfg.env.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=True,
        clip_value=True,
        value_target_clipping=0.2,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        value=("agents", "state_value")
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optim, 
        start_factor=1.0, 
        end_factor=0.1, 
        total_iters=cfg.collector.n_iters
    )

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
        logger = init_logging(cfg, model_name, run_description, run_tags)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start

        # Extract episode metrics
        with torch.no_grad():
            training_episode_metrics = extract_episode_metrics(tensordict_data, env)

        monitor_action_distribution(tensordict_data, i)

        # Log training episode metrics
        if cfg.logger.backend:
            if cfg.logger.backend == "wandb":
                logger.experiment.log({
                    # Training episode-specific metrics
                    "train/total_agent_collisions": training_episode_metrics['total_agent_collisions'],
                    "train/total_obstacle_collisions": training_episode_metrics['total_obstacle_collisions'],
                    "train/avg_agent_collisions_per_episode": training_episode_metrics['avg_agent_collisions_per_episode'],
                    "train/avg_obstacle_collisions_per_episode": training_episode_metrics['avg_obstacle_collisions_per_episode'],
                    "train/formation_success_rate": training_episode_metrics['formation_success_rate'],
                    "train/avg_time_to_formation": training_episode_metrics['avg_time_to_formation'],
                    "train/episodes_evaluated": training_episode_metrics['episodes_evaluated'],
                    "train/avg_formation_error": training_episode_metrics['avg_formation_error'],
                    "train/avg_best_formation_error": training_episode_metrics['avg_best_formation_error'],
                }, step=i)
            
            # Print training metrics
            torchrl_logger.info(
                f"Training Metrics (Iteration {i}):\n"
                f"  Formation Success Rate: {training_episode_metrics['formation_success_rate']:.3f}\n"
                f"  Avg Time to Formation: {training_episode_metrics['avg_time_to_formation']:.1f} steps\n"
                f"  Total Agent Collisions: {training_episode_metrics['total_agent_collisions']}\n"
                f"  Total Obstacle Collisions: {training_episode_metrics['total_obstacle_collisions']}\n"
                f"  Avg Collisions per Episode: Agent={training_episode_metrics['avg_agent_collisions_per_episode']:.2f}, "
                f"Obstacle={training_episode_metrics['avg_obstacle_collisions_per_episode']:.2f}"
            )


        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )

                # Add separate clipping for actor and critic
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.actor_network.parameters(), cfg.train.max_grad_norm * 0.5
                )
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.critic_network.parameters(), cfg.train.max_grad_norm * 0.5
                )

                training_tds[-1].set("grad_norm", total_norm.mean())
                training_tds[-1].set("actor_grad_norm", actor_norm.mean())
                training_tds[-1].set("critic_grad_norm", critic_norm.mean())

                optim.step()
                optim.zero_grad()
                scheduler.step()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                episode_metrics = extract_episode_metrics(rollouts, env_test)
                eval_metrics = log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

                # Log detailed episode metrics
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
                    }, step=i)
                
                # Print detailed metrics
                torchrl_logger.info(
                    f"Evaluation Metrics (Iteration {i}):\n"
                    f"  Formation Success Rate: {episode_metrics['formation_success_rate']:.3f}\n"
                    f"  Avg Time to Formation: {episode_metrics['avg_time_to_formation']:.1f} steps\n"
                    f"  Total Agent Collisions: {episode_metrics['total_agent_collisions']}\n"
                    f"  Total Obstacle Collisions: {episode_metrics['total_obstacle_collisions']}\n"
                    f"  Avg Collisions per Episode: Agent={episode_metrics['avg_agent_collisions_per_episode']:.2f}, "
                    f"Obstacle={episode_metrics['avg_obstacle_collisions_per_episode']:.2f}"
                )

                # SAVE SNAPSHOT 
                # Extract mean reward for model saving decision
                # Assuming log_evaluation returns or modifies eval_metrics with this info
                # If not, you'll need to calculate it here from rollouts
                
                # Decide whether to save based on save_best_only flag
                should_save = True
                
                if should_save:
                    save_model_snapshot(
                        policy=policy,
                        value_module=value_module,
                        iteration=i,
                    )


        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()
