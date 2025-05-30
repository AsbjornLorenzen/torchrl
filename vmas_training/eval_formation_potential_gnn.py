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
from utils.utils import DoneTransform
from models.gnn_actor import GNNActor
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


@hydra.main(version_base="1.1", config_path="", config_name="eval_mappo_gat")
def evaluate(cfg: DictConfig):
    # Add model_dir to config if not present
    if not hasattr(cfg, 'model_dir'):
        cfg.model_dir = ""  # Default empty string
        
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

    print(f"In torchrl, the given env action spec is {env_test.action_spec}")
    
    # GNN POLICY
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None)  # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2])  # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])
    
    actor_net = nn.Sequential(
        GNNActor(
            n_agent_inputs=env_test.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env_test.action_spec.shape[-1],
            gnn_hidden_dim=gnn_hidden_dim,
            n_gnn_layers=gnn_layers,
            activation_class=nn.Tanh,
            k_neighbours=k_neighbours,
            pos_indices=pos_indices,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
        ),
        NormalParamExtractor(),
    )
    
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
        pos_indices=pos_indices,
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
        logger = init_logging(cfg, model_name)

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

    if not env_test.is_closed:
        env_test.close()
    
    print("Evaluation complete!")


if __name__ == "__main__":
    evaluate()
