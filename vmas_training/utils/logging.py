# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.record.loggers.wandb import WandbLogger


def init_logging(cfg, model_name: str, run_description: str = None, tags: list = None):
    # Create experiment name incorporating description if provided
    base_exp_name = generate_exp_name(cfg.env.scenario_name, model_name)
    
    if run_description:
        # Option 1: Use description as the main experiment name
        experiment_name = run_description
        # Option 2: Combine with generated name (uncomment if preferred)
        # experiment_name = f"{base_exp_name}_{run_description}"
    else:
        experiment_name = base_exp_name
    
    # Prepare wandb kwargs with enhanced naming and metadata
    wandb_kwargs = {
        "group": cfg.logger.group_name or model_name,
        "project": cfg.logger.project_name or f"torchrl_example_{cfg.env.scenario_name}",
        "name": experiment_name,  # This becomes the run name in WandB
        "tags": tags or [],       # Tags for filtering/organizing runs
    }
    
    # Add notes/description if provided
    if run_description:
        wandb_kwargs["notes"] = run_description
    
    logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=os.getcwd(),
        experiment_name=experiment_name,
        wandb_kwargs=wandb_kwargs,
    )
    
    # Log hyperparameters including run metadata
    logger.log_hparams(cfg)
    
    # If using WandB, add run description to summary for easy access
    if cfg.logger.backend == "wandb" and hasattr(logger, 'experiment'):
        if run_description:
            logger.experiment.summary["run_description"] = run_description
        if tags:
            logger.experiment.summary["run_tags"] = tags
    
    return logger

def log_training(
    logger: Logger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int,
    total_frames: int,
    step: int,
):
    if ("next", "agents", "reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "reward"),
            sampling_td.get(("next", "reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )
    if ("next", "agents", "episode_reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "episode_reward"),
            sampling_td.get(("next", "episode_reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )

    metrics_to_log = {
        f"train/learner/{key}": value.mean().item()
        for key, value in training_td.items()
    }

    if "info" in sampling_td.get("agents").keys():
        metrics_to_log.update(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td.get(("agents", "info")).items()
            }
        )

    reward = sampling_td.get(("next", "agents", "reward")).mean(-2)  # Mean over agents
    done = sampling_td.get(("next", "done"))
    if done.ndim > reward.ndim:
        done = done[..., 0, :]  # Remove expanded agent dim
    episode_reward = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[
        done
    ]
    metrics_to_log.update(
        {
            "train/reward/reward_min": reward.min().item(),
            "train/reward/reward_mean": reward.mean().item(),
            "train/reward/reward_max": reward.max().item(),
            "train/reward/episode_reward_min": episode_reward.min().item(),
            "train/reward/episode_reward_mean": episode_reward.mean().item(),
            "train/reward/episode_reward_max": episode_reward.max().item(),
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        }
    )
    if isinstance(logger, WandbLogger):
        logger.experiment.log(metrics_to_log, commit=False)
    else:
        for key, value in metrics_to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)

    return metrics_to_log


def log_evaluation(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    evaluation_time: float,
    step: int,
):
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]

    rewards = [td.get(("next", "agents", "reward")).sum(0).mean() for td in rollouts]
    metrics_to_log = {
        "eval/episode_reward_min": min(rewards),
        "eval/episode_reward_max": max(rewards),
        "eval/episode_reward_mean": sum(rewards) / len(rollouts),
        "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
        / len(rollouts),
        "eval/evaluation_time": evaluation_time,
    }

    vid = torch.tensor(
        np.transpose(env_test.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
        dtype=torch.uint8,
    ).unsqueeze(0)

    if isinstance(logger, WandbLogger):
        import wandb

        logger.experiment.log(metrics_to_log, commit=False)
        logger.experiment.log(
            {
                "eval/video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4"),
            },
            commit=False,
        )
    else:
        for key, value in metrics_to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)
        logger.log_video("eval_video", vid, step=step)
