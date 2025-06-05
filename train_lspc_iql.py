import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from lspc_iql_configs import BC_DEFAULT_CONFIG, BCTrainConfig
from osrl.common.dataset import process_bc_dataset
from osrl.common.exp_util import auto_name, seed_all
from lspc_iql import ReplayBuffer, LSPC_IQL_Trainer, LSPC_IQL_kwargs


@pyrallis.wrap()
def train(args: BCTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix) + "_safe_iql" + f"_{args.task}"
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        # only tested on cuda
        pass

    # initialize environment
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym
    env = gym.make(args.task)

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)



    args_dict = vars(args)
    args_dict["state_dim"] = env.observation_space.shape[0]
    args_dict["action_dim"] = env.action_space.shape[0]
    iql_trainer_kwargs = LSPC_IQL_kwargs(args_dict)
    iql_trainer_kwargs["logger"] = logger
    iql_trainer_kwargs["episode_len"] = args.episode_len
    iql_trainer_kwargs["iql_tau"] = args.iql_tau
    iql_trainer_kwargs["tau"] = args.tau
    iql_trainer_kwargs["discount"] = args.discount
    iql_trainer_kwargs["exp_adv_max_cost"] = args.exp_adv_max_cost
    iql_trainer_kwargs["exp_adv_max_reward"] = args.exp_adv_max_reward

    iql_trainer_kwargs["beta_reward"] = args.beta_reward
    iql_trainer_kwargs["beta_cost"] = args.beta_cost
    iql_trainer_kwargs["safe_qc_vc_threshold"] = args.safe_qc_vc_threshold


    # model & optimizer setup
    iql_trainer = LSPC_IQL_Trainer(**iql_trainer_kwargs)

    print(f"Total actor parameters: {sum(p.numel() for p in iql_trainer.actor.parameters())}")
    print(f"Total qf parameters: {sum(p.numel() for p in iql_trainer.qf.parameters())}")
    print(f"Total vf parameters: {sum(p.numel() for p in iql_trainer.vf.parameters())}")

    def checkpoint_fn():
        return {"model_state": iql_trainer.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)



    total_dataset_size = len(data["observations"])
    buffer = ReplayBuffer(state_dim=env.observation_space.shape[0],
                            action_dim=env.action_space.shape[0],
                            buffer_size=total_dataset_size,
                            device=args.device)
    buffer.load_d4rl_dataset(data)
    logger.store(tab="train", dataset_size=total_dataset_size)


    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    # training
    for step in trange(args.max_timesteps, desc="Training"):
     

        observations, actions, rewards, costs, next_observations, done = buffer.sample(args.batch_size)
        iql_trainer.train_one_step(observations, next_observations, actions, rewards, costs, done)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.max_timesteps - 1 or step == 0:
            ret, cost, length, ret_std, cost_std, length_std = iql_trainer.evaluate(env, args.eval_episodes)
            logger.store(tab="eval", Cost=cost, Reward=ret, Length=length)
            logger.store(tab="eval", Cost_std=cost_std, Reward_std=ret_std, Length_std=length_std)
            normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
            logger.store(tab="normalized_eval", Normalized_Cost=normalized_cost, 
                                    Normalized_Reward=normalized_ret)

            safe_ret, safe_cost, safe_length, safe_ret_std, safe_cost_std, safe_length_std = iql_trainer.evaluate_safe(env, args.eval_episodes)
            logger.store(tab="eval_star", Cost=safe_cost, Reward=safe_ret, Length=safe_length)
            logger.store(tab="eval_star", Cost_std=safe_cost_std, Reward_std=safe_ret_std, Length_std=safe_length_std)
            normalized_safe_ret, normalized_safe_cost = env.get_normalized_score(safe_ret, safe_cost)
            logger.store(tab="normalized_eval_star", Normalized_Cost=normalized_safe_cost, 
                                            Normalized_Reward=normalized_safe_ret)

            logger.save_checkpoint()
            if (cost < best_cost) or (cost == best_cost and ret > best_reward):
                logger.save_checkpoint(suffix=f"best_{int(cost)}_{int(ret)}")
                best_cost = cost

            if (cost < 1.0 * args.cost_limit) and (ret > best_reward):
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write(step, display=False)


if __name__ == "__main__":
    train()
