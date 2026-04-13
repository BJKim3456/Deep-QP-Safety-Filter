"""
This code implements safe RL using PPO with a learned Deep QP Safety Filter.
The safety filter is regarded as part of the environment, so that the RL loop
becomes an MDP over the filtered actions.
"""

import argparse
import datetime
import socket
from pathlib import Path
import random

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from qpsolvers import solve_qp

from SafetyModule.SafetyCritic import SafetyCritic
from envs.gym_env import Hopper, InvertedDoublePendulumPositionBonus, InvertedDoubleMovingBonus


# ============================================================================================
# === argparse / setup
# ============================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with a learned Deep QP Safety Filter."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="hopper",
        choices=[
            "hopper",
            "inverted_double_pendulum_position_bonus",
            "inverted_double_moving_bonus",
        ],
    )
    parser.add_argument("--control-dt", type=float, default=5e-3)
    parser.add_argument("--task-time", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--baseline-type", type=str, default="PPO_Deep_QP_SafetyFilter")
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--safety-alpha", type=float, default=2.0)

    parser.add_argument("--num-rollout-steps", type=int, default=8192)
    parser.add_argument("--max-training-timesteps", type=int, default=int(10e6))
    parser.add_argument("--save-interval-mult", type=int, default=10)

    parser.add_argument("--output-dir", type=str, default="outputs/safe_rl")
    parser.add_argument("--safety-filter-path", type=str, default=None)

    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env_and_filter_info(env_name: str, dt: float, task_time: float, seed: int):
    if env_name == "hopper":
        env = Hopper(dt_=dt, T_max=task_time, seed=seed)
        default_filter_path = "./pretrained/Hopper/deep_qp_safety_filter_model"
        filter_hidden_layers = 3
        ppo_hidden_layers = 3
    elif env_name == "inverted_double_pendulum_position_bonus":
        env = InvertedDoublePendulumPositionBonus(dt_=dt, T_max=task_time, seed=seed)
        default_filter_path = "./pretrained/2D_Inverted_Pendulum/deep_qp_safety_filter_model"
        filter_hidden_layers = 2
        ppo_hidden_layers = 2
    elif env_name == "inverted_double_moving_bonus":
        env = InvertedDoubleMovingBonus(dt_=dt, T_max=task_time, seed=seed)
        default_filter_path = "./pretrained/2D_Inverted_Pendulum/deep_qp_safety_filter_model"
        filter_hidden_layers = 2
        ppo_hidden_layers = 2
    else:
        raise ValueError(f"Unknown env: {env_name}")

    return env, default_filter_path, filter_hidden_layers, ppo_hidden_layers


# ============================================================================================
# === hyperparameters
# ============================================================================================

class Hyperparameters:
    def __init__(self, args, device):
        self.time_interval = args.control_dt
        self.task_time = args.task_time
        self.max_episode_steps = int(self.task_time / self.time_interval)
        self.seed = args.seed

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.97
        self.ppo_epochs = 10
        self.num_rollout_steps = args.num_rollout_steps
        self.minibatch_size = 256
        self.clip_epsilon = 0.2
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4

        self.max_training_timesteps = args.max_training_timesteps
        self.save_interval = self.num_rollout_steps * args.save_interval_mult

        self.log_std_init = -0.5
        self.target_kl = 0.02
        self.device = device

        self.safety_margin = args.safety_margin
        self.safety_alpha = args.safety_alpha

        # PPO network width stays the same; depth changes by env
        self.hidden_dim = 64


# ============================================================================================
# === QP filter
# ============================================================================================

class QP_filter:
    def __init__(self, action_dim):
        self.ub_ = np.ones(action_dim)
        self.lb_ = -np.ones(action_dim)
        self.qp_h_col = np.ones(1)
        self.qp_G_mtx = np.ones((1, action_dim))
        self.p_ = np.identity(action_dim)

    def GetFilteredAction(
        self,
        coeff_: np.ndarray,
        scalar: np.ndarray,
        value_: np.ndarray,
        action_: np.ndarray,
        SafetyMargin=0.00,
        alpha_=1.0,
        P: np.ndarray = None,
    ):
        maximal_safest_input = np.sign(coeff_)
        alpha_v = alpha_ * (value_ - SafetyMargin)
        self.p_ = P if P is not None else np.identity(coeff_.shape[0])

        # numerically stable version of the following if-else:
        # if (-scalar - alpha_v <= np.sum(np.abs(coeff_))): # which means QP is feasible
        #    beta = -scalar - alpha_v # and solve QP
        # else: # which means QP is infeasible, and we need to return the estimated maximal safest input
        #    beta = np.sum(np.abs(coeff_)) # and solve QP, which is identical to max_{u \in U} coeff_ @ u
        
        beta = np.minimum(-scalar - alpha_v, (1.0 - 1e-6) * np.sum(np.abs(coeff_)))
        self.qp_q_col = -self.p_ @ action_
        self.qp_h_col = beta
        self.qp_G_mtx[0, :] = coeff_

        sol = solve_qp(
            P=self.p_,
            q=self.qp_q_col,
            G=-self.qp_G_mtx,
            h=-self.qp_h_col,
            A=None,
            b=None,
            lb=self.lb_,
            ub=self.ub_,
            solver="proxqp",
            max_iter=20000,
            eps_abs=1e-8,
            eps_rel=1e-8,
        )
        filtered_action_ = sol if sol is not None else maximal_safest_input
        return np.clip(filtered_action_, -1, 1)


# ============================================================================================
# === utils
# ============================================================================================

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_mlp(input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int = None):
    layers = []
    prev_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# ============================================================================================
# === PPO buffer
# ============================================================================================

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)

        self.ret_r_buf = np.zeros(size, dtype=np.float32)
        self.adv_r_buf = np.zeros(size, dtype=np.float32)

        self.val_r_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, a_squash, rew, val_r, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = a_squash
        self.rew_buf[self.ptr] = rew
        self.val_r_buf[self.ptr] = val_r
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val_r=0.0):
        sl = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[sl], last_val_r)
        vals_r = np.append(self.val_r_buf[sl], last_val_r)
        deltas_r = rews[:-1] + self.gamma * vals_r[1:] - vals_r[:-1]
        self.adv_r_buf[sl] = self._discount_cumsum(deltas_r, self.gamma * self.lam)
        self.ret_r_buf[sl] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, device, dtype):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_r_mean, adv_r_std = np.mean(self.adv_r_buf), np.std(self.adv_r_buf)
        self.adv_r_buf = (self.adv_r_buf - adv_r_mean) / (adv_r_std + 1e-8)

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret_r=self.ret_r_buf,
            adv_r=self.adv_r_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in data.items()}


# ============================================================================================
# === networks
# ============================================================================================

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_hidden_layers=3, log_std_init=-0.5, dtype=torch.float32):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dim, num_hidden_layers)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim, dtype=dtype) * log_std_init)

        self.apply(weights_init_)
        nn.init.zeros_(self.mu_head.bias)

    def _dist_params(self, state):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self._dist_params(state)
        dist = Normal(mu, std)
        a = dist.rsample()
        logp = dist.log_prob(a).sum(dim=-1)
        return a, logp, mu

    def log_prob(self, s, a):
        mu, std = self._dist_params(s)
        return Normal(mu, std).log_prob(a).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, num_hidden_layers=3):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dim, num_hidden_layers, output_dim=1)
        self.apply(weights_init_)

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ============================================================================================
# === PPO Agent
# ============================================================================================

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_space,
        hp,
        safety_critic: SafetyCritic,
        qp_filter: QP_filter,
        ppo_hidden_layers: int,
        dtype: torch.dtype,
    ):
        self.hp = hp
        self.dtype = dtype

        self.buffer = PPOBuffer(
            state_dim,
            action_space.shape[0],
            hp.num_rollout_steps,
            hp.gamma,
            hp.gae_lambda,
        )
        self.safety_critic = safety_critic
        self.qp_filter = qp_filter

        self.actor = Actor(
            state_dim,
            action_space.shape[0],
            hidden_dim=hp.hidden_dim,
            num_hidden_layers=ppo_hidden_layers,
            log_std_init=hp.log_std_init,
            dtype=dtype,
        ).to(hp.device)

        self.critic = Critic(
            state_dim,
            hidden_dim=hp.hidden_dim,
            num_hidden_layers=ppo_hidden_layers,
        ).to(hp.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=hp.lr_actor)
        self.optimizer_critic_r = torch.optim.Adam(self.critic.parameters(), lr=hp.lr_critic)

    @torch.no_grad()
    def select_action(self, state, constraint=None):
        s = torch.as_tensor(state, dtype=self.dtype, device=self.hp.device)
        a_raw, logp, _ = self.actor.sample(s)
        a_raw_clipped = torch.clamp(a_raw, -1.0, 1.0)

        if constraint is not None:
            state_tensor = torch.tensor(state, dtype=self.dtype, device=self.hp.device)
            value_, coeff_, scalar_ = self.safety_critic.GetValues(
                state_tensor,
                constraint,
                UseTarget=True,
            )
            a_env = self.qp_filter.GetFilteredAction(
                coeff_,
                scalar_,
                value_,
                a_raw_clipped.cpu().numpy(),
                SafetyMargin=self.hp.safety_margin,
                alpha_=self.hp.safety_alpha,
            )
        else:
            a_env = a_raw_clipped.cpu().numpy()

        val_r = self.critic(s)
        return a_env, a_raw.cpu().numpy(), val_r.item(), logp.item()

    def update(self):
        data = self.buffer.get(self.hp.device, self.dtype)
        obs, a_squash, ret_r, adv, logp_old = (
            data["obs"],
            data["act"],
            data["ret_r"],
            data["adv_r"],
            data["logp"],
        )

        for _ in range(self.hp.ppo_epochs):
            kl_exceeded = False
            last_approx_kl = 0.0
            idx = np.random.permutation(self.hp.num_rollout_steps)

            for start in range(0, self.hp.num_rollout_steps, self.hp.minibatch_size):
                batch = idx[start:start + self.hp.minibatch_size]
                b_obs = obs[batch]
                b_act_squash = a_squash[batch]
                b_adv = adv[batch]
                b_ret_r = ret_r[batch]
                b_logp_old = logp_old[batch]

                logp_new = self.actor.log_prob(b_obs, b_act_squash)
                ratio = torch.exp(logp_new - b_logp_old)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.hp.clip_epsilon,
                    1.0 + self.hp.clip_epsilon,
                ) * b_adv
                loss_pi = -torch.min(surr1, surr2).mean()

                self.optimizer_actor.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.optimizer_actor.step()

                loss_v_r = F.mse_loss(self.critic(b_obs), b_ret_r)
                self.optimizer_critic_r.zero_grad()
                loss_v_r.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.optimizer_critic_r.step()

                with torch.no_grad():
                    logp_new_post = self.actor.log_prob(b_obs, b_act_squash)
                    approx_kl = (b_logp_old - logp_new_post).mean().item()
                    last_approx_kl = approx_kl

                if approx_kl > self.hp.target_kl:
                    kl_exceeded = True
                    break

            if kl_exceeded:
                break

        return float(loss_v_r.item()), float(loss_pi.item()), float(last_approx_kl)


# ============================================================================================
# === main
# ============================================================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    set_seed(args.seed)

    hp = Hyperparameters(args, device)

    env, default_filter_path, filter_hidden_layers, ppo_hidden_layers = make_env_and_filter_info(
        args.env,
        hp.time_interval,
        hp.task_time,
        hp.seed,
    )

    filter_path = args.safety_filter_path or default_filter_path

    env_name_log = env.name + "/" + args.baseline_type
    run_id = f"{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}_{args.baseline_type}"
    run_dir = Path(args.output_dir) / env.name / run_id
    tb_dir = run_dir / "tensorboard"
    model_dir = run_dir / "checkpoints"
    tb_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[{args.baseline_type}] Training on {env_name_log}, Logging to {tb_dir}")

    state_dim = env.env_.observation_space.shape[0]
    action_dim = env.env_.action_space.shape[0]

    args_SF = {
        "lambda_init": 0.0,
        "lambda_final": 0.0,
        "hidden_size": 256,
        "hidden_layers": filter_hidden_layers,
        "cuda": device,
        "dtype": dtype,
    }

    safety_critic_ = SafetyCritic(state_dim, action_dim, args_SF, eval_=True)
    safety_critic_.NNload(filter_path)
    qp_filter_ = QP_filter(action_dim)

    agent = PPOAgent(
        state_dim,
        env.env_.action_space,
        hp,
        safety_critic_,
        qp_filter_,
        ppo_hidden_layers=ppo_hidden_layers,
        dtype=dtype,
    )

    state, constraint = env.reset()
    total_time_steps = 0
    i_episode = 0
    ep_ret, ep_len = 0.0, 0

    while total_time_steps < hp.max_training_timesteps:
        for t in range(hp.num_rollout_steps):
            a_env, a_raw, v_r, logp = agent.select_action(state, constraint)
            next_state, task_reward, next_constraint, terminated, truncated, violate_info = env.step(a_env)

            total_time_steps += 1
            ep_len += 1
            ep_ret += task_reward

            agent.buffer.store(state, a_raw, task_reward, v_r, logp)

            state = next_state
            constraint = next_constraint

            done = terminated or truncated
            epoch_ended = (t == hp.num_rollout_steps - 1)

            if done or epoch_ended:
                if epoch_ended and not terminated:
                    _, _, last_val_r, _ = agent.select_action(state)
                else:
                    last_val_r = 0.0

                agent.buffer.finish_path(last_val_r)

                if done:
                    writer.add_scalar(f"{env_name_log}/Rollout/Episode_Reward", ep_ret, i_episode)
                    writer.add_scalar(f"{env_name_log}/Rollout/Episode_Length", ep_len, i_episode)

                    msg = f"Ep: {i_episode} | Steps: {total_time_steps} | EpLen: {ep_len} | EpRet: {ep_ret:.2f}"
                    if terminated:
                        violation_info_str = ""
                        if isinstance(violate_info, dict):
                            violation_info_str = ", ".join([k for k, v in violate_info.items() if v])
                        print(f"[Failed]    {msg} | Violations: {violation_info_str}")
                    else:
                        print(f"[Truncated] {msg}")

                    state, constraint = env.reset()
                    ep_ret, ep_len = 0.0, 0
                    i_episode += 1

        v_loss, pi_loss, kl = agent.update()

        writer.add_scalar(f"{env_name_log}/Loss/Critic_Reward", v_loss, total_time_steps)
        writer.add_scalar(f"{env_name_log}/Loss/Actor", pi_loss, total_time_steps)
        writer.add_scalar(f"{env_name_log}/Info/KL_Divergence", kl, total_time_steps)

        if (total_time_steps // hp.num_rollout_steps) > 0 and \
           (total_time_steps // hp.num_rollout_steps) % (hp.save_interval // hp.num_rollout_steps) == 0:
            save_path = model_dir / f"model_{total_time_steps}.pth"
            torch.save(
                {
                    "actor_state_dict": agent.actor.state_dict(),
                    "critic_r_state_dict": agent.critic.state_dict(),
                },
                str(save_path),
            )
            print(f"Model saved at timestep {total_time_steps} to {save_path}")

    if hasattr(env, "close"):
        env.close()
    elif hasattr(env, "env_"):
        env.env_.close()

    writer.close()
    print(f"Training finished for {args.baseline_type}.")


if __name__ == "__main__":
    main()