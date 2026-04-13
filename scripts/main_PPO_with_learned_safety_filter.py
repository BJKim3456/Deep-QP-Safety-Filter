"""
This code implements a safe RL(using PPO) with learned safety filter.
The safety filter is regarded as part of the environment, so that the RL learning framework becomes a MDP. 
for the core logic only, we give the following pseudo-code:

get the current state s and the safety constraint c from the environment
for each u_rl from the RL policy:
    get the safety critic's value, coeff, scalar for (s, c)
    get the filtered action u_env by solving the QP with (value, coeff, scalar, u_rl)
    execute u_env in the environment, and get the next state s', reward r, next constraint c', and done signal d
    use (s, u_rl, r, s', c', d) as a transition data for the RL algorithm.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from SafetyModule.SafetyCritic import SafetyCritic

from qpsolvers import solve_qp
import numpy as np
import os
import datetime
import socket
import random
import scipy.signal

from envs.gym_env import Hopper, InvertedDoublePendulumPositionBonus, InvertedDoubleMovingBonus

# ============================================================================================
# === hyperparameters
# ============================================================================================

BASELINE_TYPE = "PPO_Deep_QP_SafetyFilter" 
SAFETY_MARGIN = 0.2
SAFETY_ALPHA = 2

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

current_datetime = datetime.datetime.now()
date_string = current_datetime.strftime("%b%d_%H-%M-%S")
hostname = socket.gethostname()
result_string = f"{date_string}_{hostname}_{BASELINE_TYPE}"

log_dir = "./log/safe_RL/"
log_filename = "baselines/"
network_dir = "./checkpoints/"+log_filename

TIME_INTERVAL = 5e-3
TASK_TIME = 10
MAX_TIME_STEP = int(TASK_TIME / TIME_INTERVAL)
DEVICE_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE_ = torch.float32

class QP_filter:
    def __init__(self, action_dim):

        self.u_max = np.ones(action_dim)
        self.ub_ = np.ones(action_dim)
        self.lb_ = -np.ones(action_dim)
        self.input_dim = action_dim
        self.qp_h_col = np.ones(1)
        self.qp_G_mtx = np.ones((1, action_dim))
        self.p_ = np.identity(action_dim)

    def GetFilteredAction(self, coeff_:np.ndarray, scalar:np.ndarray, value_:np.ndarray, action_:np.ndarray, SafetyMargin=0.00, alpha_ = 1.0, P: np.ndarray =None):
        maximal_safest_input = np.sign(coeff_)
        alpha_v = alpha_ * (value_ - SafetyMargin) 
        self.p_ = P if P is not None else np.identity(coeff_.shape[0])
        # numerically stable version of the following if-else:
        # if (-scalar - alpha_v <= np.sum(np.abs(coeff_))): # which means QP is feasible
        #    beta = -scalar - alpha_v # and solve QP
        # else: # which means QP is infeasible, and we need to return the estimated maximal safest input
        #    beta = np.sum(np.abs(coeff_)) # and solve QP, which is identical to max_{u \in U} coeff_ @ u
        
        beta = np.minimum(-scalar - alpha_v, (1.0-1e-6)*np.sum(np.abs(coeff_)))
        self.qp_q_col = - self.p_@action_ 
        self.qp_h_col = beta
        self.qp_G_mtx[0,:] = coeff_
        
        sol = solve_qp(P=self.p_, q=self.qp_q_col, G=-self.qp_G_mtx, h=-self.qp_h_col, 
                A=None, b= None, lb = self.lb_, ub=self.ub_, 
                solver="proxqp", 
                max_iter=20000, 
                eps_abs=1e-8,   
                eps_rel=1e-8    
                )
        filtered_action_ = sol if sol is not None else maximal_safest_input
        return np.clip(filtered_action_,-1,1)
        
class Hyperparameters:
    time_interval = TIME_INTERVAL
    task_time = TASK_TIME
    max_episode_steps = MAX_TIME_STEP
    seed = seed

    # PPO hyperparameters
    gamma = 0.99
    cost_gamma = 0.99
    gae_lambda = 0.97
    ppo_epochs = 10
    num_rollout_steps = 8192
    minibatch_size = 256
    clip_epsilon = 0.2
    lr_actor = 3e-4
    lr_critic = 3e-4

    max_training_timesteps = int(10e6)
    save_interval = num_rollout_steps * 10

    log_std_init = -0.5

    target_kl = 0.02

    device = DEVICE_

# utils
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma, cost_gamma, lam):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32) 
        self.rew_buf  = np.zeros(size, dtype=np.float32)

        self.ret_r_buf = np.zeros(size, dtype=np.float32)
        self.ret_c_buf = np.zeros(size, dtype=np.float32)
        self.adv_r_buf = np.zeros(size, dtype=np.float32)
        self.adv_c_buf = np.zeros(size, dtype=np.float32)

        self.val_r_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf  = np.zeros(size, dtype=np.float32)

        self.gamma, self.cost_gamma, self.lam = gamma, cost_gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, a_squash, rew, val_r, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = a_squash
        self.rew_buf[self.ptr]  = rew
        self.val_r_buf[self.ptr] = val_r
        self.logp_buf[self.ptr]  = logp
        self.ptr += 1

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val_r=0.0):
        sl = slice(self.path_start_idx, self.ptr)

        rews   = np.append(self.rew_buf[sl], last_val_r)
        vals_r = np.append(self.val_r_buf[sl], last_val_r)
        deltas_r = rews[:-1] + self.gamma * vals_r[1:] - vals_r[:-1]
        self.adv_r_buf[sl] = self._discount_cumsum(deltas_r, self.gamma * self.lam)
        self.ret_r_buf[sl] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_r_mean, adv_r_std = np.mean(self.adv_r_buf), np.std(self.adv_r_buf)
        self.adv_r_buf = (self.adv_r_buf - adv_r_mean) / (adv_r_std + 1e-8)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret_r=self.ret_r_buf,
                    adv_r=self.adv_r_buf, logp=self.logp_buf)

        return {k: torch.as_tensor(v, dtype=DTYPE_, device=DEVICE_) for k, v in data.items()}

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
EPS = 1e-6

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_init=-0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim, dtype=DTYPE_) * log_std_init) 
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
        logp  = dist.log_prob(a).sum(dim=-1)
        return a, logp, mu

    def log_prob(self, s, a):
        mu, std = self._dist_params(s)
        return Normal(mu, std).log_prob(a).sum(dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)
    def forward(self, state):
        return self.net(state).squeeze(-1)

# ============================================================================================
# === PPO Agent
# ============================================================================================
class PPOAgent:
    def __init__(self, state_dim, action_space, hp, safety_critic:SafetyCritic, qp_filter:QP_filter):
        self.hp = hp
        
        self.buffer = PPOBuffer(state_dim, action_space.shape[0],
                                hp.num_rollout_steps, hp.gamma, hp.cost_gamma, hp.gae_lambda)
        self.safety_critic = safety_critic
        self.qp_filter = qp_filter
        self.actor   = Actor(state_dim, action_space.shape[0], log_std_init=hp.log_std_init).to(hp.device)
        self.critic_r = Critic(state_dim).to(hp.device)
        self.optimizer_actor   = torch.optim.Adam(self.actor.parameters(), lr=hp.lr_actor)
        self.optimizer_critic_r= torch.optim.Adam(self.critic_r.parameters(), lr=hp.lr_critic)

    @torch.no_grad()
    def select_action(self, state, constraint = None):
        s = torch.as_tensor(state, dtype=DTYPE_, device=self.hp.device)
        a_raw, logp, a_det = self.actor.sample(s)
        a_raw_clipped = torch.clamp(a_raw, -1.0, 1.0)
        if constraint is not None:
            state_tensor = torch.tensor(state, dtype=DTYPE_, device=DEVICE_)
            value_, coeff_, scalar_ = safety_critic_.GetValues(state_tensor, constraint, UseTarget=True)
            a_env = qp_filter_.GetFilteredAction(coeff_, scalar_, value_, a_raw_clipped.cpu().numpy(), SafetyMargin=SAFETY_MARGIN, alpha_ = SAFETY_ALPHA)
        else:
            a_env = a_raw_clipped.cpu().numpy()
        
        val_r = self.critic_r(s)
        return (a_env,
                a_raw.cpu().numpy(),
                val_r.item(),
                logp.item())

    def update(self):
        data = self.buffer.get()
        obs, a_squash, ret_r, adv, logp_old = data['obs'], data['act'], data['ret_r'], data['adv_r'], data['logp']

        # PPO update
        for _ in range(self.hp.ppo_epochs):
            kl_exceeded = False
            last_approx_kl = 0.0
            idx = np.random.permutation(self.hp.num_rollout_steps)
            for start in range(0, self.hp.num_rollout_steps, self.hp.minibatch_size):
                batch = idx[start:start+self.hp.minibatch_size]
                b_obs = obs[batch]
                b_act_squash = a_squash[batch]
                b_adv = adv[batch]
                b_ret_r = ret_r[batch]
                b_logp_old = logp_old[batch]

                logp_new = self.actor.log_prob(b_obs, b_act_squash)
                ratio = torch.exp(logp_new - b_logp_old)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.hp.clip_epsilon, 1.0 + self.hp.clip_epsilon) * b_adv
                loss_pi = -torch.min(surr1, surr2).mean()

                self.optimizer_actor.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.optimizer_actor.step()

                loss_v_r = F.mse_loss(self.critic_r(b_obs), b_ret_r)
                self.optimizer_critic_r.zero_grad()
                loss_v_r.backward()
                nn.utils.clip_grad_norm_(self.critic_r.parameters(), 1.0)
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

        with torch.no_grad():
            kl = last_approx_kl 
            
        return float(loss_v_r.item()), float(loss_pi.item()), float(kl)

if __name__ == "__main__":
    hp = Hyperparameters()

    args_SF = {
                'lambda_init':0.0, 
                'lambda_final':0.0, 
                
                'hidden_size':256,  
                'hidden_layers':2, # 3 for hopper, 2 for others 
                
                'cuda': DEVICE_,
                'dtype':DTYPE_,
            }

    # env = InvertedDoublePendulumPositionBonus(dt_=hp.time_interval, T_max=hp.task_time, seed=hp.seed)
    # env = InvertedDoubleMovingBonus(dt_=hp.time_interval, T_max=hp.task_time, seed=hp.seed)
    # load_path = "./pretrained/2D_Inverted_Pendulum/"

    env = Hopper(dt_=hp.time_interval, T_max=hp.task_time, seed=hp.seed)
    load_path = "./pretrained/Hopper/"
    args_SF['hidden_layers'] = 3 # 3 for hopper, 2 for others
    
    env_name_log = env.name+"/"+BASELINE_TYPE
    full_log_dir = os.path.join(log_dir, log_filename, env_name_log, result_string)
    full_model_dir = os.path.join(network_dir, env_name_log, result_string)
    os.makedirs(full_log_dir, exist_ok=True)
    os.makedirs(full_model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=full_log_dir)
    print(f"[{BASELINE_TYPE}] Training on {env_name_log}, Logging to {full_log_dir}")

    state_dim   = env.env_.observation_space.shape[0]
    action_dim  = env.env_.action_space.shape[0]
    
    safety_critic_ = SafetyCritic(state_dim, action_dim, args_SF, eval_=True)
    safety_critic_.NNload(load_path+"deep_qp_safety_filter_model")
    qp_filter_ = QP_filter(action_dim)
    agent = PPOAgent(state_dim, env.env_.action_space, hp, safety_critic_, qp_filter_)

    state, constraint = env.reset()
    total_time_steps = 0
    i_episode = 0
    ep_ret, ep_len = 0.0, 0

    while total_time_steps < hp.max_training_timesteps:

        for t in range(hp.num_rollout_steps):
            a_env, a_raw, v_r, logp = agent.select_action(state, constraint)
            next_state, task_reward, constraint, terminated, truncated, violate_info = env.step(a_env)
            total_time_steps += 1
            ep_len += 1

            ep_ret += task_reward

            agent.buffer.store(state, a_raw, task_reward, v_r, logp)

            state = next_state

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
                        violation_info_str = ''
                        if isinstance(violate_info, dict):
                            for key in violate_info.keys():
                                if violate_info[key]:
                                    violation_info_str += key + ', '
                        print(f"[Failed]    {msg} | Violations: {violation_info_str}")
                    else:
                        print(f"[Truncated] {msg}")

                    state, c = env.reset()
                    ep_ret, ep_len = 0.0, 0
                    i_episode += 1

        # --- 3) 업데이트 ---
        v_loss, pi_loss, kl = agent.update()

        writer.add_scalar(f"{env_name_log}/Loss/Critic_Reward", v_loss, total_time_steps)
        writer.add_scalar(f"{env_name_log}/Loss/Actor", pi_loss, total_time_steps)
        writer.add_scalar(f"{env_name_log}/Info/KL_Divergence", kl, total_time_steps)
        
        # --- 4) 저장 ---
        if (total_time_steps // hp.num_rollout_steps) > 0 and \
           (total_time_steps // hp.num_rollout_steps) % (hp.save_interval // hp.num_rollout_steps) == 0:
            save_path = os.path.join(full_model_dir, f"model_{total_time_steps}.pth")
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_r_state_dict': agent.critic_r.state_dict()
            }, save_path)
            print(f"Model saved at Timestep {total_time_steps} to {save_path}")

    env.close()
    writer.close()
    print(f"Training finished for {BASELINE_TYPE}.")