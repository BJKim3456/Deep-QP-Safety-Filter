import argparse
import datetime
import socket
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from qpsolvers import solve_qp
from torch.utils.tensorboard import SummaryWriter

from SafetyModule.SafetyCritic import SafetyCritic
from scripts.common import make_env, set_seed, default_hidden_layers, OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Deep QP Safety Filter."
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_sf_args(cfg: dict, device, dtype) -> dict:
    dt = cfg["control_dt"]
    sf = cfg["sf"]

    hidden_layers = sf.get("hidden_layers", default_hidden_layers(cfg["env"]))

    return {
        "lambda_init": sf["lambda_init_scale"] / dt,
        "lambda_final": sf["lambda_final_scale"] / dt,
        "lambda_increasing_interval": sf["lambda_increasing_interval"],
        "lambda_start_to_decrease": sf["lambda_start_to_decrease"],
        "lr_decreasing_interval": sf["lr_decreasing_interval"],
        "warmup_steps": sf["warmup_steps"],
        "tau": sf["tau"],
        "dt": dt,
        "clip_grad": cfg["grad_clip"],
        "with_warm_up": sf["with_warm_up"],
        "lr_safe_value": sf["lr_safe_value"],
        "lr_safe_q_coeff": sf["lr_safe_q_coeff"],
        "lr_safe_q_scalar": sf["lr_safe_q_scalar"],
        "lr_safe_value_goal": sf["lr_safe_value_goal"],
        "lr_safe_q_coeff_goal": sf["lr_safe_q_coeff_goal"],
        "lr_safe_q_scalar_goal": sf["lr_safe_q_scalar_goal"],
        "batch_size": sf["batch_size"],
        "hidden_size": sf["hidden_size"],
        "hidden_layers": hidden_layers,
        "target_update_interval": sf["target_update_interval"],
        "replay_size": sf["replay_size"],
        "update_count": 0,
        "cuda": device,
        "dtype": dtype,
    }


class OUProcess:
    def __init__(self, action_dim, dt, kappa=7.5, sigma=1.5):
        self.kappa_ = kappa
        self.sigma_ = sigma
        self.dt_ = dt
        self.z_ = np.zeros(action_dim)
        self.theta_ = np.zeros(action_dim)

    def reset(self, theta=None):
        self.z_ = 2 * (np.random.rand(self.z_.shape[0]) - 0.5)
        self.theta_ = theta if theta is not None else np.zeros_like(self.z_)

    def generate(self):
        self.z_ = (
            self.z_
            + self.kappa_ * (self.theta_ - self.z_) * self.dt_
            + self.sigma_ * np.sqrt(self.dt_) * np.random.normal(size=self.z_.shape)
        )
        return np.clip(self.z_, -1, 1)


class QP_filter:
    def __init__(self, action_dim):
        self.u_max = np.ones(action_dim)
        self.ub_ = np.ones(action_dim)
        self.lb_ = -np.ones(action_dim)
        self.input_dim = action_dim
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
        filtered_action_, safest_count = (sol, 0) if sol is not None else (maximal_safest_input, 1)
        if safest_count == 0:
            if beta < -scalar - alpha_v:
                safest_count = 1
        return np.clip(filtered_action_, -1, 1), safest_count


@torch.no_grad()
def evaluate_safety_filter(
    given_policy,
    given_critic,
    given_env,
    given_qp_filter,
    reps: int,
    margin: float,
    alpha_: float,
    max_time_step: int,
    device,
    dtype,
):
    returns = []
    steps = []
    safest_action_counts = []
    fail_count_ = 0

    action_dim, action_high, action_low = given_env.GetActionInfo()
    action_scale = (action_high - action_low) / 2
    action_bias = (action_high + action_low) / 2

    for _ in range(reps):
        returns.append(0.0)
        steps.append(0)
        safest_action_counts.append(0)

        state, constraint = given_env.reset()
        theta_ = -0.5 + np.random.rand(action_dim)
        given_policy.reset(theta=theta_)

        while True:
            state_tensor = torch.tensor(state, dtype=dtype, device=device)
            value_, coeff_, scalar_ = given_critic.GetValues(state_tensor, constraint, UseTarget=True)

            unfiltered_action = given_policy.generate()
            action, safest_action_count = given_qp_filter.GetFilteredAction(
                coeff_,
                scalar_,
                value_,
                unfiltered_action,
                SafetyMargin=margin,
                alpha_=alpha_,
            )

            state, reward, next_constraint, fail, trunc, _ = given_env.step(
                action_scale * action + action_bias
            )
            constraint = next_constraint

            returns[-1] += reward
            steps[-1] += 1
            safest_action_counts[-1] += safest_action_count

            if fail or trunc or steps[-1] >= max_time_step:
                break

        if fail:
            fail_count_ += 1

    return (
        float(np.mean(returns)),
        float(np.mean(steps)),
        int(fail_count_),
        float(np.mean(safest_action_counts)),
    )


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    torch.cuda.empty_cache()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    set_seed(cfg["seed"])

    time_interval = cfg["control_dt"]
    task_time = cfg["task_time"]
    max_time_step = int(task_time / time_interval)

    args_SF = build_sf_args(cfg, device=device, dtype=dtype)

    env = make_env(
        env_name=cfg["env"],
        dt=time_interval,
        task_time=task_time,
        render_mode=None,
        seed=cfg["seed"],
    )
    env_eval = make_env(
        env_name=cfg["env"],
        dt=time_interval,
        task_time=task_time,
        render_mode=None,
        seed=cfg["seed"],
    )
    env_name = env.GetEnvName()

    state, _ = env.reset()
    state_dim = state.shape[0]
    action_dim, action_high, action_low = env.GetActionInfo()

    action_scale = (action_high - action_low) / 2
    action_bias = (action_high + action_low) / 2

    run_id = f"{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}"
    base_output = Path(cfg["output_dir"]) if "output_dir" in cfg else (OUTPUT_DIR / "filter_training")
    run_dir = base_output / env_name / run_id
    tb_dir = run_dir / "tensorboard"
    model_dir = run_dir / "checkpoints"
    replay_dir = run_dir / "replay_memory"

    tb_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    noise_for_train_agent = OUProcess(action_dim, time_interval, kappa=1.5, sigma=2.5)
    qp_filter_ = QP_filter(action_dim)

    safety_critic_ = SafetyCritic(state_dim, action_dim, args_SF)
    eval_critic_ = SafetyCritic(state_dim, action_dim, args_SF)

    update_started = False
    i_episode = 0
    optim_step = 0

    try:
        while optim_step < cfg["grad_step_max"]:
            i_episode += 1

            current_ep_reward = 0.0
            current_ep_safest_action_count = 0

            state, constraint = env.reset()
            c = float(constraint)
            time_step = 0

            theta_ = -0.5 + np.random.rand(action_dim)
            noise_for_train_agent.reset(theta=theta_)

            cur_alpha_ = cfg["safety_alpha"]
            cur_p_ = np.identity(action_dim)
            for i in range(action_dim - 1):
                cur_p_[i, i] = np.random.rand() * 9.9 + 0.1

            while True:
                time_step += 1
                state_tensor = torch.tensor(state, dtype=dtype, device=device)
                action = noise_for_train_agent.generate()
                safest_action_count = 0

                if update_started:
                    value_, coeff_, scalar_ = safety_critic_.GetValues(
                        state_tensor, constraint, UseTarget=True
                    )
                    action, safest_action_count = qp_filter_.GetFilteredAction(
                        coeff_,
                        scalar_,
                        value_,
                        action,
                        SafetyMargin=cfg["safety_margin"],
                        alpha_=cur_alpha_,
                        P=cur_p_,
                    )
                    current_ep_safest_action_count += safest_action_count

                next_state, reward, next_constraint, fail, trunc, violate_info = env.step(
                    action_scale * action + action_bias
                )
                safety_critic_.append_transition(
                    state, action, constraint, next_state, next_constraint, fail
                )

                if not update_started:
                    if safety_critic_.GetRMLength() >= cfg["replay_warmup_size"]:
                        update_started = True
                else:
                    if optim_step % cfg["evaluation_period"] == 0:
                        ckpt_path = model_dir / f"episode_{i_episode}"
                        safety_critic_.NNsave(str(ckpt_path))
                        safety_critic_.save_replay_memory(str(replay_dir))

                        eval_critic_.load_state_dict(safety_critic_)

                        avg_return, avg_steps, total_fails, avg_safest_count = evaluate_safety_filter(
                            given_policy=deepcopy(noise_for_train_agent),
                            given_critic=eval_critic_,
                            given_env=deepcopy(env_eval),
                            given_qp_filter=QP_filter(action_dim),
                            reps=cfg["num_eval"],
                            margin=cfg["safety_margin"],
                            alpha_=cfg["safety_alpha"],
                            max_time_step=max_time_step,
                            device=device,
                            dtype=dtype,
                        )

                        writer.add_scalar(f"{env_name}/evaluation/avg_reward", avg_return, optim_step)
                        writer.add_scalar(f"{env_name}/evaluation/avg_steps", avg_steps, optim_step)

                        print(
                            f"Eval: reps={cfg['num_eval']}, fails={total_fails}, "
                            f"avg_return={avg_return:.2f}, avg_steps={avg_steps:.2f}, "
                            f"avg_safest_count={avg_safest_count:.2f}"
                        )

                    safe_value_loss, safe_small_q_loss, consistency_loss = safety_critic_.update()
                    writer.add_scalar(f"{env_name}/train/safe_critic_loss", safe_value_loss, optim_step)
                    writer.add_scalar(f"{env_name}/train/safe_small_q_loss", safe_small_q_loss, optim_step)
                    writer.add_scalar(f"{env_name}/train/safe_value_consistency", consistency_loss, optim_step)

                    optim_step += 1

                state = next_state
                constraint = next_constraint
                c = min(c, float(constraint))
                current_ep_reward += reward * time_interval

                if fail or trunc or time_step >= max_time_step:
                    break

            if not update_started:
                i_episode = 0
            else:
                if fail:
                    violation_info_str = ""
                    if isinstance(violate_info, dict):
                        violation_info_str = ", ".join([k for k, v in violate_info.items() if v])
                    print(
                        f"Failed at episode {i_episode}, "
                        f"return={current_ep_reward / time_interval:.2f}, "
                        f"steps={time_step}, safest_count={current_ep_safest_action_count}, "
                        f"violations={violation_info_str}"
                    )
                else:
                    print(
                        f"Truncated at episode {i_episode}, "
                        f"return={current_ep_reward / time_interval:.2f}, "
                        f"min_constraint={c:.2f}, safest_count={current_ep_safest_action_count}"
                    )

            writer.add_scalar(f"{env_name}/episode/return", current_ep_reward / time_interval, i_episode)
            writer.add_scalar(f"{env_name}/episode/steps", time_step, i_episode)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        writer.close()


if __name__ == "__main__":
    main()