import argparse
from pathlib import Path
import random

import imageio.v2 as imageio
import numpy as np
import torch
from qpsolvers import solve_qp

from SafetyModule.SafetyCritic import SafetyCritic
from scripts.common import make_env, set_seed, default_hidden_layers, pretrained_ckpt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a pretrained Deep QP Safety Filter."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="hopper",
        choices=["hopper", "inverted_double_pendulum", "inverted_pendulum"],
    )
    parser.add_argument("--control-dt", type=float, default=5e-3)
    parser.add_argument("--task-time", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345678)

    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "bangbang"],
    )
    parser.add_argument("--period", type=float, default=5.0)

    parser.add_argument(
        "--human",
        action="store_true",
        help="Render the rollout in a human viewer window.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save the rollout as a GIF.",
    )
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/inference")
    parser.add_argument("--model-path", type=str, default=None)

    return parser.parse_args()


class OUProcess:
    def __init__(self, action_dim: int, dt: float, kappa: float = 7.5, sigma: float = 1.5):
        self.kappa_ = kappa
        self.sigma_ = sigma
        self.dt_ = dt
        self.z_ = np.zeros(action_dim, dtype=np.float64)

    def reset(self):
        self.z_ = 2.0 * (np.random.rand(self.z_.shape[0]) - 0.5)

    def generate(self):
        self.z_ = (
            self.z_
            + self.kappa_ * (-self.z_) * self.dt_
            + self.sigma_ * np.sqrt(self.dt_) * np.random.normal(size=self.z_.shape)
        )
        return np.clip(self.z_, -1.0, 1.0)


class QPFilter:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.lb_ = -np.ones(action_dim, dtype=np.float64)
        self.ub_ = np.ones(action_dim, dtype=np.float64)
        self.P_ = np.eye(action_dim, dtype=np.float64)
        self.G_ = np.zeros((1, action_dim), dtype=np.float64)

    def get_filtered_action(
        self,
        coeff_: np.ndarray,
        scalar_: np.ndarray,
        value_: np.ndarray,
        action_: np.ndarray,
        safety_margin: float = 0.0,
        alpha_: float = 1.0,
    ):
        coeff_ = np.asarray(coeff_, dtype=np.float64).reshape(-1)
        scalar_ = float(np.asarray(scalar_, dtype=np.float64).reshape(-1)[0])
        value_ = float(np.asarray(value_, dtype=np.float64).reshape(-1)[0])
        action_ = np.asarray(action_, dtype=np.float64).reshape(-1)

        maximal_safest_input = np.sign(coeff_)
        coeff_norm = float(np.sum(np.abs(coeff_)))
        alpha_v = alpha_ * (value_ - safety_margin)

        # numerically stable version of the following if-else:
        # if (-scalar - alpha_v <= np.sum(np.abs(coeff_))): # which means QP is feasible
        #    beta = -scalar - alpha_v # and solve QP
        # else: # which means QP is infeasible, and we need to return the estimated maximal safest input
        #    beta = np.sum(np.abs(coeff_)) # and solve QP, which is identical to max_{u \in U} coeff_ @ u
        
        beta_candidate = -scalar_ - alpha_v
        infeasible_qp = int(beta_candidate > coeff_norm)

        beta = min(beta_candidate, (1.0 - 1e-6) * coeff_norm)
        q_ = -self.P_ @ action_
        self.G_[0, :] = coeff_

        sol = solve_qp(
            P=self.P_,
            q=q_,
            G=-self.G_,
            h=np.array([-beta], dtype=np.float64),
            A=None,
            b=None,
            lb=self.lb_,
            ub=self.ub_,
            solver="proxqp",
            max_iter=20000,
            eps_abs=1e-8,
            eps_rel=1e-8,
        )

        filtered_action, safest_fallback = (
            (sol, 0) if sol is not None else (maximal_safest_input, 1)
        )
        return np.clip(filtered_action, -1.0, 1.0), safest_fallback, infeasible_qp


def make_reference_action(
    policy_type: str,
    noise_process: OUProcess,
    action_dim: int,
    step_idx: int,
    control_dt: float,
    period: float,
):
    if policy_type == "random":
        return noise_process.generate()

    if policy_type == "bangbang":
        half_period_steps = int(0.5 * period / control_dt)
        full_period_steps = int(period / control_dt)
        return (
            np.ones(action_dim, dtype=np.float64)
            if (step_idx % full_period_steps) < half_period_steps
            else -np.ones(action_dim, dtype=np.float64)
        )

    raise ValueError(f"Unknown policy_type: {policy_type}")

def save_gif(frames, gif_path: Path, sim_dt: float, gif_fps: int):
    if len(frames) == 0:
        return

    gif_path.parent.mkdir(parents=True, exist_ok=True)

    sim_fps = int(round(1.0 / sim_dt))
    save_every = max(1, sim_fps // gif_fps)
    frames_to_save = frames[::save_every]

    imageio.mimsave(
        gif_path,
        frames_to_save,
        fps=gif_fps,
        loop=0,
    )


def main():
    args = parse_args()

    if args.human and args.save_gif:
        raise ValueError(
            "--human and --save-gif cannot be used together in the current implementation. "
            "Use --human for interactive viewing, or --save-gif for file export."
        )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    set_seed(args.seed)

    if args.save_gif:
        render_mode = "rgb_array"
    elif args.human:
        render_mode = "human"
    else:
        render_mode = None

    env = make_env(
        env_name=args.env,
        dt=args.control_dt,
        task_time=args.task_time,
        render_mode=render_mode,
        seed=args.seed,
    )
    env_name = env.GetEnvName()

    state, _ = env.reset()
    state_dim = state.shape[0]
    action_dim, action_high, action_low = env.GetActionInfo()

    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    args_sf = {
        "lambda_init": 0.0,
        "lambda_final": 0.0,
        "hidden_size": 256,
        "hidden_layers": default_hidden_layers(args.env),
        "cuda": device,
        "dtype": dtype,
    }

    model_path = Path(args.model_path) if args.model_path is not None else pretrained_ckpt(args.env)
    safety_critic = SafetyCritic(state_dim, action_dim, args_sf, eval_=True)
    safety_critic.NNload(str(model_path))

    qp_filter = QPFilter(action_dim)
    noise_process = OUProcess(action_dim, args.control_dt, kappa=2.5, sigma=10.0)

    failed = False
    done = False
    step_idx = 0
    episode_reward = 0.0
    safest_action_count = 0
    qp_infeas_count = 0
    min_constraint = np.inf
    violation_info = {}

    frames = []
    state, constraint = env.reset()
    noise_process.reset()

    while not done:
        if args.save_gif:
            frames.append(env.env_.render())
        elif args.human:
            env.env_.render()

        step_idx += 1
        state_tensor = torch.tensor(state, dtype=dtype, device=device)

        raw_action = make_reference_action(
            policy_type=args.policy,
            noise_process=noise_process,
            action_dim=action_dim,
            step_idx=step_idx,
            control_dt=args.control_dt,
            period=args.period,
        )

        value_, coeff_, scalar_ = safety_critic.GetValues(
            state_tensor,
            constraint,
            UseTarget=True,
        )

        filtered_action, safest_count, infeas_count = qp_filter.get_filtered_action(
            coeff_=coeff_,
            scalar_=scalar_,
            value_=value_,
            action_=raw_action,
            safety_margin=args.margin,
            alpha_=args.alpha,
        )

        safest_action_count += safest_count
        qp_infeas_count += infeas_count

        env_action = action_scale * filtered_action + action_bias
        state, reward, next_constraint, fail, trunc, violation_info = env.step(env_action)

        done = fail or trunc
        failed = fail
        constraint = next_constraint
        episode_reward += reward * args.control_dt

        if constraint < min_constraint:
            min_constraint = constraint

    if failed:
        violated_keys = [k for k, v in violation_info.items() if v]
        violated_str = ", ".join(violated_keys) if len(violated_keys) > 0 else "unknown"
        print(
            f"[FAILED] return={episode_reward / args.control_dt:.2f}, "
            f"steps={step_idx}, safest_fallback_count={safest_action_count}, "
            f"violations={violated_str}"
        )
    else:
        print(
            f"[TRUNCATED] return={episode_reward / args.control_dt:.2f}, "
            f"min_constraint={min_constraint:.4f}, "
            f"safest_fallback_count={safest_action_count}"
        )

    print(
        f"QP infeasible count: {qp_infeas_count} / {step_idx} "
        f"({qp_infeas_count / step_idx:.4%})"
    )

    if args.save_gif:
        output_dir = Path(args.output_dir) / env_name
        gif_name = f"safety_filtering_{args.policy}.gif"
        gif_path = output_dir / gif_name
        save_gif(frames, gif_path, sim_dt=args.control_dt, gif_fps=args.gif_fps)
        print(f"Saved GIF to: {gif_path}")

if __name__ == "__main__":
    main()