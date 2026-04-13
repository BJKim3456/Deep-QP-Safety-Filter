from pathlib import Path
import random
import numpy as np
import torch

from envs.gym_env import (
    InvertedPendulum,
    InvertedDoublePendulum,
    InvertedDoublePendulumPositionBonus,
    InvertedDoubleMovingBonus,
    Hopper,
)
from envs.custom_env import DoubleIntegrator

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = ROOT / "pretrained"
OUTPUT_DIR = ROOT / "outputs"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(env_name: str, dt: float, task_time: float, render_mode=None, seed: int = 0):
    if env_name == "hopper":
        return Hopper(dt_=dt, T_max=task_time, render_=render_mode, seed=seed)
    if env_name == "inverted_double_pendulum":
        return InvertedDoublePendulum(dt_=dt, T_max=task_time, render_=render_mode, seed=seed)
    if env_name == "inverted_pendulum":
        return InvertedPendulum(dt_=dt, T_max=task_time, render_=render_mode, seed=seed)
    if env_name == "inverted_double_pendulum_position_bonus":
        return InvertedDoublePendulumPositionBonus(dt_=dt, T_max=task_time, render_=render_mode, seed=seed)
    if env_name == "inverted_double_moving_bonus":
        return InvertedDoubleMovingBonus(dt_=dt, T_max=task_time, render_=render_mode, seed=seed)
    if env_name == "double_integrator":
        return DoubleIntegrator(dt_=dt, T_max=task_time)
    raise ValueError(f"Unknown env_name: {env_name}")

def default_hidden_layers(env_name: str) -> int:
    return 3 if env_name == "hopper" else 2

def pretrained_ckpt(env_name: str) -> Path:
    mapping = {
        "hopper": PRETRAINED_DIR / "Hopper" / "deep_qp_safety_filter_model",
        "inverted_double_pendulum": PRETRAINED_DIR / "2D_Inverted_Pendulum" / "deep_qp_safety_filter_model",
        "inverted_pendulum": PRETRAINED_DIR / "1D_Inverted_Pendulum" / "deep_qp_safety_filter_model",
        "inverted_double_pendulum_position_bonus": PRETRAINED_DIR / "2D_Inverted_Pendulum" / "deep_qp_safety_filter_model",
        "inverted_double_moving_bonus": PRETRAINED_DIR / "2D_Inverted_Pendulum" / "deep_qp_safety_filter_model",
    }
    return mapping[env_name]