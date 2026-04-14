# Deep QP Safety Filter

<!-- <p align="center">
  <img src="media/Hopper/safety_filtering_bangbang.gif" width="85%" alt="Deep QP Safety Filter on Hopper"/>
</p> -->

<p align="center">
  <img src="media/Hopper/safety_filtering_random.gif" width="48%" alt="demo1"/>
  <img src="media/Hopper/safety_filtering_bangbang.gif" width="48%" alt="demo2"/>
</p>

<p align="center">
  <strong>Model-free learning of a reachability-based QP safety filter for black-box dynamical systems</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.21297">Paper</a>
</p>

---

## Overview

Deep QP Safety Filter is a fully data-driven safety layer for black-box dynamical systems.

The method learns a reachability-based QP safety filter directly from transition data, without requiring an explicit system model.
It learns a discounted safety value together with its derivative terms, and then uses them inside a quadratic program to minimally modify a raw control input into a safe control input.

This repository includes:

- online training for the Deep QP Safety Filter
- inference with pretrained safety filters
- safe reinforcement learning with PPO and a learned safety filter
- Gymnasium / MuJoCo experiments including Inverted Pendulum, Inverted Double Pendulum, and Hopper

---

## Environment Setup

This project was developed in a Conda environment.

Create the environment with:

```bash
conda env create -f requirements.yml
conda activate deep_qp_sf
```
---

## Run Inference

To run inference with a pretrained safety filter and save a GIF:

```bash
python3 run_inference.py --env hopper --save-gif
```

To render in a viewer window:

```bash
python3 run_inference.py --env hopper --human
```

For inverted double pendulum:

```bash
python3 run_inference.py --env inverted_double_pendulum --save-gif
```

Outputs are saved under:

```text
outputs/inference/<env_name>/
```

---

## Train the Safety Filter

To train the Deep QP Safety Filter for a specific system:

```bash
python3 train_filter.py --config configs/<env_name>.yaml
```
For example:
```bash
python3 train_filter.py --config configs/hopper.yaml
```

Training outputs are saved under:

```text
outputs/filter_training/<env_name>/<run_id>/
```

including:

- TensorBoard logs
- checkpoints
- replay memory

---

## Train Safe Reinforcement Learning

To train PPO with the learned safety filter on Hopper:

```bash
python3 train_safe_rl.py --env hopper
```

For the inverted double pendulum RL tasks:

```bash
python3 train_safe_rl.py --env inverted_double_pendulum_position_bonus
python3 train_safe_rl.py --env inverted_double_moving_bonus
```

Outputs are saved under:

```text
outputs/safe_rl/<env_name>/<run_id>/
```
---
<!-- 
## Notes

- The safety filter is trained task-agnostically using only transition data and safety signals.
- The same learned filter can be reused across tasks through the aggressiveness parameter.
- The default hidden width is 256 for the safety filter.
- The safety filter uses 2 hidden layers for most environments, and 3 hidden layers for Hopper.
- In safe RL, the PPO actor and critic use 2 hidden layers for the inverted double pendulum tasks, and 3 hidden layers for Hopper.

--- -->

<!-- ## Demo Asset

Place the Hopper demo GIF at:

```text
assets/hopper_demo.gif
```

so that it renders at the top of this README.

--- -->

## Citation

If you find this repository useful, please cite:

```bibtex
@article{kim2026deep,
  title   = {Deep QP Safety Filter: Model-free Learning for Reachability-based Safety Filter},
  author  = {Kim, Byeongjun and Kim, H Jin},
  journal = {arXiv preprint arXiv:2601.21297},
  year    = {2026}
}
```
