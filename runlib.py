"""训练/推理共用的小工具。train.py、bc.py、evalx.py 都从这里取，避免三份实现漂移。"""

from __future__ import annotations

import numpy as np
import torch

from models import (GRUPolicy, TransformerPolicy, CEDPolicy,
                    action_nll, greedy_actions, n_params)


def make_model(name: str, **kw):
    if name == "gru":
        return GRUPolicy(**kw)
    if name == "tf":
        return TransformerPolicy(**kw)
    if name == "ced":
        return CEDPolicy(**kw)
    raise ValueError(f"unknown model {name}")


def device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def bc_loss(model, obs, act):
    dev = device_of(model)
    obs, act = obs.to(dev), act.to(dev)
    if isinstance(model, CEDPolicy):
        return model.loss(obs, act)
    return action_nll(model(obs), act)


def train_step(model, opt, obs, act, clip: float = 1.0) -> float:
    loss = bc_loss(model, obs, act)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def policy_actions(model, obs_seq: np.ndarray) -> np.ndarray:
    """obs_seq: (T, OBS_DIM) -> (14,) 当前帧动作。"""
    dev = device_of(model)
    x = torch.as_tensor(obs_seq[None], dtype=torch.float32, device=dev)
    if isinstance(model, CEDPolicy):
        acts = model(x, None)
    else:
        acts = greedy_actions(model(x))[:, -1]
    return acts.cpu().numpy()[0]


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


__all__ = ["make_model", "bc_loss", "train_step", "policy_actions", "rss_mb",
           "device_of", "n_params", "CEDPolicy"]
