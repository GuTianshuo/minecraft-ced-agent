"""训练/推理共用的小工具。train.py、bc.py、evalx.py 都从这里取，避免三份实现漂移。"""

from __future__ import annotations

import numpy as np
import torch

import torch.nn.functional as F

from models import (GRUPolicy, TransformerPolicy, CEDPolicy,
                    action_nll, greedy_actions, n_params)
from schema_v0 import ACTION_NCLASSES, MAX_CLASSES, N_ACTION_TOKENS


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


def balanced_weights(act, alpha: float = 0.5):
    """按类别频率给每个 token 的每个类别一个权重：w = (mean/freq)^alpha。

    目的：BC 数据里"不转"占 94%，不加权时模型会退化成永远不转，
    总体准确率仍有 94%，但闭环完全不能用（实测转向样本准确率 0%）。
    """
    flat = torch.as_tensor(act).reshape(-1, N_ACTION_TOKENS)
    out = []
    for i in range(N_ACTION_TOKENS):
        c = int(ACTION_NCLASSES[i])
        counts = torch.bincount(flat[:, i].long(), minlength=c).float().clamp(min=1.0)
        out.append((counts.mean() / counts) ** alpha)
    return out


def action_nll_weighted(logits, targets, weights):
    L = logits.reshape(-1, N_ACTION_TOKENS, MAX_CLASSES)
    T = targets.reshape(-1, N_ACTION_TOKENS)
    total = logits.new_zeros(T.shape[0])
    for i in range(N_ACTION_TOKENS):
        c = int(ACTION_NCLASSES[i])
        total = total + F.cross_entropy(L[:, i, :c], T[:, i], reduction="none",
                                        weight=weights[i][:c].to(L.device))
    return total.mean()


def bc_loss(model, obs, act, class_weights=None):
    dev = device_of(model)
    obs, act = obs.to(dev), act.to(dev)
    if isinstance(model, CEDPolicy):
        logits = model(obs, act)
    else:
        logits = model(obs)
    if class_weights is None:
        return action_nll(logits, act)
    return action_nll_weighted(logits, act, class_weights)


def train_step(model, opt, obs, act, clip: float = 1.0, class_weights=None) -> float:
    loss = bc_loss(model, obs, act, class_weights)
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
           "balanced_weights", "action_nll_weighted",
           "device_of", "n_params", "CEDPolicy"]
