"""模型契约：形状、因果性、参数对齐、生成范围。

因果性测试是这套骨架里最重要的一条：未来帧不得改变当前帧的动作 logits，
否则长程记忆与在线推理的结论全部作废。
跑法：python tests/test_models.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema_v0 import OBS_DIM, N_ACTION_TOKENS, ACTION_NCLASSES
from models import (GRUPolicy, TransformerPolicy, CEDPolicy,
                    action_nll, greedy_actions, n_params)

torch.manual_seed(0)


def _obs(b, t):
    return torch.rand(b, t, OBS_DIM)


def _acts(b, t):
    a = torch.stack([torch.randint(0, int(c), (b, t)) for c in ACTION_NCLASSES], dim=-1)
    return a


def test_shapes():
    x, a = _obs(2, 8), _acts(2, 8)
    models = {
        "gru": GRUPolicy(d=64, hidden=64),
        "tf": TransformerPolicy(d=128, layers=2, heads=4, max_frames=16),
        "ced": CEDPolicy(d=128, enc_layers=2, dec_layers=1, heads=4, max_frames=16),
    }
    for name, m in models.items():
        if name == "ced":
            out = m(x, a)
        else:
            out = m(x)
        assert out.shape == (2, 8, N_ACTION_TOKENS, 121), (name, out.shape)
        assert torch.isfinite(action_nll(out, a))
        act = greedy_actions(out)[:, -1]
        assert act.shape == (2, N_ACTION_TOKENS)
    print("  shapes ok")


def test_causality():
    """未来帧不得影响当前帧输出。"""
    x, a = _obs(2, 8), _acts(2, 10)
    x_future = torch.rand(2, 2, OBS_DIM)
    x_long = torch.cat([x, x_future], dim=1)
    a_long = torch.cat([a[:, :8], _acts(2, 2)], dim=1)

    for name, m in {
        "gru": GRUPolicy(d=64, hidden=64),
        "tf": TransformerPolicy(d=128, layers=2, heads=4, max_frames=16),
        "ced": CEDPolicy(d=128, enc_layers=2, dec_layers=1, heads=4, max_frames=16),
    }.items():
        with torch.no_grad():
            if name == "ced":
                short = m(x, a[:, :8])
                long_ = m(x_long, a_long)
            else:
                short = m(x)
                long_ = m(x_long)
        assert torch.allclose(long_[:, :8], short, atol=1e-4), f"{name} leaks the future"
    print("  causality ok")


def test_param_parity():
    tf, ced = TransformerPolicy(), CEDPolicy()
    ratio = n_params(ced) / n_params(tf)
    print(f"  params: tf={n_params(tf)/1e6:.2f}M  ced={n_params(ced)/1e6:.2f}M  "
          f"gru={n_params(GRUPolicy())/1e6:.2f}M  ratio={ratio:.2f}")
    assert 0.8 <= ratio <= 1.25, "CED 与 Transformer 参数不再对齐，对照实验失效"


def test_generate_range():
    m = CEDPolicy(d=128, enc_layers=1, dec_layers=1, heads=4, max_frames=8)
    with torch.no_grad():
        acts = m(_obs(3, 4), None)
    assert acts.shape == (3, N_ACTION_TOKENS)
    assert (acts >= 0).all()
    assert (acts < torch.as_tensor(ACTION_NCLASSES)).all()
    print("  generate ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("model tests passed")
