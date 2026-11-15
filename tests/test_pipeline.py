"""端到端：采集 → 分片 → 训练 → 评测。

跑法：python tests/test_pipeline.py
它不验证学习效果（那是门禁的事），只验证数据能流过去、形状与切分不出错。
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collect
import evalx
import shards
import tasks
from runlib import make_model, train_step
from schema_v0 import OBS_DIM, N_ACTION_TOKENS


def test_shard_roundtrip():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((7, OBS_DIM)).astype(np.float32)
    act = rng.integers(0, 2, size=(7, N_ACTION_TOKENS)).astype(np.int64)
    with tempfile.TemporaryDirectory() as d:
        p = shards.write_episode(d, "T0", 3, obs, act, "unit")
        ep = shards.read_episode(p)
        assert ep["obs"].shape == (7, OBS_DIM)
        assert ep["obs"].dtype == np.float32
        assert ep["act"].dtype == np.int64
        assert np.allclose(ep["obs"], np.float16(obs).astype(np.float32), atol=1e-3)
        assert ep["meta"]["format"] == shards.FORMAT
        assert ep["meta"]["split"] == "train"


def test_unimplemented_task_is_loud():
    for tid in ("T1", "T3", "T6"):
        try:
            tasks.make_task(tid)
        except NotImplementedError:
            continue
        raise AssertionError(f"{tid} should not be silently available on dummy env")


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        paths = collect.collect(d, "T0", 12, "teacher", verbose=False)
        assert len(paths) == 12, len(paths)

        store = shards.EpisodeStore(d, "T0")
        stats = store.stats()
        assert stats["steps"] > 50, stats
        assert 0 < stats["bytes_per_step"] < stats["bytes_per_step_uncompressed"]

        splits = store.splits()
        assert len(splits["train"]) >= 8 and len(splits["val"]) >= 1, splits

        model = make_model("gru", d=32, hidden=32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(0)
        loss = float("nan")
        for _ in range(10):
            obs, act = store.sample_batch(splits["train"], 4, 8, rng)
            loss = train_step(model, opt, obs, act)
        assert np.isfinite(loss), loss

        res = evalx.evaluate(model, "T0",
                             lambda s: tasks.make_task("T0", seed=s),
                             seeds=(0,), episodes=2)
        assert 0.0 <= res["success_mean"] <= 1.0
        assert 0.0 <= res["per_seed"][0]["stuck_rate"] <= 1.0
    print("  pipeline ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("pipeline tests passed")
