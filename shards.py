"""episode 分片：写入、读取、切分、采样。

磁盘账本（重要，修正过一次）：
  观测 2568 维 × float16 = 5136 字节/决策步。规划 v2 写的"约 1 KB/步"是错的。
  实测前先按 5 KB/步 + 压缩比 3–10 倍记账：
    8 环境 × 20 步/s × 5 KB = 800 KB/s ≈ 2.9 GB/h（未压缩）
  因此必须 savez_compressed，并且 collect 结束要打印真实 bytes/step。
  配额见 configs/base.yaml 的 data.max_disk_gb，超了要轮转，不许静默增长。
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections import OrderedDict

import numpy as np

from schema_v0 import OBS_DIM, N_ACTION_TOKENS, SCHEMA_VERSION

FORMAT = "ced-shard-v1"
SPLITS = ("train", "val", "test")


def split_of(seed: int) -> str:
    r = abs(int(seed)) % 10
    return "train" if r < 8 else ("val" if r == 8 else "test")


def episode_path(root: str, task: str, seed: int) -> str:
    return os.path.join(root, task, f"{task}_s{int(seed):08d}.npz")


def write_episode(root: str, task: str, seed: int, obs: np.ndarray, act: np.ndarray,
                  policy: str, extra: dict | None = None) -> str:
    assert obs.shape[1] == OBS_DIM, obs.shape
    assert act.shape[1] == N_ACTION_TOKENS, act.shape
    path = episode_path(root, task, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "seed": int(seed),
        "split": split_of(seed),
        "policy": policy,
        "steps": int(len(act)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    np.savez_compressed(
        path,
        obs=obs.astype(np.float16),
        act=act.astype(np.int16),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return path


def read_episode(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return {
            "obs": z["obs"].astype(np.float32),
            "act": z["act"].astype(np.int64),
            "meta": meta,
        }


def list_episodes(root: str, task: str | None = None) -> list[str]:
    pattern = os.path.join(root, task, "*.npz") if task else os.path.join(root, "**", "*.npz")
    return sorted(glob.glob(pattern, recursive=True))


def split_files(files: list[str]) -> dict[str, list[str]]:
    out = {s: [] for s in SPLITS}
    for p in files:
        name = os.path.basename(p)
        try:
            seed = int(name.rsplit("_s", 1)[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        out[split_of(seed)].append(p)
    return out


class EpisodeStore:
    """按需加载 + LRU。20 万步全放内存是 1 GB 级别，不允许。"""

    def __init__(self, root: str, task: str | None = None, cache_size: int = 32):
        self.root = root
        self.files = list_episodes(root, task)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self.files)

    def splits(self) -> dict[str, list[str]]:
        return split_files(self.files)

    def get(self, path: str) -> dict:
        hit = self._cache.get(path)
        if hit is not None:
            self._cache.move_to_end(path)
            return hit
        ep = read_episode(path)
        self._cache[path] = ep
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return ep

    def sample_batch(self, files: list[str], batch: int, window: int,
                     rng: np.random.Generator):
        import torch
        if not files:
            raise ValueError("empty file list")
        obs_b, act_b = [], []
        for _ in range(batch):
            ep = self.get(files[int(rng.integers(len(files)))])
            obs, act = ep["obs"], ep["act"]
            t = len(act)
            if t >= window:
                start = int(rng.integers(0, t - window + 1))
                obs_b.append(obs[start:start + window])
                act_b.append(act[start:start + window])
            else:
                pad = window - t
                obs_b.append(np.concatenate([obs, np.repeat(obs[-1:], pad, 0)]))
                act_b.append(np.concatenate([act, np.repeat(act[-1:], pad, 0)]))
        return (torch.as_tensor(np.stack(obs_b), dtype=torch.float32),
                torch.as_tensor(np.stack(act_b), dtype=torch.long))

    def stats(self) -> dict:
        steps = 0
        nbytes = 0
        for p in self.files:
            nbytes += os.path.getsize(p)
            with np.load(p, allow_pickle=False) as z:
                steps += int(z["act"].shape[0])
        return {
            "episodes": len(self.files),
            "steps": steps,
            "bytes": nbytes,
            "bytes_per_step": (nbytes / steps) if steps else 0.0,
            "bytes_per_step_uncompressed": 2 * OBS_DIM + 2 * N_ACTION_TOKENS,
        }
