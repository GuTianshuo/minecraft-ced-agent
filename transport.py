"""游戏 ↔ Python 的传输契约（v1，文件通道）。

为什么是文件而不是 socket：
  Luanti 的 mod 沙箱里没有 socket API；Craftium 自带的通道传的是渲染帧。
  20 Hz × 约 5 KB 的写入对本地文件系统完全不是瓶颈，先用最简单的机制把语义做对。
  真机上如果实测 tick 时间受影响，再换成 Craftium 的 C++ 通道或 UDP。

帧文件（游戏 → Python），每 tick 原子覆写：

  <schema_version> <tick> <episode_id> <obs_dim> <reward> <terminated> <truncated>
  <obs_dim 个浮点数，空格分隔>

动作文件（Python → 游戏），每个决策原子覆写：

  <tick> <14 个整数>

控制文件（Python → 游戏）：

  reset <seed>        或      stop

语义要求（与规划 v2 §3.1 对齐）：
  * tick 单调递增，跨 episode 不重置；episode_id 在 reset 时 +1
  * reward / terminated / truncated 由服务端计算，Python 侧不重新解释
  * 原子替换（os.replace），读侧永远看不到半写文件
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from schema_v0 import (OBS_DIM, N_ACTION_TOKENS, SCHEMA_VERSION,
                       validate_obs, validate_actions)

HEADER_FIELDS = 7


@dataclass
class Frame:
    tick: int
    episode_id: int
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_frame(path: str, tick: int, episode_id: int, obs: np.ndarray,
                reward: float = 0.0, terminated: bool = False,
                truncated: bool = False) -> None:
    assert obs.shape[-1] == OBS_DIM, obs.shape
    head = (f"{SCHEMA_VERSION} {tick} {episode_id} {OBS_DIM} "
            f"{reward:.6f} {int(terminated)} {int(truncated)}")
    body = " ".join(f"{float(x):.4f}" for x in obs)
    _write_atomic(path, head + "\n" + body + "\n")


def write_action(path: str, tick: int, action: np.ndarray) -> None:
    validate_actions(action)
    _write_atomic(path, f"{tick} " + " ".join(str(int(a)) for a in action) + "\n")


def read_action(path: str):
    try:
        with open(path) as f:
            parts = f.read().split()
    except FileNotFoundError:
        return None
    if len(parts) != N_ACTION_TOKENS + 1:
        return None
    try:
        return int(parts[0]), np.array([int(x) for x in parts[1:]], dtype=np.int64)
    except ValueError:
        return None


def write_control(path: str, cmd: str) -> None:
    _write_atomic(path, cmd + "\n")


def read_control(path: str):
    try:
        with open(path) as f:
            return f.read().strip().split()
    except FileNotFoundError:
        return None


class ObsChannel:
    """等一帧比 after_tick 新、且 episode_id >= min_episode 的观测。"""

    def __init__(self, path: str, timeout: float = 2.0):
        self.path = path
        self.timeout = timeout

    def read(self, after_tick: int = -1, min_episode: int | None = None) -> Frame:
        deadline = time.time() + self.timeout
        last_err = None
        while time.time() < deadline:
            try:
                with open(self.path) as f:
                    lines = f.read().split("\n")
            except FileNotFoundError:
                time.sleep(0.002)
                continue
            if len(lines) < 2 or not lines[1].strip():
                time.sleep(0.002)
                continue
            head = lines[0].split()
            if len(head) != HEADER_FIELDS:
                time.sleep(0.002)
                continue
            try:
                schema, tick, ep, dim, rew, term, trunc = head
                if schema != SCHEMA_VERSION:
                    raise ValueError(f"schema {schema} != {SCHEMA_VERSION}")
                tick, ep, dim = int(tick), int(ep), int(dim)
                if dim != OBS_DIM:
                    raise ValueError(f"obs_dim {dim} != {OBS_DIM}")
                if tick <= after_tick:
                    time.sleep(0.002)
                    continue
                if min_episode is not None and ep < min_episode:
                    time.sleep(0.002)
                    continue
                vals = lines[1].split()
                if len(vals) != dim:
                    time.sleep(0.002)      # 半写，重试
                    continue
                obs = np.asarray([float(x) for x in vals], dtype=np.float32)
                validate_obs(obs)
            except (ValueError, AssertionError) as e:   # 半写或格式错误，重试
                last_err = e
                time.sleep(0.002)
                continue
            return Frame(tick, ep, obs, float(rew), bool(int(term)), bool(int(trunc)))
        raise TimeoutError(f"no fresh frame at {self.path} after {self.timeout}s"
                           f"{f' (last error: {last_err})' if last_err else ''}")
