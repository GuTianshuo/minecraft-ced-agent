"""MockGame：假游戏服务端，用来在没有 Luanti/Minecraft 的情况下测通传输契约。

它按 transport.py 的格式以固定频率写帧、读动作、读控制，物理用 DummyNavEnv。
价值在于：适配器的等待、超时、episode 边界、原子替换全都能被测到，
而这些正是接真引擎时最容易出问题的地方。

它不模拟渲染、不模拟网络、不模拟真实物理 —— 不要用它跑训练结论。
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

from envs import DummyNavEnv
from schema_v0 import N_ACTION_TOKENS
import transport


def noop_action() -> np.ndarray:
    a = np.zeros(N_ACTION_TOKENS, dtype=np.int64)
    a[11] = 60      # yaw 中心 = 不转向
    return a


class MockGame(threading.Thread):
    def __init__(self, root: str, hz: float = 20.0, seed: int = 0):
        super().__init__(daemon=True)
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.obs_path = os.path.join(root, "obs.txt")
        self.act_path = os.path.join(root, "act.txt")
        self.ctrl_path = os.path.join(root, "ctrl.txt")
        self.period = 1.0 / hz
        self.env = DummyNavEnv(seed=seed)
        self.tick = 0
        self.episode_id = 0
        self.last_applied = -1
        self.done = False
        self._stop = threading.Event()
        self._ready = threading.Event()

    # ---- 生命周期
    def start(self):
        super().start()
        assert self._ready.wait(timeout=2.0), "mock game failed to start"
        return self

    def stop(self):
        transport.write_control(self.ctrl_path, "stop")
        self._stop.set()
        self.join(timeout=2.0)

    # ---- 主循环
    def run(self):
        obs = self.env.reset(seed=0)
        self.tick = 1
        transport.write_frame(self.obs_path, self.tick, self.episode_id, obs)
        self._ready.set()

        next_t = time.time()
        while not self._stop.is_set():
            ctrl = transport.read_control(self.ctrl_path)
            if ctrl:
                if ctrl[0] == "stop":
                    break
                if ctrl[0] == "reset" and len(ctrl) >= 2:
                    self.episode_id += 1
                    seed = int(ctrl[1])
                    obs = self.env.reset(seed=seed)
                    self.tick += 1
                    self.last_applied = -1
                    self.done = False
                    transport.write_frame(self.obs_path, self.tick, self.episode_id, obs)
                    os.remove(self.ctrl_path)
                next_t = time.time()
                continue

            if not self.done:
                a = transport.read_action(self.act_path)
                if a is not None and a[0] >= self.last_applied:
                    self.last_applied = int(a[0])
                    obs, r, term, trunc, info = self.env.step(a[1])
                else:
                    obs, r, term, trunc, info = self.env.step(noop_action())
                self.tick += 1
                self.done = bool(term or trunc)
                transport.write_frame(self.obs_path, self.tick, self.episode_id,
                                      obs, r, term, trunc)

            next_t += self.period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()
