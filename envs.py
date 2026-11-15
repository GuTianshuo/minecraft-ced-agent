"""环境层。

DummyNavEnv  —— 合成 2D 导航环境，填满完整 obs 布局，只用于验证"观测 → 动作 → 数据 → 训练"链路。
TransportEnv —— 通过文件通道（transport.py）与真正的游戏进程通信的通用适配器，
                对 MockGame 与 Luanti/Minecraft 服务端是同一份代码。
"""

from __future__ import annotations

import math

import numpy as np

import transport
from schema_v0 import (
    N_RAYS, N_RAY_DIM, N_ENT, ENT_DIM, N_AUDIO, AUDIO_DIM,
    N_ACTION_TOKENS, OBS_DIM,
    OFF_RAYS, OFF_SELF, OFF_TASK,
    empty_obs, validate_actions,
    yaw_index_to_rad, rad_to_yaw_index,
)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DummyNavEnv:
    """平地到点任务。动作只用到 fwd 与 yaw，其余 token 保持 0。"""

    task_id = 0  # go_to_point

    def __init__(self, seed: int = 0, arena: float = 8.0, max_steps: int = 200,
                 step_size: float = 0.25, reach: float = 0.8):
        self.rng = np.random.default_rng(seed)
        self.arena = arena
        self.max_steps = max_steps
        self.step_size = step_size
        self.reach = reach
        self.pos = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.heading = 0.0
        self.t = 0
        self.prev_dist = 0.0
        self._obs = empty_obs()

    def _sample_goal(self):
        while True:
            g = self.rng.uniform(-self.arena, self.arena, size=2)
            if np.linalg.norm(g - self.pos) > 3.0:
                return g

    def _angle_to_goal(self) -> float:
        d = self.goal - self.pos
        return math.atan2(d[0], d[1])   # 与 heading 同约定：0 指向 +z

    def _build_obs(self) -> np.ndarray:
        obs = self._obs
        obs[:] = 0.0
        d = self.goal - self.pos
        dist = float(np.linalg.norm(d))

        rays = obs[OFF_RAYS:OFF_SELF].reshape(N_RAYS, N_RAY_DIM)
        center_row = 8 * N_RAYS // 16
        rel = _wrap(self._angle_to_goal() - self.heading)
        fov_half = math.radians(35.0)
        for u in range(16):
            ray_ang = (u - 7.5) / 7.5 * fov_half
            if abs(_wrap(rel - ray_ang)) < math.radians(6.0):
                r = rays[center_row + u]
                r[0] = 1.0
                r[4] = min(dist, 32.0) / 32.0
                r[6] = min(dist, 32.0) / 32.0
                r[7] = 1.0

        s = obs[OFF_SELF:OFF_SELF + 48]
        s[0], s[1], s[2] = self.pos[0] / 32.0, 0.0, self.pos[1] / 32.0
        s[3], s[4] = math.sin(self.heading), math.cos(self.heading)
        s[10] = 1.0
        s[13] = 1.0
        s[14] = 1.0
        s[17] = 1.0

        t = obs[OFF_TASK:]
        t[0] = self.task_id
        t[1] = d[0] / 32.0
        t[2] = d[1] / 32.0
        t[3] = min(dist, 32.0) / 32.0
        t[4] = np.clip(1.0 - dist / (2 * self.arena), 0.0, 1.0)
        return obs

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.rng.uniform(-self.arena / 2, self.arena / 2, size=2)
        self.heading = self.rng.uniform(-math.pi, math.pi)
        self.goal = self._sample_goal()
        self.t = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))
        return self._build_obs().copy()

    def step(self, action: np.ndarray):
        validate_actions(action)
        self.heading = _wrap(self.heading + float(yaw_index_to_rad(action[11])))
        if action[0] == 1:
            self.pos = np.clip(
                self.pos + self.step_size * np.array([math.sin(self.heading),
                                                      math.cos(self.heading)]),
                -self.arena, self.arena)
        self.t += 1
        dist = float(np.linalg.norm(self.goal - self.pos))
        reward = self.prev_dist - dist
        self.prev_dist = dist
        terminated = bool(dist < self.reach)
        truncated = bool(self.t >= self.max_steps)
        info = {"dist": dist, "tick": self.t, "env_id": 0, "episode_id": -1}
        return self._build_obs().copy(), float(reward), terminated, truncated, info

    def teacher_action(self) -> np.ndarray:
        delta = _wrap(self._angle_to_goal() - self.heading)
        act = np.zeros(N_ACTION_TOKENS, dtype=np.int64)
        act[11] = int(rad_to_yaw_index(np.array(delta)))
        act[0] = 1 if abs(delta) < 1.0 else 0
        return act


def rollout_teacher(env, max_steps: int = 200, seed: int | None = None):
    obs = env.reset(seed=seed)
    obs_seq, act_seq = [obs], []
    done = False
    while not done and len(act_seq) < max_steps:
        a = env.teacher_action()
        obs, _, term, trunc, _ = env.step(a)
        obs_seq.append(obs)
        act_seq.append(a)
        done = term or trunc
    return np.stack(obs_seq[:-1]), np.stack(act_seq)


class TransportEnv:
    """通过文件通道控制一个真实/模拟的游戏服务端。

    与 DummyNavEnv 的区别：观测、奖励、终止全部来自服务端，Python 不重新计算。
    这正是规划 v2 §3.1 要求的语义，也让"奖励由特权状态算出"的风险被隔离在服务端。
    """

    def __init__(self, root: str, timeout: float = 2.0, task_id: int = 0):
        import os
        self.root = root
        self.obs_path = os.path.join(root, "obs.txt")
        self.act_path = os.path.join(root, "act.txt")
        self.ctrl_path = os.path.join(root, "ctrl.txt")
        self.channel = transport.ObsChannel(self.obs_path, timeout=timeout)
        self.timeout = timeout
        self.task_id = task_id
        self.last_tick = -1
        self.episode_id = 0     # 服务端启动帧是 ep=0，第一次 reset 必须等到 ep>=1

    def reset(self, seed: int | None = None) -> np.ndarray:
        transport.write_control(self.ctrl_path, f"reset {0 if seed is None else int(seed)}")
        frame = self.channel.read(after_tick=self.last_tick,
                                  min_episode=self.episode_id + 1)
        self.last_tick, self.episode_id = frame.tick, frame.episode_id
        return frame.obs

    def step(self, action: np.ndarray):
        validate_actions(action)
        transport.write_action(self.act_path, self.last_tick, action)
        frame = self.channel.read(after_tick=self.last_tick,
                                  min_episode=self.episode_id)
        self.last_tick = frame.tick
        info = {"tick": frame.tick, "episode_id": frame.episode_id, "env_id": 0}
        return frame.obs, frame.reward, frame.terminated, frame.truncated, info

    def close(self):
        transport.write_control(self.ctrl_path, "stop")
