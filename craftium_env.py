"""Craftium 接入：把 ced_obs 结构化观测 mod 注入 chop-tree 环境。

Craftium 安装包里自带的环境目录是只读的（在 site-packages 里），所以这里把它复制到
`env/build/chop-tree`（已 gitignore），再加上本仓库的 `env/lua/ced_obs`，并在世界配置里
启用该 mod。每次 prepare() 都会从包内环境重新复制，保证不会累积手工改动。

用法：

    import craftium_env
    env = craftium_env.make_env(seed=0)
    obs, _ = env.reset()
    path = craftium_env.obs_frame_path(env)
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np

import transport
from schema_v0 import OFF_SELF, validate_actions, yaw_index_to_rad

ROOT = Path(__file__).resolve().parent
MOD_SRC = ROOT / "env" / "lua" / "ced_obs"
OBS_FILE = "ced_obs.txt"


def package_env_dir(env_name: str = "chop-tree") -> Path:
    import craftium
    return Path(craftium.__file__).resolve().parent / "craftium-envs" / env_name


def prepare(env_name: str = "chop-tree", force: bool = True) -> Path:
    """从包内环境复制一份可写的环境目录，并挂上 ced_obs。"""
    build_dir = ROOT / "env" / "build" / env_name
    if build_dir.exists():
        if not force:
            return build_dir
        shutil.rmtree(build_dir)
    shutil.copytree(package_env_dir(env_name), build_dir)
    shutil.copytree(MOD_SRC, build_dir / "mods" / "ced_obs")

    world_mt = build_dir / "worlds" / "world" / "world.mt"
    lines = world_mt.read_text().splitlines()
    if not any(l.startswith("load_mod_ced_obs") for l in lines):
        lines.append("load_mod_ced_obs = mods/ced_obs")
        world_mt.write_text("\n".join(lines) + "\n")
    return build_dir


def make_env(seed: int | None = None, env_name: str = "chop-tree", **kwargs):
    """构造一个挂了 ced_obs 的环境。"""
    from craftium import CraftiumEnv

    env_dir = prepare(env_name)
    params = dict(
        env_dir=str(env_dir),
        obs_width=64,
        obs_height=64,
        max_timesteps=2000,
        init_frames=200,
        seed=seed,
        minetest_conf=dict(give_initial_stuff=True, initial_stuff="default:axe_steel",
                           # 服务端步长决定客户端上报位置的频率：默认 ~0.09 s 时，
                           # 观测里的位置每 5–8 个决策才更新一次（动作一直在用旧状态）。
                           # 0.01 让位置每决策都刷新（0008）。
                           dedicated_server_step=0.01),
    )
    params.update(kwargs)
    return CraftiumEnv(**params)


def obs_frame_path(env) -> str:
    """ced_obs 写的帧文件路径（在当前 run_dir 的世界目录下）。"""
    return os.path.join(env.mt.run_dir, "worlds", "world", OBS_FILE)


# ---------------------------------------------------------------- 动作通道
#
# 0005 的结论：按键走 Craftium 的 Dict（每帧重发），朝向走 look.txt（绝对值，
# 由 Lua 用 set_look_horizontal/vertical 应用）。鼠标通道不用。

LOOK_FILE = "look.txt"

# 14 个 token 里前 10 个是 0/1 按键，按 schema_v0.ACTION_TOKENS 的顺序
TOKEN_KEYS = ("forward", "backward", "left", "right", "jump", "sneak",
              "aux1", "dig", "place", "drop")
PITCH_LIMIT = math.pi / 2 - 0.01


def tokens_to_action(tokens) -> dict:
    """14 个 token -> Craftium 的 Dict 动作。yaw/pitch/hold 不在这里处理。"""
    from craftium.craftium_env import ACTION_ORDER

    d = {k: 0 for k in ACTION_ORDER[:-1]}
    for i, name in enumerate(TOKEN_KEYS):
        d[name] = int(tokens[i])
    hotbar = int(tokens[10])
    if 1 <= hotbar <= 9:
        d[f"slot_{hotbar}"] = 1
    d["mouse"] = np.zeros(2, dtype=np.float32)
    return d


class CraftiumCedEnv:
    """Craftium 上的 14-token 环境，接口与 DummyNavEnv 对齐。

    - 观测：ced_obs 的 2568 维帧（不是 Craftium 的像素）
    - 动作：schema_v0 的 14 个 token
    - 一次 step = 等一帧新观测；这段时间里同一个按键动作每帧重发（约 3.4 帧）
    - yaw/pitch token 是相对量，由本类累计成绝对角度写进 look.txt

    hold token（0..3）：在新观测到达后继续重复该动作的额外帧数，用来给出
    "这个动作持续多久" 的控制维度。默认 0。
    """

    def __init__(self, seed: int | None = None, max_frames_per_decision: int = 12,
                 frame_timeout: float = 5.0, **kwargs):
        self.craftium = make_env(seed=seed, **kwargs)
        base = os.path.dirname(obs_frame_path(self.craftium))
        self.obs_path = os.path.join(base, OBS_FILE)
        self.look_path = os.path.join(base, LOOK_FILE)
        self.channel = transport.ObsChannel(self.obs_path, timeout=frame_timeout)
        self.max_frames_per_decision = max_frames_per_decision
        self.last_frame = None
        self.last_pixels = None
        self.logical_tick = 0
        self.episode_id = 0
        self.server_tick_offset = 0

    # ---- 内部
    def _server_tick(self) -> int:
        try:
            with open(self.obs_path) as f:
                head = f.readline().split()
            return int(head[1]) if len(head) >= 2 else -1
        except (OSError, ValueError, IndexError):
            return -1

    @staticmethod
    def _look_from_frame(obs) -> tuple[float, float]:
        return (float(np.arctan2(obs[OFF_SELF + 3], obs[OFF_SELF + 4])),
                float(np.arctan2(obs[OFF_SELF + 5], obs[OFF_SELF + 6])))

    def _write_look(self, yaw: float, pitch: float) -> None:
        tmp = self.look_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(f"{yaw:.6f} {pitch:.6f}\n")
        os.replace(tmp, self.look_path)

    # ---- 公共接口
    def reset(self, seed: int | None = None):
        self.last_pixels, _ = self.craftium.reset(seed=seed)   # 硬重置：服务端进程重启，tick 归零
        frame = self.channel.read(after_tick=-1)
        self.episode_id += 1
        self.server_tick_offset = self.logical_tick + 1 - frame.tick
        self.last_frame = frame
        self.logical_tick = frame.tick + self.server_tick_offset
        return frame.obs.copy()

    def step(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64)
        validate_actions(tokens)

        yaw, pitch = self._look_from_frame(self.last_frame.obs)
        yaw += float(yaw_index_to_rad(np.array(tokens[11])))
        pitch += float(yaw_index_to_rad(np.array(tokens[12])))
        pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch))
        self._write_look(yaw, pitch)

        action = tokens_to_action(tokens)
        hold = int(tokens[13])
        start_tick = self.last_frame.tick
        reward, frames = 0.0, 0
        term = trunc = False

        while True:
            pixels, r, term, trunc, _ = self.craftium.step(action)
            self.last_pixels = pixels
            reward += float(r)
            frames += 1
            tick = self._server_tick()
            if tick > start_tick or frames >= self.max_frames_per_decision or term or trunc:
                break

        if tick > start_tick and hold and not (term or trunc):
            for _ in range(hold):
                pixels, r, term, trunc, _ = self.craftium.step(action)
                self.last_pixels = pixels
                reward += float(r)
                frames += 1
                if term or trunc:
                    break

        frame = self.channel.read(after_tick=start_tick)
        # look.txt 是在两次 tick 之间写的：这一帧可能还没应用它。核对一次，
        # 不符就再等一帧（最多两次），避免把旧朝向的观测配给新动作。
        for _ in range(2):
            fy, fp = self._look_from_frame(frame.obs)
            if abs(fy - yaw) < 0.05 and abs(fp - pitch) < 0.05:
                break
            frame = self.channel.read(after_tick=frame.tick)
        self.last_frame = frame
        self.logical_tick = frame.tick + self.server_tick_offset
        info = {
            "tick": self.logical_tick,
            "env_id": 0,
            "episode_id": self.episode_id,
            "frames": frames,
            "yaw": yaw,
            "pitch": pitch,
        }
        # 奖励与终止以 ced_obs 帧为准（任务在 Lua 侧算），Craftium 自带的那套只用来
        # 提前结束（例如玩家死亡）。
        return (frame.obs.copy(), frame.reward,
                bool(frame.terminated or term), bool(frame.truncated or trunc), info)

    def close(self):
        self.craftium.close()
