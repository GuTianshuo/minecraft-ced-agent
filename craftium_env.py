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

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "env" / "build" / "chop-tree"
MOD_SRC = ROOT / "env" / "lua" / "ced_obs"
OBS_FILE = "ced_obs.txt"


def package_env_dir() -> Path:
    import craftium
    return Path(craftium.__file__).resolve().parent / "craftium-envs" / "chop-tree"


def prepare(force: bool = True) -> Path:
    """从包内环境复制一份可写的 chop-tree，并挂上 ced_obs。"""
    if BUILD_DIR.exists():
        if not force:
            return BUILD_DIR
        shutil.rmtree(BUILD_DIR)
    shutil.copytree(package_env_dir(), BUILD_DIR)
    shutil.copytree(MOD_SRC, BUILD_DIR / "mods" / "ced_obs")

    world_mt = BUILD_DIR / "worlds" / "world" / "world.mt"
    lines = world_mt.read_text().splitlines()
    if not any(l.startswith("load_mod_ced_obs") for l in lines):
        lines.append("load_mod_ced_obs = mods/ced_obs")
        world_mt.write_text("\n".join(lines) + "\n")
    return BUILD_DIR


def make_env(seed: int | None = None, **kwargs):
    """构造一个挂了 ced_obs 的 ChopTree 环境（与 Craftium/ChopTree-v0 同参）。"""
    from craftium import CraftiumEnv

    env_dir = prepare()
    params = dict(
        env_dir=str(env_dir),
        obs_width=64,
        obs_height=64,
        max_timesteps=2000,
        init_frames=200,
        seed=seed,
        minetest_conf=dict(give_initial_stuff=True, initial_stuff="default:axe_steel"),
    )
    params.update(kwargs)
    return CraftiumEnv(**params)


def obs_frame_path(env) -> str:
    """ced_obs 写的帧文件路径（在当前 run_dir 的世界目录下）。"""
    return os.path.join(env.mt.run_dir, "worlds", "world", OBS_FILE)
