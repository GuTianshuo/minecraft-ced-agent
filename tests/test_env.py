"""环境语义：确定性、教师可达、terminated/truncated 分离、观测合法。
跑法：python tests/test_env.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import DummyNavEnv, rollout_teacher
from schema_v0 import validate_obs


def test_determinism():
    a_obs, a_act = rollout_teacher(DummyNavEnv(seed=0), seed=7)
    b_obs, b_act = rollout_teacher(DummyNavEnv(seed=1), seed=7)
    assert np.array_equal(a_obs, b_obs), "same seed must give same rollout"
    assert np.array_equal(a_act, b_act)


def test_teacher_reaches_goal():
    env = DummyNavEnv(seed=0)
    wins = 0
    for i in range(20):
        obs = env.reset(seed=100 + i)
        done = success = False
        while not done:
            _, _, term, trunc, _ = env.step(env.teacher_action())
            done, success = (term or trunc), success or term
        wins += int(success)
    print(f"  teacher reached {wins}/20")
    assert wins >= 16, f"teacher only reached {wins}/20 —— 任务定义或教师有问题"


def test_truncated_is_not_terminated():
    env = DummyNavEnv(seed=0, max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, term, trunc, _ = env.step(env.teacher_action())
    assert trunc and not term, "time limit must be truncated, not terminated"


def test_obs_always_valid():
    env = DummyNavEnv(seed=3)
    obs = env.reset(seed=3)
    validate_obs(obs)
    for _ in range(50):
        obs, *_ = env.step(env.teacher_action())
        validate_obs(obs)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("env tests passed")
