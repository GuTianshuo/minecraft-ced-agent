"""传输契约：文件通道往返、超时、以及用 MockGame 跑通适配器。

跑法：python tests/test_adapter.py
它测的是"等待/超时/episode 边界/原子替换"这些接真引擎时最易出错的地方，
不测物理，也不产生任何可用的训练结论。
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transport
from envs import TransportEnv
from mock_game import MockGame
from schema_v0 import (OBS_DIM, N_ACTION_TOKENS, OFF_SELF, empty_obs, validate_obs)


def test_frame_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "obs.txt")
        obs = empty_obs()
        obs[100] = 0.5          # 随便设一个非零值，验证 roundtrip 不丢数据
        transport.write_frame(p, tick=7, episode_id=2, obs=obs, reward=1.5,
                              terminated=True)
        fr = transport.ObsChannel(p, timeout=0.5).read()
        assert fr.tick == 7 and fr.episode_id == 2
        assert np.allclose(fr.obs, obs)
        assert fr.reward == 1.5 and fr.terminated and not fr.truncated

        # 只等更新的帧，应该超时
        try:
            transport.ObsChannel(p, timeout=0.05).read(after_tick=7)
        except TimeoutError:
            pass
        else:
            raise AssertionError("stale frame should time out")


def test_action_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "act.txt")
        a = np.zeros(N_ACTION_TOKENS, dtype=np.int64)
        a[11] = 60
        transport.write_action(p, 12, a)
        tick, back = transport.read_action(p)
        assert tick == 12 and np.array_equal(back, a)
        assert transport.read_action(os.path.join(d, "missing.txt")) is None


def test_env_against_mock_game():
    with tempfile.TemporaryDirectory() as d:
        game = MockGame(d, hz=50.0, seed=0).start()
        try:
            env = TransportEnv(d, timeout=2.0)
            obs = env.reset(seed=3)
            validate_obs(obs)
            assert env.episode_id >= 1, "reset must observe a new episode"

            a = np.zeros(N_ACTION_TOKENS, dtype=np.int64)
            a[11] = 60          # 不转向
            a[0] = 1            # 前进
            tick = env.last_tick
            px0, pz0 = float(obs[OFF_SELF + 0]), float(obs[OFF_SELF + 2])
            for _ in range(6):
                obs, r, term, trunc, info = env.step(a)
                validate_obs(obs)
                assert env.last_tick > tick, "tick must advance"
                tick = env.last_tick
            moved = abs(float(obs[OFF_SELF + 0]) - px0) + abs(float(obs[OFF_SELF + 2]) - pz0)
            assert moved > 1e-4, "forward action did not move the agent"

            obs2 = env.reset(seed=4)
            assert env.episode_id >= 2, "second reset must start a new episode"
            assert env.last_tick > tick
            env.close()
        finally:
            game.stop()


def test_timeout_when_no_server():
    with tempfile.TemporaryDirectory() as d:
        env = TransportEnv(d, timeout=0.05)
        try:
            env.reset(seed=0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("should time out without a server")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("adapter tests passed")
