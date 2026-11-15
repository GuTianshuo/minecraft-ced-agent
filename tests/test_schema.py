"""schema 不变量。跑法：python tests/test_schema.py（不需要 pytest）。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schema_v0 as S


def test_offsets_contiguous():
    assert S.OFF_RAYS == 0
    assert S.OFF_SELF == S.N_RAYS * S.N_RAY_DIM
    assert S.OFF_ENT == S.OFF_SELF + S.SELF_DIM
    assert S.OFF_AUDIO == S.OFF_ENT + S.N_ENT * S.ENT_DIM
    assert S.OFF_TASK == S.OFF_AUDIO + S.N_AUDIO * S.AUDIO_DIM
    assert S.OBS_DIM == S.OFF_TASK + S.TASK_DIM == 2568


def test_feature_lists_agree_with_dims():
    assert len(S.RAY_FEATURES) == S.N_RAY_DIM
    assert len(S.SELF_FEATURES) == S.SELF_DIM == 48
    assert len(S.ENT_FEATURES) == S.ENT_DIM
    assert len(S.AUDIO_FEATURES) == S.AUDIO_DIM
    assert len(S.TASK_FEATURES) == S.TASK_DIM


def test_actions():
    assert len(S.ACTION_TOKENS) == S.N_ACTION_TOKENS == 14
    assert S.ACTION_NCLASSES.tolist() == [2] * 10 + [9, 121, 121, 4]
    assert S.MAX_CLASSES == 121


def test_yaw_roundtrip():
    idx = np.arange(121)
    rad = S.yaw_index_to_rad(idx)
    back = S.rad_to_yaw_index(rad)
    assert np.array_equal(idx, back)
    assert abs(rad[60]) < 1e-6
    assert abs(rad[0] + np.pi / 2) < 1e-6


def test_unpack_shapes():
    obs = S.empty_obs()
    b = S.unpack(obs)
    assert b["rays"].shape == (256, 9)
    assert b["self"].shape == (48,)
    assert b["ents"].shape == (16, 12)
    assert b["audio"].shape == (4, 3)
    assert b["task"].shape == (12,)
    S.validate_obs(obs)


def test_validate_rejects_bad():
    obs = S.empty_obs()
    obs[3] = np.nan
    try:
        S.validate_obs(obs)
    except AssertionError:
        pass
    else:
        raise AssertionError("nan should be rejected")

    bad = np.zeros(S.N_ACTION_TOKENS, dtype=np.int64)
    bad[11] = 121                     # yaw 越界
    try:
        S.validate_actions(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("out-of-range action should be rejected")


def test_describe_consistent():
    d = S.describe()
    assert d["obs_dim"] == S.OBS_DIM
    assert sum(x["count"] * x["dim"] for x in d["blocks"].values()) == S.OBS_DIM


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("schema tests passed")
