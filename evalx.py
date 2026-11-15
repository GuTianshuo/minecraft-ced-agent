"""评测协议。

规则（与规划 v2 §4.4 一致，改这里要同步改文档）：
  * 固定初始状态：由 (seed, episode) 决定，不随策略变化
  * 3 个种子 × 20 episode，种子间成功率标准差 > 10 个百分点视为结果不可用
  * 成功判据由环境给，不在评测侧重新解释
  * shaping 只作诊断，不参与任何比较
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from schema_v0 import OFF_SELF
from runlib import policy_actions

MAX_CTX = 32
STUCK_STEPS = 60          # 3 秒 @20Hz，连续位移 < 0.05 格


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    final_dist: float
    stuck: bool
    shaping: float
    illegal: int


def _pos(obs: np.ndarray) -> tuple[float, float]:
    # self 段前三个是 px,py,pz（单位：格/32）
    return float(obs[OFF_SELF + 0]) * 32.0, float(obs[OFF_SELF + 2]) * 32.0


def run_episode(model, env, seed: int, max_ctx: int = MAX_CTX) -> EpisodeResult:
    obs = env.reset(seed=seed)
    buf = [obs]
    success, shaping, illegal = False, 0.0, 0
    stuck_run, stuck = 0, False
    prev = _pos(obs)
    steps = 0
    while True:
        act = policy_actions(model, np.stack(buf[-max_ctx:]))
        obs, r, term, trunc, info = env.step(act)
        buf.append(obs)
        steps += 1
        shaping += float(r)
        illegal += int(info.get("illegal", 0))
        cur = _pos(obs)
        moved = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur
        stuck_run = stuck_run + 1 if moved < 0.05 else 0
        stuck = stuck or stuck_run >= STUCK_STEPS
        if term:
            success = True
        if term or trunc:
            break
    return EpisodeResult(success, steps, float(info.get("dist", 0.0)),
                         stuck, shaping, illegal)


def evaluate(model, task_id: str, make_env, seeds=(0, 1, 2), episodes: int = 20,
             max_ctx: int = MAX_CTX) -> dict:
    model.eval()
    per_seed = {}
    for s in seeds:
        rows = []
        for ep in range(episodes):
            env = make_env(s * 1000 + ep)
            rows.append(run_episode(model, env, seed=s * 1000 + ep, max_ctx=max_ctx))
        per_seed[s] = {
            "success_rate": float(np.mean([r.success for r in rows])),
            "mean_steps": float(np.mean([r.steps for r in rows])),
            "stuck_rate": float(np.mean([r.stuck for r in rows])),
            "mean_shaping": float(np.mean([r.shaping for r in rows])),
            "illegal_rate": float(np.mean([r.illegal for r in rows])),
        }
    model.train()
    succ = [per_seed[s]["success_rate"] for s in seeds]
    return {
        "task": task_id,
        "per_seed": per_seed,
        "success_mean": float(np.mean(succ)),
        "success_std": float(np.std(succ)),
        "usable": bool(np.std(succ) <= 0.10),
        "episodes_per_seed": episodes,
    }


def check_gate(result: dict, min_success: float) -> tuple[bool, str]:
    if not result["usable"]:
        return False, (f"种子间方差过大 (std={result['success_std']:.3f} > 0.10)，"
                       f"结果不可用；先加数据或减少随机性")
    if result["success_mean"] < min_success:
        return False, (f"成功率 {result['success_mean']:.1%} < 门禁 {min_success:.0%}；"
                       f"按规划 §4.3，先查数据管线，不改架构")
    return True, "pass"


def format_table(result: dict) -> str:
    lines = [f"task={result['task']}  episodes/seed={result['episodes_per_seed']}"]
    lines.append(f"{'seed':>4}  {'succ':>6}  {'steps':>6}  {'stuck':>6}  {'illegal':>7}")
    for s, r in result["per_seed"].items():
        lines.append(f"{s:>4}  {r['success_rate']:>6.1%}  {r['mean_steps']:>6.1f}  "
                     f"{r['stuck_rate']:>6.1%}  {r['illegal_rate']:>7.2f}")
    lines.append(f"mean={result['success_mean']:.1%}  std={result['success_std']:.3f}  "
                 f"usable={result['usable']}")
    return "\n".join(lines)
