"""三档模型：B0 GRU / B1 纯解码器 Transformer / B2 CED。

公平性约束（改这三档时不许破坏）：
  * 共用同一个 ObsEncoder 与同一个动作头形状
  * 参数量对齐在同一量级：GRU ≈ 3.7M，Transformer ≈ 18.0M，CED ≈ 18.6M
  * 训练时三档都对**窗口内每一帧**出 loss（不是只算最后一帧），
    否则逐帧预测的 CED 会以 16 倍的样本效率优势赢得不公平
  * 推理时三档都只预测当前帧
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema_v0 import (
    OBS_DIM, N_RAYS, RAY_GRID, N_RAY_DIM, SELF_DIM, N_ENT, ENT_DIM,
    N_AUDIO, AUDIO_DIM, TASK_DIM,
    OFF_SELF, OFF_ENT, OFF_AUDIO, OFF_TASK,
    N_ACTION_TOKENS, ACTION_NCLASSES, MAX_CLASSES,
)

TOKENS_PER_FRAME = 16 + 1 + N_ENT + 1 + 1   # 35：16 射线几何 + 自身 + 16 实体 + 听觉 + 任务


# --------------------------------------------------------------------------
# 观测编码器：N,obs -> N,35,d
# --------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.d = d
        self.ray_proj = nn.Linear(N_RAY_DIM, d)
        self.ray_conv1 = nn.Conv2d(d, d, 3, stride=2, padding=1)
        self.ray_conv2 = nn.Conv2d(d, d, 3, stride=2, padding=1)
        self.ray_pos = nn.Parameter(torch.zeros(1, 16, d))
        self.self_proj = nn.Linear(SELF_DIM, d)
        self.ent_proj = nn.Linear(ENT_DIM, d)
        self.audio_proj = nn.Linear(N_AUDIO * AUDIO_DIM, d)
        self.task_proj = nn.Linear(TASK_DIM, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.shape[0]
        rays = obs[:, :OFF_SELF].view(n, N_RAYS, N_RAY_DIM)
        h = self.ray_proj(rays)
        h = h.view(n, RAY_GRID, RAY_GRID, self.d).permute(0, 3, 1, 2)
        h = F.gelu(self.ray_conv1(h))
        h = F.gelu(self.ray_conv2(h))                            # N,d,4,4
        h = h.flatten(2).transpose(1, 2) + self.ray_pos          # N,16,d
        s = self.self_proj(obs[:, OFF_SELF:OFF_ENT]).unsqueeze(1)
        e = self.ent_proj(obs[:, OFF_ENT:OFF_AUDIO].view(n, N_ENT, ENT_DIM))
        a = self.audio_proj(obs[:, OFF_AUDIO:OFF_TASK]).unsqueeze(1)
        t = self.task_proj(obs[:, OFF_TASK:]).unsqueeze(1)
        return self.norm(torch.cat([h, s, e, a, t], dim=1))      # N,35,d

    def frames(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """B,T,OBS -> B,T,35,d"""
        b, t = obs_seq.shape[:2]
        out = self(obs_seq.reshape(b * t, OBS_DIM))
        return out.view(b, t, TOKENS_PER_FRAME, self.d)


# --------------------------------------------------------------------------
# 动作头与损失
# --------------------------------------------------------------------------

class ActionHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, N_ACTION_TOKENS * MAX_CLASSES)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h)
        return logits.view(*h.shape[:-1], N_ACTION_TOKENS, MAX_CLASSES)


def action_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (...,14,121)，targets (...,14) long -> 标量。"""
    L = logits.reshape(-1, N_ACTION_TOKENS, MAX_CLASSES)
    T = targets.reshape(-1, N_ACTION_TOKENS)
    total = logits.new_zeros(T.shape[0])
    for i in range(N_ACTION_TOKENS):
        c = int(ACTION_NCLASSES[i])
        total = total + F.cross_entropy(L[:, i, :c], T[:, i], reduction="none")
    return total.mean()


@torch.no_grad()
def greedy_actions(logits: torch.Tensor) -> torch.Tensor:
    """(...,14,121) -> (...,14)，按各 token 的合法类别数取 argmax。"""
    lead = logits.shape[:-2]
    L = logits.reshape(-1, N_ACTION_TOKENS, MAX_CLASSES)
    out = [L[:, i, :int(ACTION_NCLASSES[i])].argmax(-1)
           for i in range(N_ACTION_TOKENS)]
    return torch.stack(out, dim=-1).reshape(*lead, N_ACTION_TOKENS)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# B0：GRU
# --------------------------------------------------------------------------

class GRUPolicy(nn.Module):
    def __init__(self, d: int = 256, hidden: int = 256):
        super().__init__()
        self.enc = ObsEncoder(d)
        self.gru = nn.GRU(d, hidden, num_layers=1, batch_first=True)
        self.head = ActionHead(hidden)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """B,T,OBS -> B,T,14,121"""
        toks = self.enc.frames(obs_seq)                          # B,T,35,d
        out, _ = self.gru(toks.mean(dim=2))
        return self.head(out)


# --------------------------------------------------------------------------
# B1：纯解码器 Transformer
# --------------------------------------------------------------------------

class TransformerPolicy(nn.Module):
    def __init__(self, d: int = 384, layers: int = 8, heads: int = 6,
                 max_frames: int = 32):
        super().__init__()
        self.max_frames = max_frames
        self.enc = ObsEncoder(d)
        self.pos = nn.Parameter(torch.zeros(1, max_frames * TOKENS_PER_FRAME, d))
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = ActionHead(d)

    @staticmethod
    def causal_mask(s: int, device) -> torch.Tensor:
        m = torch.full((s, s), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        b, t = obs_seq.shape[:2]
        assert t <= self.max_frames, f"t={t} > max_frames={self.max_frames}"
        toks = self.enc.frames(obs_seq).view(b, t * TOKENS_PER_FRAME, -1)
        toks = toks + self.pos[:, :t * TOKENS_PER_FRAME]
        out = self.body(toks, mask=self.causal_mask(toks.shape[1], toks.device))
        per_frame = out.view(b, t, TOKENS_PER_FRAME, -1).mean(dim=2)   # B,T,d
        return self.head(per_frame)                                    # B,T,14,121


# --------------------------------------------------------------------------
# B2：CED（编码器读、解码器写）
# --------------------------------------------------------------------------

class CEDPolicy(nn.Module):
    """编码器：帧内双向 + 帧间因果。解码器：只交叉注意该帧的 35 个编码状态。

    动作生成不重新穿透编码器 —— 这是与 B1 唯一的结构性差别。
    """

    def __init__(self, d: int = 384, enc_layers: int = 6, dec_layers: int = 2,
                 heads: int = 6, max_frames: int = 32):
        super().__init__()
        self.max_frames = max_frames
        self.enc = ObsEncoder(d)
        self.in_pos = nn.Parameter(torch.zeros(1, max_frames, TOKENS_PER_FRAME, d))
        enc_layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers,
                                             enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(
            d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.tok_emb = nn.Embedding(MAX_CLASSES, d)
        self.tok_pos = nn.Parameter(torch.zeros(1, N_ACTION_TOKENS, d))
        self.out = nn.Linear(d, MAX_CLASSES)

    @staticmethod
    def block_causal_mask(t_frames: int, device) -> torch.Tensor:
        f = torch.arange(t_frames, device=device).repeat_interleave(TOKENS_PER_FRAME)
        allow = f[None, :] <= f[:, None]
        m = torch.full((f.numel(), f.numel()), float("-inf"), device=device)
        m[allow] = 0.0
        return m

    def encode(self, obs_seq: torch.Tensor) -> torch.Tensor:
        b, t = obs_seq.shape[:2]
        assert t <= self.max_frames, f"t={t} > max_frames={self.max_frames}"
        toks = self.enc.frames(obs_seq) + self.in_pos[:, :t]      # B,T,35,d
        toks = toks.reshape(b, t * TOKENS_PER_FRAME, -1)
        return self.encoder(toks, mask=self.block_causal_mask(t, toks.device))

    def decode(self, memory: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """teacher forcing。memory N,35,d；prev_actions N,14 -> N,14,121"""
        n = memory.shape[0]
        inp = torch.cat([self.bos.expand(n, -1, -1),
                         self.tok_emb(prev_actions[:, :-1].clamp(0, MAX_CLASSES - 1))],
                        dim=1)
        q = inp + self.tok_pos[:, :inp.shape[1]]
        s = q.shape[1]
        qmask = torch.triu(torch.full((s, s), float("-inf"), device=q.device), 1)
        return self.out(self.decoder(q, memory, tgt_mask=qmask))

    def forward(self, obs_seq: torch.Tensor, actions: torch.Tensor | None = None):
        """训练：B,T,OBS + B,T,14 -> B,T,14,121；推理：-> B,14"""
        b, t = obs_seq.shape[:2]
        mem = self.encode(obs_seq).view(b, t, TOKENS_PER_FRAME, -1)
        if actions is None:
            return self.generate(mem[:, -1])
        logits = self.decode(mem.reshape(b * t, TOKENS_PER_FRAME, -1),
                             actions.reshape(b * t, N_ACTION_TOKENS))
        return logits.view(b, t, N_ACTION_TOKENS, MAX_CLASSES)

    def loss(self, obs_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return action_nll(self.forward(obs_seq, actions), actions)

    @torch.no_grad()
    def generate(self, memory: torch.Tensor) -> torch.Tensor:
        """朴素自回归，仅评测用。14 次解码器前向，不重跑编码器。"""
        n = memory.shape[0]
        acts = torch.zeros(n, 0, dtype=torch.long, device=memory.device)
        for i in range(N_ACTION_TOKENS):
            inp = torch.cat([self.bos.expand(n, -1, -1), self.tok_emb(acts)], dim=1)
            q = inp + self.tok_pos[:, :inp.shape[1]]
            s = q.shape[1]
            qmask = torch.triu(torch.full((s, s), float("-inf"), device=q.device), 1)
            logits = self.out(self.decoder(q, memory, tgt_mask=qmask))[:, -1]
            acts = torch.cat([acts, logits[:, :int(ACTION_NCLASSES[i])]
                              .argmax(-1, keepdim=True)], dim=1)
        return acts
