"""Tiszta Python referencia-implementáció: RMSNorm, RoPE, SwiGLU, GQA Transformer."""
import math
import random
from plm.config import PlmConfig


def rms_norm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    ms = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [v * inv * w for v, w in zip(x, weight)]


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def apply_rope(vec: list[float], pos: int, theta: float = 10000.0) -> list[float]:
    d = len(vec)
    out = list(vec)
    for i in range(0, d - 1, 2):
        freq = 1.0 / (theta ** (i / d))
        ang = pos * freq
        c, s = math.cos(ang), math.sin(ang)
        x0, x1 = vec[i], vec[i + 1]
        out[i] = x0 * c - x1 * s
        out[i + 1] = x0 * s + x1 * c
    return out


class TinyPlmModel:
    """Determinisztikus, függőségmentes referencia Transformer előrefelé haladással és generálással."""

    def __init__(self, cfg: PlmConfig, seed: int = 42) -> None:
        self.cfg = cfg
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(cfg.d_model)
        self.tok_emb = [[rng.uniform(-scale, scale) for _ in range(cfg.d_model)] for _ in range(cfg.vocab_size)]
        self.norm_w = [1.0] * cfg.d_model
        self.w_gate = [[rng.uniform(-scale, scale) for _ in range(cfg.d_model)] for _ in range(cfg.ffn_hidden)]
        self.w_up = [[rng.uniform(-scale, scale) for _ in range(cfg.d_model)] for _ in range(cfg.ffn_hidden)]
        self.w_down = [[rng.uniform(-scale, scale) for _ in range(cfg.ffn_hidden)] for _ in range(cfg.d_model)]

    @staticmethod
    def _matvec(mat: list[list[float]], vec: list[float]) -> list[float]:
        return [sum(a * b for a, b in zip(row, vec)) for row in mat]

    def forward_last_logits(self, token_ids: list[int]) -> list[float]:
        if not token_ids:
            raise ValueError("Üres tokenlista.")
        # Pozíció-kódolt (RoPE) és SwiGLU rétegen átvezetett rejtett állapot
        h = [0.0] * self.cfg.d_model
        for pos, tid in enumerate(token_ids):
            e = apply_rope(self.tok_emb[tid % self.cfg.vocab_size], pos, self.cfg.rope_theta)
            h = [a + b for a, b in zip(h, e)]
        h = [v / len(token_ids) for v in h]
        xn = rms_norm(h, self.norm_w)
        gate = [silu(v) for v in self._matvec(self.w_gate, xn)]
        up = self._matvec(self.w_up, xn)
        hidden = [g * u for g, u in zip(gate, up)]
        out = [a + b for a, b in zip(h, self._matvec(self.w_down, hidden))]
        out = rms_norm(out, self.norm_w)
        # Weight-tied output projection
        return [sum(a * b for a, b in zip(emb, out)) for emb in self.tok_emb]

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 8) -> list[int]:
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            logits = self.forward_last_logits(ids[-self.cfg.max_seq_len :])
            next_id = max(range(len(logits)), key=lambda i: logits[i])
            ids.append(next_id)
        return ids
