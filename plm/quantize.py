"""Szimmetrikus INT8 és blokkos INT4 (Q4_0) kvantálás mobil futtatáshoz."""
import math


def quantize_q8(weights: list[float]) -> tuple[list[int], float]:
    max_abs = max((abs(w) for w in weights), default=0.0)
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = [max(-127, min(127, int(round(w / scale)))) for w in weights]
    return q, scale


def dequantize_q8(q: list[int], scale: float) -> list[float]:
    return [val * scale for val in q]


def quantize_q4_blocks(weights: list[float], block_size: int = 32) -> list[tuple[list[int], float]]:
    blocks = []
    for i in range(0, len(weights), block_size):
        chunk = weights[i : i + block_size]
        max_abs = max((abs(w) for w in chunk), default=0.0)
        scale = max_abs / 7.0 if max_abs > 0 else 1.0
        q = [max(-7, min(7, int(round(w / scale)))) for w in chunk]
        blocks.append((q, scale))
    return blocks


def dequantize_q4_blocks(blocks: list[tuple[list[int], float]]) -> list[float]:
    out: list[float] = []
    for q, scale in blocks:
        out.extend(v * scale for v in q)
    return out


def mse(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a))
