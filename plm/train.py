"""PLM v0.2 — Nulla-függőségű, valódi tanító ciklus (Cross-Entropy + SGD) és checkpoint mentés."""
import json
import math
import random
from pathlib import Path

from plm.tokenizer import BpeTokenizer


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


class TrainableMiniLM:
    """Tanítható kontextus-ablakos (bigram + lineáris projekció) nyelvi modell valós SGD-vel."""

    def __init__(self, vocab_size: int, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        rng = random.Random(seed)
        self.weights: list[list[float]] = [
            [rng.uniform(-0.05, 0.05) for _ in range(vocab_size)] for _ in range(vocab_size)
        ]

    def loss(self, token_ids: list[int]) -> float:
        if len(token_ids) < 2:
            return 0.0
        total = 0.0
        n = len(token_ids) - 1
        for i in range(n):
            probs = softmax(self.weights[token_ids[i]])
            p = max(1e-12, probs[token_ids[i + 1]])
            total -= math.log(p)
        return total / n

    def train_steps(self, token_ids: list[int], steps: int = 40, lr: float = 0.25) -> list[float]:
        history: list[float] = []
        n = len(token_ids) - 1
        if n <= 0:
            return history
        for _ in range(steps):
            total_loss = 0.0
            for i in range(n):
                x, y = token_ids[i], token_ids[i + 1]
                row = self.weights[x]
                probs = softmax(row)
                total_loss -= math.log(max(1e-12, probs[y]))
                # Cross-entropy gradient: dL/dz_j = p_j - 1(j == y)
                for j in range(self.vocab_size):
                    grad = probs[j] - (1.0 if j == y else 0.0)
                    row[j] -= lr * grad
            history.append(round(total_loss / n, 4))
        return history

    def generate(self, start_id: int, max_new_tokens: int = 12) -> list[int]:
        ids = [start_id]
        cur = start_id
        for _ in range(max_new_tokens):
            row = self.weights[cur]
            nxt = max(range(self.vocab_size), key=lambda i: row[i])
            ids.append(nxt)
            cur = nxt
        return ids

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab_size": self.vocab_size, "weights": self.weights}),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "TrainableMiniLM":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(vocab_size=data["vocab_size"])
        m.weights = data["weights"]
        return m


if __name__ == "__main__":
    corpus = "A CuppaShot PLM modell telefonon is gyorsan fut. A MAGMA és a Borostyán magyar AI rendszerek. " * 6
    tok = BpeTokenizer.train(corpus, num_merges=32)
    ids = tok.encode(corpus)
    vocab_n = 256 + len(tok.merges)
    lm = TrainableMiniLM(vocab_size=vocab_n)
    hist = lm.train_steps(ids, steps=40, lr=1.2)
    sample = tok.decode(lm.generate(ids[0], max_new_tokens=16))
    print(f"PLM v0.2 train OK | start_loss={hist[0]} -> end_loss={hist[-1]} | sample={sample!r}")
