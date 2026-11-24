"""Egyszerű, nulla-függőségű Byte-Pair Encoding (BPE) tokenizáló magyar és angol szövegre."""
import json
from collections import Counter
from pathlib import Path


class BpeTokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = merges or []
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for idx, (a, b) in enumerate(self.merges, start=256):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @staticmethod
    def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, num_merges: int = 64) -> "BpeTokenizer":
        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        for step in range(num_merges):
            if len(ids) < 2:
                break
            counts = Counter(zip(ids, ids[1:]))
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + step
            ids = cls._merge_ids(ids, pair, new_id)
            merges.append(pair)
        return cls(merges=merges)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for idx, pair in enumerate(self.merges, start=256):
            ids = self._merge_ids(ids, pair, idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(p) for p in data["merges"]])
