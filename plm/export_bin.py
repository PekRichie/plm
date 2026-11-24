"""Bináris `.plm` formátum exportáló a C99 mobil futtatókörnyezethez."""
import struct
from pathlib import Path
from plm.config import PlmConfig

MAGIC = b"PLM1"


def export_plm_header(cfg: PlmConfig, weights_sample: list[float], out_path: str | Path) -> Path:
    p = Path(out_path)
    header = struct.pack(
        "<4sIIIIIII",
        MAGIC,
        cfg.vocab_size,
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.ffn_hidden,
        cfg.max_seq_len,
    )
    body = struct.pack(f"<{len(weights_sample)}f", *weights_sample)
    p.write_bytes(header + body)
    return p
