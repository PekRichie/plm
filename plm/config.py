"""PLM architektúra-konfigurációk és mobil RAM-becslő."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlmConfig:
    name: str
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4  # Grouped-Query Attention (GQA)
    ffn_hidden: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0

    def param_count(self) -> int:
        # Weight tying: input & output embeddings share vocab_size * d_model
        embed = self.vocab_size * self.d_model
        head_dim = self.d_model // self.n_heads
        kv_dim = self.n_kv_heads * head_dim
        attn_per_layer = (self.d_model * self.d_model) + 2 * (self.d_model * kv_dim) + (self.d_model * self.d_model)
        # SwiGLU FFN: gate, up, down projections
        ffn_per_layer = 3 * self.d_model * self.ffn_hidden
        norms_per_layer = 2 * self.d_model
        return embed + self.n_layers * (attn_per_layer + ffn_per_layer + norms_per_layer) + self.d_model

    def estimated_size_mb(self, bits_per_weight: float = 4.5) -> float:
        return round((self.param_count() * bits_per_weight / 8.0) / (1024 * 1024), 2)


PRESETS = {
    "tiny-test": PlmConfig(name="plm-tiny-test", vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64, max_seq_len=64),
    "nano-30m": PlmConfig(name="plm-nano-30m", vocab_size=16384, d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, ffn_hidden=1024, max_seq_len=1024),
    "micro-180m": PlmConfig(name="plm-micro-180m", vocab_size=32768, d_model=768, n_layers=14, n_heads=12, n_kv_heads=4, ffn_hidden=2048, max_seq_len=2048),
}
