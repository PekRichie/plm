import subprocess
import tempfile
import unittest
from pathlib import Path

from plm.config import PRESETS
from plm.export_bin import export_plm_header
from plm.model import TinyPlmModel
from plm.quantize import dequantize_q4_blocks, dequantize_q8, mse, quantize_q4_blocks, quantize_q8
from plm.tokenizer import BpeTokenizer
from plm.train import TrainableMiniLM

ROOT = Path(__file__).resolve().parent.parent


class TestPlm(unittest.TestCase):
    def test_presets_and_mobile_ram(self):
        m180 = PRESETS["micro-180m"]
        self.assertGreater(m180.param_count(), 100_000_000)
        self.assertLess(m180.estimated_size_mb(4.5), 140.0)

    def test_bpe_roundtrip_hu_en(self):
        sample = "A CuppaShot MAGMA és a PLM modell telefonon is gyorsan fut. CuppaShot PLM!"
        tok = BpeTokenizer.train(sample * 4, num_merges=24)
        ids = tok.encode(sample)
        self.assertLess(len(ids), len(sample.encode("utf-8")))
        self.assertEqual(tok.decode(ids), sample)

    def test_transformer_forward_and_generate(self):
        cfg = PRESETS["tiny-test"]
        model = TinyPlmModel(cfg, seed=7)
        out = model.generate([10, 20, 30], max_new_tokens=4)
        self.assertEqual(len(out), 7)

    def test_q8_and_q4_quantization(self):
        weights = [((i % 19) - 9) / 10.0 for i in range(128)]
        q8, s8 = quantize_q8(weights)
        self.assertLess(mse(weights, dequantize_q8(q8, s8)), 1e-4)
        q4 = quantize_q4_blocks(weights, block_size=32)
        self.assertLess(mse(weights, dequantize_q4_blocks(q4)), 5e-3)

    def test_binary_export_and_c99_runtime(self):
        cfg = PRESETS["tiny-test"]
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "tiny.plm"
            exe_path = Path(td) / "plm_run"
            export_plm_header(cfg, [0.25 * (i + 1) for i in range(cfg.d_model)], bin_path)
            subprocess.run(["gcc", "-O3", "-std=c99", "-Wall", "-Wextra", str(ROOT / "c_runtime/plm.c"), "-lm", "-o", str(exe_path)], check=True)
            res = subprocess.run([str(exe_path), str(bin_path)], capture_output=True, text=True, check=True)
            self.assertIn("PLM1 OK", res.stdout)
            self.assertIn("q4_bench_ms=", res.stdout)

    def test_training_loop_reduces_loss_and_saves_checkpoint(self):
        corpus = "CuppaShot PLM mobil AI modell. CuppaShot PLM gyors! " * 4
        tok = BpeTokenizer.train(corpus, num_merges=16)
        ids = tok.encode(corpus)
        lm = TrainableMiniLM(vocab_size=256 + len(tok.merges), seed=1)
        hist = lm.train_steps(ids, steps=35, lr=1.5)
        self.assertGreater(hist[0], hist[-1] * 2.0)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "plm_ckpt.json"
            lm.save_checkpoint(ckpt)
            loaded = TrainableMiniLM.load_checkpoint(ckpt)
            self.assertAlmostEqual(lm.loss(ids), loaded.loss(ids), places=6)
            self.assertEqual(lm.generate(ids[0], 6), loaded.generate(ids[0], 6))


if __name__ == "__main__":
    unittest.main()
