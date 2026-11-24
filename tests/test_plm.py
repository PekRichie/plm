import subprocess
import tempfile
import unittest
from pathlib import Path

from plm.config import PRESETS
from plm.export_bin import export_plm_header
from plm.model import TinyPlmModel
from plm.quantize import dequantize_q4_blocks, dequantize_q8, mse, quantize_q4_blocks, quantize_q8
from plm.tokenizer import BpeTokenizer

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
            subprocess.run(["gcc", "-O2", "-std=c99", str(ROOT / "c_runtime/plm.c"), "-lm", "-o", str(exe_path)], check=True)
            res = subprocess.run([str(exe_path), str(bin_path)], capture_output=True, text=True, check=True)
            self.assertIn("PLM1 OK", res.stdout)


if __name__ == "__main__":
    unittest.main()
