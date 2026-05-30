"""Interface tests for the evaluator-facing baseline."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from workspace.engine import create_engine
from workspace.runtime.model import LlamaLikeConfig, LlamaLikeForCausalLM


def make_toy_artifacts() -> tuple[dict, str]:
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    model = LlamaLikeForCausalLM(LlamaLikeConfig.from_dict(config))
    for index, parameter in enumerate(model.parameters()):
        torch.manual_seed(index + 1)
        parameter.data.copy_(torch.randn_like(parameter) * 0.05)

    temp_root = Path(__file__).resolve().parents[2] / ".tmp_tests"
    temp_root.mkdir(exist_ok=True)
    weight_dir = temp_root / "phase1_weights"
    shutil.rmtree(weight_dir, ignore_errors=True)
    weight_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "model.pt")
    return config, str(weight_dir)


class EngineInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.weight_dir = make_toy_artifacts()

    def tearDown(self) -> None:
        shutil.rmtree(self.weight_dir, ignore_errors=True)

    def test_create_engine_exposes_required_methods(self) -> None:
        engine = create_engine(self.config, self.weight_dir, device="cpu")
        self.assertTrue(callable(engine.prefill))
        self.assertTrue(callable(engine.decode))
        self.assertTrue(callable(engine.remove))


if __name__ == "__main__":
    unittest.main()
