"""Request insertion/removal lifecycle tests for the Phase 1 baseline."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from workspace.engine import create_engine
from workspace.runtime.model import LlamaLikeConfig, LlamaLikeForCausalLM


def make_toy_artifacts() -> tuple[dict, str]:
    config = {
        "vocab_size": 48,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    model = LlamaLikeForCausalLM(LlamaLikeConfig.from_dict(config))
    torch.manual_seed(11)
    for parameter in model.parameters():
        parameter.data.copy_(torch.randn_like(parameter) * 0.03)
    temp_root = Path(__file__).resolve().parents[2] / ".tmp_tests"
    temp_root.mkdir(exist_ok=True)
    weight_dir = temp_root / "phase1_lifecycle"
    shutil.rmtree(weight_dir, ignore_errors=True)
    weight_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "model.pt")
    return config, str(weight_dir)


class RequestLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.weight_dir = make_toy_artifacts()
        self.engine = create_engine(self.config, self.weight_dir, device="cpu")

    def tearDown(self) -> None:
        shutil.rmtree(self.weight_dir, ignore_errors=True)

    def test_insert_remove_and_continue_decode(self) -> None:
        self.engine.prefill([1], [torch.tensor([1, 2, 3], dtype=torch.long)])
        self.engine.prefill([2], [torch.tensor([4, 5], dtype=torch.long)])
        self.engine.decode([1, 2], torch.tensor([6, 7], dtype=torch.long))
        self.engine.remove([1])

        remaining = self.engine.decode([2], torch.tensor([8], dtype=torch.long))
        self.assertEqual(tuple(remaining.shape), (1, self.config["vocab_size"]))

    def test_decode_unknown_request_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.engine.decode([999], torch.tensor([1], dtype=torch.long))

    def test_cache_slot_reused_after_remove(self) -> None:
        self.engine.prefill([10], [torch.tensor([1, 2], dtype=torch.long)])
        first_slot = self.engine.requests.require(10).cache_slot
        self.engine.remove([10])

        self.engine.prefill([11], [torch.tensor([3, 4], dtype=torch.long)])
        second_slot = self.engine.requests.require(11).cache_slot
        self.assertEqual(first_slot, second_slot)

    def test_mixed_trace_insert_remove_and_continue(self) -> None:
        reference = LlamaLikeForCausalLM(LlamaLikeConfig.from_dict(self.config))
        state_dict = torch.load(Path(self.weight_dir) / "model.pt", map_location="cpu")
        reference.load_state_dict(state_dict)
        reference.eval()

        self.engine.prefill([1], [torch.tensor([1, 2, 3], dtype=torch.long)])
        self.engine.decode([1], torch.tensor([4], dtype=torch.long))

        inserted_prompt = torch.tensor([7, 8], dtype=torch.long)
        inserted_logits = self.engine.prefill([2], [inserted_prompt])
        expected_inserted = reference.logits_for_last_token(inserted_prompt.view(1, -1))
        self.assertTrue(torch.allclose(inserted_logits, expected_inserted, atol=1e-5, rtol=1e-5))

        self.engine.remove([1])
        remaining = self.engine.decode([2], torch.tensor([9], dtype=torch.long))
        expected_remaining = reference.logits_for_last_token(torch.tensor([7, 8, 9], dtype=torch.long).view(1, -1))
        self.assertTrue(torch.allclose(remaining, expected_remaining, atol=1e-5, rtol=1e-5))
        self.assertEqual(self.engine.requests.active_request_ids(), [2])


if __name__ == "__main__":
    unittest.main()
