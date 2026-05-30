"""Local correctness regression tests for the Phase 1 baseline."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from workspace.engine import create_engine
from workspace.runtime.model import LlamaLikeConfig, LlamaLikeForCausalLM


def build_reference_model(config: dict, weight_dir: str) -> LlamaLikeForCausalLM:
    model = LlamaLikeForCausalLM(LlamaLikeConfig.from_dict(config))
    state_dict = torch.load(Path(weight_dir) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return model.eval()


def make_toy_artifacts() -> tuple[dict, str]:
    config = {
        "vocab_size": 64,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    model = LlamaLikeForCausalLM(LlamaLikeConfig.from_dict(config))
    torch.manual_seed(7)
    for parameter in model.parameters():
        parameter.data.copy_(torch.randn_like(parameter) * 0.04)

    temp_root = Path(__file__).resolve().parents[2] / ".tmp_tests"
    temp_root.mkdir(exist_ok=True)
    weight_dir = temp_root / "phase1_correctness"
    shutil.rmtree(weight_dir, ignore_errors=True)
    weight_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "model.pt")
    return config, str(weight_dir)


class BaselineCorrectnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.weight_dir = make_toy_artifacts()
        self.reference = build_reference_model(self.config, self.weight_dir)
        self.engine = create_engine(self.config, self.weight_dir, device="cpu")

    def tearDown(self) -> None:
        shutil.rmtree(self.weight_dir, ignore_errors=True)

    def assert_logits_close(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), msg=f"\nactual={actual}\nexpected={expected}")

    def test_single_request_prefill_and_decode(self) -> None:
        prompt = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        expected_prefill = self.reference.logits_for_last_token(prompt.view(1, -1))
        actual_prefill = self.engine.prefill([101], [prompt])
        self.assert_logits_close(actual_prefill, expected_prefill)
        self.assertEqual(self.engine.requests.require(101).kv_cache.seq_len, 4)

        token = torch.tensor([5], dtype=torch.long)
        full_sequence = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        expected_decode = self.reference.logits_for_last_token(full_sequence.view(1, -1))
        actual_decode = self.engine.decode([101], token)
        self.assert_logits_close(actual_decode, expected_decode)
        self.assertEqual(self.engine.requests.require(101).kv_cache.seq_len, 5)

    def test_single_request_multiple_incremental_decode_steps(self) -> None:
        prompt = torch.tensor([8, 9, 10], dtype=torch.long)
        self.engine.prefill([303], [prompt])

        running = prompt.clone()
        for token in [11, 12, 13]:
            running = torch.cat([running, torch.tensor([token], dtype=torch.long)])
            expected = self.reference.logits_for_last_token(running.view(1, -1))
            actual = self.engine.decode([303], torch.tensor([token], dtype=torch.long))
            self.assert_logits_close(actual, expected)
        self.assertEqual(self.engine.requests.require(303).kv_cache.seq_len, running.numel())

    def test_multi_request_prefill_and_decode(self) -> None:
        prompts = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
        ]
        expected_prefill = torch.cat(
            [self.reference.logits_for_last_token(prompt.view(1, -1)) for prompt in prompts],
            dim=0,
        )
        actual_prefill = self.engine.prefill([201, 202], prompts)
        self.assert_logits_close(actual_prefill, expected_prefill)

        decode_tokens = torch.tensor([6, 7], dtype=torch.long)
        sequences = [
            torch.tensor([1, 2, 3, 6], dtype=torch.long),
            torch.tensor([4, 5, 7], dtype=torch.long),
        ]
        expected_decode = torch.cat(
            [self.reference.logits_for_last_token(sequence.view(1, -1)) for sequence in sequences],
            dim=0,
        )
        actual_decode = self.engine.decode([201, 202], decode_tokens)
        self.assert_logits_close(actual_decode, expected_decode)

    def test_batched_incremental_decode_for_same_length_requests(self) -> None:
        prompts = [
            torch.tensor([2, 3, 4], dtype=torch.long),
            torch.tensor([5, 6, 7], dtype=torch.long),
        ]
        self.engine.prefill([401, 402], prompts)

        original = self.engine.model.logits_for_decode_batch_with_manager
        call_counter = {"count": 0}

        def wrapped(*args, **kwargs):
            call_counter["count"] += 1
            return original(*args, **kwargs)

        self.engine.model.logits_for_decode_batch_with_manager = wrapped
        try:
            decode_tokens = torch.tensor([8, 9], dtype=torch.long)
            sequences = [
                torch.tensor([2, 3, 4, 8], dtype=torch.long),
                torch.tensor([5, 6, 7, 9], dtype=torch.long),
            ]
            expected = torch.cat(
                [self.reference.logits_for_last_token(sequence.view(1, -1)) for sequence in sequences],
                dim=0,
            )
            actual = self.engine.decode([401, 402], decode_tokens)
        finally:
            self.engine.model.logits_for_decode_batch_with_manager = original

        self.assert_logits_close(actual, expected)
        self.assertEqual(call_counter["count"], 1)

    def test_batched_prefill_for_same_length_requests(self) -> None:
        prompts = [
            torch.tensor([10, 11, 12, 13], dtype=torch.long),
            torch.tensor([14, 15, 16, 17], dtype=torch.long),
        ]
        expected = torch.cat(
            [self.reference.logits_for_last_token(prompt.view(1, -1)) for prompt in prompts],
            dim=0,
        )

        original = self.engine.model.logits_and_cache_for_prefill_batch
        call_counter = {"count": 0}

        def wrapped(*args, **kwargs):
            call_counter["count"] += 1
            return original(*args, **kwargs)

        self.engine.model.logits_and_cache_for_prefill_batch = wrapped
        try:
            actual = self.engine.prefill([501, 502], prompts)
        finally:
            self.engine.model.logits_and_cache_for_prefill_batch = original

        self.assert_logits_close(actual, expected)
        self.assertEqual(call_counter["count"], 1)
        self.assertEqual(self.engine.requests.require(501).kv_cache.seq_len, 4)
        self.assertEqual(self.engine.requests.require(502).kv_cache.seq_len, 4)


if __name__ == "__main__":
    unittest.main()
