"""Decode profiling helpers."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace.engine import create_engine
from workspace.tools.runtime_fixture import create_toy_runtime_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile decode-heavy local traces.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=24)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    config, weight_dir = create_toy_runtime_artifacts("profile_decode")
    engine = create_engine(config, weight_dir, device=args.device)
    vocab_size = config["vocab_size"]

    request_ids = list(range(args.batch_size))
    prompts = [
        torch.randint(0, vocab_size, (args.prompt_len,), dtype=torch.long)
        for _ in request_ids
    ]
    engine.prefill(request_ids, prompts)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.decode_steps):
        token_ids = torch.randint(0, vocab_size, (len(request_ids),), dtype=torch.long)
        engine.decode(request_ids, token_ids)
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
