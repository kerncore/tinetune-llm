#!/usr/bin/env python3
"""Embedding benchmark using sentence-transformers.

This script replicates the functionality of the original MLX benchmark
but relies solely on the ``SentenceTransformer`` library. It measures
embedding throughput for batches of text of different token lengths.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL_ID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
PREFIX_TXT = "Instruct: Focus on most important parts of the text\nQuery: "

TARGET_LENS = [250, 500, 1000, 1500]
N_RUNS = 250
N_ROUNDS = 3

# ---------------------------------------------------------------------------
# setup model and tokenizer
# ---------------------------------------------------------------------------
model = SentenceTransformer(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def build_batch(token_len: int, run_id: int = 0) -> Tuple[str, int]:
    """Return a text sample with an exact token length."""
    prefix_ids = tokenizer.encode(PREFIX_TXT, add_special_tokens=False)
    need = token_len - len(prefix_ids)

    unique_part = f"batch_{run_id}_data "
    unique_ids = tokenizer.encode(unique_part, add_special_tokens=False)

    base_text = "embedding benchmark test content "
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)

    remaining = need - len(unique_ids)
    if remaining > 0:
        reps = math.ceil(remaining / len(base_ids))
        repeated_ids = (base_ids * reps)[:remaining]
        body = unique_ids + repeated_ids
    else:
        body = unique_ids[:need]

    ids = prefix_ids + body
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    return text, len(ids)

# ---------------------------------------------------------------------------
# embed helper
# ---------------------------------------------------------------------------
def embed(text: str) -> None:
    model.encode([text], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)

# ---------------------------------------------------------------------------
# benchmark loop
# ---------------------------------------------------------------------------
results = []

for tok_len in TARGET_LENS:
    print(f"\nBenchmarking {tok_len} tokens...")
    print("  Warming up with diverse content...")
    for _ in range(10):
        warm_text, _ = build_batch(tok_len, random.randint(0, 1000))
        embed(warm_text)

    round_results = []
    for round_num in range(N_ROUNDS):
        print(f"  Round {round_num + 1}/{N_ROUNDS}...", flush=True)

        batches: List[str] = []
        for run in range(N_RUNS):
            text, _ = build_batch(tok_len, run + round_num * N_RUNS)
            batches.append(text)

        times: List[float] = []
        for text in batches:
            t0 = time.perf_counter()
            embed(text)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        total_time = sum(times)
        avg_time = total_time / N_RUNS
        embeds_per_sec = N_RUNS / total_time if total_time else 0
        tokens_per_sec = (N_RUNS * tok_len) / total_time if total_time else 0

        round_results.append({
            "elapsed": total_time,
            "avg_time_per_embed": avg_time,
            "embeds_per_sec": embeds_per_sec,
            "tokens_per_sec": tokens_per_sec,
        })

        print(f"    {embeds_per_sec:.1f} embeds/s, {tokens_per_sec:.0f} tokens/s, {avg_time*1000:.2f}ms per embed")

    avg_embeds_per_sec = sum(r["embeds_per_sec"] for r in round_results) / N_ROUNDS
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in round_results) / N_ROUNDS
    avg_time_per_embed = sum(r["avg_time_per_embed"] for r in round_results) / N_ROUNDS

    results.append({
        "tokens": tok_len,
        "embeds_per_sec": avg_embeds_per_sec,
        "tokens_per_sec": avg_tokens_per_sec,
        "ms_per_embed": avg_time_per_embed * 1000,
    })

print("\n" + "=" * 70)
print("FINAL BENCHMARK RESULTS")
print("=" * 70)
print(f"{'Tokens':<8} {'Embeds/sec':<12} {'Tokens/sec':<12} {'ms/embed':<10}")
print("-" * 42)
for result in results:
    print(f"{result['tokens']:<8} {result['embeds_per_sec']:<12.1f} {result['tokens_per_sec']:<12.0f} {result['ms_per_embed']:<10.2f}")
print("=" * 70)
