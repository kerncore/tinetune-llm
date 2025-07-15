#!/usr/bin/env python3
"""
SentenceTransformer embedding-throughput benchmark

Example
-------
python st_benchmark.py --embed /path/to/project/src
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, __version__ as st_version
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Maximum sequence length for tokenization
MAX_LENGTH = 8192

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def gather_source_files(root_dir: str) -> List[str]:
    exclude_dirs = {"node_modules", ".venv", "dist", "__pycache__"}
    valid_ext    = {".ts", ".tsx", ".js", ".jsx", ".mjs"}

    matches: List[str] = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]          # in-place
        matches.extend(
            os.path.join(root, f)
            for f in files
            if os.path.splitext(f)[1] in valid_ext
        )
    return matches

# --------------------------------------------------------------------------------------
# Benchmark class
# --------------------------------------------------------------------------------------

class STEmbeddingBenchmark:
    INDEX_PROMPTS = [
        "Parse and index all symbol definitions including functions, classes, interfaces, and types for code navigation",
        "Extract all function declarations, their signatures, parameters, and return types for API documentation",
        "Build an index of all navigable symbols including their locations and relationships for IDE features",
        "Identify all top-level and nested symbol declarations with their scopes and accessibility modifiers",
    ]

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id   = model_id
        self.device     = device

        # Accumulators
        self.total_tokens          = 0
        self.total_embeddings      = 0
        self.total_time            = 0.0
        self.total_chunk_sizes: List[int] = []
        self.total_chunks_per_file: List[int] = []
        self.total_time_per_file:  List[float] = []
        self.total_files_processed = 0

        if tuple(int(x) for x in st_version.split(".")[:2]) < (2, 2):
            raise RuntimeError(
                f"sentence-transformers >=2.2.0 required, found {st_version}"
            )

        print("⏳ Loading model …")
        self.model = SentenceTransformer(model_id, device=self.device)
        self.model.max_seq_length = MAX_LENGTH

        print("⏳ Loading tokenizer from 🤗 Transformers …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise RuntimeError("Tokenizer is not a valid 🤗 tokenizer")

        print("✓ Model & tokenizer ready")

    # ------------------------------------------------------------------ token handling
    def encode_len(self, text: str) -> int:
        """Real token length (no heuristics)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------ code chunking
    def chunk_source_code(
        self,
        code: str,
        min_tokens: int = 100,
        max_tokens: int = 1000,
    ) -> List[str]:
        lines   = code.splitlines()
        chunks  = []
        buffer  = []
        buf_tok = 0

        for line in lines:
            line_tok = self.encode_len(line)
            if buf_tok + line_tok > max_tokens:
                if buf_tok >= min_tokens:
                    chunks.append("\n".join(buffer).rstrip())
                    buffer, buf_tok = [line], line_tok
                else:                                    # merge small chunk
                    buffer.append(line)
                    buf_tok += line_tok
            else:
                buffer.append(line)
                buf_tok += line_tok

        if buffer:
            # either append or merge with previous small chunk
            if buf_tok < min_tokens and chunks:
                chunks[-1] = chunks[-1] + "\n" + "\n".join(buffer)
            else:
                chunks.append("\n".join(buffer).rstrip())

        return chunks

    # ------------------------------------------------------------------ batching helpers
    def _make_batches(
        self, chunks: List[str], batch_size: int = 32
    ) -> List[List[str]]:
        """Expand each chunk × #prompts and split into batches."""
        expanded: List[str] = []
        for ch in chunks:
            expanded.extend([ch] * len(self.INDEX_PROMPTS))
        return [expanded[i : i + batch_size] for i in range(0, len(expanded), batch_size)]

    # ------------------------------------------------------------------ embedding core
    def _embed(self, texts: List[str]) -> tuple[list[list[float]], int, float]:
        """
        Forward-pass a list of strings and return:
        • embeddings  (List[List[float]])
        • total token count  (int)
        • elapsed wall-clock time (float seconds)
        """
        start = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=128,
        )
        elapsed = time.time() - start

        tokenizer = self.tokenizer
        max_length = MAX_LENGTH

        batch_dict = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        token_count = int(batch_dict["attention_mask"].sum().item())

        return embeddings.tolist(), token_count, elapsed

    # ------------------------------------------------------------------ public driver
    def benchmark_file(self, path: str) -> None:
        try:
            code = open(path, "r", encoding="utf-8").read()
        except FileNotFoundError:
            print(f"✗ File not found: {path}")
            return

        chunks = self.chunk_source_code(code)
        batches = self._make_batches(chunks)

        file_tokens      = 0
        file_embeddings  = 0
        file_time        = 0.0
        chunk_token_lens = []

        for batch in batches:
            embeds, tokens, t_elapsed = self._embed(batch)

            file_tokens     += tokens
            file_embeddings += len(embeds)
            file_time       += t_elapsed
            chunk_token_lens.extend([self.encode_len(txt) for txt in batch])

        # ------------ per-file report -------------------------------------------------
        tps = file_tokens / file_time if file_time else 0
        eps = file_embeddings / file_time if file_time else 0
        avg_chunk = np.mean(chunk_token_lens) if chunk_token_lens else 0

        print(f"✓ {os.path.basename(path)}")
        print(f"    chunks       : {file_embeddings}")
        print(f"    tokens       : {file_tokens}")
        print(f"    time         : {file_time:.2f}s")
        print(f"    tokens/sec   : {tps:.2f}")
        print(f"    embeddings/s : {eps:.2f}")
        print(f"    avg chunk sz : {avg_chunk:.1f} tokens")

        # ------------ global accumulators --------------------------------------------
        self.total_files_processed += 1
        self.total_tokens          += file_tokens
        self.total_embeddings      += file_embeddings
        self.total_time            += file_time
        self.total_chunk_sizes.extend(chunk_token_lens)
        self.total_chunks_per_file.append(file_embeddings)
        self.total_time_per_file.append(file_time)

    # ------------------------------------------------------------------ summary
    def print_summary(self) -> None:
        if self.total_files_processed == 0:
            print("❌ No files processed.")
            return

        avg_tokens_per_file = self.total_tokens / self.total_files_processed
        avg_chunks_per_file = np.mean(self.total_chunks_per_file)
        avg_tokens_per_chunk = np.mean(self.total_chunk_sizes)
        avg_time_per_file = np.mean(self.total_time_per_file)
        tps = self.total_tokens / self.total_time if self.total_time else 0
        eps = self.total_embeddings / self.total_time if self.total_time else 0

        print("\n📊  Overall summary")
        print("────────────────────────────────────────────────────")
        print(f"Files processed          : {self.total_files_processed}")
        print(f"Total chunks embedded     : {self.total_embeddings}")
        print(f"Total tokens              : {self.total_tokens}")
        print(f"Total time                : {self.total_time:.2f}s")
        print(f"Avg tokens / file         : {avg_tokens_per_file:.1f}")
        print(f"Avg chunks / file         : {avg_chunks_per_file:.1f}")
        print(f"Avg tokens / chunk        : {avg_tokens_per_chunk:.1f}")
        print(f"Avg embedding time / file : {avg_time_per_file:.2f}s")
        print(f"Throughput tokens / sec   : {tps:.2f}")
        print(f"Throughput embeds / sec   : {eps:.2f}")
        print(f"Min / Max chunk size      : {min(self.total_chunk_sizes)} / "
              f"{max(self.total_chunk_sizes)}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SentenceTransformer embedding benchmark")
    parser.add_argument(
        "--embed",
        required=True,
        help="File or directory to embed",
    )
    parser.add_argument(
        "--model",
        default="kerncore/Qwen3-Embedding-0.6B-MXL-4bit",
        help="HF model id (must be sentence-embedding capable)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, or mps)",
    )
    args = parser.parse_args()

    bench = STEmbeddingBenchmark(args.model, device=args.device)

    if os.path.isfile(args.embed):
        bench.benchmark_file(args.embed)
    elif os.path.isdir(args.embed):
        files = gather_source_files(args.embed)
        print(f"\n📁 Found {len(files)} source files\n")
        for f in files:
            bench.benchmark_file(f)
    else:
        raise FileNotFoundError(f"{args.embed} is neither a file nor a directory")

    bench.print_summary()

if __name__ == "__main__":
    main()
