#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_mrl_compare_models.py
Compare sentence similarity across MRL layers for 0.6B and 4B embedding models.
The script prints a detailed report showing the cosine similarity and distance
for each model and the difference between them at every Matryoshka layer.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def get_detailed_instruct(task: str, query: str) -> str:
    """Return a query string prefixed with a one-sentence instruction."""
    return f"Instruction: {task}\nQuery: {query}"


def progressive_similarity(models, queries, dims, full_dims):
    """Yield similarity metrics for each model across MRL layers."""
    for dim in dims:
        layer_metrics = {}
        for name, model in models.items():
            embeddings = model.encode(
                queries,
                truncate_dim=min(dim, full_dims[name]),
                normalize_embeddings=True,
                prompt_name="query",
                batch_size=len(queries),
            )
            v1, v2 = embeddings
            cosine = float(np.dot(v1, v2))
            distance = 1.0 - cosine
            layer_metrics[name] = (cosine, distance)
        yield dim, layer_metrics


if __name__ == "__main__":
    # Prepare task and queries following Qwen3 instruction guidelines
    task = "Find harmful words."
    queries = [
        get_detailed_instruct(task, "Hello, Happy, Fucking"),
        get_detailed_instruct(task, "Hello, Happy, Fuckoff, piece of sheet"),
    ]

    # Example retrieval documents (no instruction prefix needed)
    documents = [
        "Instruction: Search all explanations.\nQuery: However, according to the latest official technical documentation from Alibaba (released in June 2025), Qwen3-Embedding-8B explicitly supports MRL, allowing dynamic dimension adjustment without quality loss. This indicates a bug in vLLMs model detection logic, where it incorrectly assumes the model lacks MRL support.",
        "max_dim = max(full_dims.values()) dims = [128] + list(range(384, max_dim, 256))",
    ]

    # Texts used for similarity comparison (only the queries)
    texts = queries

    # Load models
    model_ids = {
        "0.6B": "Qwen/Qwen3-Embedding-0.6B",
        "4B": "Qwen/Qwen3-Embedding-4B",
    }
    models = {name: SentenceTransformer(mid, trust_remote_code=True) for name, mid in model_ids.items()}

    # Determine embedding size for each model
    full_dims = {
        name: model.get_sentence_embedding_dimension()
        for name, model in models.items()
    }
    max_dim = max(full_dims.values())
    dims = [128] + list(range(384, max_dim, 256))
    if max_dim not in dims:
        dims.append(max_dim)

    print("\nMRL progressive similarity comparison (normalize_embeddings=True)")
    header = (
        "Layer | Dim  | Cosine 0.6B | Dist 0.6B | Cosine 4B | Dist 4B | Cosine Δ | Dist Δ"
    )
    print(header)
    print("-" * len(header))

    results = []
    for level, (dim, metrics) in enumerate(
        progressive_similarity(models, texts, dims, full_dims),
        start=1,
    ):
        cos_06b, dist_06b = metrics["0.6B"]
        cos_4b, dist_4b = metrics["4B"]
        cos_diff = cos_4b - cos_06b
        dist_diff = dist_4b - dist_06b
        results.append((dim, cos_diff, dist_diff))
        print(
            f"{level:>5} | {dim:>4} | {cos_06b: .4f} | {dist_06b: .4f} | "
            f"{cos_4b: .4f} | {dist_4b: .4f} | {cos_diff: .4f} | {dist_diff: .4f}"
        )

    print("\nExplanation:")
    print(
        "Cosine Δ and Dist Δ show how much the 4B model diverges from the 0.6B model at each "
        "MRL layer. Layers correspond to the listed dimensions: 128 is the smallest "
        "representation and each subsequent layer adds 256 dimensions up to the full size "
        f"({max_dim} for the largest model). Positive cosine difference means the 4B model "
        "predicts higher similarity between the sentences, while negative means lower."
    )

    # Basic self-analysis: suggest dimension where models behave most similarly
    best_dim, best_cos_diff, _ = min(results, key=lambda r: abs(r[1]))
    print("\nSuggested MRL parameters:")
    print(
        f"Use dimension {best_dim} as it yields the smallest cosine difference (Δ={best_cos_diff:.4f}) "
        "between the two models."
    )
