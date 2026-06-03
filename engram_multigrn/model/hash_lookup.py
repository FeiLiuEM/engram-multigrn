"""
Multi-head hashing module. Adapted from DeepSeek Engram.
Maps gene regulatory contexts (analogous to n-grams) to embedding table indices
using deterministic prime-number hashing.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from typing import Dict, List, Optional


def find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


def compute_regulatory_hash(
    gene_ids: np.ndarray,
    multipliers: np.ndarray,
    pad_id: int,
) -> np.ndarray:
    """
    Hash analog of Engram's n-gram hash for gene regulatory contexts.
    gene_ids: [B, context_size]  - gene IDs forming a regulatory context
    multipliers: [context_size]  - per-position multipliers
    """
    mixed = (gene_ids[:, 0] * multipliers[0]).astype(np.int64)
    for k in range(1, gene_ids.shape[1]):
        term = (gene_ids[:, k] * multipliers[k]).astype(np.int64)
        mixed = np.bitwise_xor(mixed, term)
    return mixed


class GeneContextHasher:
    """
    Maps gene regulatory contexts to hash indices.
    For each 'order' (analogous to n-gram size: 2-genes, 3-genes),
    uses multi-head hashing with prime-number vocabulary sizes.
    """
    def __init__(
        self,
        engram_vocab_size: List[int],
        max_context_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        pad_id: int,
        seed: int,
    ):
        self.vocab_size_per_order = engram_vocab_size
        self.max_context_size = max_context_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids
        self.rng = np.random.default_rng(seed)

        # Build per-layer multipliers (one per context position)
        self.layer_multipliers: Dict[int, np.ndarray] = {}
        PRIME_1 = 10007
        for layer_id in layer_ids:
            base_seed = seed + PRIME_1 * layer_id
            g = np.random.default_rng(int(base_seed))
            mults = g.integers(0, np.iinfo(np.int64).max // 2,
                               size=(max_context_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = mults * 2 + 1

        # Build prime-number vocab sizes for each layer/order/head
        self.vocab_size_across_layers = self._build_vocab_sizes()

    def _build_vocab_sizes(self) -> Dict[int, List[List[int]]]:
        """Build prime-number embedding table sizes for each layer."""
        seen_primes: set = set()
        result: Dict[int, List[List[int]]] = {}
        for layer_id in self.layer_ids:
            all_order_sizes = []
            for order_idx in range(self.max_context_size - 1):
                head_sizes = []
                base = self.vocab_size_per_order[order_idx]
                current = base - 1
                for _ in range(self.n_head_per_ngram):
                    p = find_next_prime(current, seen_primes)
                    seen_primes.add(p)
                    head_sizes.append(p)
                    current = p
                all_order_sizes.append(head_sizes)
            result[layer_id] = all_order_sizes
        return result

    def hash(
        self,
        context_ids: np.ndarray,
        layer_id: int = 0,
    ) -> np.ndarray:
        """
        Hash regulatory context IDs to multi-head indices.
        context_ids: [B, context_size]
        Returns: [B, num_orders * num_heads] hash indices
        """
        B = context_ids.shape[0]
        multipliers = self.layer_multipliers[layer_id]
        order_sizes = self.vocab_size_across_layers[layer_id]
        all_hashes = []

        for order_idx in range(self.max_context_size - 1):
            order_size = order_idx + 2
            tokens = context_ids[:, :order_size]
            mixed = compute_regulatory_hash(tokens, multipliers[:order_size], self.pad_id)
            head_vocab_sizes = order_sizes[order_idx]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mixed % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=1)


class MultiHeadEmbedding(nn.Module):
    """
    Embedding table with multi-head support.
    Multiple hash heads each index into a contiguous sub-table.
    """
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted = input_ids + self.offsets
        return self.embedding(shifted)
