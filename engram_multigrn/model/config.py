"""
Engram-GRN Configuration.
Adapted from DeepSeek Engram for gene regulatory networks.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramGRNConfig:
    """Main configuration for Engram-GRN model."""

    # ── Gene Vocabulary ──
    vocab_size: int = 20000          # Number of gene tokens (human protein-coding)
    gene_embed_dim: int = 128        # Gene embedding dimension (d_model)

    # ── Regulatory Context / "N-gram" ──
    max_context_size: int = 3        # 3 → 2 orders: 2-gene, 3-gene (like bigram+trigram)
    n_context_orders: int = 2        # Must equal max_context_size - 1

    # ── Hash Lookup (Engram core) ──
    engram_vocab_size: List[int] = field(
        default_factory=lambda: [500000, 500000]  # Table size per n-gram order
    )
    n_embed_per_ngram: int = 128     # Total embedding dim retrieved per n-gram order
    n_head_per_ngram: int = 4        # Multi-head hashing parallelism
    layer_ids: List[int] = field(
        default_factory=lambda: [0]   # Which "layers" (one for now)
    )
    pad_id: int = 0
    seed: int = 42

    # ── Context Gating ──
    condition_dim: int = 32          # Dimension of cell condition vector
    gate_hidden_dim: int = 64        # Hidden dim for gate projection

    # ── ShortConv ──
    kernel_size: int = 3
    conv_hc_mult: int = 2            # Group multiplier for depthwise conv

    # ── Output ──
    output_dim: int = 1              # Predict Kla score or expression level

    # ── Training ──
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 1e-5


@dataclass
class DatasetConfig:
    """Configuration for data sources."""

    # Data paths
    data_dir: str = "data"
    gene_vocab_file: str = "data/gene_vocabulary.pt"
    pathway_file: str = "data/pathways.pt"
    coexpression_file: str = "data/coexpression.pt"
    kla_data_file: str = "data/kla_datasets.pt"

    # Data sources
    ensembl_gtf_url: str = (
        "https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/"
        "Homo_sapiens.GRCh38.113.gtf.gz"
    )
    download_kla: bool = True
