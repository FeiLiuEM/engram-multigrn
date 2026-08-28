# Engram-MultiGRN

**Domain-Isolated Conditional Memory for Multi-Species Gene Regulatory Network Inference**

A neural architecture that learns gene regulatory networks across biological domains without forgetting — no gradient freezing, no elastic weight consolidation, no rehearsal. Each domain (species × cell type × histone modification) gets independent gate and output head parameters, stored in domain-keyed ModuleDict entries. The shared foundation (gene embeddings, hash memory, context encoder) stays fixed. Adding a new domain costs ~43K parameters (<0.2% of the 41.6M total).

<div align="center">

| Model | Cross-Domain Retention | Self-Training R (10 conditions) |
|-------|----------------------|-------------------------------|
| **MultiGRN** | **100%** | **0.34–0.81** |
| GNN | 0% | −0.01 to 0.10 |
| MLP | 0% | −0.01 to 0.09 |

</div>

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Figure Reproduction](#figure-reproduction)
- [Reference Data](#reference-data)
- [Citation](#citation)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/FeiLiuEM/engram-multigrn.git
cd engram-multigrn

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- CUDA-capable GPU with 8+ GB VRAM (NVIDIA RTX 4090 recommended)
- `pyBigWig` for ChIP-seq bigWig processing (C extension, requires system `libcurl`)

---

## Quick Start

```python
import torch
from engram_multigrn.model.multigrn import EngramMultiGRN, MultiGRNConfig
from engram_multigrn.data_pipeline.gene_vocab import GeneVocabulary
from engram_multigrn.data_pipeline.regulatory_context import RegulatoryContextBuilder

# Initialize model
cfg = MultiGRNConfig()
cfg.n_ortho_groups = 30000  # orthology groups (human+mouse)
model = EngramMultiGRN(
    n_ortho_groups=cfg.n_ortho_groups,
    d_ctx=102,           # context feature dimension
    ctx_input_dim=102
).cuda()

model.add_species("human", num_genes=19298, ctx_input_dim=102)
model.add_species("mouse", num_genes=26264, ctx_input_dim=102)

# Forward pass
ortho_ids = torch.randint(0, cfg.n_ortho_groups, (64,)).cuda()
sp_gene_ids = torch.randint(0, 20000, (64,)).cuda()
ctx_features = torch.randn(64, 102).cuda()
cond_ids = torch.zeros(64, dtype=torch.long).cuda()

predictions = model(
    ortho_ids, sp_gene_ids, ctx_features, cond_ids,
    species="human", cell_type="hepg2", mark="H4K5la"
)
# predictions.shape = (64, 1)
```

---

## Architecture

Engram-MultiGRN has five components:

| Component | Description | Parameters |
|-----------|-------------|------------|
| 1. Ortho-group gene embedding | Shared across species (29,114 groups × 128d) + per-species offset | 5.8M (shared) |
| 2. Per-species context encoder | STRING PPI (64d) + genomic features (8d) + chromatin states (30d) → 64d | 0.33M |
| 3. Domain-isolated hash memory | 4-head deterministic hashing → 1M-slot × 32d embedding table | 32M |
| 4. Domain-isolated multi-head gate | Per-domain RMSNorm + projections (key/value Linear 32→128) | ~17K/domain |
| 5. Domain-isolated output head | Per-domain 2-layer MLP (256→128→1) with GELU + dropout | ~33K/domain |

**Total shared**: ~41.6M parameters  
**Per new domain**: ~43K parameters (<0.2% overhead)

For a detailed architecture description, see the manuscript (`files/manuscript_humanized.md`).

---

## Dataset Preparation

### 1. Download gene annotations and STRING data

```bash
python scripts/download_data.py --download-genes --download-pathways --build-vocab
```

Downloads HGNC gene list, KEGG pathway data, and builds gene vocabulary from Ensembl release-113 GTF.

### 2. Download ChIP-seq datasets from GEO

The following GEO datasets are used:

| GEO ID | Histone mark | Cell line | Conditions | Processing method |
|--------|-------------|-----------|------------|-------------------|
| GSE314769 | H4K5la | HepG2 (HCC) | NM2, NM3, LAC2, LAC3 | narrowPeak overlap |
| GSE269142 | H3K18la | MDA-MB-231 (breast) | CON, HYP, HIF-KD | peak-gene.txt mouse→human |
| GSE314155 | H3K18la | HepG2 (HCC) | NC | TSS-centered bigWig |
| GSE247800 | H3K18la | HEK293T | K192R, WT | TSS-centered bigWig |
| GSE219045 | — | Mouse (84 tissues) | FPKM expression | Per-tissue average |

Download data files from [GEO](https://www.ncbi.nlm.nih.gov/geo/) and place them in `data/kla_chip/`.

### 3. Process ChIP-seq data to gene-level scores

```bash
python scripts/process_kla_data.py
python scripts/preprocess_bigwig_improved.py
```

Converts bigWig, bedgraph, and narrowPeak files to per-gene enrichment scores.

### 4. Build training features

```bash
python scripts/build_features.py
python scripts/build_multispecies_data.py
```

Generates `multigrn_features_full.json` with orthology mapping, STRING context, and genomic/chromatin features.

---

## Training

### Single-domain training

```bash
python scripts/train_full.py
```

Trains Engram-MultiGRN on H4K5la ChIP-seq prediction (HepG2, 4 conditions).

### Cross-species incremental training

```bash
python scripts/train_multigrn_v5.py
```

Two-stage incremental training: human H4K5la → mouse FPKM (GSE219045, 84 tissues).

### Multi-dataset incremental training (with baseline comparison)

```bash
python scripts/incremental_multigrn_pipeline.py
```

Full pipeline: preprocessing → feature building → 10-condition incremental training with MultiGRN, GNN, and MLP baselines.

**Training details:**
- Optimizer: AdamW (lr=3×10⁻⁴, weight decay=10⁻⁵)
- Epochs: 25 per condition (incremental) or 100 (single-domain convergence)
- Batch size: 256
- Loss: Mean Squared Error
- Hardware: Single NVIDIA RTX 4090 (24 GB), ~30 seconds per domain per 50 epochs

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint data/engram_grn_best.pt --test-set data/test_split.json
```

**Metrics:**
- Primary: Pearson R between predicted and actual ChIP-seq scores
- Knowledge retention: (Stage 2 R / Stage 1 R) × 100% on held-out test sets

**Key results:**

| Experiment | MultiGRN | GNN | MLP |
|-----------|----------|-----|-----|
| H4K5la ablation (full model) | R=0.830 | R=0.552 | R=0.282 |
| Cross-species retention (H→M) | 100% | 0% | 0% |
| Cross-species retention (M→H) | 99.3% | — | — |
| 15-round incremental (final) | R=0.878 | R=0.676 | R=0.442 |
| 10-condition self-training range | 0.34–0.81 | −0.01–0.09 | −0.01–0.10 |

---

## Testing

```bash
# Run integration tests (requires pre-computed feature files in data/)
ENGRAM_DATA_DIR=$(pwd)/data python tests/test_integration.py
```

The integration test verifies three things:
1. **Model instantiation** — forward pass succeeds with correct output dimensions
2. **Training convergence** — model achieves Pearson R > 0.05 on H4K5la NM2+NM3 after 15 epochs (~4 seconds on RTX 4090)
3. **Checkpoint save/load** — model weights are identical after save→reload

Expected output:
```
============================================================
Engram-MultiGRN Integration Tests
============================================================
Device: cuda
  ✅ Model instantiation: 38,400,771 params, output [8]
  Loaded H4K5la_NM2 (8132 genes) + NM3 (8400 genes)
  Train: 13225, Test: 3307
  Pearson R after 15 epochs: 0.3180 (4s)
  ✅ Checkpoint save/load: weights identical
============================================================
Results: 3 passed, 0 failed out of 3
✅ All tests passed.
```

---

## Figure Reproduction

```bash
python scripts/generate_figures.py
```

Generates Figure 5 (multi-dataset incremental learning heatmaps + baseline comparison) from pre-computed results in `data/`.

All figures are exported as PNG (600 dpi), SVG, PDF, and TIFF.

---

## Reference Data

### Pre-Computed Feature Files (included in `data/`)

The following files are pre-computed and included in the repository. They are sufficient to run all training scripts and the integration test without re-downloading external databases.

| File | Size | Description |
|------|------|-------------|
| `multigrn_features_full.json` | 34 MB | Master feature registry: orthology-to-ID mapping, per-species vocabulary sizes, human→orthoid and mouse→orthoid mappings for ~19K human and ~26K mouse genes (29,114 orthology groups total) |
| `gene_vocabulary.pt` | 0.6 MB | PyTorch checkpoint: HGNC gene vocabulary (19,295 human protein-coding genes → integer indices), loaded by `GeneVocabulary.load()` |
| `regulatory_contexts.pt` | 1.5 MB | PyTorch checkpoint: STRING PPI + KEGG pathway context per gene, loaded by `RegulatoryContextBuilder.load()`. Contains 451,924 high-confidence edges (combined score ≥ 700) |
| `string_high_conf.json` | 21 MB | STRING v11 human PPI edges at combined score ≥ 700. Each entry is `[protein_a, protein_b, score]` using Ensembl protein IDs (ENSP...) |
| `string_gene_interactions.json` | 4.1 MB | Pre-processed STRING interactions keyed by HGNC gene symbols, used by the context builder pipeline |
| `gene_genomic_features.json` | 6.3 MB | Per-gene genomic features: gene length (log), exon count, intron count, intron length (log), CDS length (log), transcript count, exon density, CDS ratio. Computed from UCSC refGene (GRCh38) |
| `hepg2_chromatin_features.json` | 5.3 MB | HepG2-specific ChromHMM 30-state chromatin features per gene, used as cell-type-specific context in 102-dim input |
| `mouse_human_ortholog_map.json` | 0.4 MB | Mouse→human ortholog pairs from Ensembl Compara (18,985 pairs). Key is mouse gene symbol, value is human gene symbol |
| `refGene.txt.gz` | 8.3 MB | UCSC refGene gene annotation (GRCh38), used to compute TSS positions and genomic feature boundaries |
| `incremental_pipeline_results/all_preprocessed_scores.json` | 5.0 MB | Pre-processed ChIP-seq scores for 14 conditions: H4K5la (4 conditions, narrowPeak), H3K18la peak-gene (3 conditions), H3K18la TSS bigWig (4 conditions), H3K18la sepsis/bladder (2+1 conditions). Each entry is `{gene_symbol: enrichment_score}` |
| `bigwig_preprocessed_v2/bigwig_scores_v2.json` | 4.4 MB | Improved TSS-centered bigWig scores (v2) with rank-percentile normalization for 11 conditions across GSE314155, GSE247800, GSE328660, GSE325983 |

### Feature Dimension Reference

| Feature Group | Dimensions | Source | Used In |
|--------------|-----------|--------|---------|
| STRING PPI context | 64 | `string_high_conf.json` → `RegulatoryContextBuilder` | All experiments |
| Genomic features | 8 | `gene_genomic_features.json` | All experiments |
| ChromHMM states (HepG2) | 30 | `hepg2_chromatin_features.json` | Multi-dataset (§2.5) only |
| **Total (single-domain)** | **72** | | **§2.2, §2.3, §2.4** |
| **Total (multi-dataset)** | **102** | | **§2.5** |

### External Databases

The model uses data from the following databases. Feature files are pre-processed from these sources:



---

## Citation

If you use Engram-MultiGRN in your research, please cite:

```bibtex
@article{engram2026,
  title={Engram-MultiGRN: Domain-Isolated Conditional Memory for Cross-Species Gene Regulatory Network Inference},
  author={...},
  journal={bioRxiv},
  year={2026},
  doi={...}
}
```

---

## License

[License information to be added]

## Contact

[Corresponding author email to be added]
