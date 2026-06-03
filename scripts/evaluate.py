"""
Evaluate Engram-GRN on Kla prediction tasks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, json, random
import numpy as np
from pathlib import Path

from engram_grn.model.config import EngramGRNConfig
from engram_grn.model.engram_grn import EngramGRN
from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder

DATA_DIR = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
vocab = GeneVocabulary(str(DATA_DIR)); vocab.load()
ctx = RegulatoryContextBuilder(vocab, str(DATA_DIR)); ctx.load()
kla_labels = torch.load(DATA_DIR / "kla_gene_labels.pt", weights_only=False)
kla_high = [g for g in kla_labels['high_confidence'] if g in vocab.gene_to_idx]
kla_medium = [g for g in kla_labels['medium_confidence'] if g in vocab.gene_to_idx]
kla_set = set(kla_high + kla_medium)

# Load model
cfg = EngramGRNConfig()
cfg.engram_vocab_size = [1000000, 1000000]
cfg.gene_embed_dim = 128
cfg.vocab_size = vocab.vocab_size
model = EngramGRN(cfg).to(device)
state = torch.load(DATA_DIR / "engram_grn_best.pt", map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Conditions to test
conditions = {
    'high_lactate': [0.0, 0.0, 0.9, 0.1],
    'low_lactate':  [0.0, 0.0, 0.1, 0.0],
    'hypoxia':      [0.9, 0.0, 0.8, 0.0],
    'normoxia':     [0.0, 0.0, 0.2, 0.0],
    'm1_mac':       [0.0, 0.9, 0.85, 0.8],
    'm2_mac':       [0.0, 0.0, 0.3, 0.9],
}

def pad_cond(vec, dim=32):
    v = torch.zeros(dim, dtype=torch.float32, device=device)
    v[:len(vec)] = torch.tensor(vec, dtype=torch.float32, device=device)
    return v.unsqueeze(0)

print("\n=== Kla Prediction by Condition ===")
results = {}
for cond_name, cond_vec in conditions.items():
    cond_tensor = pad_cond(cond_vec)

    high_preds, medium_preds, non_kla_preds = [], [], []
    for gene in kla_high:
        gidx = torch.tensor([vocab.gene_to_idx[gene]], device=device)
        cids = ctx.get_context_for_genes(gidx, max_context=cfg.max_context_size).to(device)
        with torch.no_grad():
            p = model.predict_kla(gidx, cids, cond_tensor).item()
        high_preds.append(p)

    for gene in kla_medium:
        gidx = torch.tensor([vocab.gene_to_idx[gene]], device=device)
        cids = ctx.get_context_for_genes(gidx, max_context=cfg.max_context_size).to(device)
        with torch.no_grad():
            p = model.predict_kla(gidx, cids, cond_tensor).item()
        medium_preds.append(p)

    non_kla_sample = [g for g in vocab.gene_list if g not in kla_set][:200]
    for gene in non_kla_sample:
        gidx = torch.tensor([vocab.gene_to_idx[gene]], device=device)
        cids = ctx.get_context_for_genes(gidx, max_context=cfg.max_context_size).to(device)
        with torch.no_grad():
            p = model.predict_kla(gidx, cids, cond_tensor).item()
        non_kla_preds.append(p)

    hp = np.mean(high_preds) if high_preds else 0
    mp = np.mean(medium_preds) if medium_preds else 0
    np_ = np.mean(non_kla_preds) if non_kla_preds else 0
    results[cond_name] = (hp, mp, np_)
    print(f"  {cond_name:15s}  Kla-high: {hp:.3f}  Kla-med: {mp:.3f}  Non-Kla: {np_:.3f}")

# Qualitative evaluation
print("\n=== Top 20 predicted Kla targets by condition ===")
test_cond = 'high_lactate'
cond_tensor = pad_cond(conditions[test_cond])
all_scores = []
for gene in vocab.gene_list[:2000]:
    gidx = torch.tensor([vocab.gene_to_idx[gene]], device=device)
    cids = ctx.get_context_for_genes(gidx, max_context=cfg.max_context_size).to(device)
    with torch.no_grad():
        p = model.predict_kla(gidx, cids, cond_tensor).item()
    all_scores.append((gene, p))
all_scores.sort(key=lambda x: -x[1])
print(f"  Condition: {test_cond}")
for i, (gene, score) in enumerate(all_scores[:20]):
    is_kla = "✓" if gene in kla_set else " "
    print(f"  {i+1:2d}. [{is_kla}] {gene:12s}  Kla={score:.4f}")

# Separation metrics
print("\n=== Kla vs Non-Kla Separation ===")
non_kla_sample = [g for g in vocab.gene_list if g not in kla_set][:200]
for cond_name in conditions:
    cond_t = pad_cond(conditions[cond_name])
    kla_scores, non_scores = [], []
    for gene in kla_high:
        gidx = torch.tensor([vocab.gene_to_idx[gene]], device=device)
        cids = ctx.get_context_for_genes(gidx, max_context=cfg.max_context_size).to(device)
        with torch.no_grad():
            kla_scores.append(model.predict_kla(gidx, cids, cond_t).item())
    for gene in non_kla_sample:
        gidx = torch.tensor([vocab.gene_to_idx[gene]], device=device)
        cids = ctx.get_context_for_genes(gidx, max_context=cfg.max_context_size).to(device)
        with torch.no_grad():
            non_scores.append(model.predict_kla(gidx, cids, cond_t).item())
    sep = np.mean(kla_scores) - np.mean(non_scores)
    print(f"  {cond_name:15s}  Δ={sep:.4f}  Kla: {np.mean(kla_scores):.3f}±{np.std(kla_scores):.3f}  Non: {np.mean(non_scores):.3f}±{np.std(non_scores):.3f}")

print("\nDone.")
