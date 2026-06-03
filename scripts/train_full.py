"""
Full-scale training of Engram-GRN with HGNC vocabulary and STRING contexts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from engram_grn.model.config import EngramGRNConfig
from engram_grn.model.engram_grn import EngramGRN
from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder
from engram_grn.data_pipeline.dataset import (
    EngramGRNDataset, KlaSample, create_dataloaders
)

DATA_DIR = Path("data")


def build_realistic_dataset(gene_vocab, context_builder, num_neg_per_pos=10):
    """Build dataset from known Kla targets + STRING contexts."""
    import random
    random.seed(42)

    kla_labels = torch.load(DATA_DIR / "kla_gene_labels.pt", weights_only=False)
    gene_to_idx = gene_vocab.gene_to_idx

    kla_high = [g for g in kla_labels['high_confidence'] if g in gene_to_idx]
    kla_medium = [g for g in kla_labels['medium_confidence'] if g in gene_to_idx]
    kla_high_idx = [gene_to_idx[g] for g in kla_high]
    kla_medium_idx = [gene_to_idx[g] for g in kla_medium]
    kla_combined_idx = set(kla_high_idx + kla_medium_idx)

    all_genes = list(gene_to_idx.keys())
    non_kla_genes = [g for g in all_genes if gene_to_idx.get(g, -1) not in kla_combined_idx]

    conditions = {
        'high_lactate': [0.0, 0.0, 0.9, 0.1],
        'low_lactate':  [0.0, 0.0, 0.1, 0.0],
        'hypoxia':      [0.9, 0.0, 0.8, 0.0],
        'normoxia':     [0.0, 0.0, 0.2, 0.0],
        'm1_mac':       [0.0, 0.9, 0.85, 0.8],
        'm2_mac':       [0.0, 0.0, 0.3, 0.9],
    }
    cond_keys = list(conditions.keys())

    samples = []
    for gene in kla_high:
        for cond_name, cond_vec in conditions.items():
            if cond_name in ('high_lactate', 'hypoxia', 'm1_mac'):
                score = random.uniform(0.7, 0.95)
            else:
                score = random.uniform(0.2, 0.45)
            samples.append(KlaSample(gene, score, cond_name, cond_vec, [gene]))

    for gene in kla_medium:
        for cond_name, cond_vec in conditions.items():
            if cond_name in ('high_lactate', 'hypoxia', 'm1_mac'):
                score = random.uniform(0.4, 0.7)
            else:
                score = random.uniform(0.1, 0.3)
            samples.append(KlaSample(gene, score, cond_name, cond_vec, [gene]))

    neg_genes_needed = len(kla_combined_idx) * num_neg_per_pos * len(conditions)
    available_neg = [g for g in non_kla_genes if g in gene_to_idx]
    sampled_neg = random.sample(available_neg, min(neg_genes_needed, len(available_neg)))
    for gene in sampled_neg:
        cond_name = random.choice(cond_keys)
        cond_vec = conditions[cond_name]
        score = random.uniform(0.0, 0.1)
        samples.append(KlaSample(gene, score, cond_name, cond_vec, [gene]))

    random.shuffle(samples)
    print(f"Total samples: {len(samples)}")
    print(f"  Kla high: {len(kla_high) * len(conditions)}")
    print(f"  Kla medium: {len(kla_medium) * len(conditions)}")
    print(f"  Non-Kla: {len(sampled_neg)}")
    return samples


class ConditionedGRNDataset(EngramGRNDataset):
    def __init__(self, samples, gene_vocab, context_builder, max_context=5, condition_dim=32):
        super().__init__(samples, gene_vocab, context_builder, max_context, condition_dim)

    def __getitem__(self, idx):
        s = self.samples[idx]
        gidx = self.gene_vocab.gene_to_idx.get(s.gene_name, 0)
        ctx = self.context_builder.get_context_for_genes(
            torch.tensor([gidx]), max_context=self.max_context).squeeze(0)
        cond = torch.tensor(s.condition_vec[:self.condition_dim], dtype=torch.float32)
        if len(cond) < self.condition_dim:
            cond = torch.nn.functional.pad(cond, (0, self.condition_dim - len(cond)))
        label = torch.tensor([s.kla_score], dtype=torch.float32)
        return gidx, ctx, cond, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    vocab = GeneVocabulary(str(DATA_DIR))
    if not vocab.load():
        print("ERROR: Gene vocabulary not found. Build it first.")
        return
    print(f"Vocabulary: {vocab.vocab_size} genes")

    ctx_builder = RegulatoryContextBuilder(vocab, str(DATA_DIR))
    if not ctx_builder.load():
        print("ERROR: Regulatory contexts not found.")
        return
    ctx_stats = sum(len(v) for v in ctx_builder.gene_context_map.values())
    print(f"Contexts: {len(ctx_builder.gene_context_map)} genes, {ctx_stats} pairs")

    samples = build_realistic_dataset(vocab, ctx_builder)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples, val_samples = samples[:split], samples[split:]

    cfg = EngramGRNConfig()
    cfg.vocab_size = vocab.vocab_size
    cfg.max_context_size = min(cfg.max_context_size, 5)
    cfg.condition_dim = 32
    cfg.gene_embed_dim = 128
    cfg.n_embed_per_ngram = 128
    cfg.n_head_per_ngram = 4
    cfg.engram_vocab_size = [1000000, 1000000]
    cfg.learning_rate = 5e-4
    cfg.batch_size = 128
    cfg.max_epochs = 50

    model = EngramGRN(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    total_mb = total * 4 / 1024 / 1024
    print(f"Model: {total:,} params ({total_mb:.1f}MB @ FP32)")

    # Batch size calibration
    if device.type == "cuda":
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_gb = free_mem / 1024**3
        print(f"Free VRAM: {free_gb:.1f}GB")
        suggested_bs = max(1, min(cfg.batch_size, int(free_gb * 32)))
        cfg.batch_size = min(cfg.batch_size, suggested_bs)
        print(f"Batch size: {cfg.batch_size}")

    train_ds = ConditionedGRNDataset(train_samples, vocab, ctx_builder, max_context=cfg.max_context_size)
    val_ds = ConditionedGRNDataset(val_samples, vocab, ctx_builder, max_context=cfg.max_context_size)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss = 0.0
        for g, c, cond, lbl in train_loader:
            g, c, cond, lbl = g.to(device), c.to(device), cond.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(g, c, cond), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g, c, cond, lbl in val_loader:
                g, c, cond, lbl = g.to(device), c.to(device), cond.to(device), lbl.to(device)
                loss = criterion(model(g, c, cond), lbl)
                val_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), DATA_DIR / "engram_grn_best.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{cfg.max_epochs}  Train: {train_loss:.5f}  Val: {val_loss:.5f}  Best: {best_val:.5f}")

    print(f"\nTraining complete. Best val: {best_val:.5f}")

    # Evaluation metrics
    model.load_state_dict(torch.load(DATA_DIR / "engram_grn_best.pt"))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for g, c, cond, lbl in val_loader:
            g, c, cond = g.to(device), c.to(device), cond.to(device)
            pred = model(g, c, cond).cpu()
            all_pred.append(pred)
            all_true.append(lbl)
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    mse = nn.functional.mse_loss(all_pred, all_true).item()
    mae = nn.functional.l1_loss(all_pred, all_true).item()
    print(f"Final test MSE: {mse:.6f}, MAE: {mae:.6f}")


if __name__ == "__main__":
    main()
