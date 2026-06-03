"""
Dataset for Engram-GRN training and Kla prediction.
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class KlaSample:
    def __init__(self, gene_name: str, kla_score: float, condition: str,
                 condition_vec: List[float], context_genes: List[str]):
        self.gene_name = gene_name
        self.kla_score = kla_score
        self.condition = condition
        self.condition_vec = condition_vec
        self.context_genes = context_genes


class EngramGRNDataset(Dataset):
    """
    Dataset for gene regulatory network training.
    Each sample: (target_gene, context_genes, condition_vector, label)
    """
    def __init__(
        self,
        samples: List[KlaSample],
        gene_vocab,
        context_builder,
        max_context: int = 5,
        condition_dim: int = 32,
    ):
        self.samples = samples
        self.gene_vocab = gene_vocab
        self.context_builder = context_builder
        self.max_context = max_context
        self.condition_dim = condition_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        gene_idx = self.gene_vocab.gene_to_idx.get(sample.gene_name, 0)
        context = self.context_builder.get_context_for_genes(
            torch.tensor([gene_idx]),
            max_context=self.max_context,
        ).squeeze(0)
        cond = torch.tensor(sample.condition_vec, dtype=torch.float32)
        if cond.shape[0] < self.condition_dim:
            cond = torch.nn.functional.pad(cond, (0, self.condition_dim - cond.shape[0]))
        elif cond.shape[0] > self.condition_dim:
            cond = cond[:self.condition_dim]
        label = torch.tensor([sample.kla_score], dtype=torch.float32)
        return gene_idx, context, cond, label


def create_synthetic_kla_data(
    gene_vocab,
    context_builder,
    num_samples: int = 1000,
) -> List[KlaSample]:
    """
    Create synthetic Kla data for testing the training pipeline.
    Scenarios: M1 macrophages (high glycolysis → high Kla),
               M2 macrophages (moderate), control (low).
    """
    import random
    samples = []

    m1_cond = [1.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0] + [0.0] * 24
    m2_cond = [0.0, 1.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0] + [0.0] * 24
    control_cond = [0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0] + [0.0] * 24

    kla_genes = ["H2AC1", "H2BC1", "H3C1", "H4C1", "H3F3A",
                 "LDHA", "PKM", "HIF1A", "MYC", "STAT3"]

    non_kla = ["TP53", "AKT1", "BRCA1", "CTNNB1", "EGFR",
               "KRAS", "PTEN", "RB1", "SRC", "VEGFA"]

    for g in kla_genes:
        samples.append(KlaSample(g, 0.85, "M1", m1_cond, [g]))
        samples.append(KlaSample(g, 0.45, "M2", m2_cond, [g]))
        samples.append(KlaSample(g, 0.10, "Ctrl", control_cond, [g]))

    for g in non_kla:
        samples.append(KlaSample(g, 0.08, "M1", m1_cond, [g]))
        samples.append(KlaSample(g, 0.05, "M2", m2_cond, [g]))
        samples.append(KlaSample(g, 0.03, "Ctrl", control_cond, [g]))

    for _ in range(num_samples - len(samples)):
        gene = random.choice(list(gene_vocab.gene_to_idx.keys()))
        cond_type = random.choice(["M1", "M2", "Ctrl"])
        if cond_type == "M1":
            cond = m1_cond
            score = 0.7 if gene in kla_genes else 0.1
        elif cond_type == "M2":
            cond = m2_cond
            score = 0.4 if gene in kla_genes else 0.05
        else:
            cond = control_cond
            score = 0.08 if gene in kla_genes else 0.02
        samples.append(KlaSample(gene, score, cond_type, cond, [gene]))

    return samples


def create_dataloaders(
    gene_vocab,
    context_builder,
    batch_size: int = 64,
    num_samples: int = 2000,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    all_samples = create_synthetic_kla_data(gene_vocab, context_builder, num_samples)
    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * train_ratio)
    train_dataset = EngramGRNDataset(all_samples[:split], gene_vocab, context_builder)
    val_dataset = EngramGRNDataset(all_samples[split:], gene_vocab, context_builder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
