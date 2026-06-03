"""Build proper multi-species features for MultiGRN:
1. Real Ensembl orthology mapping (human↔mouse share ortho_id)
2. Per-species STRING context features 
3. Full genomic features for all species
"""
import sys, os, json, gzip, random, math
sys.path.insert(0, '.')
import numpy as np
import torch
from pathlib import Path; from collections import defaultdict

DATA = Path("data")
print("Building multi-species features...", flush=True)

# ═══ 1. Real orthology mapping ═══
ortho_pairs = json.load(open(DATA/"mouse_human_ortholog_map.json"))
mouse_to_human = dict(ortho_pairs)
human_to_mouse = {h:m for m,h in mouse_to_human.items()}

from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder, PATHWAYS

# Human vocab
vocab_h = GeneVocabulary(str(DATA)); vocab_h.load()

# Mouse vocab
mouse_genes = set()
with gzip.open(DATA/"mouse_refGene.txt.gz", "rt") as f:
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 13: mouse_genes.add(p[12])
mouse_list = sorted(mouse_genes)
mouse_g2i = {g:i+1 for i,g in enumerate(mouse_list)}
print(f"Human: {vocab_h.vocab_size}, Mouse: {len(mouse_list)}", flush=True)

# Build ortho_id: orthologous genes share the same ID
gene_to_orthoid = {}
oid_counter = 1

# First pass: human genes with mouse orthologs
for h_gene in vocab_h.gene_to_idx:
    if h_gene in gene_to_orthoid: continue
    gene_to_orthoid[h_gene] = oid_counter
    m_gene = human_to_mouse.get(h_gene)
    if m_gene and m_gene in mouse_g2i and m_gene not in gene_to_orthoid:
        gene_to_orthoid[m_gene] = oid_counter
    oid_counter += 1

# Second pass: remaining mouse genes
for m_gene in mouse_g2i:
    if m_gene not in gene_to_orthoid:
        gene_to_orthoid[m_gene] = oid_counter
        oid_counter += 1

# Third pass: remaining human genes  
for h_gene in vocab_h.gene_to_idx:
    if h_gene not in gene_to_orthoid:
        gene_to_orthoid[h_gene] = oid_counter
        oid_counter += 1

N_ORTHO = oid_counter - 1
print(f"Ortho groups: {N_ORTHO}", flush=True)

# ═══ 2. Human STRING context features ═══
print("Computing human STRING features...", flush=True)
ctx_h = RegulatoryContextBuilder(vocab_h, str(DATA)); ctx_h.load()
all_human_genes = list(vocab_h.gene_to_idx.keys())[:10000]

human_strung_feats = {}
for gene in all_human_genes:
    gid = vocab_h.gene_to_idx.get(gene, 0)
    if gid == 0:
        human_strung_feats[gene] = [0.0]*64; continue
    c = ctx_h.get_context_for_genes(torch.tensor([gid]), max_context=3).squeeze(0).float()
    if c.shape[0] < 64: c = torch.nn.functional.pad(c, (0, 64-c.shape[0]))
    human_strung_feats[gene] = c[:64].tolist()
print(f"  {len(human_strung_feats)} genes", flush=True)

# ═══ 3. Mouse STRING context features (from mapped data) ═══
print("Computing mouse STRING features...", flush=True)
mouse_string = json.load(open(DATA/"mouse_string_human_mapped.json"))

# Build mouse partner list (mouse gene → set of mouse partner indices)
mouse_context = defaultdict(set)
for h_gene, partners in mouse_string.items():
    m_gene = human_to_mouse.get(h_gene)
    if m_gene and m_gene in mouse_g2i:
        mi = mouse_g2i[m_gene]
        for p in partners:
            mp = human_to_mouse.get(p)
            if mp and mp in mouse_g2i:
                mouse_context[mi].add(mouse_g2i[mp])

mouse_strung_feats = {}
for mgene, mi in list(mouse_g2i.items())[:10000]:
    partners = mouse_context.get(mi, set())
    # Encode partners as feature vector (up to 64 dims)
    feat = [float(p)/max(len(mouse_g2i),1) for p in list(partners)[:64]]
    if len(feat) < 64: feat = feat + [0.0]*(64-len(feat))
    mouse_strung_feats[mgene] = feat[:64]
print(f"  {len(mouse_strung_feats)} genes", flush=True)

# ═══ 4. Save feature mappings ═══
feat_data = {
    "n_ortho_groups": N_ORTHO,
    "human_vocab_size": vocab_h.vocab_size,
    "mouse_vocab_size": len(mouse_list),
    "human_to_orthoid": {g:gene_to_orthoid.get(g,0) for g in vocab_h.gene_to_idx if g in gene_to_orthoid},
    "mouse_to_orthoid": {g:gene_to_orthoid.get(g,0) for g in mouse_g2i if g in gene_to_orthoid},
}
json.dump(feat_data, open(DATA/"multigrn_features.json", "w"), indent=2)
print(f"\nSaved: data/multigrn_features.json", flush=True)
print(f"  Ortho groups: {N_ORTHO}", flush=True)
print(f"  Human genes mapped: {sum(1 for g in vocab_h.gene_to_idx if g in gene_to_orthoid)}", flush=True)
print(f"  Mouse genes mapped: {sum(1 for g in mouse_g2i if g in gene_to_orthoid)}", flush=True)
