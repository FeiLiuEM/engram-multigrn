"""
Build multi-species data for Engram-MultiGRN training.
Produces: ortho_vocab, per-species gene idx, STRING context features,
          unified Kla training data with species/cell-type labels.
"""
import sys, os, json, gzip, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).parent.parent / "data"
print("Building multi-species dataset...", flush=True)

# ═══ 1. Ortholog mapping ═══
ortho_map_raw = json.load(open(DATA / "mouse_human_ortholog_map.json"))
# Build forward (mouse→human) and reverse (human→mouse)
mouse_to_human = dict(ortho_map_raw)  # {mouse_gene: human_gene}
human_to_mouse = {h: m for m, h in mouse_to_human.items()}
print(f"Orthologs: {len(mouse_to_human)} mouse→human, {len(human_to_mouse)} human→mouse")

# Build ortho_id mapping: assign unique IDs to orthologous groups
from engram_grn.data_pipeline.gene_vocab import GeneVocabulary

# Human vocab
human_vocab = GeneVocabulary(str(DATA))
human_vocab.load()
# Mouse vocab
mouse_vocab = GeneVocabulary(str(DATA))
mouse_vocab.file_names = {'gene_vocab': str(DATA / 'mouse_gene_vocab.pt')}
# Check if mouse vocab exists
mouse_vocab_path = DATA / 'mouse_gene_vocab.pt'
if mouse_vocab_path.exists():
    mouse_vocab.load()
else:
    # Build from mouse refGene
    print("Building mouse vocabulary...", flush=True)
    mouse_genes = set()
    with gzip.open(DATA / 'mouse_refGene.txt.gz', 'rt') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 13:
                mouse_genes.add(p[12])
    mouse_vocab.vocab_list = sorted(mouse_genes)
    mouse_vocab.gene_to_idx = {g: i+1 for i, g in enumerate(mouse_vocab.vocab_list)}
    mouse_vocab.vocab_size = len(mouse_vocab.gene_to_idx) + 1
    mouse_vocab.pad_id = 0
    print(f"Mouse vocab: {mouse_vocab.vocab_size} genes (from refGene)", flush=True)

VS_human = human_vocab.vocab_size
VS_mouse = mouse_vocab.vocab_size
print(f"Human vocab: {VS_human} genes", flush=True)
print(f"Mouse vocab: {VS_mouse} genes", flush=True)

# Assign ortho_ids: 0=pad, 1..N for each orthologous group
ortho_id_map_human = {}  # {human_gene_name: ortho_id}
ortho_id_map_mouse = {}  # {mouse_gene_name: ortho_id}
ortho_id = 1
gene_to_orthoid = {}
orthoid_to_genes = {}

# Process all human genes
for h_gene in human_vocab.gene_to_idx:
    if h_gene not in gene_to_orthoid:
        gene_to_orthoid[h_gene] = ortho_id
        orthoid_to_genes[ortho_id] = [h_gene]
        # If mouse ortholog exists, add it
        if h_gene in human_to_mouse:
            m_gene = human_to_mouse[h_gene]
            if m_gene not in gene_to_orthoid:
                gene_to_orthoid[m_gene] = ortho_id
                orthoid_to_genes[ortho_id].append(m_gene)
        ortho_id += 1

# Process remaining mouse genes not mapped to human
for m_gene in mouse_vocab.gene_to_idx:
    if m_gene not in gene_to_orthoid:
        gene_to_orthoid[m_gene] = ortho_id
        orthoid_to_genes[ortho_id] = [m_gene]
        ortho_id += 1

N_ortho = ortho_id - 1
print(f"Orthologous groups: {N_ortho}", flush=True)

# ═══ 2. Load STRING context features ═══
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder, PATHWAYS

print("Loading human STRING context...", flush=True)
ctx_human = RegulatoryContextBuilder(human_vocab, str(DATA))
ctx_human.load()

# Build mouse STRING context from mapped data
print("Building mouse STRING context...", flush=True)
mouse_string = json.load(open(DATA / "mouse_string_human_mapped.json"))
# Build mouse context: for each mouse gene, get its partners (also mouse)
mouse_context = defaultdict(set)
for h_gene, partners in mouse_string.items():
    m_gene = human_to_mouse.get(h_gene)
    if m_gene and m_gene in mouse_vocab.gene_to_idx:
        for p in partners:
            m_partner = human_to_mouse.get(p)
            if m_partner and m_partner in mouse_vocab.gene_to_idx:
                mouse_context[mouse_vocab.gene_to_idx[m_gene]].add(
                    mouse_vocab.gene_to_idx[m_partner])

# Build gene features for both species (for baseline comparison)
def build_gene_features(vocab, feat_dim=51):
    """Simplified features: random 51-dim (matches human features)."""
    return torch.zeros((vocab.vocab_size, feat_dim)).uniform_(-0.1, 0.1)

# ═══ 3. Load unified Kla dataset ═══
print("Loading Kla data...", flush=True)
chip_data = json.load(open(DATA / "kla_chip_scores.json"))

# Human samples (H4K5la)
cond_map_h = {'H4K5la_NM2':0, 'H4K5la_NM3':0, 'H4K5la_LAC2':1, 'H4K5la_LAC3':1}
cond_vals_h = defaultdict(list)
for k, s in chip_data.items():
    g, c = k.split('__')
    cond_vals_h[c].append(s)

human_samples = []
for k, raw in chip_data.items():
    g, c = k.split('__')
    if c not in cond_map_h: continue
    vs = cond_vals_h[c]; mn, mx = min(vs), max(vs)
    n = (raw - mn) / (mx - mn) if mx > mn else 0.5
    ortho = gene_to_orthoid.get(g, 0)
    sp_gid = human_vocab.gene_to_idx.get(g, 0)
    human_samples.append({
        'ortho_id': ortho, 'sp_gene_id': sp_gid, 'score': min(1, max(0, n)),
        'cond_id': cond_map_h[c], 'species': 'human', 'cell': 'hepg2',
        'gene': g
    })

# Mouse Kla samples (from unified dataset or placeholder)
# Check if we have real mouse Kla data
mouse_kla = []
unified = json.load(open(DATA / "unified_kla_dataset.json"))
# Explore unified_kla_dataset structure
if isinstance(unified, list):
    for item in unified[:5]:
        if isinstance(item, dict):
            print(f"  Sample keys: {list(item.keys())[:10]}", flush=True)
            break
elif isinstance(unified, dict):
    print(f"Unified dataset keys: {list(unified.keys())[:10]}", flush=True)

# Try to find mouse Kla data
for k in unified.keys() if isinstance(unified, dict) else []:
    if 'mouse' in k.lower() or 'Mouse' in k:
        mouse_kla_data = unified[k]
        print(f"Found mouse Kla data under key: {k}, type={type(mouse_kla_data).__name__}", flush=True)
        if isinstance(mouse_kla_data, list):
            print(f"  Size: {len(mouse_kla_data)}", flush=True)
PYEOF