#!/usr/bin/env python3
"""
Engram-MultiGRN: Full preprocessing + incremental multi-dataset training pipeline.

Flow:
  1. Preprocess all 7 GEO ChIP-seq datasets → unified gene-level scores
  2. Build training samples with STRING(64)+genomic(8)+chromatin(30)=102d features
  3. Incremental training: sequential multi-task with catastrophic forgetting evaluation

Datasets:
  D1: GSE314769   H4K5la HCC (HepG2)           - narrowPeak
  D2: GSE269142   HIF1a/H3K18la (MDA-MB-231)    - peak-gene.txt (mouse → human ortholog)
  D3: GSE314155   H3K18la/BRD9 HCC (HepG2)      - bigWig
  D4: GSE247800   GTPSCS K192R H3K18la (HEK293T) - bigWig
  D5: GSE328660   H3K18la sepsis (THP-1)         - bedgraph + RNA-seq
  D6: GSE325983   H3K18la bladder (T24)          - bigWig
"""

import sys, os, json, gzip, random, time, bisect, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch, torch.nn as nn
import pyBigWig
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
DATA = Path(__file__).parent.parent / "data"
KLA_DIR = DATA / "kla_chip" / "extracted"
OUTPUT = DATA / "incremental_pipeline_results"
OUTPUT.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "CPU only")

# ═══════════════════════════════════════════════════════════════════════
# PART 1: GENE INTERVAL PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_refgene(path):
    """Parse UCSC refGene to get gene intervals (chr, min_start, max_end)."""
    genes = {}
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 13: continue
            name, chrom, s, e = p[12], p[2], int(p[4]), int(p[5])
            if name not in genes:
                genes[name] = {'chrom': chrom, 'min_s': s, 'max_e': e}
            else:
                genes[name]['min_s'] = min(genes[name]['min_s'], s)
                genes[name]['max_e'] = max(genes[name]['max_e'], e)
    return genes

# ═══════════════════════════════════════════════════════════════════════
# PART 2: PER-DATASET PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_bigwig_gene_scores(bw_path, gene_intervals, gene_list, label=""):
    """Compute mean ChIP-seq signal per gene from bigWig (±2kb window).
    Uses bw.stats() (fast batch API, 300x faster than bw.values())."""
    try:
        bw = pyBigWig.open(str(bw_path))
    except Exception as e:
        print(f"  [SKIP] Cannot open {Path(bw_path).name}: {e}")
        return {}
    bw_chroms = set(bw.chroms().keys())
    scores = {}
    skipped = 0
    for gene in gene_list:
        info = gene_intervals.get(gene)
        if not info: continue
        chrom = info['chrom']
        candidates = [chrom]
        if chrom.startswith('chr'): candidates.append(chrom[3:])
        else: candidates.append('chr' + chrom)
        matched = None
        for c in candidates:
            if c in bw_chroms:
                matched = c; break
        if not matched: continue
        try:
            # Use stats() for fast server-side mean computation
            result = bw.stats(matched,
                              max(0, info['min_s'] - 2000),
                              info['max_e'] + 2000,
                              type='mean', nBins=1)
            if result and result[0] is not None and not np.isnan(result[0]):
                scores[gene] = float(result[0])
        except:
            skipped += 1
    bw.close()
    if label:
        print(f"  {label}: {len(scores)} genes ({skipped} skipped)")
    return scores


def compute_bedgraph_gene_scores(bg_path, gene_intervals, gene_list, label=""):
    """Compute mean ChIP-seq signal per gene from bedGraph.
    Handles both 'chr1' and '1' chromosome naming. Skips header lines."""
    chrom_bg = defaultdict(list)
    opener = gzip.open(bg_path, 'rt') if str(bg_path).endswith('.gz') else open(bg_path)
    with opener as f:
        for line in f:
            if line.startswith('track') or line.startswith('browser') or line.startswith('#'):
                continue
            p = line.strip().split()
            if len(p) < 4: continue
            # Skip header lines like "type=bedGraph"
            try:
                s_pos = int(p[1])
            except ValueError:
                continue
            chrom_bg[p[0]].append({'s': s_pos, 'e': int(p[2]), 'v': float(p[3])})
    scores = {}
    for gene in gene_list:
        info = gene_intervals.get(gene)
        if not info: continue
        chrom = info['chrom']
        candidates = [chrom]
        if chrom.startswith('chr'): candidates.append(chrom[3:])
        else: candidates.append('chr' + chrom)
        gs, ge = info['min_s'], info['max_e']
        for c in candidates:
            if c not in chrom_bg: continue
            vals = [bg['v'] for bg in chrom_bg[c] if bg['s'] < ge and bg['e'] > gs]
            if vals:
                scores[gene] = np.mean(vals)
                break
    if label:
        print(f"  {label}: {len(scores)} genes")
    return scores


def compute_narrowpeak_gene_scores(np_path, gene_intervals, gene_list):
    """Score genes by peak overlap using narrowPeak signalValue column."""
    peaks = []
    opener = gzip.open(np_path, 'rt') if str(np_path).endswith('.gz') else open(np_path)
    with opener as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 7:
                peaks.append({'chr': p[0], 's': int(p[1]), 'e': int(p[2]),
                              'score': float(p[6])})
    gene_scores = {}
    for gene in gene_list:
        info = gene_intervals.get(gene)
        if not info: continue
        total = 0.0
        for pk in peaks:
            if pk['chr'] == info['chrom'] and pk['s'] < info['max_e'] and pk['e'] > info['min_s']:
                overlap = min(pk['e'], info['max_e']) - max(pk['s'], info['min_s'])
                if overlap > 0:
                    gene_len = max(1, info['max_e'] - info['min_s'])
                    total += pk['score'] * (overlap / gene_len)
        if total > 0: gene_scores[gene] = total
    return gene_scores


def process_d1_gse314769(gene_intervals, gene_list):
    """D1: GSE314769 - H4K5la HCC (HepG2), narrowPeak files.
    Conditions: NM (normal medium) = baseline, LAC (lactate) = high Kla."""
    print("\n[D1] GSE314769 - H4K5la HCC (narrowPeak)")
    np_dir = KLA_DIR / "GSE314769"
    conditions = {}
    for label, fname in [('H4K5la_NM2', 'GSM9410262_NM2_peaks_RIAS.narrowPeak.gz'),
                          ('H4K5la_NM3', 'GSM9410263_NM3_peaks_RIAS.narrowPeak.gz'),
                          ('H4K5la_LAC2', 'GSM9410264_LAC2_peaks_RIAS.narrowPeak.gz'),
                          ('H4K5la_LAC3', 'GSM9410265_LAC3_peaks_RIAS.narrowPeak.gz')]:
        scores = compute_narrowpeak_gene_scores(str(np_dir / fname), gene_intervals, gene_list)
        conditions[label] = scores
        print(f"  {label}: {len(scores)} genes, mean={np.mean(list(scores.values())):.3f}")
    return conditions


def process_d2_gse269142(gene_intervals, gene_list, ortholog_map):
    """D2: GSE269142 - HIF1a/H3K18la (MDA-MB-231 breast), peak-gene.txt.gz.
    Mouse data → human ortholog mapping.
    Conditions: CON (control), HYP (hypoxia), HIF-KD (HIF1a knockdown)."""
    print("\n[D2] GSE269142 - HIF1a/H3K18la breast cancer (peak-gene.txt)")
    peak_dir = KLA_DIR / "GSE269142"
    conditions = {}
    # Count peaks per gene from peak-gene.txt.gz
    for label, fname in [('H3K18la_CON', 'GSM8306937_CON_peak-gene.txt.gz'),
                          ('H3K18la_HYP', 'GSM8306939_HYP_peak-gene.txt.gz'),
                          ('H3K18la_HIFKD', 'GSM8306941_HIF-KD_peak-gene.txt.gz')]:
        gene_counts = defaultdict(int)
        with gzip.open(str(peak_dir / fname), 'rt') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4:
                    mouse_gene = p[4]  # Gene symbol (mouse)
                    human_gene = ortholog_map.get(mouse_gene, None)
                    if human_gene:
                        gene_counts[human_gene] += 1
        if gene_counts:
            # Normalize: peak count → score (0-1 via log normalization)
            max_count = max(gene_counts.values())
            conditions[label] = {g: float(np.log2(c + 1) / max(np.log2(max_count + 1), 0.001))
                                 for g, c in gene_counts.items()}
            print(f"  {label}: {len(gene_counts)} genes (mouse→human mapped), max peaks={max_count}")
        else:
            print(f"  {label}: NO genes mapped")
    return conditions


def process_d3_gse314155(gene_intervals, gene_list):
    """D3: GSE314155 - H3K18la/BRD9 HCC HepG2 + NC vs 2-DG treatment, bigWig.
    Conditions: H3K18la_NC, H3K18la_2DG, BRD9_NC, BRD9_2DG, ATAC_NC, ATAC_2DG."""
    print("\n[D3] GSE314155 - H3K18la/BRD9/ATAC HCC (bigWig)")
    bw_dir = KLA_DIR / "GSE314155"
    conditions = {}

    # H3K18la NC
    ip = compute_bigwig_gene_scores(str(bw_dir / 'GSM9382816_ChIP-seq_HepG2_H3K18la_NC.bw'),
                                     gene_intervals, gene_list, label="H3K18la_IP")
    inp = compute_bigwig_gene_scores(str(bw_dir / 'GSM9382815_ChIP-seq_HepG2_Input_NC.bw'),
                                      gene_intervals, gene_list, label="Input_NC")
    common = set(ip.keys()) & set(inp.keys())
    if common:
        conditions['H3K18la_NC'] = {g: float(np.log2((ip[g] + 0.01) / max(inp[g], 0.01)))
                                     for g in common}
        vals = list(conditions['H3K18la_NC'].values())
        print(f"  H3K18la_NC: {len(common)} genes, mean enrichment={np.mean(vals):.3f}")
    else:
        print("  H3K18la_NC: no overlap with refGene")

    # BRD9 NC (file may be corrupted)
    brd9_path = bw_dir / 'GSM9382817_ChIP-seq_HepG2_BRD9_NC.bw'
    try:
        brd9_ip = compute_bigwig_gene_scores(str(brd9_path), gene_intervals, gene_list, label="BRD9_IP")
        common = set(brd9_ip.keys()) & set(inp.keys())
        if common:
            conditions['BRD9_NC'] = {g: float(np.log2((brd9_ip[g] + 0.01) / max(inp[g], 0.01))) for g in common}
            print(f"  BRD9_NC: {len(common)} genes enriched")
    except Exception as e:
        print(f"  BRD9_NC: CORRUPTED - skipping ({e})")

    # Try RNA-seq integration if available
    rnaseq = bw_dir / 'GSM9382806_All_RNA-seq_RPKM.xlsx'
    if rnaseq.exists():
        print(f"  RNA-seq RPKM: available at {rnaseq.name} ({rnaseq.stat().st_size/1e6:.1f} MB)")

    return conditions


def process_d4_gse247800(gene_intervals, gene_list):
    """D4: GSE247800 - GTPSCS K192R mutant H3K18la (HEK293T), bigWig.
    Conditions: K192R (mutant, 3 reps), WT (wild-type, 3 reps)."""
    print("\n[D4] GSE247800 - GTPSCS K192R H3K18la (bigWig)")
    bw_dir = KLA_DIR / "GSE247800"
    conditions = {}

    # K192R mutant (3 replicates)
    k192r_reps = ['GSM7900969_K192R_ChIP_1.bw', 'GSM7900970_K192R_ChIP_2.bw',
                   'GSM7900971_K192R_ChIP_3.bw']
    wt_reps = ['GSM7900973_WT_ChIP_1.bw', 'GSM7900974_WT_ChIP_2.bw',
                'GSM7900975_WT_ChIP_3.bw']
    inp = compute_bigwig_gene_scores(str(bw_dir / 'GSM7900972_K192R_Input.bw'),
                                      gene_intervals, gene_list)

    # Average across K192R replicates
    k192r_all = defaultdict(list)
    for fname in k192r_reps:
        scores = compute_bigwig_gene_scores(str(bw_dir / fname), gene_intervals, gene_list)
        for g, s in scores.items(): k192r_all[g].append(s)
    k192r_mean = {g: np.mean(v) for g, v in k192r_all.items()}
    common = set(k192r_mean.keys()) & set(inp.keys())
    conditions['H3K18la_K192R'] = {g: float(np.log2((k192r_mean[g] + 0.01) / max(inp[g], 0.01))) for g in common}
    print(f"  H3K18la_K192R: {len(common)} genes enriched (3 reps averaged)")

    # Average across WT replicates
    wt_all = defaultdict(list)
    for fname in wt_reps:
        scores = compute_bigwig_gene_scores(str(bw_dir / fname), gene_intervals, gene_list)
        for g, s in scores.items(): wt_all[g].append(s)
    wt_mean = {g: np.mean(v) for g, v in wt_all.items()}
    common = set(wt_mean.keys()) & set(inp.keys())
    conditions['H3K18la_WT'] = {g: float(np.log2((wt_mean[g] + 0.01) / max(inp[g], 0.01))) for g in common}
    print(f"  H3K18la_WT: {len(common)} genes enriched (3 reps averaged)")

    # Differential: K192R vs WT
    both = set(conditions.get('H3K18la_K192R', {}).keys()) & set(conditions.get('H3K18la_WT', {}).keys())
    if both:
        conditions['H3K18la_K192R_vs_WT'] = {
            g: float(conditions['H3K18la_K192R'].get(g, 0.0) - conditions['H3K18la_WT'].get(g, 0.0))
            for g in both}
        vals = list(conditions['H3K18la_K192R_vs_WT'].values())
        print(f"  H3K18la_K192R_vs_WT: {len(both)} genes differential, "
              f"mean={np.mean(vals):.4f}")

    return conditions


def process_d5_gse328660(gene_intervals, gene_list):
    """D5: GSE328660 - H3K18la sepsis (THP-1), bedgraph + RNA-seq.
    Conditions: control, lactate."""
    print("\n[D5] GSE328660 - H3K18la sepsis (bedGraph)")
    bg_dir = KLA_DIR / "GSE328660"
    conditions = {}

    scores_con = compute_bedgraph_gene_scores(str(bg_dir / 'GSM9686847_ChIP-Con.bedgraph.gz'),
                                               gene_intervals, gene_list)
    scores_lac = compute_bedgraph_gene_scores(str(bg_dir / 'GSM9686848_ChIP-LAC.bedgraph.gz'),
                                               gene_intervals, gene_list)

    # Enrichment: lactate / control
    common = set(scores_con.keys()) & set(scores_lac.keys())
    conditions['H3K18la_sepsis'] = {g: float(np.log2((scores_lac[g] + 0.01) / max(scores_con[g], 0.01))) for g in common}
    print(f"  H3K18la_sepsis: {len(common)} genes enriched")

    # Additionally save raw absolute values
    conditions['H3K18la_sepsis_lac_raw'] = scores_lac
    print(f"  H3K18la_sepsis_lac_raw: {len(scores_lac)} genes (absolute)")

    # Try RNA-seq integration
    rnaseq_path = DATA / "kla_chip" / "GSE328660_RNAseq_count.txt.gz"
    if rnaseq_path.exists():
        print(f"  RNA-seq found at {rnaseq_path}")
    else:
        print("  No RNA-seq file found")

    return conditions


def process_d6_gse325983(gene_intervals, gene_list):
    """D6: GSE325983 - H3K18la bladder cancer (T24), bigWig.
    Enrichment: IP / Input."""
    print("\n[D6] GSE325983 - H3K18la bladder cancer (bigWig)")
    bw_dir = KLA_DIR / "GSE325983"
    conditions = {}

    ip = compute_bigwig_gene_scores(str(bw_dir / 'GSM9618377_T24_H3K18la_IP.bw'),
                                     gene_intervals, gene_list)
    inp = compute_bigwig_gene_scores(str(bw_dir / 'GSM9618378_T24_H3K18la_Input.bw'),
                                      gene_intervals, gene_list)
    common = set(ip.keys()) & set(inp.keys())
    conditions['H3K18la_bladder'] = {g: float(np.log2((ip[g] + 0.01) / max(inp[g], 0.01))) for g in common}
    print(f"  H3K18la_bladder: {len(common)} genes enriched")
    return conditions


# ═══════════════════════════════════════════════════════════════════════
# PART 3: BUILD UNIFIED TRAINING DATASET
# ═══════════════════════════════════════════════════════════════════════

def build_unified_dataset(all_conditions):
    """Convert per-condition gene scores into training samples with context features."""
    print("\n" + "=" * 60)
    print("BUILDING UNIFIED TRAINING DATASET")
    print("=" * 60)

    from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
    from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder

    # Load features
    feat = json.load(open(DATA / "multigrn_features_full.json"))
    human_oid = feat["human_to_orthoid"]

    # STRING context
    print("Loading STRING context...", end=" ", flush=True)
    vocab_h = GeneVocabulary(str(DATA)); vocab_h.load()
    ctx_h = RegulatoryContextBuilder(vocab_h, str(DATA)); ctx_h.load()
    human_ctx_cache = {}
    for gene in list(vocab_h.gene_to_idx.keys())[:15000]:
        gid = vocab_h.gene_to_idx.get(gene, 0)
        if gid == 0:
            human_ctx_cache[gene] = [0.0] * 64; continue
        c = ctx_h.get_context_for_genes(torch.tensor([gid]), max_context=3).squeeze(0).float()
        if c.shape[0] < 64: c = nn.functional.pad(c, (0, 64 - c.shape[0]))
        human_ctx_cache[gene] = c[:64].tolist()
    print(f"{len(human_ctx_cache)} genes")

    # Genomic + Chromatin features
    gfeats = json.load(open(DATA / "gene_genomic_features.json"))
    cfeats = json.load(open(DATA / "hepg2_chromatin_features.json"))
    GFK = ["gene_len", "n_exons", "n_introns", "intron_len", "cds_len",
           "n_transcripts", "exon_density", "cds_ratio"]

    def get_ctx_feat(gene):
        s = human_ctx_cache.get(gene, [0.0] * 64)
        if len(s) < 64: s = s + [0.0] * (64 - len(s))
        gf = gfeats.get(gene, {k: 0.0 for k in GFK})
        cf = cfeats.get(gene, [0.0] * 30)
        if len(cf) < 30: cf = cf + [0.0] * (30 - len(cf))
        return s[:64] + [gf.get(k, 0.0) for k in GFK] + cf[:30]  # 102d

    # Build samples
    random.seed(42)
    samples = []
    cond_id_map = {}
    cond_counter = 0

    for cond_name, gene_scores in all_conditions.items():
        if not gene_scores: continue
        cond_id_map[cond_name] = cond_counter
        # Normalize scores to [0, 1]
        vals = list(gene_scores.values())
        log_vals = np.log2(np.array(vals) + 1)
        if log_vals.max() > log_vals.min():
            norm_vals = (log_vals - log_vals.min()) / (log_vals.max() - log_vals.min())
        else:
            norm_vals = np.full_like(log_vals, 0.5)

        for (gene, _), norm_score in zip(gene_scores.items(), norm_vals):
            oid = human_oid.get(gene, 0)
            sid = hash(gene) % 19000 + 1
            ctx_feat = get_ctx_feat(gene)
            samples.append({
                "ortho_id": int(oid),
                "sp_gene_id": sid,
                "score": float(norm_score),
                "cond_id": cond_counter,
                "species": "human",
                "cell": "mixed",
                "domain": cond_name,
                "mark": cond_name.split('_')[0],
                "ctx_feat": ctx_feat,
            })
        print(f"  {cond_name}: {len(gene_scores)} genes → {len(vals)} samples")
        cond_counter += 1

    print(f"\nTotal conditions: {cond_counter}")
    print(f"Total samples: {len(samples)}")
    print(f"Condition map: {cond_id_map}")
    return samples, cond_id_map, len(samples[0]['ctx_feat']) if samples else 102


# ═══════════════════════════════════════════════════════════════════════
# PART 4: INCREMENTAL TRAINING
# ═══════════════════════════════════════════════════════════════════════

def run_incremental_training(samples, cond_id_map, input_dim):
    """Sequential multi-task incremental training with catastrophic forgetting eval."""
    print("\n" + "=" * 60)
    print("INCREMENTAL TRAINING")
    print("=" * 60)



    # Split: 80/20 per condition
    random.seed(42)
    random.shuffle(samples)
    cond_samples = defaultdict(list)
    for s in samples:
        cond_samples[s['domain']].append(s)

    train_samples = {}
    test_samples = {}
    for cond, ss in cond_samples.items():
        n_train = int(len(ss) * 0.8)
        train_samples[cond] = ss[:n_train]
        test_samples[cond] = ss[n_train:]
        print(f"  {cond}: train={n_train}, test={len(ss)-n_train}")

    # Dataset class
    class ChIPDataset(torch.utils.data.Dataset):
        def __init__(self, samples_list):
            self.samples = samples_list
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            s = self.samples[i]
            return (torch.tensor(s['ortho_id'], dtype=torch.long),
                    torch.tensor(s['sp_gene_id'], dtype=torch.long),
                    torch.tensor(s['ctx_feat'], dtype=torch.float32),
                    torch.tensor(s['cond_id'], dtype=torch.long),
                    torch.tensor(s['score'], dtype=torch.float32),
                    s['species'], s['cell'], s['mark'], s['domain'])

    def collate_fn(batch):
        return (torch.stack([x[0] for x in batch]),   # ortho_ids
                torch.stack([x[1] for x in batch]),   # sp_gene_ids
                torch.stack([x[2] for x in batch]),   # ctx_feat
                torch.stack([x[3] for x in batch]),   # cond_ids
                torch.stack([x[4] for x in batch]),   # scores
                [x[5] for x in batch],                # species list
                [x[6] for x in batch],                # cell list
                [x[7] for x in batch],                # mark list
                [x[8] for x in batch])                # domain list

    def evaluate(model, test_dict):
        """Evaluate model on all test sets. Returns {condition: Pearson R}."""
        model.eval()
        results = {}
        for cond, ss in test_dict.items():
            if len(ss) < 5: continue
            loader = torch.utils.data.DataLoader(
                ChIPDataset(ss), batch_size=512, collate_fn=collate_fn)
            preds, targets = [], []
            with torch.no_grad():
                for o, sg, c, co, sc, sp, cl, mx, dm in loader:
                    o, sg, c, co = [x.to(device) for x in [o, sg, c, co]]
                    pr = model(o, sg, c, co, sp[0], cl[0], mx[0])
                    preds.extend(pr.cpu().numpy().flatten().tolist())
                    targets.extend(sc.numpy().tolist())
            preds = np.array(preds)
            targets = np.array(targets)
            if len(set(targets)) > 1 and len(preds) > 0:
                r = float(np.corrcoef(preds, targets)[0, 1])
            else:
                r = 0.0
            results[cond] = r
        return results

    # Model init
    from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig, ConditionEncoder
    feat = json.load(open(DATA / "multigrn_features_full.json"))
    cfg = MultiGRNConfig()
    cfg.n_ortho_groups = feat["n_ortho_groups"] + 1000
    model = EngramMultiGRN(cfg.n_ortho_groups, d_ctx=input_dim,
                            ctx_input_dim=input_dim).to(device)
    model.add_species("human", feat["human_vocab_size"], input_dim)
    # Fix: replace condition encoder with enough capacity for all conditions
    n_conds = len(cond_id_map)
    model.cond_encoder = ConditionEncoder(n_conditions=max(n_conds, 8), d_cond=cfg.d_cond).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Condition encoder: {n_conds} conditions")

    criterion = nn.MSELoss()
    all_results = []  # [{stage, condition, R}]
    cond_list = sorted(cond_id_map.keys())

    # ═══ INCREMENTAL LOOP ═══
    for stage_idx, cond_name in enumerate(cond_list):
        cid = cond_id_map[cond_name]
        train_data = train_samples.get(cond_name, [])
        if len(train_data) < 100:
            print(f"\n--- Stage {stage_idx + 1}: {cond_name} (SKIP: {len(train_data)} samples < 100) ---")
            continue

        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_idx + 1}/{len(cond_list)}: Training on {cond_name}")
        print(f"  Samples: {len(train_data)}, Cond ID: {cid}")
        print(f"{'=' * 60}")

        loader = torch.utils.data.DataLoader(
            ChIPDataset(train_data), batch_size=256, shuffle=True,
            collate_fn=collate_fn)

        # Train 25 epochs (quick evaluation)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        model.train()
        for ep in range(25):
            total_loss = 0.0
            for o, sg, c, co, sc, sp, cl, mx, dm in loader:
                o, sg, c, co, sc = [x.to(device) for x in [o, sg, c, co, sc]]
                optimizer.zero_grad()
                loss = criterion(model(o, sg, c, co, sp[0], cl[0], mx[0]), sc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if (ep + 1) % 5 == 0:
                print(f"  ep {ep + 1}/25 loss={total_loss / len(loader):.4f}")

        # Evaluate on ALL test sets
        r_dict = evaluate(model, test_samples)
        print(f"\n  === Post-Stage {stage_idx + 1} Evaluation (all datasets) ===")
        for cn, r_val in sorted(r_dict.items()):
            marker = "← TRAINED" if cn == cond_name else ""
            all_results.append({"stage": stage_idx + 1, "trained_on": cond_name,
                                "eval_condition": cn, "pearson_r": r_val})
            print(f"    {cn:30s} R={r_val:.4f} {marker}")

    # ═══ FINAL SUMMARY ═══
    print("\n" + "=" * 70)
    print("INCREMENTAL TRAINING SUMMARY (Catastrophic Forgetting Analysis)")
    print("=" * 70)

    # Build matrix: rows=eval condition, cols=training stage
    all_conds = sorted(set(r['eval_condition'] for r in all_results))
    header = f"{'Eval Condition':30s}"
    for stage in sorted(set(r['stage'] for r in all_results)):
        header += f" | S{stage}"
    print(header)
    print("-" * len(header))

    results_by_eval = defaultdict(dict)
    for r in all_results:
        results_by_eval[r['eval_condition']][r['stage']] = r['pearson_r']

    max_stage = max(set(r['stage'] for r in all_results)) if all_results else 0
    for ec in sorted(all_conds):
        row = f"{ec:30s}"
        for s in range(1, max_stage + 1):
            val = results_by_eval[ec].get(s, None)
            row += f" | {val:.4f}" if val is not None else " |   N/A "
        print(row)

    # Save results
    json.dump({"all_results": all_results, "condition_map": cond_id_map},
              open(OUTPUT / "incremental_results.json", "w"), indent=2)
    print(f"\nResults saved to {OUTPUT / 'incremental_results.json'}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # --- 1. Load gene intervals ---
    print("=" * 60)
    print("STEP 1: Loading gene annotations")
    print("=" * 60)
    genes_raw = parse_refgene(DATA / "refGene.txt.gz")
    print(f"  refGene entries: {len(genes_raw)}")

    # Load HGNC gene list
    vocab = torch.load(DATA / "gene_vocabulary.pt", weights_only=False)
    hgnc_set = set(vocab['gene_list'])
    gene_intervals = {g: v for g, v in genes_raw.items() if g in hgnc_set}
    gene_list = sorted(gene_intervals.keys())
    print(f"  HGNC genes with coordinates: {len(gene_intervals)}")

    # --- 2. Load ortholog map (mouse→human for GSE269142) ---
    ortho_pairs = json.load(open(DATA / "mouse_human_ortholog_map.json"))
    # mouse_to_human: {mouse_gene: human_gene}
    mouse_to_human = dict(ortho_pairs)
    # Also add reverse: human→mouse (for completeness)
    # Actually GSE269142 peak-gene.txt uses mouse gene symbols
    # The format is: chr start end ENSMUSG mouse_gene_symbol
    # So mouse_to_human maps mouse gene → human gene
    # But the gene list has human gene symbols (HGNC), so we need to map mouse→human
    print(f"\n  Ortholog pairs: {len(mouse_to_human)} mouse→human")

    # --- 3. Preprocess all datasets ---
    all_conditions = {}

    # D1: GSE314769 H4K5la HCC
    d1 = process_d1_gse314769(gene_intervals, gene_list)
    for k, v in d1.items():
        if v: all_conditions[k] = v

    # D2: GSE269142 HIF1a/H3K18la (mouse→human)
    d2 = process_d2_gse269142(gene_intervals, gene_list, mouse_to_human)
    for k, v in d2.items():
        if v: all_conditions[k] = v

    # D3: GSE314155 H3K18la/BRD9 HCC
    d3 = process_d3_gse314155(gene_intervals, gene_list)
    for k, v in d3.items():
        if v: all_conditions[k] = v

    # D4: GSE247800 GTPSCS K192R
    d4 = process_d4_gse247800(gene_intervals, gene_list)
    for k, v in d4.items():
        if v: all_conditions[k] = v

    # D5: GSE328660 H3K18la sepsis
    d5 = process_d5_gse328660(gene_intervals, gene_list)
    for k, v in d5.items():
        if v: all_conditions[k] = v

    # D6: GSE325983 H3K18la bladder
    d6 = process_d6_gse325983(gene_intervals, gene_list)
    for k, v in d6.items():
        if v: all_conditions[k] = v

    # Save raw preprocessed scores
    serializable = {}
    for cond, scores in all_conditions.items():
        serializable[cond] = {g: float(s) for g, s in scores.items()}
    json.dump(serializable, open(OUTPUT / "all_preprocessed_scores.json", "w"), indent=1)
    print(f"\nTotal conditions after preprocessing: {len(all_conditions)}")
    for cond, scores in all_conditions.items():
        print(f"  {cond}: {len(scores)} genes")

    # --- 4. Build training data ---
    samples, cond_id_map, input_dim = build_unified_dataset(all_conditions)

    # Save training dataset summary
    json.dump({
        "n_conditions": len(cond_id_map),
        "n_samples": len(samples),
        "input_dim": input_dim,
        "condition_map": cond_id_map,
    }, open(OUTPUT / "dataset_summary.json", "w"), indent=2)

    # --- 5. Incremental training ---
    results = run_incremental_training(samples, cond_id_map, input_dim)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
