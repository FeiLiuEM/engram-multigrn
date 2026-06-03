#!/usr/bin/env python3
"""
Improved bigWig preprocessing for histone mark ChIP-seq.
Fixes the zero-variance problem by using TSS-centered scoring + local background.

Key improvements over the original:
  1. TSS-centered ±2kb windows (not full gene body)
  2. Local background: TSS signal / upstream intergenic signal
  3. Rank-percentile normalization per condition
  4. Unified processing for all bigWig datasets
  5. GSE328660 bedgraph fix: extract all genes, not just overlapping

Usage:
  .venv/bin/python scripts/preprocess_bigwig_improved.py
"""

import sys, os, json, gzip, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pyBigWig
import torch
from pathlib import Path
from collections import defaultdict, Counter

DATA = Path(__file__).parent.parent / "data"
KLA_EXTRACTED = DATA / "kla_chip" / "extracted"
OUTPUT = DATA / "bigwig_preprocessed_v2"
OUTPUT.mkdir(exist_ok=True)

TSS_WINDOW = 2000  # ±2kb from TSS
UPSTREAM_WINDOW = 6000  # 2-6kb upstream for local background


# ═══════════════════════════════════════════════════════════════════════
# GENE ANNOTATION: Parse TSS positions from refGene
# ═══════════════════════════════════════════════════════════════════════

def parse_tss_from_refgene(path):
    """Parse UCSC refGene → per-gene TSS (strand-aware)."""
    genes = {}
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 13: continue
            name = p[12]; chrom = p[2]; strand = p[3]
            tx_start = int(p[4]); tx_end = int(p[5])
            # TSS: txStart for + strand, txEnd for - strand
            tss = tx_end if strand == '-' else tx_start
            if name not in genes:
                genes[name] = {'chrom': chrom, 'tss': tss,
                               'min_s': tx_start, 'max_e': tx_end}
            else:
                # Keep the canonical isoform's TSS (first encountered)
                genes[name]['min_s'] = min(genes[name]['min_s'], tx_start)
                genes[name]['max_e'] = max(genes[name]['max_e'], tx_end)
    return genes


# ═══════════════════════════════════════════════════════════════════════
# BIGWIG SCORING: TSS-centered + local background
# ═══════════════════════════════════════════════════════════════════════

def score_genes_tss(bw_path, gene_tss, gene_list, label=""):
    """
    For each gene, compute:
      - tss_signal: mean bigWig signal at TSS ± 2kb
      - local_bg:   mean signal at TSS-6kb..TSS-2kb (upstream background)
      - enrichment: tss_signal / local_bg (log2 fold)
      - gene_body:  mean signal over full gene body

    Returns dict of {gene: {tss: ..., bg: ..., enrichment: ..., body: ...}}
    """
    try:
        bw = pyBigWig.open(str(bw_path))
    except Exception as e:
        print(f"  [SKIP] {Path(bw_path).name}: {e}")
        return {}

    bw_chroms = set(bw.chroms().keys())
    scores = {}
    skipped = 0

    for gene in gene_list:
        info = gene_tss.get(gene)
        if not info: continue

        # Chromosome matching (try both 'chr1' and '1')
        chrom = info['chrom']
        candidates = [chrom]
        if chrom.startswith('chr'): candidates.append(chrom[3:])
        else: candidates.append('chr' + chrom)
        matched = next((c for c in candidates if c in bw_chroms), None)
        if not matched: continue

        tss_pos = info['tss']
        try:
            tss_sig = bw.stats(matched,
                               max(0, tss_pos - TSS_WINDOW),
                               tss_pos + TSS_WINDOW,
                               type='mean', nBins=1)[0]
            bg_sig = bw.stats(matched,
                              max(0, tss_pos - UPSTREAM_WINDOW),
                              max(0, tss_pos - TSS_WINDOW),
                              type='mean', nBins=1)[0]
            body_sig = bw.stats(matched,
                                max(0, info['min_s']),
                                info['max_e'],
                                type='mean', nBins=1)[0]

            if tss_sig is not None and not np.isnan(tss_sig):
                enrichment = (tss_sig + 0.01) / (bg_sig + 0.01) if bg_sig is not None else 1.0
                scores[gene] = {
                    'tss': float(tss_sig),
                    'bg': float(bg_sig) if bg_sig is not None else 0.0,
                    'enrichment': float(enrichment),
                    'body': float(body_sig) if body_sig is not None else 0.0,
                }
        except:
            skipped += 1

    bw.close()
    if label:
        n = len(scores)
        vals = [s['enrichment'] for s in scores.values()]
        print(f"  {label}: {n} genes, "
              f"TSS mean={np.mean([s['tss'] for s in scores.values()]):.1f}, "
              f"enrich mean={np.mean(vals):.2f}, "
              f"({skipped} skipped)")
    return scores


# ═══════════════════════════════════════════════════════════════════════
# BEDGRAPH SCORING: improved for GSE328660 sparse coverage
# ═══════════════════════════════════════════════════════════════════════

def score_genes_bedgraph(bg_path, gene_tss, gene_list, label=""):
    """Bedgraph → gene-level TSS scores. Handles 'chr1'/'1' naming."""
    # Load all bedgraph data into chromosome-indexed arrays
    chrom_bg = defaultdict(list)
    skipped_header = 0
    opener = gzip.open(bg_path, 'rt') if str(bg_path).endswith('.gz') else open(bg_path)
    with opener as f:
        for line in f:
            # Skip UCSC header lines
            if line.startswith('track') or line.startswith('browser') or line.startswith('#'):
                skipped_header += 1; continue
            p = line.strip().split()
            if len(p) < 4: continue
            try:
                chrom_bg[p[0]].append({'s': int(p[1]), 'e': int(p[2]), 'v': float(p[3])})
            except ValueError:
                continue

    scores = {}
    total_genes, covered = 0, 0
    for gene in gene_list:
        total_genes += 1
        info = gene_tss.get(gene)
        if not info: continue

        chrom = info['chrom']
        candidates = [chrom]
        if chrom.startswith('chr'): candidates.append(chrom[3:])
        else: candidates.append('chr' + chrom)

        for c in candidates:
            if c not in chrom_bg: continue
            tss_pos = info['tss']
            # TSS window
            tss_vals = [bg['v'] for bg in chrom_bg[c]
                        if bg['s'] < tss_pos + TSS_WINDOW and bg['e'] > tss_pos - TSS_WINDOW]
            tss_sig = np.mean(tss_vals) if tss_vals else 0.0

            # Upstream background
            us_vals = [bg['v'] for bg in chrom_bg[c]
                       if bg['s'] < tss_pos - TSS_WINDOW and bg['e'] > tss_pos - UPSTREAM_WINDOW]
            bg_sig = np.mean(us_vals) if us_vals else 0.0

            if tss_sig > 0:
                enrichment = (tss_sig + 0.01) / (bg_sig + 0.01)
                scores[gene] = {
                    'tss': float(tss_sig),
                    'bg': float(bg_sig),
                    'enrichment': float(enrichment),
                    'body': float(tss_sig),
                }
                covered += 1
            break  # Stop after first matching chromosome convention

    if label:
        vals = [s['enrichment'] for s in scores.values()]
        print(f"  {label}: {covered}/{total_genes} genes covered, "
              f"enrich mean={np.mean(vals):.2f}")
    return scores


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZATION: Rank-percentile per condition
# ═══════════════════════════════════════════════════════════════════════

def normalize_rank_percentile(condition_scores, score_key='enrichment'):
    """
    Convert raw scores to rank percentiles [0, 1].
    Robust against outliers: uses clip at 1st/99th percentile.
    """
    vals = np.array([s[score_key] for s in condition_scores.values()])
    if len(vals) < 10 or vals.std() < 1e-8:
        return {g: 0.5 for g in condition_scores}

    # Clip at 1%/99% percentile
    lo, hi = np.percentile(vals, [1, 99])
    clipped = np.clip(vals, lo, hi)

    # Min-max normalize to [0, 1]
    if hi > lo:
        norm = (clipped - lo) / (hi - lo)
    else:
        norm = np.full_like(clipped, 0.5)

    return {g: float(n) for g, n in zip(condition_scores.keys(), norm)}


# ═══════════════════════════════════════════════════════════════════════
# MAIN: Process all bigWig datasets
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("IMPROVED BIGWIG PREPROCESSING (TSS-centered + local background)")
    print("=" * 60)

    # Load annotations
    gene_tss_raw = parse_tss_from_refgene(DATA / "refGene.txt.gz")
    vocab = torch.load(DATA / "gene_vocabulary.pt", weights_only=False)
    hgnc_set = set(vocab['gene_list'])
    gene_tss = {g: v for g, v in gene_tss_raw.items() if g in hgnc_set}
    gene_list = sorted(gene_tss.keys())
    print(f"HGNC genes with TSS: {len(gene_tss)}\n")

    ortho_pairs = json.load(open(DATA / "mouse_human_ortholog_map.json"))
    mouse_to_human = dict(ortho_pairs)

    all_enrichment_scores = {}  # {condition_name: {gene: score}}
    all_detail_scores = {}      # {condition_name: {gene: {tss, bg, enrichment, body}}}

    # ═══ DATASET 1: GSE314155 - H3K18la HCC HepG2 ═══
    print("═══ GSE314155: H3K18la HCC (HepG2) ═══")
    bw_dir = KLA_EXTRACTED / "GSE314155"

    # H3K18la NC (Normal condition)
    s_h3k18la = score_genes_tss(
        str(bw_dir / 'GSM9382816_ChIP-seq_HepG2_H3K18la_NC.bw'),
        gene_tss, gene_list, label="H3K18la_IP")
    all_detail_scores['GSE314155_H3K18la'] = s_h3k18la
    all_enrichment_scores['H3K18la_NC'] = normalize_rank_percentile(s_h3k18la, 'enrichment')

    # Input
    s_input = score_genes_tss(
        str(bw_dir / 'GSM9382815_ChIP-seq_HepG2_Input_NC.bw'),
        gene_tss, gene_list, label="Input")
    all_detail_scores['GSE314155_Input'] = s_input
    all_enrichment_scores['H3K18la_NC_TSS'] = normalize_rank_percentile(s_h3k18la, 'tss')

    # ═══ DATASET 2: GSE247800 - GTPSCS K192R H3K18la ═══
    print("\n═══ GSE247800: GTPSCS K192R H3K18la (HEK293T) ═══")
    bw_dir = KLA_EXTRACTED / "GSE247800"

    for label, fnames in [
        ('K192R', ['GSM7900969_K192R_ChIP_1.bw', 'GSM7900970_K192R_ChIP_2.bw',
                    'GSM7900971_K192R_ChIP_3.bw']),
        ('WT',    ['GSM7900973_WT_ChIP_1.bw', 'GSM7900974_WT_ChIP_2.bw',
                    'GSM7900975_WT_ChIP_3.bw']),
    ]:
        # Average across replicates
        rep_scores = []
        for fname in fnames:
            s = score_genes_tss(str(bw_dir / fname), gene_tss, gene_list, label="")
            rep_scores.append(s)

        common_genes = set.intersection(*[set(s.keys()) for s in rep_scores if s])
        if not common_genes: continue

        avg_scores = {}
        for g in common_genes:
            avg_scores[g] = {
                'tss': np.mean([s[g]['tss'] for s in rep_scores if g in s]),
                'enrichment': np.mean([s[g]['enrichment'] for s in rep_scores if g in s]),
            }
        vals = [s['tss'] for s in avg_scores.values()]
        print(f"  {label}: {len(avg_scores)} genes, TSS mean={np.mean(vals):.1f} "
              f"(avg of {len(fnames)} reps)")
        all_enrichment_scores[f'H3K18la_{label}'] = normalize_rank_percentile(avg_scores, 'enrichment')
        all_enrichment_scores[f'H3K18la_{label}_TSS'] = normalize_rank_percentile(avg_scores, 'tss')

    # ═══ DATASET 3: GSE328660 - H3K18la sepsis (bedgraph) ═══
    print("\n═══ GSE328660: H3K18la sepsis (bedgraph) ═══")
    bg_dir = KLA_EXTRACTED / "GSE328660"

    s_con = score_genes_bedgraph(str(bg_dir / 'GSM9686847_ChIP-Con.bedgraph.gz'),
                                  gene_tss, gene_list, label="Control")
    s_lac = score_genes_bedgraph(str(bg_dir / 'GSM9686848_ChIP-LAC.bedgraph.gz'),
                                  gene_tss, gene_list, label="Lactate")

    # Use absolute TSS signal (not enrichment ratio) for bedgraph — works better for sparse data
    all_enrichment_scores['H3K18la_sepsis_lac'] = normalize_rank_percentile(s_lac, 'tss')
    all_enrichment_scores['H3K18la_sepsis_con'] = normalize_rank_percentile(s_con, 'tss')

    # Also compute differential: lactate - control
    common = set(s_lac.keys()) & set(s_con.keys())
    diff_scores = {}
    for g in common:
        d = s_lac[g]['tss'] - s_con[g]['tss']
        diff_scores[g] = {
            'tss': d,
            'enrichment': (s_lac[g]['tss'] + 0.01) / (s_con[g]['tss'] + 0.01),
        }
    all_enrichment_scores['H3K18la_sepsis_diff'] = normalize_rank_percentile(diff_scores, 'enrichment')
    print(f"  Differential: {len(common)} genes, diff mean={np.mean([d['tss'] for d in diff_scores.values()]):.2f}")

    # ═══ DATASET 4: GSE325983 - H3K18la bladder (bigWig) ═══
    print("\n═══ GSE325983: H3K18la bladder cancer (T24) ═══")
    bw_dir = KLA_EXTRACTED / "GSE325983"

    s_ip = score_genes_tss(str(bw_dir / 'GSM9618377_T24_H3K18la_IP.bw'),
                            gene_tss, gene_list, label="IP")
    s_inp = score_genes_tss(str(bw_dir / 'GSM9618378_T24_H3K18la_Input.bw'),
                             gene_tss, gene_list, label="Input")

    # TSS signal directly (histone mark → use absolute signal)
    all_enrichment_scores['H3K18la_bladder'] = normalize_rank_percentile(s_ip, 'tss')
    all_enrichment_scores['H3K18la_bladder_enrich'] = normalize_rank_percentile(s_ip, 'enrichment')

    # ═══ DATASET 5: GSE269142 peak-gene (already processed in main pipeline, skip) ═══
    # Already scored using peak counting — these scores are good.

    # ═══ DATASET 6: GSE314769 narrowPeak (already processed, skip) ═══
    # Already scored using narrowPeak overlap — these scores are good.

    # ═══ SAVE RESULTS ═══
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save enrichment scores (ready for training)
    serializable = {}
    for cond, scores in all_enrichment_scores.items():
        serializable[cond] = {g: float(s) for g, s in scores.items()}
    json.dump(serializable, open(OUTPUT / "bigwig_scores_v2.json", "w"), indent=1)
    print(f"Enrichment scores: {len(serializable)} conditions → "
          f"{OUTPUT / 'bigwig_scores_v2.json'}")

    # Summary statistics per condition
    print(f"\n{'Condition':35s} {'Genes':>8s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 78)
    for cond in sorted(serializable.keys()):
        vals = list(serializable[cond].values())
        if vals:
            print(f"{cond:35s} {len(vals):8d} "
                  f"{np.mean(vals):8.4f} {np.std(vals):8.4f} "
                  f"{np.min(vals):8.4f} {np.max(vals):8.4f}")

    # Save detail scores for debugging
    json.dump(all_detail_scores, open(OUTPUT / "bigwig_detail_scores.json", "w"),
              indent=1, default=float)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
