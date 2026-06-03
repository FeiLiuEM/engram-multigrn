"""
Process multiple Kla ChIP-seq datasets + RNA-seq into gene-level training data.
Integrates: bedGraph, bigWig, narrowPeak, and expression data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gzip, pickle, json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch

DATA = Path("data")
KLA_DIR = DATA / "kla_chip" / "extracted"


def parse_refgene(path):
    """Parse UCSC refGene to get gene intervals."""
    genes = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 13:
                continue
            name = p[12]
            chrom = p[2]
            start, end = int(p[4]), int(p[5])
            if name not in genes:
                genes[name] = {'chrom': chrom, 'min_s': start, 'max_e': end}
            else:
                genes[name]['min_s'] = min(genes[name]['min_s'], start)
                genes[name]['max_e'] = max(genes[name]['max_e'], end)
    return genes


def compute_bedgraph_gene_scores(bg_path, gene_intervals, gene_list):
    """Compute mean ChIP-seq signal per gene from bedGraph."""
    chrom_bg = defaultdict(list)
    with gzip.open(bg_path, 'rt') if bg_path.endswith('.gz') else open(bg_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 4 and p[0].startswith('chr'):
                chrom_bg[p[0]].append({'s': int(p[1]), 'e': int(p[2]), 'v': float(p[3])})
    
    scores = {}
    for gene in gene_list:
        info = gene_intervals.get(gene)
        if not info:
            continue
        chrom = info['chrom']
        if chrom not in chrom_bg:
            continue
        gs, ge = info['min_s'], info['max_e']
        vals = [bg['v'] for bg in chrom_bg[chrom] if bg['s'] < ge and bg['e'] > gs]
        if vals:
            scores[gene] = np.mean(vals)
    return scores


def compute_bigwig_gene_scores(bw_path, gene_intervals, gene_list):
    """Compute mean ChIP-seq signal per gene from bigWig."""
    import pyBigWig
    bw = pyBigWig.open(str(bw_path))
    scores = {}
    for gene in gene_list:
        info = gene_intervals.get(gene)
        if not info or info['chrom'] not in bw.chroms():
            continue
        try:
            vals = bw.values(info['chrom'], max(0, info['min_s']-2000), info['max_e']+2000)
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if vals:
                scores[gene] = np.mean(vals)
        except:
            pass
    bw.close()
    return scores


def parse_narrowpeak_gene_scores(peak_path, gene_intervals, gene_list, score_col=6):
    """Score genes by proximity to ChIP-seq peaks from narrowPeak."""
    import pyBigWig
    peaks = []
    f = gzip.open(peak_path, 'rt') if peak_path.endswith('.gz') else open(peak_path)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 7:
            peaks.append({'chr': p[0], 's': int(p[1]), 'e': int(p[2]),
                          'score': float(p[6])})
    
    gene_peak_scores = {}
    for gene in gene_list:
        info = gene_intervals.get(gene)
        if not info:
            continue
        total = 0
        for pk in peaks:
            if pk['chr'] == info['chrom'] and pk['s'] < info['max_e'] and pk['e'] > info['min_s']:
                overlap = min(pk['e'], info['max_e']) - max(pk['s'], info['min_s'])
                if overlap > 0:
                    total += pk['score'] * (overlap / (info['max_e'] - info['min_s']))
        if total > 0:
            gene_peak_scores[gene] = total
    return gene_peak_scores


def process_all_datasets():
    """Process all Kla ChIP-seq datasets and create unified training data."""
    print("=" * 60)
    print("STEP 1: Parse gene annotations")
    genes = parse_refgene(DATA / "refGene.txt.gz")
    print(f"  {len(genes)} gene transcripts in refGene")

    vocab = torch.load(DATA / "gene_vocabulary.pt", weights_only=False)
    hgnc_set = set(vocab['gene_list'])

    # Filter to HGNC genes present in our vocab
    gene_intervals = {}
    for g, v in genes.items():
        if g in hgnc_set:
            gene_intervals[g] = v
    print(f"  {len(gene_intervals)} HGNC genes with coordinates")

    gene_list = sorted(gene_intervals.keys())
    all_kla_scores = {}  # (gene, condition) -> score

    print("\n" + "=" * 60)
    print("STEP 2: Process GSE328660 (H3K18la sepsis, bedGraph)")
    bg_dir = KLA_DIR / "GSE328660"
    if bg_dir.exists():
        for label, fname in [('control', 'GSM9686847_ChIP-Con.bedgraph.gz'),
                              ('lactate', 'GSM9686848_ChIP-LAC.bedgraph.gz')]:
            scores = compute_bedgraph_gene_scores(str(bg_dir / fname), gene_intervals, gene_list)
            for g, s in scores.items():
                all_kla_scores[(g, f'H3K18la_{label}')] = s
            print(f"  {label}: {len(scores)} genes scored")

    print("\n" + "=" * 60)
    print("STEP 3: Process GSE325983 (H3K18la bladder cancer, bigWig)")
    bw_dir = KLA_DIR / "GSE325983"
    if bw_dir.exists():
        scores_ip = compute_bigwig_gene_scores(str(bw_dir / 'GSM9618377_T24_H3K18la_IP.bw'),
                                                gene_intervals, gene_list)
        scores_in = compute_bigwig_gene_scores(str(bw_dir / 'GSM9618378_T24_H3K18la_Input.bw'),
                                                gene_intervals, gene_list)
        # Enrichment = IP / Input
        common = set(scores_ip.keys()) & set(scores_in.keys())
        for g in common:
            if scores_in[g] > 0:
                all_kla_scores[(g, 'bladder_H3K18la')] = scores_ip[g] / max(scores_in[g], 0.01)
        print(f"  bladder_H3K18la: {len(common)} genes scored")

    print("\n" + "=" * 60)
    print("STEP 4: Process GSE314769 (H4K5la HCC, narrowPeak)")
    np_dir = KLA_DIR / "GSE314769"
    if np_dir.exists():
        for label, fname in [('NM2', 'GSM9410262_NM2_peaks_RIAS.narrowPeak.gz'),
                              ('NM3', 'GSM9410263_NM3_peaks_RIAS.narrowPeak.gz'),
                              ('LAC2', 'GSM9410264_LAC2_peaks_RIAS.narrowPeak.gz'),
                              ('LAC3', 'GSM9410265_LAC3_peaks_RIAS.narrowPeak.gz')]:
            scores = parse_narrowpeak_gene_scores(str(np_dir / fname), gene_intervals, gene_list)
            for g, s in scores.items():
                all_kla_scores[(g, f'H4K5la_{label}')] = s
            print(f"  H4K5la_{label}: {len(scores)} genes scored")

    print("\n" + "=" * 60)
    print("STEP 5: Save unified Kla scores")
    # Convert to serializable format
    serializable = {f"{g}__{c}": float(s) for (g, c), s in all_kla_scores.items()}
    json.dump(serializable, open(DATA / "kla_chip_scores.json", "w"), indent=1)
    print(f"  Total (gene, condition) pairs: {len(serializable)}")
    print(f"  Unique genes: {len(set(g for g, c in all_kla_scores.keys()))}")
    print(f"  Unique conditions: {len(set(c for g, c in all_kla_scores.keys()))}")
    
    # Summary per condition
    from collections import Counter
    conditions = Counter(c for g, c in all_kla_scores.keys())
    for cond, count in conditions.most_common():
        vals = [all_kla_scores[(g, c)] for (g, c) in all_kla_scores.keys() if c == cond]
        print(f"    {cond}: {count} genes, mean={np.mean(vals):.3f}, max={np.max(vals):.3f}")

    return all_kla_scores


if __name__ == "__main__":
    scores = process_all_datasets()
    print("\nDone! Ready for model training.")
