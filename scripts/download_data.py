"""
Download real datasets for Engram-GRN:
1. Human gene list (HGNC)
2. KEGG pathway data
3. Lactylation (Kla) datasets from GEO
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gzip
import csv
import io
import requests
from pathlib import Path
from typing import Dict, List, Optional


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_hgnc_genes(max_genes: int = 20000):
    """Download human gene list from HGNC."""
    url = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
    print(f"Downloading HGNC gene list from {url}...")
    r = requests.get(url, timeout=60)
    genes = []
    for line in r.text.split('\n')[1:]:
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) > 19:
            symbol = parts[1]
            status = parts[19]
            if status == 'Approved' and parts[6] == 'protein-coding gene':
                genes.append(symbol)
        if len(genes) >= max_genes:
            break
    print(f"Downloaded {len(genes)} protein-coding genes")
    with open(DATA_DIR / "hgnc_genes.txt", "w") as f:
        for g in genes:
            f.write(g + "\n")
    return genes


def download_kegg_pathways():
    """Download KEGG pathway gene memberships via REST API."""
    print("Downloading KEGG pathway data...")
    org = "hsa"
    resp = requests.get(f"http://rest.kegg.jp/list/pathway/{org}", timeout=30)
    pathways = {}
    for line in resp.text.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            path_id = parts[0].replace('path:', '')
            path_name = parts[1]
            pathways[path_id] = {"name": path_name, "genes": []}

    for pid in list(pathways.keys())[:]:
        try:
            resp = requests.get(f"http://rest.kegg.jp/get/{pid}", timeout=30)
            for line in resp.text.split('\n'):
                if line.startswith("GENE"):
                    gene_part = line[4:].strip()
                    gene_symbol = gene_part.split()[0] if gene_part else ""
                    if gene_symbol:
                        pathways[pid]["genes"].append(gene_symbol)
        except Exception as e:
            print(f"  Failed to get {pid}: {e}")

    pathways = {k: v for k, v in pathways.items() if len(v["genes"]) >= 3}
    print(f"Downloaded {len(pathways)} pathways with gene memberships")
    json.dump(pathways, open(DATA_DIR / "kegg_pathways.json", "w"), indent=1)
    return pathways


def download_kla_datasets():
    """
    Download processed lactylation (Kla) datasets from public repositories.
    Targets:
    1. Zhang et al. 2019 Nature - histone Kla in macrophages (GSE126224)
    2. Multi-omics lactylation in gastric cancer (PRIDE: PXD050906)
    3. Additional Kla datasets from GEO
    """
    datasets = {}

    # Dataset 1: GSE126224 - Zhang et al. 2019 Nature
    # This is the seminal histone Kla paper
    print("\n--- Dataset 1: GSE126224 (Zhang 2019 Nature: Macrophage Kla) ---")
    try:
        url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE126224&format=file"
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            datasets["GSE126224"] = {"status": "available", "source": "GEO"}
            print(f"  GSE126224: {len(r.content)} bytes downloaded")
    except Exception as e:
        print(f"  GSE126224 download failed: {e}")

    # Dataset 2: Search for processed Kla peak data
    try:
        print("\n--- Searching for additional Kla datasets ---")
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            "?db=gds&term=lactylation+OR+Kla+OR+histone+lactylation&retmax=20"
            "&retmode=json"
        )
        r = requests.get(search_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            print(f"  Found {len(id_list)} GEO datasets related to lactylation")
            datasets["geo_ids"] = id_list
    except Exception as e:
        print(f"  Search failed: {e}")

    # Save dataset metadata
    with open(DATA_DIR / "kla_datasets.json", "w") as f:
        json.dump(datasets, f, indent=2)
    return datasets


def build_gene_vocab_with_ensembl(max_protein_coding=20000):
    """Build gene vocabulary with real Ensembl gene IDs."""
    from engram_grn.data_pipeline.gene_vocab import GeneVocabulary

    vocab = GeneVocabulary(str(DATA_DIR))

    gtf_path = DATA_DIR / "Homo_sapiens.GRCh38.113.gtf.gz"
    if not gtf_path.exists():
        url = ("https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/"
               "Homo_sapiens.GRCh38.113.gtf.gz")
        print(f"Downloading Ensembl GTF from {url}...")
        r = requests.get(url, timeout=600, stream=True)
        with open(gtf_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {gtf_path}")

    n = vocab.build_from_ensembl_gtf(str(gtf_path), max_genes=max_protein_coding)
    print(f"Built vocabulary with {n} genes")
    return vocab


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-genes", action="store_true", help="Download gene list")
    parser.add_argument("--download-pathways", action="store_true", help="Download KEGG pathways")
    parser.add_argument("--download-kla", action="store_true", help="Download Kla datasets")
    parser.add_argument("--build-vocab", action="store_true", help="Build gene vocabulary from Ensembl")
    args = parser.parse_args()

    if args.download_genes:
        download_hgnc_genes()

    if args.download_pathways:
        download_kegg_pathways()

    if args.download_kla:
        download_kla_datasets()

    if args.build_vocab:
        build_gene_vocab_with_ensembl()

    if not any([args.download_genes, args.download_pathways, args.download_kla, args.build_vocab]):
        parser.print_help()
