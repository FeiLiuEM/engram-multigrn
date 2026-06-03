"""
Build gene vocabulary from Ensembl GTF and create mock vocabulary
for testing when GTF is not available.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Fallback: 500 key human genes for testing
FALLBACK_GENES = [
    "TP53", "MYC", "EGFR", "KRAS", "PTEN", "AKT1", "MTOR", "MAPK1", "MAPK3",
    "STAT3", "STAT1", "NFKB1", "RELA", "JUN", "FOS", "HIF1A", "VEGFA", "TNF",
    "IL6", "IL1B", "TGFB1", "CDKN1A", "CDKN2A", "RB1", "BRCA1", "BRCA2",
    "DNMT1", "DNMT3A", "TET1", "TET2", "HDAC1", "HDAC2", "EP300", "CREBBP",
    "CTCF", "RAD21", "SMC1A", "SMC3", "ARID1A", "ARID1B", "SMARCA4", "SMARCB1",
    "EZH2", "SUZ12", "KMT2A", "KMT2D", "KDM1A", "KDM6A", "KDM6B",
    "H2AC1", "H2BC1", "H3C1", "H4C1", "H3F3A", "H2AFY", "H2AFZ",
    "LDHA", "LDHB", "PKM", "PKLR", "HK1", "HK2", "PFKL", "PFKP",
    "SLC16A1", "SLC16A3", "SLC16A7", "MCT1", "MCT4",
    "GAPDH", "ACTB", "B2M", "GUSB", "HPRT1", "TBP", "POLR2A",
    "ALDOA", "ENO1", "PDK1", "PDHA1", "ACACA", "FASN", "SREBF1",
    "NRF1", "PPARGC1A", "ESRRA", "PPARA", "PPARG",
    "HNF4A", "FOXA1", "FOXA2", "CEBPA", "CEBPB", "SP1", "MYOD1",
    "NANOG", "POU5F1", "SOX2", "KLF4", "MYCN", "CCND1", "CDK4",
    "CDK6", "CCNE1", "CDK2", "MDM2", "MDM4", "BAX", "BAK1", "BCL2",
    "BCL2L1", "MCL1", "BID", "BIM", "NOXA", "PUMA",
    "VHL", "WT1", "APC", "AXIN1", "CTNNB1", "GSK3B",
    "DVL1", "DVL2", "DVL3", "LRP5", "LRP6", "FZD1",
    "SMO", "GLI1", "GLI2", "GLI3", "PTCH1", "SHH", "IHH",
    "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1", "DLL4",
    "WNT1", "WNT2", "WNT3", "WNT3A", "WNT5A", "WNT7A", "WNT10B",
    "SMAD2", "SMAD3", "SMAD4", "SMAD7", "ACVR1", "ACVR1B", "BMPR1A", "BMPR2",
    "TGFBR1", "TGFBR2", "TGFBR3", "ENG",
    "FLT1", "KDR", "PDGFRA", "PDGFRB", "FGFR1", "FGFR2", "FGFR3",
    "IGF1R", "INSR", "MET", "ERBB2", "ERBB3", "ERBB4",
    "SRC", "ABL1", "LCK", "FYN", "LYN", "SYK", "ZAP70",
    "PIK3CA", "PIK3CB", "PIK3R1", "PTEN", "AKT2", "AKT3",
    "RHEB", "TSC1", "TSC2", "RPTOR", "RICTOR", "DEPTOR",
    "RAF1", "BRAF", "ARAF", "HRAS", "NRAS", "RASA1",
    "SOS1", "GRB2", "SHC1", "IRS1", "IRS2",
    "MKNK1", "MKNK2", "RPS6KA1", "RPS6KA3", "RPS6KB1",
    "EIF4E", "EIF4G1", "EIF4A1", "EIF4EBP1",
    "JAK1", "JAK2", "JAK3", "TYK2",
    "IRF1", "IRF3", "IRF4", "IRF5", "IRF7", "IRF8", "IRF9",
    "MYD88", "TIRAP", "TRIF", "TLR1", "TLR2", "TLR3", "TLR4",
    "TLR5", "TLR6", "TLR7", "TLR8", "TLR9",
    "NLRP3", "PYCARD", "CASP1", "IL18", "IL1A", "IL1R1",
    "CCL2", "CCL5", "CXCL8", "CXCL10", "ICAM1", "VCAM1", "SELE",
    "IFNG", "IL2", "IL4", "IL10", "IL12A", "IL12B", "IL17A",
    "CD4", "CD8A", "CD8B", "CD3E", "CD19", "CD14", "CD68",
    "FOXP3", "GATA3", "TBX21", "RORC", "BCL6",
    "MEF2C", "PAX5", "EBF1", "TCF4", "TCF3", "TCF12",
    "RUNX1", "RUNX2", "RUNX3", "CBFB", "ETV6", "FLI1", "ERG",
    "GATA1", "GATA2", "TAL1", "LMO2", "LDB1", "KIT",
    "CEBPA", "CEBPE", "PU.1", "IRF8", "CSF1R", "CSF2RA",
    "CDK1", "CDK2", "CDK4", "CDK6", "CDK7", "CDK8", "CDK9",
    "CDKN1B", "CDKN2B", "CDKN2C", "CDKN2D",
    "CHEK1", "CHEK2", "ATM", "ATR", "RAD51", "BRCA1", "BRCA2",
    "FANCA", "FANCC", "FANCD2", "FANCG",
    "MLH1", "MSH2", "MSH6", "PMS2", "MUTYH",
    "TOP1", "TOP2A", "TOP2B",
    "TERT", "TERC", "POT1", "TIN2", "TPP1",
    "RAC1", "RHOA", "CDC42", "ROCK1", "ROCK2",
    "LIMK1", "LIMK2", "CFL1", "WASF1", "WASF2",
    "ACTN1", "ACTN4", "TLN1", "VCL", "PXN",
    "ITGA1", "ITGA2", "ITGB1", "ITGB3", "ITGAV",
    "MMP1", "MMP2", "MMP3", "MMP7", "MMP9", "MMP13", "MMP14",
    "TIMP1", "TIMP2", "TIMP3",
    "FN1", "LAMA1", "LAMA2", "LAMB1", "LAMC1", "COL1A1", "COL1A2",
    "CDH1", "CDH2", "CTNNA1", "CTNNA2",
    "SNAI1", "SNAI2", "TWIST1", "TWIST2", "ZEB1", "ZEB2",
    "VIM", "DES", "KRT14", "KRT5", "KRT8", "KRT18",
    "PROM1", "CD44", "ALDH1A1", "ABCG2",
    "SOX9", "SOX10", "SNAI1", "SNAI2",
    "CXCR4", "CCR5", "CCR7", "CXCL12",
    "IL7R", "IL2RA", "IL6ST", "IL10RA",
    "ARG1", "ARG2", "NOS2", "NOS3",
    "CA9", "SLC2A1", "SLC2A3", "SLC16A1", "SLC16A4",
]

# Known Kla-related genes
KLA_RELATED = [
    "LDHA", "LDHB", "PKM", "SLC16A1", "SLC16A3", "HIF1A",
    "H2AC1", "H2BC1", "H3C1", "H4C1", "H3F3A",
    "EP300", "CREBBP", "HDAC1", "HDAC2", "HDAC3",
    "SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5", "SIRT6", "SIRT7",
    "GCN5", "PCAF", "KAT2A", "KAT2B", "KAT5",
    "MYC", "TP53", "STAT3", "NFKB1", "RELA",
    "IL6", "TNF", "TGFB1", "VEGFA",
    "ARG1", "MRC1", "CD163", "IL10", "CCL22",
    "NOS2", "IL1B", "IL12A", "CD80", "CD86",
    "PDK1", "PDK4", "PKLR", "PCK1", "PCK2",
    "GLS", "GLS2", "GLUD1", "GLUL",
    "HAT1", "KAT6A", "KAT6B", "ATAT1",
    "HDAC8", "HDAC9", "HDAC10", "HDAC11",
    "TET1", "TET2", "TET3",
]


class GeneVocabulary:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.gene_to_idx: Dict[str, int] = {}
        self.idx_to_gene: Dict[int, str] = {}
        self.gene_list: List[str] = []

    def build_from_fallback(self, include_kla: bool = True) -> int:
        genes = set(FALLBACK_GENES)
        if include_kla:
            genes.update(KLA_RELATED)
        genes = sorted(genes)
        self.gene_list = genes
        self.gene_to_idx = {g: i for i, g in enumerate(genes)}
        self.idx_to_gene = {i: g for g, i in self.gene_to_idx.items()}
        self._save()
        return len(self.gene_list)

    def build_from_ensembl_gtf(self, gtf_path: str, max_genes: int = 20000) -> int:
        import gzip
        print(f"Loading GTF from {gtf_path}...")
        genes = set()
        open_fn = gzip.open if gtf_path.endswith('.gz') else open
        with open_fn(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                if parts[2] != 'gene':
                    continue
                attrs = parts[8]
                gene_name = None
                gene_type = None
                for attr in attrs.split(';'):
                    attr = attr.strip()
                    if attr.startswith('gene_name'):
                        gene_name = attr.split('"')[1] if '"' in attr else attr.split()[1]
                    elif attr.startswith('gene_biotype'):
                        gene_type = attr.split('"')[1] if '"' in attr else attr.split()[1]
                if gene_name and 'protein_coding' in (gene_type or ''):
                    genes.add(gene_name)
                if len(genes) >= max_genes:
                    break
        genes = sorted(genes)
        self.gene_list = genes
        self.gene_to_idx = {g: i for i, g in enumerate(genes)}
        self.idx_to_gene = {i: g for g, i in self.gene_to_idx.items()}
        self._save()
        print(f"Built vocabulary with {len(genes)} protein-coding genes")
        return len(genes)

    def _save(self):
        torch.save({
            'gene_to_idx': self.gene_to_idx,
            'idx_to_gene': self.idx_to_gene,
            'gene_list': self.gene_list,
        }, self.data_dir / 'gene_vocabulary.pt')

    def load(self) -> bool:
        path = self.data_dir / 'gene_vocabulary.pt'
        if not path.exists():
            return False
        data = torch.load(path, weights_only=False)
        self.gene_to_idx = data['gene_to_idx']
        self.idx_to_gene = data['idx_to_gene']
        self.gene_list = data['gene_list']
        return True

    @property
    def vocab_size(self) -> int:
        return len(self.gene_list)

    def encode(self, genes: List[str]) -> torch.Tensor:
        indices = [self.gene_to_idx.get(g, 0) for g in genes]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> List[str]:
        return [self.idx_to_gene.get(int(i), 'UNKNOWN') for i in indices]
