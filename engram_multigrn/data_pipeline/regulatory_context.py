"""
Build regulatory context (analogous to n-grams) from pathway data
and co-expression patterns.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .gene_vocab import GeneVocabulary


# Pre-defined KEGG-like pathway memberships for key genes
PATHWAYS = {
    "p53_signaling": ["TP53", "MDM2", "MDM4", "CDKN1A", "BAX", "BBC3", "PMAIP1",
                       "PTEN", "ATM", "ATR", "CHEK1", "CHEK2", "SESN1", "SESN2"],
    "pi3k_akt": ["PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "AKT3", "PTEN",
                 "MTOR", "RHEB", "TSC1", "TSC2", "RPTOR", "RICTOR", "EIF4E",
                 "RPS6KB1", "GSK3B", "FOXO1", "FOXO3", "FOXO4", "BAD"],
    "mapk": ["KRAS", "HRAS", "NRAS", "RAF1", "BRAF", "ARAF", "MAP2K1", "MAP2K2",
             "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10", "MAPK14", "MAPK11",
             "JUN", "FOS", "ELK1", "MYC", "RPS6KA1", "RPS6KA3"],
    "wnt": ["CTNNB1", "AXIN1", "AXIN2", "APC", "GSK3B", "DVL1", "DVL2", "DVL3",
            "LRP5", "LRP6", "FZD1", "FZD2", "FZD3", "FZD4", "FZD5", "FZD6",
            "FZD7", "TCF7", "TCF7L1", "TCF7L2", "LEF1", "MYC", "CCND1"],
    "hedgehog": ["SHH", "IHH", "DHH", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2",
                 "GLI3", "SUFU", "KIF7"],
    "notch": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1",
              "DLL4", "DLL3", "MAML1", "MAML2", "MAML3", "CSL", "HES1", "HES5",
              "HEY1", "HEY2"],
    "TNF_NFKB": ["TNF", "TNFR1", "TRADD", "TRAF2", "RIPK1", "IKBKB", "IKBKG",
                 "NFKB1", "RELA", "NFKBIA", "NFKBIB", "NFKBIE"],
    "jak_stat": ["JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3",
                 "STAT4", "STAT5A", "STAT5B", "STAT6", "IFNG", "IL6", "IL10",
                 "IL2", "IL4", "IL12A", "IFNAR1", "IFNAR2", "IL6ST", "IL2RG"],
    "tlr": ["TLR1", "TLR2", "TLR3", "TLR4", "TLR5", "TLR6", "TLR7", "TLR8",
            "TLR9", "MYD88", "TIRAP", "TRIF", "IRAK1", "IRAK4", "TRAF6",
            "IRF3", "IRF7", "NFKB1"],
    "inflammasome": ["NLRP3", "PYCARD", "CASP1", "CASP4", "CASP5", "IL18",
                     "IL1B", "IL1A", "GSDMD", "PANX1"],
    "glycolysis": ["LDHA", "LDHB", "PKM", "PKLR", "HK1", "HK2", "PFKL", "PFKP",
                   "PFKM", "ALDOA", "ALDOB", "ALDOC", "GAPDH", "ENO1", "ENO2",
                   "PGK1", "PGAM1", "PGAM2", "TPI1", "SLC2A1", "SLC2A3", "SLC2A4"],
    "lactate_metabolism": ["LDHA", "LDHB", "SLC16A1", "SLC16A3", "SLC16A4",
                           "SLC16A7", "SLC16A8", "PKM", "PDK1", "PDK2", "PDK3",
                           "PDK4", "HIF1A", "HIF1B", "VEGFA"],
    "histone_modification": ["EP300", "CREBBP", "HDAC1", "HDAC2", "HDAC3", "HDAC8",
                              "SIRT1", "SIRT2", "SIRT3", "SIRT6", "KAT2A", "KAT2B",
                              "KAT5", "KAT6A", "KAT6B", "KAT7", "KAT8",
                              "EZH2", "SUZ12", "EED", "DOT1L", "KMT2A", "KMT2D",
                              "KDM1A", "KDM6A", "KDM6B", "SETD1A", "SETD1B"],
    "lactylation": ["H2AC1", "H2BC1", "H3C1", "H4C1", "H3F3A", "H2AFY", "H2AFZ",
                    "LDHA", "SIRT1", "SIRT2", "SIRT3", "SIRT6", "EP300",
                    "HIF1A", "NFKB1", "RELA", "MYC", "TP53", "STAT3"],
    "macrophage_m1": ["NOS2", "IL1B", "TNF", "IL6", "CXCL10", "CXCL9", "CCL5",
                      "IRF5", "IRF1", "STAT1", "NFKB1", "CD80", "CD86", "IL12A",
                      "IL23A", "CCL2", "CCL3", "CCL4"],
    "macrophage_m2": ["ARG1", "MRC1", "CD163", "IL10", "CCL22", "CCL17", "CCL18",
                      "CCL24", "TGFB1", "VEGFA", "IRF4", "STAT3", "STAT6",
                      "PPARG", "KLF4", "MYC"],
    "hypoxia": ["HIF1A", "HIF1B", "HIF3A", "VEGFA", "VEGFB", "VEGFC", "EPO",
                "SLC2A1", "SLC2A3", "LDHA", "PKM", "PDK1", "CA9", "CA12",
                "NOS2", "EDN1", "ADM", "BNIP3", "BNIP3L"],
    "autophagy": ["ATG1", "ATG3", "ATG5", "ATG7", "ATG12", "ATG16L1", "BECN1",
                  "PIK3C3", "SQSTM1", "MAP1LC3A", "MAP1LC3B", "LAMP1", "LAMP2",
                  "ULK1", "ULK2", "RB1CC1"],
    "cell_cycle": ["CCND1", "CCND2", "CCND3", "CCNE1", "CCNE2", "CDK4", "CDK6",
                   "CDK2", "CDK1", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
                   "RB1", "E2F1", "E2F2", "E2F3", "MCM2", "MCM3", "MCM4"],
    "apoptosis": ["BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BOK", "BID", "BIM",
                  "BAD", "NOXA", "PUMA", "CASP3", "CASP6", "CASP7", "CASP8",
                  "CASP9", "CYCS", "APAF1", "XIAP", "BIRC5", "BNIP3"],
    "DNA_repair": ["ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2", "RAD51",
                   "RAD52", "RAD54", "MLH1", "MSH2", "MSH6", "PMS2", "MSH3",
                   "ERCC1", "ERCC2", "XPA", "XPC", "POLK", "POLH"],
    "TGFB": ["TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "TGFBR3",
             "SMAD2", "SMAD3", "SMAD4", "SMAD7", "SKIL", "ENG",
             "BMP2", "BMP4", "BMP7", "ACVR1", "ACVR2A"],
    "epithelial_mesenchymal": ["CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "TWIST2",
                                "ZEB1", "ZEB2", "VIM", "FN1", "MMP2", "MMP9",
                                "CTNNB1", "SIP1", "SNAI3", "TCF3", "TCF4"],
}


class RegulatoryContextBuilder:
    def __init__(self, gene_vocab: 'GeneVocabulary', data_dir: str = "data"):
        self.gene_vocab = gene_vocab
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pathway_gene_map: Dict[str, List[int]] = {}
        self.gene_context_map: Dict[int, List[int]] = {}

    def build_from_pathways(self, pathways=None) -> int:
        if pathways is None:
            pathways = PATHWAYS
        vocab = self.gene_vocab
        for name, genes in pathways.items():
            idxs = []
            for g in genes:
                idx = vocab.gene_to_idx.get(g)
                if idx is not None:
                    idxs.append(idx)
            if idxs:
                self.pathway_gene_map[name] = idxs
        for gene_idx in range(vocab.vocab_size):
            partners: Set[int] = set()
            for pathway_genes in self.pathway_gene_map.values():
                if gene_idx in pathway_genes:
                    partners.update(pathway_genes)
            partners.discard(gene_idx)
            self.gene_context_map[gene_idx] = sorted(partners)
        self._save()
        total = sum(len(v) for v in self.gene_context_map.values())
        print(f"Built contexts for {len(self.gene_context_map)} genes, "
              f"{total} total context pairs")
        return total

    def _save(self):
        torch.save({
            'pathway_gene_map': self.pathway_gene_map,
            'gene_context_map': self.gene_context_map,
        }, self.data_dir / 'regulatory_contexts.pt')

    def load(self) -> bool:
        path = self.data_dir / 'regulatory_contexts.pt'
        if not path.exists():
            return False
        data = torch.load(path, weights_only=False)
        self.pathway_gene_map = data['pathway_gene_map']
        self.gene_context_map = data['gene_context_map']
        return True

    def get_context_for_genes(
        self,
        gene_indices: torch.Tensor,
        max_context: int = 5,
    ) -> torch.Tensor:
        """
        For each gene, get its top regulatory context partners.
        gene_indices: [B]
        Returns: [B, max_context] padded with pad_id (0)
        """
        B = gene_indices.shape[0]
        result = torch.zeros(B, max_context, dtype=torch.long)
        for i in range(B):
            idx = int(gene_indices[i])
            partners = self.gene_context_map.get(idx, [])
            result[i, :min(len(partners), max_context)] = torch.tensor(
                partners[:max_context], dtype=torch.long
            )
        return result

    def get_pathway_context_multihot(self, gene_idx: int, vocab_size: int) -> torch.Tensor:
        """Get binary vector of pathway membership."""
        vec = torch.zeros(vocab_size)
        partners = self.gene_context_map.get(int(gene_idx), [])
        if partners:
            vec[partners] = 1.0
        return vec
