"""
Engram-MultiGRN: Multi-species, multi-cell-type continuous learning framework.
Extends Engram-GRN with orthologous gene embeddings, domain-aware hashing,
and hypernetwork-based gating for cross-species knowledge accumulation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
# 1. ORTHOLOGOUS GENE EMBEDDING
# ═══════════════════════════════════════════════════════════════════
class OrthoGeneEmbedding(nn.Module):
    """
    Gene embedding with orthologous base + species-specific offset.
    
    gene_vec = LayerNorm( Embedding_ortho(ortho_id) + Embedding_sp(sp_local_id) )
    
    Embedding_ortho: shared across all species (N_ortho_groups, d=128)
    Embedding_sp:    per-species table (num_genes_sp, d=128), stores residual
    """
    def __init__(self, n_ortho_groups: int, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.ortho_embed = nn.Embedding(n_ortho_groups, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)
        # Per-species embeddings (added dynamically)
        self.species_embeds = nn.ModuleDict()
        
    def add_species(self, species_name: str, num_genes: int):
        """Add a new species-specific embedding table."""
        if species_name in self.species_embeds:
            raise ValueError(f"Species {species_name} already exists")
        self.species_embeds[species_name] = nn.Embedding(num_genes, self.d_model, padding_idx=0)
        
    def forward(self, ortho_ids: torch.Tensor, sp_local_ids: torch.Tensor, 
                species_name: str) -> torch.Tensor:
        """
        ortho_ids:   [B] or [B, 1] - orthologous group IDs
        sp_local_ids: [B] or [B, 1] - species-specific gene IDs
        species_name: str - which species embedding to use
        """
        if ortho_ids.dim() == 1:
            ortho_ids = ortho_ids.unsqueeze(-1)
        if sp_local_ids.dim() == 1:
            sp_local_ids = sp_local_ids.unsqueeze(-1)
        
        # Clamp to safe ranges
        ortho_ids = ortho_ids.clamp(0, self.ortho_embed.num_embeddings - 1)
        max_sp = self.species_embeds[species_name].num_embeddings - 1
        sp_local_ids = sp_local_ids.clamp(0, max_sp)
        
        u_ortho = self.ortho_embed(ortho_ids).squeeze(1)
        u_sp = self.species_embeds[species_name](sp_local_ids).squeeze(1)
        return self.norm(u_ortho + u_sp)


# ═══════════════════════════════════════════════════════════════════
# 2. PER-SPECIES REGULATORY CONTEXT ENCODER
# ═══════════════════════════════════════════════════════════════════
class RegContextEncoder(nn.Module):
    """
    Encodes STRING regulatory context into a context vector.
    Per-species: each species gets its own encoder.
    Shared base + LoRA adapter optional.
    """
    def __init__(self, input_dim: int, d_ctx: int = 64, n_layers: int = 2):
        super().__init__()
        layers = []
        dims = [input_dim] + [d_ctx * 2] * (n_layers - 1) + [d_ctx]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpeciesContextEncoderBank(nn.Module):
    """
    Manages per-species context encoders.
    New species = new encoder; old ones frozen during incremental training.
    """
    def __init__(self, input_dim: int, d_ctx: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.d_ctx = d_ctx
        self.encoders = nn.ModuleDict()
        
    def add_species(self, species_name: str):
        if species_name in self.encoders:
            return
        self.encoders[species_name] = RegContextEncoder(self.input_dim, self.d_ctx)
        
    def forward(self, x: torch.Tensor, species_name: str) -> torch.Tensor:
        return self.encoders[species_name](x)


# ═══════════════════════════════════════════════════════════════════
# 3. CONDITION ENCODER (shared across all species)
# ═══════════════════════════════════════════════════════════════════
class ConditionEncoder(nn.Module):
    """Shared condition encoder: one-hot or learned embedding → cond_vec (d=32)."""
    def __init__(self, n_conditions: int = 8, d_cond: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_conditions, d_cond)
        
    def forward(self, cond_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(cond_ids)


# ═══════════════════════════════════════════════════════════════════
# 4. DOMAIN EMBEDDING
# ═══════════════════════════════════════════════════════════════════
class DomainEmbedding(nn.Module):
    """
    Joint species × cell-type embedding table.
    d_dom = Embed_domain[species_idx, cell_type_idx] ∈ R^32
    """
    def __init__(self, d_dom: int = 32):
        super().__init__()
        self.d_dom = d_dom
        self.domain_to_idx = {}
        self.idx_to_domain = {}
        self._device = None
        self.embed = None  # Initialized lazily
        
    def register_domain(self, species: str, cell_type: str) -> int:
        key = (species, cell_type)
        if key not in self.domain_to_idx:
            idx = len(self.domain_to_idx)
            self.domain_to_idx[key] = idx
            self.idx_to_domain[idx] = key
            self._rebuild_embed()
        return self.domain_to_idx[key]
    
    def _rebuild_embed(self):
        n = len(self.domain_to_idx)
        old_weight = None
        if self.embed is not None:
            old_weight = self.embed.weight.data.clone()
        self.embed = nn.Embedding(max(n, 1), self.d_dom)
        if self._device is not None:
            self.embed = self.embed.to(self._device)
        if old_weight is not None:
            self.embed.weight.data[:old_weight.shape[0]] = old_weight
            
    def forward(self, domain_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(domain_idx)


# ═══════════════════════════════════════════════════════════════════
# 5. DETERMINISTIC HASH (using sympy primes) + STANDARD EMBEDDING
# ═══════════════════════════════════════════════════════════════════
class DeterministicHash(nn.Module):
    """
    Deterministic hash with domain string → single combined offset.
    Species and cell type are joined as 'species_cell' to form one domain key.
    A single offset per domain ensures each (species, cell) pair has its own
    isolated region in the hash space — no cross-domain collisions.
    """
    def __init__(self, max_domains: int = 20, n_hash_tables: int = 4, 
                 table_size: int = 500000):
        super().__init__()
        self.n_hash_tables = n_hash_tables
        self.table_size = table_size
        
        # Domain multipliers: each (species_cell) gets its own fixed multiplier
        # Stored as map from domain_string → list of per-head multipliers
        self.domain_multipliers = {}  # filled at runtime
        
    def register_domain(self, domain_key: str):
        """Register a new 'species_cell' domain with fixed hash multipliers."""
        if domain_key in self.domain_multipliers:
            return
        mults = []
        for h in range(self.n_hash_tables):
            m = float(hash(f"domain_{domain_key}_h{h}") % 100007 + 1)
            mults.append(m)
        self.domain_multipliers[domain_key] = mults
        
    def forward(self, input_ids: torch.Tensor, domain_key: str) -> torch.Tensor:
        """
        input_ids:  [B] - gene IDs to hash
        domain_key: str - 'species_cell' combined domain identifier
        Returns: [B, n_hash_tables] - slot indices
        """
        B = input_ids.shape[0]
        multipliers = self.domain_multipliers.get(domain_key, [1]*self.n_hash_tables)
        mult_t = torch.tensor(multipliers, device=input_ids.device, dtype=torch.long)
        
        indices = []
        for h in range(self.n_hash_tables):
            offset = mult_t[h] % self.table_size
            hash_val = (input_ids.long() * (100003 + h * 7) + offset) % self.table_size
            indices.append(hash_val.unsqueeze(1))
        return torch.cat(indices, dim=1)


# ═══════════════════════════════════════════════════════════════════
# 6. MEMORY TABLE (Standard Embedding, gradient-based)
# ═══════════════════════════════════════════════════════════════════
class MemoryTable(nn.Module):
    """
    Hash memory table using standard nn.Embedding.
    Read via embedding lookup; gradients flow through embedding weights.
    No manual EMA — standard PyTorch gradient descent.
    """
    def __init__(self, n_slots: int = 4000000, d_value: int = 32):
        super().__init__()
        self.n_slots = n_slots
        self.d_value = d_value
        self.embed = nn.Embedding(n_slots, d_value)
        self.embed.weight.data.fill_(0.0)
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, n_heads]
        Returns: [B, n_heads, d_value]
        """
        vals = self.embed(indices)  # [B, n_heads, d_value]
        return vals


# ═══════════════════════════════════════════════════════════════════
# 7. HYPERNETWORK (domain-conditional weight generation)
# ═══════════════════════════════════════════════════════════════════
class HyperNetwork(nn.Module):
    """
    Generates gating and output head parameters from domain embedding.
    
    Input: d_dom (32d) → MLP → {gate_weight_shift, out_weight, out_bias}
    """
    def __init__(self, d_dom: int = 32, d_hidden: int = 64, 
                 d_query: int = 256, d_key: int = 32, d_out: int = 1):
        super().__init__()
        self.d_out = d_out
        
        # Shared MLP
        self.net = nn.Sequential(
            nn.Linear(d_dom, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        
        # Gate Q projection shift
        self.gate_q_shift = nn.Linear(d_hidden, d_query * d_key)
        # Gate K projection (not shifted, shared base)
        self.gate_k = nn.Linear(d_key, d_key)
        
        # Output head: out_input = [v*α (32d) + cond (32d) + gene (128d)] = 192d
        self.out_dim = 192  # d_mem + d_cond + d_model
        self.out_weight_gen = nn.Linear(d_hidden, self.out_dim * d_out)
        self.out_bias_gen = nn.Linear(d_hidden, d_out)
        
    def forward(self, d_dom: torch.Tensor):
        """
        d_dom: [B, d_dom]
        Returns: gate_Q_weight_shift, gate_K_weight (shared), out_weight, out_bias
        """
        h = self.net(d_dom)  # [B, d_hidden]
        
        # Mean pool over batch for weight generation (one set per batch)
        h_pooled = h.mean(dim=0, keepdim=True)  # [1, d_hidden]
        
        gate_q_shift = self.gate_q_shift(h_pooled)  # [1, d_query*d_key]
        out_w = self.out_weight_gen(h_pooled)  # [1, d_query*d_out]
        out_b = self.out_bias_gen(h_pooled)  # [1, d_out]
        
        return gate_q_shift.squeeze(0), self.gate_k.weight, out_w.squeeze(0), out_b.squeeze(0)


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# 8. ENGRAM-MULTI-GRN: COMPLETE MODEL
# ═══════════════════════════════════════════════════════════════════
class EngramMultiGRN(nn.Module):
    """
    Full Engram-MultiGRN architecture.
    
    Forward pass:
    1. Gene embedding: OrthoBase + species offset
    2. Context encoding: per-species encoder
    3. Condition encoding: shared
    4. Domain embedding
    5. Domain-aware multi-head hash → slot indices
    6. Memory read + gated retrieval (HyperNet)
    7. Output: domain-specific head (HyperNet-generated)
    """
    def __init__(self, n_ortho_groups: int, d_model: int = 128, d_ctx: int = 72,
                 d_cond: int = 32, d_dom: int = 32, d_mem: int = 32,
                 n_heads: int = 4, hash_len: int = 32, n_memory_slots: int = 1000000,
                 ctx_input_dim: int = 72):
        super().__init__()
        
        self.d_model = d_model
        self.d_ctx = d_ctx
        self.d_cond = d_cond
        self.d_dom = d_dom
        self.d_mem = d_mem
        self.n_heads = n_heads
        
        # Core modules
        self.gene_embed = OrthoGeneEmbedding(n_ortho_groups, d_model)
        self.context_encoders = SpeciesContextEncoderBank(ctx_input_dim, d_ctx)
        self.cond_encoder = ConditionEncoder(8, d_cond)
        
        # 2a. Mark embedding: lazy registration via forward
        self.mark_embed = nn.Embedding(50, d_cond)
        self._mark_name_to_idx: Dict[str, int] = {}
        
        # Deterministic hash: each 'species_cell' gets a unique multiplier
        self.hasher = DeterministicHash(max_domains=20)
        
        # Memory (standard nn.Embedding with gradient)
        self.memory = MemoryTable(n_memory_slots, d_mem)
        
        # Gate + Output head: DOMAIN-ISOLATED (each species_cell_mark has its own)
        self.gate_norm_q = nn.ModuleDict()
        self.gate_norm_k = nn.ModuleDict()
        self.gate_k_proj = nn.ModuleDict()
        self.gate_v_proj = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        self.gate_scale = math.sqrt(d_model)
        self._d_model = d_model
        self._d_mem = d_mem
        self._d_cond = d_cond
        self._d_ctx = d_ctx
        
    def _ensure_domain_gate(self, domain_key: str):
        if domain_key not in self.gate_norm_q:
            d = self._d_model
            self.gate_norm_q[domain_key] = nn.RMSNorm(d)
            self.gate_norm_k[domain_key] = nn.RMSNorm(d)
            self.gate_k_proj[domain_key] = nn.Linear(self._d_mem, d)
            self.gate_v_proj[domain_key] = nn.Linear(self._d_mem, d)
            out_dim = d + self._d_ctx + self._d_cond + self._d_cond
            self.output_heads[domain_key] = nn.Sequential(
                nn.Linear(out_dim, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1)
            )
            if next(self.parameters(), None) is not None:
                dev = next(self.parameters()).device
                for m in [self.gate_norm_q[domain_key], self.gate_norm_k[domain_key],
                          self.gate_k_proj[domain_key], self.gate_v_proj[domain_key],
                          self.output_heads[domain_key]]:
                    m.to(dev)
        
    
    @staticmethod
    def _get_domain_key(species: str, cell_type: str, mark: str = "") -> str:
        """Combine species, cell type, and histone mark into domain key: 'species_cell_mark'."""
        return f"{species}_{cell_type}_{mark}" if mark else f"{species}_{cell_type}"
    
    def add_species(self, species_name: str, num_genes: int, 
                    ctx_input_dim: Optional[int] = None):
        """Add a new species (gene embedding + context encoder)."""
        self.gene_embed.add_species(species_name, num_genes)
        self.context_encoders.add_species(species_name)
        if list(self.parameters()):
            dev = next(self.parameters()).device
            self.gene_embed.species_embeds[species_name].to(dev)
            self.context_encoders.encoders[species_name].to(dev)
    
    def forward(self, ortho_ids: torch.Tensor, sp_gene_ids: torch.Tensor,
                ctx_features: torch.Tensor, cond_ids: torch.Tensor,
                species: str, cell_type: str,
                mark: str = "") -> torch.Tensor:
        """
        Forward pass using deterministic hash + standard embedding + direct gate.
        Like Engram-GRN: hash → embed → gate → output (no HyperNet).
        domain_key = 'species_cell_mark' for full domain isolation.
        """
        B = ortho_ids.shape[0]
        domain_key = self._get_domain_key(species, cell_type, mark)
        
        # Register domain on first use
        if domain_key not in self.hasher.domain_multipliers:
            self.hasher.register_domain(domain_key)
        
        # 1. Gene embedding (ortho base + species offset)
        g = self.gene_embed(ortho_ids, sp_gene_ids, species)  # [B, d_model]
        
        # 2. Context encoding
        ctx = self.context_encoders(ctx_features, species)  # [B, d_ctx]
        
        # 3. Condition encoding
        cond = self.cond_encoder(cond_ids)  # [B, d_cond]
        
        # 4. Deterministic hash with domain isolation (single key)
        # 1b. Hash on sp_gene_ids (unique per gene) not ortho_ids (many-to-one)
        slot_indices = self.hasher(sp_gene_ids, domain_key)  # [B, n_heads]
        
        # 5. Memory read (standard embedding, gradient flows)
        v = self.memory(slot_indices)  # [B, n_heads, d_mem]
        
        # 6. Multi-head gate (domain-isolated: each species_cell_mark has its own)
        self._ensure_domain_gate(domain_key)
        gnq = self.gate_norm_q[domain_key]
        gnk = self.gate_norm_k[domain_key]
        gkp = self.gate_k_proj[domain_key]
        gvp = self.gate_v_proj[domain_key]
        
        q = gnq(g)                                   # [B, d_model]
        k_h = gnk(gkp(v))                            # [B, n_heads, d_model]
        alpha_h = torch.sigmoid((q.unsqueeze(1) * k_h).sum(dim=-1) / self.gate_scale)  # [B, n_heads]
        
        B, H, D = v.shape
        v_proj = gvp(v.reshape(-1, D)).reshape(B, H, -1)  # [B, H, d_model]
        gated = g + (alpha_h.unsqueeze(-1) * v_proj).sum(dim=1)  # [B, d_model]
        
        # 2a. Mark embedding lookup
        if mark not in self._mark_name_to_idx:
            self._mark_name_to_idx[mark] = len(self._mark_name_to_idx)
        midx = self._mark_name_to_idx[mark]
        mark_t = torch.full((B,), midx, device=ortho_ids.device, dtype=torch.long)
        mark_vec = self.mark_embed(mark_t)  # [B, d_cond]
        
        # 7. Domain-isolated output head: gene + ctx + cond + mark
        out_input = torch.cat([gated, ctx, cond, mark_vec], dim=-1)
        return self.output_heads[domain_key](out_input).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# 10. CONFIG
# ═══════════════════════════════════════════════════════════════════
class MultiGRNConfig:
    def __init__(self):
        self.n_ortho_groups: int = 20000
        self.d_model: int = 128
        self.d_ctx: int = 64
        self.d_cond: int = 32
        self.d_dom: int = 32
        self.d_mem: int = 32
        self.n_heads: int = 4
        self.hash_len: int = 32
        self.n_memory_slots: int = 1000000
        self.ctx_input_dim: int = 64
        self.ema_alpha: float = 0.1
        self.lambda_ewc: float = 1000.0
        self.dropout: float = 0.1
