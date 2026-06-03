#!/usr/bin/env python3
"""
Integration test for Engram-MultiGRN.

Uses pre-processed data to verify:
  1. Model instantiation succeeds
  2. Training converges (Pearson R > 0 after 10 epochs on H4K5la_NM2)
  3. The model can save and reload checkpoints

Requirements:
  - All pre-processed data files in data/
  - CUDA GPU (falls back to CPU with a warning)
  - ~2 minutes on RTX 4090

Usage:
  python tests/test_integration.py
"""

import sys, os, json, random, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict

TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent
DATA_DIR = Path(os.environ.get("ENGRAM_DATA_DIR", str(PROJECT_DIR / ".." / "data")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    warnings.warn("CUDA not available — test will run on CPU (slower).")


def test_model_instantiation():
    """Test 1: Model can be instantiated with correct dimensions."""
    from engram_multigrn.model.multigrn import EngramMultiGRN, MultiGRNConfig

    cfg = MultiGRNConfig()
    cfg.n_ortho_groups = 30000
    model = EngramMultiGRN(
        cfg.n_ortho_groups, d_ctx=102, ctx_input_dim=102
    ).to(device)
    model.add_species("human", 19298, 102)

    # Forward pass
    B = 8
    o = torch.randint(0, cfg.n_ortho_groups, (B,), device=device)
    g = torch.randint(0, 19000, (B,), device=device)
    c = torch.randn(B, 102, device=device)
    cond = torch.zeros(B, dtype=torch.long, device=device)
    out = model(o, g, c, cond, "human", "hepg2", "H4K5la")

    n_params = sum(p.numel() for p in model.parameters())
    assert out.shape == (B, 1) or out.shape == (B,), f"Expected {(B,1)} or {(B,)}, got {out.shape}"
    assert n_params > 10_000_000, f"Too few params: {n_params}"

    print(f"  ✅ Model instantiation: {n_params:,} params, output {list(out.shape)}")
    return model


def test_training_on_h4k5la():
    """Test 2: Model converges on H4K5la (using 2 conditions for stronger signal)."""
    from engram_multigrn.model.multigrn import EngramMultiGRN, MultiGRNConfig, ConditionEncoder
    from engram_multigrn.data_pipeline.gene_vocab import GeneVocabulary
    from engram_multigrn.data_pipeline.regulatory_context import RegulatoryContextBuilder

    # Load feature data
    feat = json.load(open(DATA_DIR / "multigrn_features_full.json"))
    human_oid = feat["human_to_orthoid"]

    scores_raw = json.load(
        open(DATA_DIR / "incremental_pipeline_results/all_preprocessed_scores.json")
    )
    # Use 2 conditions to give the model more signal
    all_gene_scores = {
        "NM2": scores_raw["H4K5la_NM2"],
        "NM3": scores_raw["H4K5la_NM3"],
    }
    print(f"  Loaded H4K5la_NM2 ({len(all_gene_scores['NM2'])} genes) + NM3 ({len(all_gene_scores['NM3'])} genes)")

    # Build STRING + genomic + chromatin features
    vocab_h = GeneVocabulary(str(DATA_DIR))
    vocab_h.load()
    ctx_h = RegulatoryContextBuilder(vocab_h, str(DATA_DIR))
    ctx_h.load()

    gfeats = json.load(open(DATA_DIR / "gene_genomic_features.json"))
    cfeats = json.load(open(DATA_DIR / "hepg2_chromatin_features.json"))
    GFK = [
        "gene_len", "n_exons", "n_introns", "intron_len",
        "cds_len", "n_transcripts", "exon_density", "cds_ratio",
    ]

    human_ctx_cache = {}
    for gene in list(vocab_h.gene_to_idx.keys())[:15000]:
        gid = vocab_h.gene_to_idx.get(gene, 0)
        if gid == 0:
            human_ctx_cache[gene] = [0.0] * 64
            continue
        c = (
            ctx_h.get_context_for_genes(torch.tensor([gid]), max_context=3)
            .squeeze(0)
            .float()
        )
        if c.shape[0] < 64:
            c = nn.functional.pad(c, (0, 64 - c.shape[0]))
        human_ctx_cache[gene] = c[:64].tolist()

    def get_ctx(gene):
        s = human_ctx_cache.get(gene, [0.0] * 64)
        if len(s) < 64:
            s = s + [0.0] * (64 - len(s))
        gf = gfeats.get(gene, {k: 0.0 for k in GFK})
        cf = cfeats.get(gene, [0.0] * 30)
        if len(cf) < 30:
            cf = cf + [0.0] * (30 - len(cf))
        return s[:64] + [gf.get(k, 0.0) for k in GFK] + cf[:30]

    # Build samples from both conditions
    random.seed(42)
    samples = []
    cond_map = {}
    for cid, (cond_name, gene_scores) in enumerate(sorted(all_gene_scores.items())):
        cond_map[cond_name] = cid
        vals = list(gene_scores.values())
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [1, 99])
        ac = np.clip(arr, lo, hi)
        norm = (ac - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)
        for (gene, _), nscore in zip(gene_scores.items(), norm):
            oid = int(human_oid.get(gene, 0))
            sid = hash(gene) % 19000 + 1
            ctx = get_ctx(gene)
            samples.append({
                "ortho_id": oid, "sp_gene_id": sid,
                "score": float(nscore), "ctx_feat": ctx,
                "cond_id": cid, "species": "human",
                "cell": "hepg2", "mark": "H4K5la", "domain": cond_name,
            })

    random.shuffle(samples)
    n_train = int(len(samples) * 0.8)
    train_s = samples[:n_train]
    test_s = samples[n_train:]
    print(f"  Train: {len(train_s)}, Test: {len(test_s)}")

    INPUT_DIM = len(samples[0]["ctx_feat"])

    class DS(torch.utils.data.Dataset):
        def __init__(self, sl):
            self.s = sl

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            s = self.s[i]
            return (
                torch.tensor(s["ortho_id"], dtype=torch.long),
                torch.tensor(s["sp_gene_id"], dtype=torch.long),
                torch.tensor(s["ctx_feat"], dtype=torch.float32),
                torch.tensor(s["cond_id"], dtype=torch.long),
                torch.tensor(s["score"], dtype=torch.float32),
                s["species"], s["cell"], s["mark"], s["domain"],
            )

    def collate(b):
        return (
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
            torch.stack([x[2] for x in b]),
            torch.stack([x[3] for x in b]),
            torch.stack([x[4] for x in b]),
            [x[5] for x in b], [x[6] for x in b],
            [x[7] for x in b], [x[8] for x in b],
        )

    # Model
    cfg = MultiGRNConfig()
    cfg.n_ortho_groups = feat["n_ortho_groups"] + 1000
    model = EngramMultiGRN(
        cfg.n_ortho_groups, d_ctx=INPUT_DIM, ctx_input_dim=INPUT_DIM
    ).to(device)
    model.add_species("human", feat["human_vocab_size"], INPUT_DIM)
    model.cond_encoder = ConditionEncoder(
        n_conditions=max(len(cond_map), 8), d_cond=cfg.d_cond
    ).to(device)

    train_loader = torch.utils.data.DataLoader(
        DS(train_s), batch_size=256, shuffle=True, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        DS(test_s), batch_size=512, shuffle=False, collate_fn=collate
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-5
    )

    t0 = time.time()
    for epoch in range(15):
        model.train()
        for o, g, c, co, sc, sp, cl, mx, dm in train_loader:
            o, g, c, co, sc = [x.to(device) for x in [o, g, c, co, sc]]
            optimizer.zero_grad()
            loss = criterion(
                model(o, g, c, co, sp[0], cl[0], mx[0]), sc
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for o, g, c, co, sc, sp, cl, mx, dm in test_loader:
            o, g, c, co = [x.to(device) for x in [o, g, c, co]]
            pr = model(o, g, c, co, sp[0], cl[0], mx[0])
            preds.extend(pr.cpu().numpy().flatten().tolist())
            targets.extend(sc.numpy().tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    r = float(np.corrcoef(preds, targets)[0, 1])
    elapsed = time.time() - t0

    print(f"  Pearson R after 15 epochs: {r:.4f} ({elapsed:.0f}s)")

    assert r > 0.05, f"R={r:.4f} — model did not learn (expected R > 0.05)"
    assert r < 0.95, f"R={r:.4f} — suspiciously high, possible data leak"
    return r


def test_checkpoint_save_load():
    """Test 3: Model can save and reload checkpoints without degrading performance."""
    from engram_multigrn.model.multigrn import EngramMultiGRN, MultiGRNConfig

    cfg = MultiGRNConfig()
    cfg.n_ortho_groups = 30000
    model = EngramMultiGRN(cfg.n_ortho_groups, d_ctx=102, ctx_input_dim=102).to(device)

    # Save
    tmp_path = "/tmp/engram_test_checkpoint.pt"
    torch.save(model.state_dict(), tmp_path)

    # Load into new model
    model2 = EngramMultiGRN(cfg.n_ortho_groups, d_ctx=102, ctx_input_dim=102).to(device)
    model2.load_state_dict(torch.load(tmp_path, map_location=device))

    # Verify weights are identical
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Checkpoint save/load changed weights"

    os.remove(tmp_path)
    print(f"  ✅ Checkpoint save/load: weights identical")


def main():
    print("=" * 60)
    print("Engram-MultiGRN Integration Tests")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        print("❌ Data directory not found. Set ENGRAM_DATA_DIR environment variable.")
        sys.exit(1)

    passed = 0
    failed = 0

    for name, fn in [
        ("Model instantiation", test_model_instantiation),
        ("Training convergence (H4K5la_NM2)", test_training_on_h4k5la),
        ("Checkpoint save/load", test_checkpoint_save_load),
    ]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed:
        print("❌ Some tests failed.")
        sys.exit(1)
    else:
        print("✅ All tests passed.")


if __name__ == "__main__":
    main()
