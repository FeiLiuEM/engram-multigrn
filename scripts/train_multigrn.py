"""
Engram-MultiGRN: Two-stage incremental training pipeline.
Stage 1: Train on Human HepG2 (H4K5la)
Stage 2: Incrementally add Mouse Kla data with EWC protection
"""
import sys, os, json, random, time, gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).parent.parent / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder
from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig, EWC

# ── Load data ──
vocab = GeneVocabulary(str(DATA)); vocab.load()
ctx = RegulatoryContextBuilder(vocab, str(DATA)); ctx.load()

# Human vocab
human_genes = vocab.gene_to_idx
VS_human = vocab.vocab_size
print(f"Human vocab: {VS_human} genes", flush=True)

# Human Kla data
chip_data = json.load(open(DATA/"kla_chip_scores.json"))
# Use only H4K5la conditions
cond_map = {'H4K5la_NM2':[0,0,.2,0,0],'H4K5la_NM3':[0,0,.2,0,0],
            'H4K5la_LAC2':[0,0,.8,1,0],'H4K5la_LAC3':[0,0,.8,1,0]}
cond_vals = defaultdict(list)
for k,s in chip_data.items():
    g,c = k.split('__'); cond_vals[c].append(s)

random.seed(42)
human_samples = []
for k,raw in chip_data.items():
    g,c = k.split('__')
    if c not in cond_map: continue
    vs=cond_vals[c]; mn,mx=min(vs),max(vs)
    n=(raw-mn)/(mx-mn) if mx>mn else 0.5
    human_samples.append({'gene':g,'cond':c,'score':min(1,max(0,n)),
                         'cond_id':0 if 'NM' in c else 1, 'species':'human','cell':'hepg2'})
random.shuffle(human_samples)
sp_h = int(len(human_samples)*0.8)
train_h, test_h = human_samples[:sp_h], human_samples[sp_h:]
print(f"Human: {len(train_h)} train, {len(test_h)} test", flush=True)

# ── Mouse data ──
# Check if mouse Kla exists
mouse_kla_file = DATA / "mouse_kla_data.json"
if mouse_kla_file.exists():
    mouse_data = json.load(open(mouse_kla_file))
    print(f"Mouse Kla data: {len(mouse_data)} samples", flush=True)
else:
    # Generate synthetic mouse data for testing (based on human data mapped via orthologs)
    print("No mouse Kla data found. Creating placeholder from human data.", flush=True)
    # This is a fallback for development
    mouse_samples = []
    import copy
    mouse_genes = list(set(k.split('__')[0] for k in chip_data.keys()))
    random.shuffle(mouse_genes)
    # Map human genes to mouse orthologs
    try:
        ortho_map = json.load(open(DATA/"mouse_human_ortholog_map.json"))
    except: ortho_map = {}
    # Reverse: human→mouse
    human_to_mouse = {v:k for k,v in ortho_map.items()}
    
    for g in mouse_genes[:2000]:
        mouse_g = human_to_mouse.get(g, g.lower())
        for cid, (cond_name, cond_vec) in enumerate([('NM',[0,0,.2,0,0]),('LAC',[0,0,.8,1,0])]):
            score = random.uniform(0.1, 0.9)  # Placeholder
            mouse_samples.append({'gene':mouse_g,'cond':cond_name,'score':score,
                                 'cond_id':cid, 'species':'mouse','cell':'t_cell'})
    random.shuffle(mouse_samples); sp_m = int(len(mouse_samples)*0.8)
    train_m, test_m = mouse_samples[:sp_m], mouse_samples[sp_m:]
    print(f"Mouse (placeholder): {len(train_m)} train, {len(test_m)} test", flush=True)

# ── Dataset ──
class MultiSpeciesDS(torch.utils.data.Dataset):
    def __init__(self, samples, ortho_map, ctx_builder):
        self.samples = samples
        self.ortho_map = ortho_map  # {species_gene: ortho_id}
        self.ctx = ctx_builder
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        # Ortho ID (use gene name as fallback hash)
        ortho_id = hash(s['gene']) % 19999 + 1  # temporary
        sp_gene_id = hash(s['gene']) % 15000 + 1  # temporary
        cond_id = s['cond_id']
        score = s['score']
        species = s['species']
        cell = s['cell']
        
        # Context features (simplified for now)
        ctx_feat = torch.zeros(64).uniform_(-0.1, 0.1)
        if species == 'human':
            gid = human_genes.get(s['gene'], 0)
            if gid > 0:
                ctx_feat = ctx.get_context_for_genes(
                    torch.tensor([gid]), max_context=3).squeeze(0).float()
                if ctx_feat.shape[0] < 64:
                    ctx_feat = torch.nn.functional.pad(ctx_feat, (0, 64 - ctx_feat.shape[0]))
                elif ctx_feat.shape[0] > 64:
                    ctx_feat = ctx_feat[:64].float()
        
        return (torch.tensor(ortho_id, dtype=torch.long),
                torch.tensor(sp_gene_id, dtype=torch.long),
                ctx_feat[:64].float(),
                torch.tensor(cond_id, dtype=torch.long),
                torch.tensor(score, dtype=torch.float32),
                species, cell)

def collate_fn(batch):
    ortho = torch.stack([b[0] for b in batch])
    sp_gene = torch.stack([b[1] for b in batch])
    ctx = torch.stack([b[2] for b in batch])
    cond = torch.stack([b[3] for b in batch])
    score = torch.stack([b[4] for b in batch])
    species = [b[5] for b in batch]
    cell = [b[6] for b in batch]
    return ortho, sp_gene, ctx, cond, score, species, cell

# ── Initialize model ──
cfg = MultiGRNConfig()
cfg.n_ortho_groups = 20000
model = EngramMultiGRN(cfg.n_ortho_groups).to(device)
model.domain_embed._device = device
model.add_species('human', VS_human, 64)
model.add_species('mouse', 20000, 64)

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ═══════════════════════════════════════════════════
# STAGE 1: Train on Human data
# ═══════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STAGE 1: Training on Human HepG2", flush=True)
print("="*60, flush=True)

train_dl_h = torch.utils.data.DataLoader(
    MultiSpeciesDS(train_h, {}, ctx), batch_size=128, shuffle=True, collate_fn=collate_fn)
test_dl_h = torch.utils.data.DataLoader(
    MultiSpeciesDS(test_h, {}, ctx), batch_size=128, collate_fn=collate_fn)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
crit = nn.MSELoss()

def evaluate(model, dataloader):
    model.eval()
    pp, tt = [], []
    with torch.no_grad():
        for ortho, sp_gene, ctx_f, cond, score, sp, cell in dataloader:
            ortho, sp_gene, ctx_f, cond = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond]]
            pred = model(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
            pp.extend(pred.cpu().numpy())
            tt.extend(score.numpy())
    pp = np.array(pp); tt = np.array(tt)
    return float(np.corrcoef(pp, tt)[0, 1]) if len(set(tt)) > 1 else 0.0

t0 = time.time()
EPOCHS_1 = 30
for ep in range(EPOCHS_1):
    model.train()
    for ortho, sp_gene, ctx_f, cond, score, sp, cell in train_dl_h:
        ortho, sp_gene, ctx_f, cond, score = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond, score]]
        opt.zero_grad()
        pred = model(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
        loss = crit(pred, score)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    if (ep+1) % 10 == 0:
        r = evaluate(model, test_dl_h)
        print(f"  ep {ep+1}/{EPOCHS_1} R={r:.4f} ({time.time()-t0:.0f}s)", flush=True)

r_h_final = evaluate(model, test_dl_h)
print(f"Stage 1 final R: {r_h_final:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ── Save Stage 1 ──
torch.save(model.state_dict(), DATA/"multigrn_stage1.pt")
print("Saved: data/multigrn_stage1.pt", flush=True)

# ═══════════════════════════════════════════════════
# STAGE 2: Incremental addition of Mouse data
# ═══════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STAGE 2: Incremental Training + Mouse TCell", flush=True)
print("="*60, flush=True)

# Compute EWC for shared parameters
shared_names = model.get_shared_params()
print(f"Shared params protected by EWC: {len(shared_names)}", flush=True)

# Estimate Fisher on human data
ewc = EWC(model, lambda_ewc=500.0)
print("Estimating Fisher information matrix...", flush=True)
ewc.estimate_fisher(train_dl_h, crit, device,
                     ['ortho_embed', 'cond_encoder', 'hasher',
                      'hypernet', 'gate_', 'W_value'])
print("Fisher estimation complete.", flush=True)

# Freeze human-specific parameters
for name, param in model.named_parameters():
    if 'species_embeds.human' in name or 'encoders.human' in name:
        param.requires_grad = False
    elif any(sp in name for sp in ['ortho_embed', 'cond_encoder', 'hasher', 
                                    'hypernet', 'gate_', 'memory', 'W_value']):
        param.requires_grad = True  # shared, with EWC
    elif 'species_embeds.mouse' in name or 'encoders.mouse' in name:
        param.requires_grad = True  # new species

# Train on mouse
train_dl_m = torch.utils.data.DataLoader(
    MultiSpeciesDS(train_m, {}, ctx), batch_size=128, shuffle=True, collate_fn=collate_fn)
test_dl_m = torch.utils.data.DataLoader(
    MultiSpeciesDS(test_m, {}, ctx), batch_size=128, collate_fn=collate_fn)

opt2 = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 
                'species_embeds.mouse' in n or 'encoders.mouse' in n]},
    {'params': [p for n,p in model.named_parameters() if p.requires_grad and 
                not ('species_embeds.mouse' in n or 'encoders.mouse' in n)], 'lr': 1e-5}
], lr=3e-4, weight_decay=1e-5)

EPOCHS_2 = 20
for ep in range(EPOCHS_2):
    model.train()
    for ortho, sp_gene, ctx_f, cond, score, sp, cell in train_dl_m:
        ortho, sp_gene, ctx_f, cond, score = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond, score]]
        opt2.zero_grad()
        pred = model(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
        loss = crit(pred, score) + ewc.penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
    
    if (ep+1) % 5 == 0:
        r_h = evaluate(model, test_dl_h)
        r_m = evaluate(model, test_dl_m)
        print(f"  ep {ep+1}/{EPOCHS_2} Human R={r_h:.4f} Mouse R={r_m:.4f}", flush=True)

r_h_after = evaluate(model, test_dl_h)
r_m_final = evaluate(model, test_dl_m)
print(f"\nStage 2 final: Human R={r_h_after:.4f} (was {r_h_final:.4f}) Mouse R={r_m_final:.4f}", flush=True)

# Save results
results = {
    "stage1_human_R": r_h_final,
    "stage2_human_R_after_incremental": r_h_after,
    "stage2_mouse_R": r_m_final,
    "human_forgetting": r_h_final - r_h_after,
    "stage1_epochs": EPOCHS_1,
    "stage2_epochs": EPOCHS_2,
    "total_params": sum(p.numel() for p in model.parameters()),
    "shared_params_protected": len(shared_names),
}
json.dump(results, open(DATA/"multigrn_results.json", "w"), indent=2)
print(f"\nSaved: data/multigrn_results.json", flush=True)
print(json.dumps(results, indent=2), flush=True)
