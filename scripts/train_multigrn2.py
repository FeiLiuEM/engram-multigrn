"""Engram-MultiGRN training with real human+mouse Kla data."""
import sys, os, json, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch, torch.nn as nn
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig, EWC

# Load dataset
ds = json.load(open(DATA / "multispecies_dataset.json"))
train = ds['train']
test_h = ds['test_human']
test_m = ds['test_mouse']
stats = ds['stats']
print(f"Train: {len(train)} Human_test: {len(test_h)} Mouse_test: {len(test_m)}", flush=True)

class MultiSpeciesDS(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s['ortho_id'], dtype=torch.long),
                torch.tensor(s['sp_gene_id'], dtype=torch.long),
                torch.tensor(s['ctx_feat'], dtype=torch.float32),
                torch.tensor(s['cond_id'], dtype=torch.long),
                torch.tensor(s['score'], dtype=torch.float32),
                s['species'], s['cell'])

def collate_fn(batch):
    ortho = torch.stack([b[0] for b in batch])
    sp = torch.stack([b[1] for b in batch])
    ctx = torch.stack([b[2] for b in batch])
    cond = torch.stack([b[3] for b in batch])
    score = torch.stack([b[4] for b in batch])
    species = [b[5] for b in batch]
    cell = [b[6] for b in batch]
    return ortho, sp, ctx, cond, score, species, cell

# Datasets
train_ds = MultiSpeciesDS(train)
test_h_ds = MultiSpeciesDS(test_h)
test_m_ds = MultiSpeciesDS(test_m)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
test_h_dl = torch.utils.data.DataLoader(test_h_ds, batch_size=256, collate_fn=collate_fn)
test_m_dl = torch.utils.data.DataLoader(test_m_ds, batch_size=256, collate_fn=collate_fn)

# ═══ Initialize model ═══
n_ortho = stats['n_ortho_groups']
cfg = MultiGRNConfig()
cfg.n_ortho_groups = n_ortho + 1000  # buffer for novel genes

model = EngramMultiGRN(cfg.n_ortho_groups, d_ctx=64).to(device)
model.domain_embed._device = device
model.add_species('human', stats['human_vocab_size'], 64)
model.add_species('mouse', stats['mouse_vocab_size'], 64)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

crit = nn.MSELoss()

def evaluate(model, dataloader, species_filter=None):
    model.eval()
    pp, tt = [], []
    with torch.no_grad():
        for ortho, sp_gene, ctx_f, cond, score, sp, cell in dataloader:
            if species_filter and sp[0] != species_filter: continue
            ortho, sp_gene, ctx_f, cond = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond]]
            pred = model(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
            pp.extend(pred.cpu().numpy())
            tt.extend(score.numpy())
    pp=np.array(pp);tt=np.array(tt)
    return float(np.corrcoef(pp,tt)[0,1]) if len(set(tt))>1 else 0.0

# ═══ FREE FULL TRAINING (both species) ═══
print("\n"+"="*60, flush=True)
print("FULL TRAINING: Human + Mouse jointly", flush=True)
print("="*60, flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
t0 = time.time()
n_epochs = 40

for ep in range(n_epochs):
    model.train()
    epoch_loss = 0
    for ortho, sp_gene, ctx_f, cond, score, sp, cell in train_dl:
        ortho, sp_gene, ctx_f, cond, score = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond, score]]
        opt.zero_grad()
        pred = model(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
        loss = crit(pred, score)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item()
    
    if (ep+1) % 10 == 0:
        r_h = evaluate(model, test_h_dl)
        r_m = evaluate(model, test_m_dl)
        print(f"  ep {ep+1}/{n_epochs} loss={epoch_loss/len(train_dl):.5f} "
              f"Human R={r_h:.4f} Mouse R={r_m:.4f} ({time.time()-t0:.0f}s)", flush=True)

r_h_full = evaluate(model, test_h_dl)
r_m_full = evaluate(model, test_m_dl)
print(f"\nFull training done: Human R={r_h_full:.4f} Mouse R={r_m_full:.4f}", flush=True)

# Save
torch.save(model.state_dict(), DATA/"multigrn_full.pt")
results = {"full_human_R": r_h_full, "full_mouse_R": r_m_full,
           "epochs": n_epochs, "params": sum(p.numel() for p in model.parameters()),
           "train_samples": len(train)}

# ═══ INCREMENTAL: Train human first, then add mouse ═══
print("\n"+"="*60, flush=True)
print("STAGE 1: Human only", flush=True)
print("="*60, flush=True)

# Reset model
model2 = EngramMultiGRN(cfg.n_ortho_groups, d_ctx=64).to(device)
model2.domain_embed._device = device
model2.add_species('human', stats['human_vocab_size'], 64)
model2.add_species('mouse', stats['mouse_vocab_size'], 64)

human_train = [s for s in train if s['species']=='human']
human_dl = torch.utils.data.DataLoader(
    MultiSpeciesDS(human_train), batch_size=256, shuffle=True, collate_fn=collate_fn)

opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4, weight_decay=1e-5)

t0 = time.time()
for ep in range(30):
    model2.train()
    for ortho, sp_gene, ctx_f, cond, score, sp, cell in human_dl:
        ortho, sp_gene, ctx_f, cond, score = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond, score]]
        opt2.zero_grad()
        pred = model2(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
        crit(pred, score).backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt2.step()
    if (ep+1)%10==0:
        r = evaluate(model2, test_h_dl)
        print(f"  ep {ep+1}/30 Human R={r:.4f} ({time.time()-t0:.0f}s)", flush=True)

r_h_stage1 = evaluate(model2, test_h_dl)
print(f"Stage 1: Human R={r_h_stage1:.4f}", flush=True)

# EWC
ewc = EWC(model2, lambda_ewc=500.0)
print("Computing EWC Fisher...", flush=True)
ewc.estimate_fisher(human_dl, crit, device,
                     ['ortho_embed', 'cond_encoder', 'hasher',
                      'hypernet', 'gate_', 'W_value'])
print("Fisher done.", flush=True)

# Freeze human-specific, add mouse
for name, p in model2.named_parameters():
    if 'species_embeds.human' in name or 'encoders.human' in name:
        p.requires_grad = False

opt3 = torch.optim.AdamW([
    {'params': [p for n,p in model2.named_parameters()
                if 'species_embeds.mouse' in n or 'encoders.mouse' in n]},
    {'params': [p for n,p in model2.named_parameters()
                if p.requires_grad and not('species_embeds.mouse' in n or 'encoders.mouse' in n)],
     'lr': 1e-5}
], lr=3e-4, weight_decay=1e-5)

mouse_train = [s for s in train if s['species']=='mouse']
mouse_dl = torch.utils.data.DataLoader(
    MultiSpeciesDS(mouse_train), batch_size=256, shuffle=True, collate_fn=collate_fn)

print("\n"+"="*60, flush=True)
print("STAGE 2: Add Mouse (incremental)", flush=True)
print("="*60, flush=True)

t0 = time.time()
for ep in range(20):
    model2.train()
    for ortho, sp_gene, ctx_f, cond, score, sp, cell in mouse_dl:
        ortho, sp_gene, ctx_f, cond, score = [x.to(device) for x in [ortho, sp_gene, ctx_f, cond, score]]
        opt3.zero_grad()
        pred = model2(ortho, sp_gene, ctx_f, cond, sp[0], cell[0])
        loss = crit(pred, score) + ewc.penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt3.step()
    if (ep+1)%5==0:
        rh = evaluate(model2, test_h_dl)
        rm = evaluate(model2, test_m_dl)
        print(f"  ep {ep+1}/20 Human R={rh:.4f} Mouse R={rm:.4f} ({time.time()-t0:.0f}s)", flush=True)

r_h_inc, r_m_inc = evaluate(model2, test_h_dl), evaluate(model2, test_m_dl)
print(f"\nIncremental done: Human R={r_h_inc:.4f} (was {r_h_stage1:.4f}, Δ={r_h_inc-r_h_stage1:+.4f})", flush=True)
print(f"                Mouse R={r_m_inc:.4f}", flush=True)

# Save all results
results.update({
    "stage1_human_R": r_h_stage1,
    "stage2_human_R": r_h_inc,
    "stage2_mouse_R": r_m_inc,
    "human_forgetting": r_h_stage1 - r_h_inc,
    "full_human_R": r_h_full,
    "full_mouse_R": r_m_full,
})
json.dump(results, open(DATA/"multigrn_results.json","w"), indent=2)
print(f"\nResults:\n{json.dumps(results,indent=2)}", flush=True)
