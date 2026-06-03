"""MultiGRN v2 with deterministic hash + standard embedding."""
import sys, os, json, random, time, gzip, tarfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch, torch.nn as nn
from pathlib import Path; from collections import defaultdict

DATA = Path(__file__).parent.parent / "data"
device = torch.device("cuda")
print(f"Device: {device}", flush=True)

from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder
from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig

vocab = GeneVocabulary(str(DATA)); vocab.load()
ctx = RegulatoryContextBuilder(vocab, str(DATA)); ctx.load()

def get_cf(gene):
    gid = vocab.gene_to_idx.get(gene, 0)
    if gid == 0: return [0.0]*64
    c = ctx.get_context_for_genes(torch.tensor([gid]), max_context=3).squeeze(0).float()
    if c.shape[0] < 64: c = nn.functional.pad(c, (0, 64-c.shape[0]))
    return c[:64].tolist()

all_g = list(set(k.split("__")[0] for k in json.load(open(DATA/"kla_chip_scores.json")).keys()))
cfc = {g: get_cf(g) for g in all_g[:10000]}
print(f"CF: {len(cfc)} genes", flush=True)

hr = defaultdict(list)
with gzip.open(DATA/"refGene.txt.gz", "rt") as f:
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 13: hr[p[2]].append((int(p[4]), int(p[5]), p[12]))
mr = defaultdict(list)
with gzip.open(DATA/"mouse_refGene.txt.gz", "rt") as f:
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 13: mr[p[2]].append((int(p[4]), int(p[5]), p[12]))

def p2s(tp, rg, cp=""):
    sc = defaultdict(float)
    ta = tarfile.open(tp)
    for m in ta.getmembers():
        if not m.name.endswith("broadPeak.gz"): continue
        raw_bytes = ta.extractfile(m).read()
        ct = gzip.decompress(raw_bytes).decode()
        for l in ct.strip().split("\n"):
            ps = l.split("\t")
            if len(ps) >= 7:
                ch = cp+ps[0]; s = int(ps[1]); e = int(ps[2]); scv = float(ps[6])
                if ch in rg:
                    for gs, ge, gn in rg[ch]:
                        if s < ge and e > gs: sc[gn] += scv
    return sc

h99 = p2s(DATA/"geo/GSE207814_RAW.tar", hr, "chr")
bmd = p2s(DATA/"geo/GSE115354_RAW.tar", mr, "")
def nz(d):
    v = list(d.values())
    if not v: return {}
    mn, mx = min(v), max(v)
    return {g: (s-mn)/(mx-mn) if mx>mn else 0.5 for g,s in d.items()}
h99n = nz(h99); bmdn = nz(bmd)
print(f"H1299:{len(h99n)} BMDM:{len(bmdn)}", flush=True)

random.seed(42); samples = []
cd = json.load(open(DATA/"kla_chip_scores.json"))
cm = {"H4K5la_NM2":0,"H4K5la_NM3":0,"H4K5la_LAC2":1,"H4K5la_LAC3":1}
cv_ = defaultdict(list)
for k,s in cd.items(): g,c = k.split("__"); cv_[c].append(s)

def make_sample(g, s, ci, sp, cl, dom, mk_t=""):
    cf = cfc.get(g, [0.0]*64)
    if len(cf) < 64: cf = cf + [0.0]*(64-len(cf))
    return {"ortho_id":hash(g)%30000+1, "sp_gene_id":hash(g)%(19000 if sp=="human" else 26000)+1,
            "score":s, "cond_id":ci, "species":sp, "cell":cl, "domain":dom, 
            "mark":mk_t, "ctx_feat":cf}

for k,raw in cd.items():
    g,c = k.split("__")
    if c not in cm: continue
    vs = cv_[c]; mn,mx = min(vs), max(vs); n = (raw-mn)/(mx-mn) if mx>mn else 0.5
    samples.append(make_sample(g, min(1,max(0,n)), cm[c], "human", "hepg2", "human_hepg2", "H4K5la"))
for g,s in h99n.items(): samples.append(make_sample(g, s, 2, "human", "h1299", "human_h1299", "H3K18la"))
for g,s in bmdn.items(): samples.append(make_sample(g, s, 3, "mouse", "bmdm", "mouse_bmdm", "H3K18la"))

random.shuffle(samples)
sp = int(len(samples)*0.8); tr_s, te_s = samples[:sp], samples[sp:]
td = defaultdict(list)
for s in te_s: td[s["domain"]].append(s)
print(f"Total:{len(samples)}", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        return (torch.tensor(s["ortho_id"], dtype=torch.long),
                torch.tensor(s["sp_gene_id"], dtype=torch.long),
                torch.tensor(s["ctx_feat"], dtype=torch.float32),
                torch.tensor(s["cond_id"], dtype=torch.long),
                torch.tensor(s["score"], dtype=torch.float32),
                s["species"], s["cell"], s["domain"], s["mark"])

def collate(b):
    o=torch.stack([x[0] for x in b]); sg=torch.stack([x[1] for x in b])
    c=torch.stack([x[2] for x in b]); co=torch.stack([x[3] for x in b]); sc=torch.stack([x[4] for x in b])
    return o,sg,c,co,sc,[x[5] for x in b],[x[6] for x in b],[x[7] for x in b],[x[8] for x in b]

def ev(model, td):
    model.eval(); rr = {}
    for dm, ss in td.items():
        if not ss: continue
        dl = torch.utils.data.DataLoader(DS(ss), batch_size=512, collate_fn=collate)
        pp, tt = [], []
        with torch.no_grad():
            for o,sg,c,co,sc,sp,cl,_,mk in dl:
                o,sg,c,co = [x.to(device) for x in [o,sg,c,co]]
                pr = model(o,sg,c,co,sp[0],cl[0],mk[0])
                pp.extend(pr.cpu().numpy()); tt.extend(sc.numpy())
        pp = np.array(pp); tt = np.array(tt)
        rr[dm] = float(np.corrcoef(pp,tt)[0,1]) if len(set(tt))>1 else 0.0
    return rr

cfg = MultiGRNConfig(); cfg.n_ortho_groups = 30000
model = EngramMultiGRN(cfg.n_ortho_groups, d_ctx=64).to(device)
model.add_species("human", 20000, 64); model.add_species("mouse", 27000, 64)
print(f"Params:{sum(p.numel() for p in model.parameters()):,}", flush=True)
crit = nn.MSELoss(); t0 = time.time()

print("\n=== STAGE 1: Human HCC ===", flush=True)
htr = [s for s in tr_s if s["domain"]=="human_hepg2"]
hdl = torch.utils.data.DataLoader(DS(htr), batch_size=256, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
for ep in range(50):
    model.train()
    for o,sg,c,co,sc,sp,cl,_,mk in hdl:
        o,sg,c,co,sc = [x.to(device) for x in [o,sg,c,co,sc]]
        opt.zero_grad(); crit(model(o,sg,c,co,sp[0],cl[0],mk[0]),sc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (ep+1)%10==0:
        r = ev(model, {"human_hepg2":td["human_hepg2"]})
        print(f"  ep{ep+1}/50 HCC R={r['human_hepg2']:.4f} ({time.time()-t0:.0f}s)", flush=True)
r1 = ev(model, td)
print(f"Stage1: {' '.join(f'{k}={v:.4f}' for k,v in r1.items())}", flush=True)

print("\n=== STAGE 2: +H1299 (same species, full tuning) ===", flush=True)
h99tr = [s for s in tr_s if s["domain"]=="human_h1299"]
h99dl = torch.utils.data.DataLoader(DS(h99tr), batch_size=256, shuffle=True, collate_fn=collate)
opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
for ep in range(50):
    model.train()
    for o,sg,c,co,sc,sp,cl,_,mk in h99dl:
        o,sg,c,co,sc = [x.to(device) for x in [o,sg,c,co,sc]]
        opt2.zero_grad(); crit(model(o,sg,c,co,sp[0],cl[0],mk[0]),sc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt2.step()
    if (ep+1)%10==0:
        r = ev(model, {"human_hepg2":td["human_hepg2"], "human_h1299":td["human_h1299"]})
        print(f"  ep{ep+1}/50 hcc={r['human_hepg2']:.4f} h1299={r['human_h1299']:.4f}", flush=True)
r2 = ev(model, td)
print(f"Stage2: {' '.join(f'{k}={v:.4f}' for k,v in r2.items())}", flush=True)

print("\n=== STAGE 3: +Mouse BMDM (no freeze, hash isolation protects domains) ===", flush=True)
# No parameter freezing. domain_key isolates each species_cell_mark's memory slots.
# Shared embeddings can adapt to new domains without corrupting existing memory entries.
btr = [s for s in tr_s if s["domain"]=="mouse_bmdm"]
bdl = torch.utils.data.DataLoader(DS(btr), batch_size=256, shuffle=True, collate_fn=collate)
opt3 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
for ep in range(50):
    model.train()
    for o,sg,c,co,sc,sp,cl,_,mk in bdl:
        o,sg,c,co,sc = [x.to(device) for x in [o,sg,c,co,sc]]
        opt3.zero_grad(); crit(model(o,sg,c,co,sp[0],cl[0],mk[0]),sc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt3.step()
    if (ep+1)%10==0:
        r = ev(model, td)
        print(f"  ep{ep+1}/50 hcc={r['human_hepg2']:.4f} h1299={r['human_h1299']:.4f} bmdm={r['mouse_bmdm']:.4f}", flush=True)
r3 = ev(model, td)
print(f"\n{'='*60}\nFINAL\n{'='*60}", flush=True)
for k in sorted(r3.keys()):
    b = r1.get(k,0); a = r3.get(k,0)
    print(f"  {k:20s}: {b:.4f} -> {a:.4f} d={a-b:+.4f}", flush=True)

json.dump({"s1":{k:float(v) for k,v in r1.items()},"s2":{k:float(v) for k,v in r2.items()},"s3":{k:float(v) for k,v in r3.items()}},
    open(DATA/"multigrn_v2_results.json","w"), indent=2)
print("\nDone", flush=True)
