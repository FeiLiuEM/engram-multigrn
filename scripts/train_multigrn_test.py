"""MultiGRN test: HCC H4K5la -> HepG2 H3K27ac cross-mark."""
import sys, os, json, random, time, gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch, torch.nn as nn
from pathlib import Path; from collections import defaultdict

DATA=Path(__file__).parent.parent/"data"; device=torch.device("cuda")
from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig

feat=json.load(open(DATA/"multigrn_features_full.json"))
human_oid=feat["human_to_orthoid"]; mouse_oid=feat["mouse_to_orthoid"]

from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder

vocab_h=GeneVocabulary(str(DATA));vocab_h.load()
ctx_h=RegulatoryContextBuilder(vocab_h,str(DATA));ctx_h.load()
human_ctx_cache={}
for gene in list(vocab_h.gene_to_idx.keys())[:15000]:
    gid=vocab_h.gene_to_idx.get(gene,0)
    if gid==0: human_ctx_cache[gene]=[0.0]*64;continue
    c=ctx_h.get_context_for_genes(torch.tensor([gid]),max_context=3).squeeze(0).float()
    if c.shape[0]<64:c=nn.functional.pad(c,(0,64-c.shape[0]))
    human_ctx_cache[gene]=c[:64].tolist()

# ═══ Data ═══
random.seed(42); samples=[]
cd=json.load(open(DATA/"kla_chip_scores.json"))
cm={"H4K5la_NM2":0,"H4K5la_NM3":0,"H4K5la_LAC2":1,"H4K5la_LAC3":1}
cv_=defaultdict(list)
for k,s in cd.items():g,c=k.split("__");cv_[c].append(s)

def mk(g,s,ci,sp,cl,dom,mk_t):
    oid=human_oid.get(g,mouse_oid.get(g,0))
    sid=hash(g)%19000+1
    cf=human_ctx_cache.get(g,[0.0]*64)
    if len(cf)<64:cf=cf+[0.0]*(64-len(cf))
    return{"ortho_id":oid,"sp_gene_id":sid,"score":s,"cond_id":ci,
           "species":sp,"cell":cl,"domain":dom,"mark":mk_t,"ctx_feat":cf[:64]}

for k,raw in cd.items():
    g,c=k.split("__")
    if c not in cm:continue
    vs=cv_[c];mn,mx=min(vs),max(vs);n=(raw-mn)/(mx-mn) if mx>mn else 0.5
    samples.append(mk(g,min(1,max(0,n)),cm[c],"human","hepg2","human_hepg2","H4K5la"))

# H3K27ac
k27=json.load(open(DATA/"hepg2_h3k27ac_scores.json"))
for g,s in k27.items():samples.append(mk(g,s,4,"human","hepg2","human_hepg2_k27","H3K27ac"))

random.shuffle(samples)
sp=int(len(samples)*0.8);tr_s,te_s=samples[:sp],samples[sp:]
td=defaultdict(list)
for s in te_s:td[s["domain"]].append(s)
print(f"Total:{len(samples)} | H4K5la:{sum(1 for s in tr_s if s['mark']=='H4K5la')} H3K27ac:{sum(1 for s in tr_s if s['mark']=='H3K27ac')}",flush=True)

D=torch.utils.data.Dataset
class DS(D):
    def __init__(self,s):self.s=s
    def __len__(self):return len(self.s)
    def __getitem__(self,i):
        s=self.s[i]
        return(torch.tensor(s["ortho_id"],dtype=torch.long),torch.tensor(s["sp_gene_id"],dtype=torch.long),
               torch.tensor(s["ctx_feat"],dtype=torch.float32),torch.tensor(s["cond_id"],dtype=torch.long),
               torch.tensor(s["score"],dtype=torch.float32),s["species"],s["cell"],s["domain"],s["mark"])

def collate(b):
    o=torch.stack([x[0] for x in b]);sg=torch.stack([x[1] for x in b])
    c=torch.stack([x[2] for x in b]);co=torch.stack([x[3] for x in b]);sc=torch.stack([x[4] for x in b])
    return o,sg,c,co,sc,[x[5] for x in b],[x[6] for x in b],[x[7] for x in b],[x[8] for x in b]

def ev(model,td):
    model.eval();rr={}
    for dm,ss in td.items():
        if not ss:continue
        dl=torch.utils.data.DataLoader(DS(ss),batch_size=512,collate_fn=collate)
        pp,tt=[],[]
        with torch.no_grad():
            for o,sg,c,co,sc,sp,cl,_,mx in dl:
                o,sg,c,co=[x.to(device) for x in [o,sg,c,co]]
                pr=model(o,sg,c,co,sp[0],cl[0],mx[0])
                pp.extend(pr.cpu().numpy());tt.extend(sc.numpy())
        pp=np.array(pp);tt=np.array(tt)
        rr[dm]=float(np.corrcoef(pp,tt)[0,1]) if len(set(tt))>1 else 0.0
    return rr

# ═══ Model ═══
cfg=MultiGRNConfig();cfg.n_ortho_groups=feat["n_ortho_groups"]+1000
model=EngramMultiGRN(cfg.n_ortho_groups,d_ctx=64).to(device)
model.add_species("human",feat["human_vocab_size"],64)
print(f"Params:{sum(p.numel() for p in model.parameters()):,}",flush=True)
crit=nn.MSELoss();t0=time.time()

# Stage 1: HCC H4K5la
print("\n=== STAGE 1: HCC H4K5la ===",flush=True)
htr=[s for s in tr_s if s["domain"]=="human_hepg2"]
hdl=torch.utils.data.DataLoader(DS(htr),batch_size=256,shuffle=True,collate_fn=collate)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
for ep in range(50):
    model.train()
    for o,sg,c,co,sc,sp,cl,_,mx in hdl:
        o,sg,c,co,sc=[x.to(device) for x in [o,sg,c,co,sc]]
        opt.zero_grad();crit(model(o,sg,c,co,sp[0],cl[0],mx[0]),sc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    if (ep+1)%10==0:
        r=ev(model,{"human_hepg2":td["human_hepg2"]})
        print(f"  ep{ep+1}/50 HCC_H4K5la R={r['human_hepg2']:.4f} ({time.time()-t0:.0f}s)",flush=True)
r1=ev(model,td)
print(f"S1: {' '.join(f'{k}={v:.4f}' for k,v in r1.items())}",flush=True)

# Stage 2: +HepG2 H3K27ac (same cell, different mark)
print("\n=== STAGE 2: +HepG2 H3K27ac ===",flush=True)
k27tr=[s for s in tr_s if s["domain"]=="human_hepg2_k27"]
k27dl=torch.utils.data.DataLoader(DS(k27tr),batch_size=256,shuffle=True,collate_fn=collate)
opt2=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
for ep in range(50):
    model.train()
    for o,sg,c,co,sc,sp,cl,_,mx in k27dl:
        o,sg,c,co,sc=[x.to(device) for x in [o,sg,c,co,sc]]
        opt2.zero_grad();crit(model(o,sg,c,co,sp[0],cl[0],mx[0]),sc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt2.step()
    if (ep+1)%10==0:
        r=ev(model,td)
        h4=r.get("human_hepg2",0);k27=r.get("human_hepg2_k27",0)
        print(f"  ep{ep+1}/50 H4K5la={h4:.4f} H3K27ac={k27:.4f}",flush=True)
r2=ev(model,td)

print(f"\n{'='*60}\nCROSS-MARK TEST: H4K5la vs H3K27ac\n{'='*60}",flush=True)
for k in sorted(r2.keys()):
    b=r1.get(k,0);a=r2.get(k,0)
    print(f"  {k:25s}: {b:.4f} -> {a:.4f} d={a-b:+.4f}")
# H3K27ac Stage 1 standalone (for comparison)
print(f"  {'H3K27ac (Stage 2 only)':25s}: -> {r2.get('human_hepg2_k27',0):.4f}",flush=True)

json.dump({"s1":{k:float(v) for k,v in r1.items()},"s2":{k:float(v) for k,v in r2.items()}},
    open(DATA/"multigrn_cross_mark_test.json","w"),indent=2)
print("\nDone",flush=True)
