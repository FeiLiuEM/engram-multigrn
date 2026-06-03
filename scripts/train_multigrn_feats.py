"""MultiGRN + genomic features: test cross-mark H4K5la -> H3K27ac."""
import sys, os, json, random, time, gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch, torch.nn as nn
from pathlib import Path; from collections import defaultdict

DATA=Path(__file__).parent.parent/"data"; device=torch.device("cuda")
from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig
from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder

feat=json.load(open(DATA/"multigrn_features_full.json"))
human_oid=feat["human_to_orthoid"]; mouse_oid=feat["mouse_to_orthoid"]

# STRING features
vocab_h=GeneVocabulary(str(DATA));vocab_h.load()
ctx_h=RegulatoryContextBuilder(vocab_h,str(DATA));ctx_h.load()
human_ctx_cache={}
for gene in list(vocab_h.gene_to_idx.keys())[:15000]:
    gid=vocab_h.gene_to_idx.get(gene,0)
    if gid==0: human_ctx_cache[gene]=[0.0]*64;continue
    c=ctx_h.get_context_for_genes(torch.tensor([gid]),max_context=3).squeeze(0).float()
    if c.shape[0]<64:c=nn.functional.pad(c,(0,64-c.shape[0]))
    human_ctx_cache[gene]=c[:64].tolist()

# Genomic features (8d)
gfeats=json.load(open(DATA/"gene_genomic_features.json"))
GFEAT_KEYS=["gene_len","n_exons","n_introns","intron_len","cds_len","n_transcripts","exon_density","cds_ratio"]

def get_ctx(g,sp="human"):
    """STRING(64) + genomic(8) = 72d"""
    s=human_ctx_cache.get(g,[0.0]*64)
    if len(s)<64:s=s+[0.0]*(64-len(s))
    s=s[:64]
    gf=gfeats.get(g,{k:0.0 for k in GFEAT_KEYS})
    gv=[gf.get(k,0.0) for k in GFEAT_KEYS]
    return s+gv

# Data
random.seed(42);samples=[]
cd=json.load(open(DATA/"kla_chip_scores.json"))
cm={"H4K5la_NM2":0,"H4K5la_NM3":0,"H4K5la_LAC2":1,"H4K5la_LAC3":1}
cv_=defaultdict(list)
for k,s in cd.items():g,c=k.split("__");cv_[c].append(s)

def mk(g,raw,ci,sp,cl,dom,mk_t):
    oid=human_oid.get(g,mouse_oid.get(g,0))
    sid=hash(g)%19000+1
    return{"ortho_id":oid,"sp_gene_id":sid,"score":raw,"cond_id":ci,
           "species":sp,"cell":cl,"domain":dom,"mark":mk_t,"ctx_feat":get_ctx(g,sp)}

def norm_log(vals):
    """log2(score+1) then min-max → [0,1]"""
    log=np.log2(np.array(vals)+1)
    mn,mx=log.min(),log.max()
    return [(l-mn)/(mx-mn) if mx>mn else 0.5 for l in log]

# HCC H4K5la (per-condition log normalization)
for c_name, c_idx in cm.items():
    cond_samples=[(k,raw) for k,raw in cd.items() if k.endswith(f"__{c_name}")]
    if not cond_samples: continue
    raw_vals=[raw for _,raw in cond_samples]
    log_vals=norm_log(raw_vals)
    for (k,raw),n in zip(cond_samples,log_vals):
        g,c_=k.split("__")
        samples.append(mk(g,min(1,max(0,n)),c_idx,"human","hepg2","human_hepg2","H4K5la"))

# H3K27ac (ENCODE narrowPeak)
k27=json.load(open(DATA/"hepg2_h3k27ac_scores.json"))
k27_log=norm_log(list(k27.values()))
for (g,raw),n in zip(k27.items(),k27_log):
    samples.append(mk(g,n,4,"human","hepg2","human_hepg2_k27","H3K27ac"))

random.shuffle(samples)
sp=int(len(samples)*0.8);tr_s,te_s=samples[:sp],samples[sp:]
td=defaultdict(list)
for s in te_s:td[s["domain"]].append(s)
n4=sum(1 for s in tr_s if s["mark"]=="H4K5la")
n27=sum(1 for s in tr_s if s["mark"]=="H3K27ac")
print(f"Total:{len(samples)} H4K5la:{n4} H3K27ac:{n27}",flush=True)

class DS(torch.utils.data.Dataset):
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

# Model with 72d ctx
cfg=MultiGRNConfig();cfg.n_ortho_groups=feat["n_ortho_groups"]+1000
model=EngramMultiGRN(cfg.n_ortho_groups,d_ctx=72,ctx_input_dim=72).to(device)
model.add_species("human",feat["human_vocab_size"],72)
print(f"Params:{sum(p.numel() for p in model.parameters()):,}",flush=True)
crit=nn.MSELoss();t0=time.time()

# Stage 1: HCC H4K5la
print("\n=== STAGE 1: HCC H4K5la (string+genomic) ===",flush=True)
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
        print(f"  ep{ep+1}/50 H4K5la R={r['human_hepg2']:.4f} ({time.time()-t0:.0f}s)",flush=True)
r1=ev(model,td)
print(f"S1: {' '.join(f'{k}={v:.4f}' for k,v in r1.items())}",flush=True)

# Stage 2: +HepG2 H3K27ac
print("\n=== STAGE 2: +HepG2 H3K27ac (string+genomic) ===",flush=True)
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

print(f"\n{'='*60}\nCROSS-MARK: H4K5la vs H3K27ac (+genomic feats)\n{'='*60}")
for k in sorted(r2.keys()):
    b=r1.get(k,0);a=r2.get(k,0)
    print(f"  {k:25s}: {b:.4f} -> {a:.4f} d={a-b:+.4f}")

# H3K27ac standalone training (for baseline)
print("\n=== STANDALONE: H3K27ac (scratch) ===",flush=True)
model2=EngramMultiGRN(cfg.n_ortho_groups,d_ctx=72,ctx_input_dim=72).to(device)
model2.add_species("human",feat["human_vocab_size"],72)
opt3=torch.optim.AdamW(model2.parameters(),lr=3e-4,weight_decay=1e-5)
for ep in range(50):
    model2.train()
    for o,sg,c,co,sc,sp,cl,_,mx in k27dl:
        o,sg,c,co,sc=[x.to(device) for x in [o,sg,c,co,sc]]
        opt3.zero_grad();crit(model2(o,sg,c,co,sp[0],cl[0],mx[0]),sc).backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(),1.0);opt3.step()
    if (ep+1)%10==0:
        r=ev(model2,{"human_hepg2_k27":td["human_hepg2_k27"]})
        print(f"  ep{ep+1}/50 H3K27ac (scratch) R={r['human_hepg2_k27']:.4f}",flush=True)
r3=ev(model2,td)
print(f"Standalone H3K27ac: {r3.get('human_hepg2_k27',0):.4f}",flush=True)

print(f"\nSummary:")
print(f"  H4K5la (string only):         0.806 (previous run)")
print(f"  H4K5la (+genomic feats):      {r1.get('human_hepg2',0):.4f}")
print(f"  H3K27ac (string only):        -0.009 (previous run)")
print(f"  H3K27ac (+genomic, incr):     {r2.get('human_hepg2_k27',0):.4f}")
print(f"  H3K27ac (+genomic, scratch):  {r3.get('human_hepg2_k27',0):.4f}")

json.dump({"s1":{k:float(v) for k,v in r1.items()},"s2":{k:float(v) for k,v in r2.items()},"s3":{k:float(v) for k,v in r3.items()}},
    open(DATA/"multigrn_genomic_test.json","w"),indent=2)
print("\nDone",flush=True)
