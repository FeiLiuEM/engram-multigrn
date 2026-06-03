"""MultiGRN multi-gate: incremental test both directions."""
import sys, os, json, random, time, gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np; import torch, torch.nn as nn
from pathlib import Path; from collections import defaultdict
DATA=Path(__file__).parent.parent/"data"; device=torch.device("cuda")
from engram_grn.model.multigrn import EngramMultiGRN, MultiGRNConfig, EWC
from engram_grn.data_pipeline.gene_vocab import GeneVocabulary
from engram_grn.data_pipeline.regulatory_context import RegulatoryContextBuilder

feat=json.load(open(DATA/"multigrn_features_full.json"))
human_oid=feat["human_to_orthoid"]; mouse_oid=feat["mouse_to_orthoid"]
GFK=["gene_len","n_exons","n_introns","intron_len","cds_len","n_transcripts","exon_density","cds_ratio"]
P_DIM=72

vocab_h=GeneVocabulary(str(DATA));vocab_h.load()
ctx_h=RegulatoryContextBuilder(vocab_h,str(DATA));ctx_h.load()
human_ctx={}
for gene in list(vocab_h.gene_to_idx.keys())[:15000]:
    gid=vocab_h.gene_to_idx.get(gene,0)
    if gid==0: human_ctx[gene]=[0.0]*P_DIM;continue
    c=ctx_h.get_context_for_genes(torch.tensor([gid]),max_context=3).squeeze(0).float()
    if c.shape[0]<64:c=nn.functional.pad(c,(0,64-c.shape[0]))
    human_ctx[gene]=c[:64].tolist()

mouse_string=feat.get("mouse_string_feats_sample",{})
mouse_ctx={k:v if len(v)==64 else v+[0.0]*(64-len(v)) for k,v in mouse_string.items()}
hgfeats=json.load(open(DATA/"gene_genomic_features.json"))
mgfeats=json.load(open(DATA/"mouse_genomic_features.json"))

def mk_human_ctx(g):
    s=human_ctx.get(g,[0.0]*64);gf=hgfeats.get(g,{k:0.0 for k in GFK})
    return (s[:64] if len(s)>=64 else s+[0.0]*(64-len(s)))+[gf.get(k,0.0) for k in GFK]
def mk_mouse_ctx(g):
    s=mouse_ctx.get(g,[0.0]*64);gf=mgfeats.get(g,{k:0.0 for k in GFK})
    return (s[:64] if len(s)>=64 else s+[0.0]*(64-len(s)))+[gf.get(k,0.0) for k in GFK]

# Data
random.seed(42);samples1=[]
cd=json.load(open(DATA/"kla_chip_scores.json"))
cm={"H4K5la_NM2":0,"H4K5la_NM3":0,"H4K5la_LAC2":1,"H4K5la_LAC3":1}
def norm_log(vals):
    log=np.log2(np.array(vals)+1);mn,mx=log.min(),log.max()
    return [(l-mn)/(mx-mn) if mx>mn else 0.5 for l in log]
for c_name,c_idx in cm.items():
    cond_samples=[(k,raw) for k,raw in cd.items() if k.endswith(f"__{c_name}")]
    for (k,_),n in zip(cond_samples,norm_log([raw for _,raw in cond_samples])):
        g,_=k.split("__")
        samples1.append({"ortho_id":human_oid.get(g,0),"sp_gene_id":hash(g)%19000+1,
            "score":min(1,max(0,n)),"cond_id":c_idx,"species":"human","cell":"hepg2",
            "domain":"human_hepg2","mark":"H4K5la","ctx_feat":mk_human_ctx(g)})

expr=json.load(open(DATA/"gse219045_mouse_fpkm.json"))
overlap=[g for g in expr if g in mouse_oid and g in mouse_ctx]
scores=[expr[g] for g in overlap];mn,mx=min(scores),max(scores)
samples2=[]
for g in overlap:
    oid=mouse_oid.get(g,0);sid=hash(g)%26000+1
    samples2.append({"ortho_id":oid,"sp_gene_id":sid,"score":(expr[g]-mn)/(mx-mn) if mx>mn else 0.5,
        "cond_id":0,"species":"mouse","cell":"multitissue","domain":"mouse_fpkm","mark":"FPKM","ctx_feat":mk_mouse_ctx(g)})

random.shuffle(samples1);sp=int(len(samples1)*0.8);tr1,te1=samples1[:sp],samples1[sp:]
random.shuffle(samples2);sp=int(len(samples2)*0.8);tr2,te2=samples2[:sp],samples2[sp:]
td={"human_hepg2":te1,"mouse_fpkm":te2}

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

def train_stage(model, dl, opt, n_ep, label, eval_every=10):
    for ep in range(n_ep):
        model.train()
        for o,sg,c,co,sc,sp,cl,_,mx in dl:
            o,sg,c,co,sc=[x.to(device) for x in [o,sg,c,co,sc]]
            opt.zero_grad();crit(model(o,sg,c,co,sp[0],cl[0],mx[0]),sc).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (ep+1)%eval_every==0:
            r=ev(model,td)
            print(f"  {label} ep{ep+1}/{n_ep} H4K5la={r.get('human_hepg2',0):.4f} Mouse={r.get('mouse_fpkm',0):.4f}", flush=True)

def run_test(name, s1_data, s2_data, use_ewc=False):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    cfg=MultiGRNConfig();cfg.n_ortho_groups=feat["n_ortho_groups"]+1000
    model=EngramMultiGRN(cfg.n_ortho_groups,d_ctx=P_DIM,ctx_input_dim=P_DIM).to(device)
    model.add_species("human",feat["human_vocab_size"],P_DIM)
    model.add_species("mouse",feat.get("mouse_vocab_size",26264),P_DIM)
    
    dl1=torch.utils.data.DataLoader(DS(s1_data),batch_size=256,shuffle=True,collate_fn=collate)
    dl2=torch.utils.data.DataLoader(DS(s2_data),batch_size=256,shuffle=True,collate_fn=collate)
    crit=nn.MSELoss()
    
    # Stage 1
    train_stage(model, dl1, torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5), 50, "S1", 10)
    r1=ev(model,td)
    print(f"  >> S1 final: H4K5la={r1.get('human_hepg2',0):.4f} Mouse={r1.get('mouse_fpkm',0):.4f}")
    
    # Stage 2
    ewc=EWC(model,lambda_ewc=5000.0) if use_ewc else None
    if ewc:
        ewc.estimate_fisher(dl1, crit, device)
        print(f"  EWC: {len(ewc.params)} params protected, λ={ewc.lambda_ewc}")
    
    opt2=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    for ep in range(50):
        model.train()
        for o,sg,c,co,sc,sp,cl,_,mx in dl2:
            o,sg,c,co,sc=[x.to(device) for x in [o,sg,c,co,sc]]
            opt2.zero_grad()
            loss=crit(model(o,sg,c,co,sp[0],cl[0],mx[0]),sc)
            if ewc: loss=loss+ewc.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt2.step()
        if (ep+1)%10==0:
            r=ev(model,td)
            print(f"  S2 ep{ep+1}/50 H4K5la={r.get('human_hepg2',0):.4f} Mouse={r.get('mouse_fpkm',0):.4f}", flush=True)
    r2=ev(model,td)
    
    # Report
    print(f"  >> S2 final: H4K5la={r2.get('human_hepg2',0):.4f} Mouse={r2.get('mouse_fpkm',0):.4f}")
    h4_ret=r2.get('human_hepg2',0)/max(r1.get('human_hepg2',0),0.001)*100
    ms_ret=r2.get('mouse_fpkm',0)/max(r1.get('mouse_fpkm',0),0.001)*100
    print(f"  >> H4K5la Δ={r2.get('human_hepg2',0)-r1.get('human_hepg2',0):+.4f}  retention={h4_ret:.1f}%")
    print(f"  >> Mouse   Δ={r2.get('mouse_fpkm',0)-r1.get('mouse_fpkm',0):+.4f}  retention={ms_ret:.1f}%")
    return r1, r2

crit=nn.MSELoss()

# Test 1: Human → Mouse
r1a,r2a=run_test("HUMAN → MOUSE (multi-gate)", tr1, tr2, use_ewc=False)

# Test 2: Mouse → Human (with EWC)
r1b,r2b=run_test("MOUSE → HUMAN + EWC (multi-gate)", tr2, tr1, use_ewc=True)

print(f"\n{'='*70}")
print(f"MULTI-GATE SUMMARY")
print(f"{'='*70}")
print(f"  {'':30s}  {'S1':>8s}  {'S2':>8s}  {'Δ':>8s}  {'Ret':>6s}")
print(f"  {'Human→Mouse H4K5la':30s}  {r1a.get('human_hepg2',0):>8.4f}  {r2a.get('human_hepg2',0):>8.4f}  {r2a.get('human_hepg2',0)-r1a.get('human_hepg2',0):>+8.4f}")
ms_ret_a=r2a.get('mouse_fpkm',0)/max(r1a.get('mouse_fpkm',0),0.001)*100
print(f"  {'Human→Mouse FPKM':30s}  {r1a.get('mouse_fpkm',0):>8.4f}  {r2a.get('mouse_fpkm',0):>8.4f}  {r2a.get('mouse_fpkm',0)-r1a.get('mouse_fpkm',0):>+8.4f}  {ms_ret_a:>5.0f}%")
print(f"  {'Mouse→Human H4K5la':30s}  {r1b.get('human_hepg2',0):>8.4f}  {r2b.get('human_hepg2',0):>8.4f}  {r2b.get('human_hepg2',0)-r1b.get('human_hepg2',0):>+8.4f}")
ms_ret_b=r2b.get('mouse_fpkm',0)/max(r1b.get('mouse_fpkm',0),0.001)*100
print(f"  {'Mouse→Human FPKM':30s}  {r1b.get('mouse_fpkm',0):>8.4f}  {r2b.get('mouse_fpkm',0):>8.4f}  {r2b.get('mouse_fpkm',0)-r1b.get('mouse_fpkm',0):>+8.4f}  {ms_ret_b:>5.0f}%")
