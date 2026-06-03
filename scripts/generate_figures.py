#!/usr/bin/env python3
"""Figure 5 v2: Multi-dataset incremental learning — MultiGRN vs GNN vs MLP."""

import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
fd_multi = json.load(open(DATA / "baseline_comparison" / "all_models_incremental.json"))
fd_fig  = json.load(open(DATA / "figure_10cond" / "figure_data.json"))

# ── MATPLOTLIB STYLE ──
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":6.5,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.6,
    "legend.frameon":False,"xtick.major.width":0.5,"ytick.major.width":0.5,
    "xtick.major.size":2,"ytick.major.size":2,
})

C = {"blue":"#1B5A9C","green":"#2E8B57","red":"#C44E52","orange":"#D98C2E",
     "grey":"#666666","grey_l":"#BBBBBB","white":"#FFFFFF"}

# ── DATA ──
cond_list = fd_fig["conditions"]
meta = fd_fig["condition_metadata"]
# Extract per-model results
def get_matrix(model_name):
    evals = sorted(set(r['eval'] for r in fd_multi[model_name]))
    stages = sorted(set(r['stage'] for r in fd_multi[model_name]))
    mat = np.zeros((len(evals), len(stages)))
    for i,ec in enumerate(evals):
        for j,s in enumerate(stages):
            rr = [r['R'] for r in fd_multi[model_name] if r['eval']==ec and r['stage']==s]
            mat[i][j] = rr[0] if rr else 0.0
    return evals, stages, mat

evals_m, stages_m, mat_multi = get_matrix("MultiGRN")
evals_g, stages_g, mat_gnn   = get_matrix("GNN")
evals_l, stages_l, mat_mlp   = get_matrix("MLP")

# H3K18la forgetting: track CON across stages
def track_cond(model_name, cond="H3K18la_CON"):
    stages = sorted(set(r['stage'] for r in fd_multi[model_name]))
    vals = []
    for s in stages:
        rr = [r['R'] for r in fd_multi[model_name] if r['eval']==cond and r['stage']==s]
        vals.append(rr[0] if rr else 0.0)
    return stages, vals

# Short labels
short = lambda ec: ec.replace("H3K18la_","").replace("H4K5la_","").replace("_TSS","")

# ── BUILD FIGURE ──
fig = plt.figure(figsize=(7.48, 8.0), dpi=300, facecolor="white")
gs = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.38,
                       height_ratios=[1, 0.85, 0.85])

# ═══ PANEL A: MultiGRN Heatmap (hero) ═══
ax_a = fig.add_subplot(gs[0, :])
cmap = LinearSegmentedColormap.from_list("div",[(0,C["red"]),(0.38,C["white"]),(0.55,C["grey_l"]),(0.7,C["green"]),(1,C["green"])],N=256)
im = ax_a.imshow(mat_multi, cmap=cmap, aspect="auto", vmin=-0.05, vmax=0.90)
ax_a.set_xticks(range(len(stages_m)))
ax_a.set_xticklabels([f"S{s}" for s in stages_m], fontsize=5.5)
ax_a.set_yticks(range(len(evals_m)))
ax_a.set_yticklabels([short(ec) for ec in evals_m], fontsize=5.2)
ax_a.set_xlabel("Training Stage", fontsize=6, fontweight="bold")
cbar = plt.colorbar(im, ax=ax_a, shrink=0.92, aspect=25, pad=0.02)
cbar.set_label("Pearson R", fontsize=5.5); cbar.ax.tick_params(labelsize=5)
for i in range(len(evals_m)):
    for j in range(len(stages_m)):
        v = mat_multi[i][j]
        if abs(v) > 0.1:
            ax_a.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=4.2,
                     color="white" if abs(v)>0.55 else "black", weight="bold")
# Block outlines
for bs,be,lc,lab in [(0,2.5,C["orange"],"H3K18la peak-gene"),(3,5.5,C["red"],"H3K18la TSS"),(6,9.5,C["blue"],"H4K5la")]:
    ax_a.add_patch(plt.Rectangle((-0.5,bs-0.5),len(stages_m),be-bs,linewidth=1.3,edgecolor=lc,facecolor="none",linestyle="-",clip_on=False))
ax_a.set_title("A  MultiGRN: Domain-Isolated Incremental Learning (10 conditions)",fontsize=7,fontweight="bold",loc="left")

# ═══ PANEL B: MLP Heatmap ═══
ax_b = fig.add_subplot(gs[1, 0])
im_b = ax_b.imshow(mat_mlp, cmap=cmap, aspect="auto", vmin=-0.05, vmax=0.90)
ax_b.set_xticks(range(len(stages_l)))
ax_b.set_xticklabels([f"S{s}" for s in stages_l], fontsize=5.5)
ax_b.set_yticks(range(len(evals_l)))
ax_b.set_yticklabels([short(ec) for ec in evals_l], fontsize=5.2)
ax_b.set_xlabel("Training Stage", fontsize=6, fontweight="bold")
for i in range(len(evals_l)):
    for j in range(len(stages_l)):
        v = mat_mlp[i][j]
        if abs(v) > 0.04: ax_b.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=4.2, color="black" if abs(v)<0.6 else "white", weight="bold")
ax_b.set_title("B  MLP: Context-Features Only (No Gene Embedding, No Memory)",fontsize=6.5,fontweight="bold",loc="left")

# ═══ PANEL C: GNN Heatmap ═══
ax_c = fig.add_subplot(gs[1, 1])
im_c = ax_c.imshow(mat_gnn, cmap=cmap, aspect="auto", vmin=-0.05, vmax=0.90)
ax_c.set_xticks(range(len(stages_g)))
ax_c.set_xticklabels([f"S{s}" for s in stages_g], fontsize=5.5)
ax_c.set_yticks(range(len(evals_g)))
ax_c.set_yticklabels([short(ec) for ec in evals_g], fontsize=5.2)
ax_c.set_xlabel("Training Stage", fontsize=6, fontweight="bold")
for i in range(len(evals_g)):
    for j in range(len(stages_g)):
        v = mat_gnn[i][j]
        if abs(v) > 0.04: ax_c.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=4.2, color="black" if abs(v)<0.6 else "white", weight="bold")
ax_c.set_title("C  GNN: Gene Embedding + Context (No Domain Isolation)",fontsize=6.5,fontweight="bold",loc="left")

# ═══ PANEL D: Forgetting Trajectory (H3K18la_CON) ═══
ax_d = fig.add_subplot(gs[2, 0])
colors_d = [C["green"], C["orange"], C["red"]]
markers_d = ["o", "s", "D"]
for mi,(mn,clr,mkr) in enumerate(zip(["MultiGRN","GNN","MLP"],colors_d,markers_d)):
    st,vl = track_cond(mn, "H3K18la_CON")
    ax_d.plot(st, vl, color=clr, marker=mkr, markersize=3.5, linewidth=1.0, label=mn, markevery=1)
ax_d.axvline(x=3.5, color=C["grey_l"], linestyle="--", linewidth=0.7)
ax_d.text(3.7, 0.80, "bigWig\nstarts", fontsize=4.5, color=C["grey"], va="top")
ax_d.set_xlabel("Training Stage", fontsize=6, fontweight="bold")
ax_d.set_ylabel("Pearson R (H3K18la_CON test)", fontsize=6, fontweight="bold")
ax_d.set_ylim(-0.1, 0.90); ax_d.legend(fontsize=5)
ax_d.set_title("D  Forgetting: H3K18la_CON Retention",fontsize=6.5,fontweight="bold",loc="left")

# ═══ PANEL E: Self-Training R Comparison (Bar) ═══
ax_e = fig.add_subplot(gs[2, 1])
clist_short = [short(c) for c in cond_list]
bar_w = 0.22; x = np.arange(len(cond_list))
for bi,(mn,clr,off) in enumerate(zip(["MultiGRN","GNN","MLP"],[C["blue"],C["orange"],C["red"]],[-bar_w,0,bar_w])):
    vals = []
    for cn in cond_list:
        sr=next((r['R'] for r in fd_multi[mn] if r['trained_on']==cn and r['eval']==cn),0)
        vals.append(sr)
    ax_e.bar(x+off, vals, bar_w, color=clr, label=mn, edgecolor="white", linewidth=0.3)
ax_e.set_xticks(x); ax_e.set_xticklabels(clist_short, fontsize=4.5, rotation=45, ha="right")
ax_e.set_ylabel("Self-Training R", fontsize=6, fontweight="bold")
ax_e.legend(fontsize=5.5, loc="upper right")
ax_e.set_ylim(-0.05, 0.95)
ax_e.axhline(y=0, color=C["grey_l"], linewidth=0.5, linestyle="-")
ax_e.set_title("E  Self-Training Performance by Condition",fontsize=6.5,fontweight="bold",loc="left")

# ═══ EXPORT ═══
out_dir = Path("figures")
for fmt in ["png","svg","pdf"]:
    path = out_dir / f"fig5_multi_dataset_v2.{fmt}"
    fig.savefig(str(path), bbox_inches="tight", dpi=600 if fmt=="png" else None, facecolor="white", edgecolor="none")
    print(f"Saved: {path}")
fig.savefig(str(out_dir / "fig5_multi_dataset_v2.tiff"), bbox_inches="tight", dpi=600, facecolor="white", edgecolor="none", pil_kwargs={"compression":"tiff_lzw"})
print(f"Saved: {out_dir / 'fig5_multi_dataset_v2.tiff'}")
plt.close()
print("Done.")
