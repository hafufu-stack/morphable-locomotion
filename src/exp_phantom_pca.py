"""
Phantom Synchronization: PCA Brain Analysis
============================================
Mechanism explanation for WHY Phantom Sync works.
Records hidden layer activations (32 units) for Body 0 and Body 2
during 5:1:5 simulation, then PCA-projects to 2D.

If bodies trace identical neural manifolds, it proves sync is
deterministic computation, not physical coupling.
"""
import numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import os, sys, json

# Add src to path to import season6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from season6_experiments import build_bodies, evolve_3body

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures"); RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN=32; INPUT_SIZE=7

@torch.no_grad()
def phantom_replay_pca(genes, data, nsteps, mass_ratios, cut_step=None):
    """Replay 5:1:5 and record hidden activations for each body."""
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt; OUT=3; NW1=INPUT_SIZE*HIDDEN
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    g=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gi=0;W1=g[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
    b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2=g[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
    fv=g[:,gi].abs().item()
    bid_vals=bit.float()/2.0; bid=bid_vals.unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    masks=[(bit==b) for b in range(3)]
    idxs=[masks[b].nonzero(as_tuple=True)[0] for b in range(3)]
    cd01=False;cd12=False;comb=torch.zeros(B,1,1,device=DEVICE)
    mass=torch.ones(B,N,device=DEVICE)
    for b in range(3): mass[:,masks[b]]=mass_ratios[b]
    # Storage for hidden activations (mean over particles per body)
    h_act = {0:[], 1:[], 2:[]}
    fx = {0:[], 1:[], 2:[]}
    for step in range(nsteps):
        t=step*DT
        if not cd01 and step%10==0:
            p0=pos[0,idxs[0]];p1=pos[0,idxs[1]];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,idxs[0][cl[:nn_,0]]]);csb=torch.cat([csb,idxs[1][cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]]); cd01=True
        if not cd12 and step%10==0:
            p1=pos[0,idxs[1]];p2=pos[0,idxs[2]];ds=torch.cdist(p1,p2)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,idxs[1][cl[:nn_,0]]]);csb=torch.cat([csb,idxs[2][cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]]); cd12=True
        if cd01 or cd12: comb=torch.ones(B,1,1,device=DEVICE)
        # CUT SPRINGS
        if cut_step is not None and step==cut_step:
            csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
        sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
        st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        # Record hidden activations BEFORE output
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g)  # shape: (1, N, 32)
        o=torch.tanh(torch.bmm(h,W2)+b2g)
        # Mean hidden activation per body
        for b in range(3):
            h_mean = h[0, masks[b]].mean(dim=0).cpu().numpy()  # (32,)
            h_act[b].append(h_mean)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        for b in range(3): fx[b].append(ext[0,masks[b],0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        acc=f/(mass.clamp(min=0.01).unsqueeze(2))
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    # Convert to arrays
    for b in range(3): h_act[b]=np.array(h_act[b])  # (nsteps, 32)
    pairs={}
    for(a,b_)in[(0,1),(1,2),(0,2)]:
        if np.std(fx[a])<1e-10 or np.std(fx[b_])<1e-10: pairs[f"r{a}{b_}"]=0.0
        else: r_,_=pearsonr(fx[a],fx[b_]);pairs[f"r{a}{b_}"]=round(r_,3)
    return h_act, pairs, fx


def make_pca_figure(h_nocut, h_cut, pairs_nocut, pairs_cut, fx_nocut, fx_cut, cut_step):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Phantom Synchronization: Neural Mechanism Analysis\n"
                 "5:1:5 Mass Ratio — Hidden Layer (32 units) PCA Projection",
                 fontsize=14, fontweight="bold")

    # Row 1: PCA trajectories
    # Fit PCA on combined Body 0 + Body 2 data (no cut)
    combined = np.vstack([h_nocut[0], h_nocut[2]])
    pca = PCA(n_components=2)
    pca.fit(combined)

    # Panel 1: No cut PCA
    ax1 = fig.add_subplot(2, 3, 1)
    proj0 = pca.transform(h_nocut[0])
    proj1 = pca.transform(h_nocut[1])
    proj2 = pca.transform(h_nocut[2])
    nsteps = len(proj0)
    cols = plt.cm.viridis(np.linspace(0, 1, nsteps))
    ax1.scatter(proj0[:,0], proj0[:,1], c=cols, s=3, alpha=0.7, label="Body 0 (m=5)")
    ax1.scatter(proj2[:,0], proj2[:,1], c=cols, s=3, alpha=0.7, marker="x", label="Body 2 (m=5)")
    ax1.scatter(proj1[:,0], proj1[:,1], c=cols, s=3, alpha=0.3, marker="^", label="Body 1 (m=1)")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.set_title(f"No Cut (control)\nr02={pairs_nocut['r02']}")
    ax1.legend(fontsize=7, markerscale=3); ax1.grid(alpha=0.3)

    # Panel 2: Cut PCA
    ax2 = fig.add_subplot(2, 3, 2)
    proj0c = pca.transform(h_cut[0])
    proj1c = pca.transform(h_cut[1])
    proj2c = pca.transform(h_cut[2])
    # Before cut
    ax2.scatter(proj0c[:cut_step,0], proj0c[:cut_step,1], c=cols[:cut_step], s=3, alpha=0.5, label="B0 pre-cut")
    ax2.scatter(proj2c[:cut_step,0], proj2c[:cut_step,1], c=cols[:cut_step], s=3, alpha=0.5, marker="x", label="B2 pre-cut")
    # After cut
    ax2.scatter(proj0c[cut_step:,0], proj0c[cut_step:,1], c=cols[cut_step:], s=8, alpha=0.9, label="B0 POST-CUT")
    ax2.scatter(proj2c[cut_step:,0], proj2c[cut_step:,1], c=cols[cut_step:], s=8, alpha=0.9, marker="x", label="B2 POST-CUT")
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.2)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title(f"Cut at step {cut_step}\nr02={pairs_cut['r02']}")
    ax2.legend(fontsize=6, markerscale=2); ax2.grid(alpha=0.3)

    # Panel 3: Overlay — Body 0 vs Body 2 difference
    ax3 = fig.add_subplot(2, 3, 3)
    diff_nocut = np.linalg.norm(h_nocut[0] - h_nocut[2], axis=1)
    diff_cut = np.linalg.norm(h_cut[0] - h_cut[2], axis=1)
    diff_01_nocut = np.linalg.norm(h_nocut[0] - h_nocut[1], axis=1)
    ax3.plot(diff_nocut, color="#3498db", alpha=0.7, linewidth=0.8, label=f"B0-B2 no cut (mean={diff_nocut.mean():.3f})")
    ax3.plot(diff_cut, color="#e74c3c", alpha=0.7, linewidth=0.8, label=f"B0-B2 cut@{cut_step} (mean={diff_cut.mean():.3f})")
    ax3.plot(diff_01_nocut, color="#95a5a6", alpha=0.4, linewidth=0.5, label=f"B0-B1 no cut (mean={diff_01_nocut.mean():.3f})")
    ax3.axvline(x=cut_step, color="red", linestyle="--", alpha=0.5, label=f"Cut point")
    ax3.set_xlabel("Timestep"); ax3.set_ylabel("||h(Body0) - h(BodyX)|| (L2)")
    ax3.set_title("Neural Distance Over Time\n(low = same brain state)")
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # Row 2: Force output traces
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(fx_nocut[0], color="#e74c3c", alpha=0.5, linewidth=0.5, label="B0 Fx")
    ax4.plot(fx_nocut[2], color="#3498db", alpha=0.5, linewidth=0.5, label="B2 Fx")
    ax4.set_xlabel("Step"); ax4.set_ylabel("Fx (mean)")
    ax4.set_title(f"Force Output (No Cut)\nr02={pairs_nocut['r02']}")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(fx_cut[0], color="#e74c3c", alpha=0.5, linewidth=0.5, label="B0 Fx")
    ax5.plot(fx_cut[2], color="#3498db", alpha=0.5, linewidth=0.5, label="B2 Fx")
    ax5.axvline(x=cut_step, color="red", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Step"); ax5.set_ylabel("Fx (mean)")
    ax5.set_title(f"Force Output (Cut@{cut_step})\nr02={pairs_cut['r02']}")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # Panel 6: PCA variance explained
    ax6 = fig.add_subplot(2, 3, 6)
    pca_full = PCA(n_components=min(32, combined.shape[0]))
    pca_full.fit(combined)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax6.bar(range(len(cumvar)), pca_full.explained_variance_ratio_, color="#3498db", alpha=0.7, label="Individual")
    ax6.plot(range(len(cumvar)), cumvar, "o-", color="#e74c3c", markersize=4, label="Cumulative")
    ax6.set_xlabel("Principal Component"); ax6.set_ylabel("Variance Explained")
    ax6.set_title(f"PCA Spectrum\n({cumvar[1]:.1%} in 2 PCs, {cumvar[4]:.1%} in 5 PCs)")
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(OUTPUT_DIR, "phantom_pca_analysis.png")
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    print(f"\nFigure: {fp}")


if __name__ == "__main__":
    print("="*70)
    print("PHANTOM SYNCHRONIZATION: PCA BRAIN ANALYSIS")
    print("="*70)

    NSTEPS=600; GAP=0.5; PSZ=200; NGENS=300
    gx,gy,gz,sp = 10,5,4,0.35
    mrs = (5.0, 1.0, 5.0)
    CUT_STEP = 300

    # Evolve a 5:1:5 controller
    print("\nEvolving 5:1:5 controller...")
    torch.manual_seed(42); np.random.seed(42)
    data3 = build_bodies(gx,gy,gz,sp,GAP,3)
    best_genes = evolve_3body(data3, NSTEPS, NGENS, PSZ, "pca_5:1:5", mrs)

    # Replay: no cut
    print("\nReplaying without cut...")
    h_nocut, pairs_nocut, fx_nocut = phantom_replay_pca(best_genes, data3, NSTEPS, mrs, cut_step=None)
    print(f"  No cut: r01={pairs_nocut['r01']}, r12={pairs_nocut['r12']}, r02={pairs_nocut['r02']}")

    # Replay: cut at 300
    print(f"\nReplaying with cut at step {CUT_STEP}...")
    h_cut, pairs_cut, fx_cut = phantom_replay_pca(best_genes, data3, NSTEPS, mrs, cut_step=CUT_STEP)
    print(f"  Cut@{CUT_STEP}: r01={pairs_cut['r01']}, r12={pairs_cut['r12']}, r02={pairs_cut['r02']}")

    # Neural distance analysis
    dist_nocut = np.linalg.norm(h_nocut[0] - h_nocut[2], axis=1)
    dist_cut_before = np.linalg.norm(h_cut[0][:CUT_STEP] - h_cut[2][:CUT_STEP], axis=1)
    dist_cut_after = np.linalg.norm(h_cut[0][CUT_STEP:] - h_cut[2][CUT_STEP:], axis=1)

    print(f"\n  Neural distance B0-B2:")
    print(f"    No cut mean: {dist_nocut.mean():.6f}")
    print(f"    Cut before:  {dist_cut_before.mean():.6f}")
    print(f"    Cut after:   {dist_cut_after.mean():.6f}")
    print(f"    Ratio (after/before): {dist_cut_after.mean()/max(dist_cut_before.mean(),1e-10):.3f}")

    # PCA explained variance
    combined = np.vstack([h_nocut[0], h_nocut[2]])
    pca = PCA(n_components=min(10, combined.shape[0]))
    pca.fit(combined)
    print(f"\n  PCA variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
          f"2PCs={sum(pca.explained_variance_ratio_[:2]):.3f}")

    make_pca_figure(h_nocut, h_cut, pairs_nocut, pairs_cut, fx_nocut, fx_cut, CUT_STEP)

    # Save results
    results = {
        "pairs_nocut": pairs_nocut, "pairs_cut": pairs_cut,
        "neural_dist_nocut_mean": float(dist_nocut.mean()),
        "neural_dist_cut_before_mean": float(dist_cut_before.mean()),
        "neural_dist_cut_after_mean": float(dist_cut_after.mean()),
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    }
    log_path = os.path.join(RESULTS_DIR, "phantom_pca_log.json")
    with open(log_path, "w") as f: json.dump(results, f, indent=2)
    print(f"Log: {log_path}")

    print(f"\n{'='*70}")
    print("PCA BRAIN ANALYSIS COMPLETE")
    print(f"{'='*70}")

    try:
        import winsound
        for _ in range(3): winsound.Beep(800, 300); import time; time.sleep(0.2)
    except: pass
