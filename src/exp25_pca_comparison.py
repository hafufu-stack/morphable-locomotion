"""
Exp 25: PCA Manifold Comparison — Neural Geometry of Symmetry Locks
===================================================================
Compare hidden-layer PCA trajectories across mass ratios:
  - 1:1 symmetric → single closed orbit (sync)
  - 3:1 asymmetric → partially split orbits
  - 10:1 extreme → fully separated trajectories (differentiation)

Provides the neural-geometric proof of the Symmetry Locks Theorem.
"""
import numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial import Delaunay
import os, time, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures"); RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN=32; INPUT_SIZE=7

def build_bodies_2(gx,gy,gz,sp,gap):
    nper=gx*gy*gz; nt=nper*2; ap=np.zeros((nt,3)); bi=np.zeros(nt,dtype=np.int64)
    bw=(gx-1)*sp; idx=0
    for b in range(2):
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp=(-(gap/2+bw)+x*sp) if b==0 else ((gap/2+bw)-x*sp)
                    ap[idx]=[xp,2.0+y*sp,z*sp-(gz-1)*sp/2]; bi[idx]=b; idx+=1
    sa,sb,rl=[],[],[]
    for b in range(2):
        m=np.where(bi==b)[0]; bp=ap[m]; tri=Delaunay(bp); edges=set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a);sb.append(bb);rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_=np.zeros_like(ap)
    for b in range(2):
        m=bi==b
        for d in range(3):
            vn,vx=ap[m,d].min(),ap[m,d].max(); np_[m,d]=2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap,np_,bi,np.array(sa),np.array(sb),np.array(rl),nper,nt


@torch.no_grad()
def evolve_2body(data, nsteps, ngens, psz, label, mass_ratios):
    """Evolve a shared-NN 2-body controller with given mass ratios."""
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt; OUT=3; NW1=INPUT_SIZE*HIDDEN
    NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1; s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    masks=[(bit==b) for b in range(2)]
    mass=torch.ones(1,N,device=DEVICE)
    for b in range(2): mass[:,masks[b]]=mass_ratios[b]
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    pop[:,:NW1]*=s1/0.3; pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf')).nonzero(as_tuple=True)[0]
        if nd.shape[0]>0:
            g=pop[nd]; B=g.shape[0]
            pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
            vel=torch.zeros(B,N,3,device=DEVICE)
            gi=0;W1=g[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
            b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
            W2=g[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
            b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
            fv=g[:,gi].abs()
            bid_vals=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
            ni=npt.unsqueeze(0).expand(B,-1,-1)
            csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
            comb=torch.zeros(B,1,1,device=DEVICE);cd=False
            sx=pos[:,:,0].mean(dim=1)
            mb=mass.expand(B,-1)
            for step in range(nsteps):
                t=step*DT
                if not cd and step%10==0:
                    p0=pos[0,masks[0]];p1=pos[0,masks[1]];ds=torch.cdist(p0,p1)
                    cl=(ds<1.2).nonzero(as_tuple=False)
                    if cl.shape[0]>0:
                        i0=masks[0].nonzero(as_tuple=True)[0];i1=masks[1].nonzero(as_tuple=True)[0]
                        nn_=min(cl.shape[0],500)
                        csa=torch.cat([csa,i0[cl[:nn_,0]]]);csb=torch.cat([csb,i1[cl[:nn_,1]]])
                        crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]]);comb[:]=1;cd=True
                sv=torch.sin(2*np.pi*fv*t).reshape(B,1,1).expand(B,N,1)
                cv=torch.cos(2*np.pi*fv*t).reshape(B,1,1).expand(B,N,1)
                nn_in=torch.cat([sv,cv,ni,bid_vals,comb.expand(B,N,1)],dim=2)
                h=torch.tanh(torch.bmm(nn_in,W1)+b1g)
                o=torch.tanh(torch.bmm(h,W2)+b2g)
                og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
                ext=torch.zeros(B,N,3,device=DEVICE)
                ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
                ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
                f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mb
                pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa
                di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
                dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
                rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
                ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
                f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
                f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
                pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
                bl=(pos[:,:,1]<GROUND_Y).float()
                f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
                f-=DRAG*vel;f+=ext
                acc=f/(mb.unsqueeze(2).clamp(min=0.01))
                vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
            fit=pos[:,:,0].mean(dim=1)-sx
            fit=torch.where(torch.isnan(fit),torch.tensor(-9999.0,device=DEVICE),fit)
            pf[nd]=fit
        o_=pf.argsort(descending=True);pop=pop[o_];pf=pf[o_]
        if gen%50==0 or gen==ngens-1:
            print(f"  [{label}] Gen {gen:>4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=2;fr=torch.randn(nfr,NG,device=DEVICE)*0.3
        fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE);p1_=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE);p2_=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,NG,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1_],pop[p2_])
        mt=torch.rand(nc,NG,device=DEVICE)<0.15;ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    print(f"  [{label}] Done: {(time.time()-t0)/60:.1f}min | Best={pf[0].item():+.2f}")
    return pop[0].cpu().numpy()


@torch.no_grad()
def replay_pca(genes, data, nsteps, mass_ratios):
    """Replay and record hidden activations for PCA."""
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
    bid_vals=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    masks=[(bit==b) for b in range(2)]
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone();cd=False;comb=torch.zeros(B,1,1,device=DEVICE)
    mass=torch.ones(B,N,device=DEVICE)
    for b in range(2): mass[:,masks[b]]=mass_ratios[b]
    h_act={0:[],1:[]};fx={0:[],1:[]}
    for step in range(nsteps):
        t=step*DT
        if not cd and step%10==0:
            i0=masks[0].nonzero(as_tuple=True)[0];i1=masks[1].nonzero(as_tuple=True)[0]
            p0=pos[0,i0];p1=pos[0,i1];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,i0[cl[:nn_,0]]]);csb=torch.cat([csb,i1[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]]);comb[:]=1;cd=True
        sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
        st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid_vals,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g)
        o=torch.tanh(torch.bmm(h,W2)+b2g)
        for b in range(2):
            h_mean=h[0,masks[b]].mean(dim=0).cpu().numpy()
            h_act[b].append(h_mean)
            fx[b].append(o[0,masks[b],0].mean().item())
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa
        di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        acc=f/(mass.unsqueeze(2).clamp(min=0.01))
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    for b in range(2): h_act[b]=np.array(h_act[b])
    r_fx,_=pearsonr(fx[0],fx[1]) if np.std(fx[0])>1e-10 and np.std(fx[1])>1e-10 else (0.0, 1.0)
    return h_act, round(r_fx,3), fx


def make_comparison_figure(results):
    """3-column figure comparing PCA manifolds for different mass ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(21,12))
    fig.suptitle("Exp 25: Neural Geometry of the Symmetry Locks Theorem\n"
                 "PCA Projection of Hidden Layer (32 units) — 2-Body System",
                 fontsize=14, fontweight="bold")
    conditions = list(results.keys())
    colors_map = {"1:1":"#3498db", "3:1":"#e67e22", "10:1":"#e74c3c"}

    for col, cond in enumerate(conditions):
        r = results[cond]; h_act = r["h_act"]; r_fx = r["r_fx"]; fx = r["fx"]
        # Fit PCA on combined B0+B1
        combined = np.vstack([h_act[0], h_act[1]])
        pca = PCA(n_components=2)
        pca.fit(combined)
        proj0 = pca.transform(h_act[0])
        proj1 = pca.transform(h_act[1])
        nsteps = len(proj0)
        cols = plt.cm.viridis(np.linspace(0,1,nsteps))
        color = colors_map.get(cond, "#3498db")

        # Top row: PCA trajectories
        ax = axes[0, col]
        ax.scatter(proj0[:,0], proj0[:,1], c=cols, s=3, alpha=0.7, label="Body 0")
        ax.scatter(proj1[:,0], proj1[:,1], c=cols, s=3, alpha=0.7, marker="x", label="Body 1")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        # Neural distance
        dist = np.linalg.norm(h_act[0] - h_act[1], axis=1)
        ax.set_title(f"Mass {cond}\nr(Fx)={r_fx} | neural dist={dist.mean():.2f}",
                    fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)
        # Highlight key insight
        if cond == "1:1":
            ax.annotate("OVERLAPPING\n(Same manifold)", xy=(0.5, 0.02), xycoords="axes fraction",
                       ha="center", fontsize=9, color="green", fontweight="bold",
                       bbox=dict(facecolor="white", alpha=0.8, edgecolor="green"))
        elif cond == "10:1":
            ax.annotate("SEPARATED\n(Different manifolds)", xy=(0.5, 0.02), xycoords="axes fraction",
                       ha="center", fontsize=9, color="red", fontweight="bold",
                       bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"))

        # Bottom row: Force output traces
        ax2 = axes[1, col]
        ax2.plot(fx[0], color="#e74c3c", alpha=0.5, linewidth=0.5, label="B0 Fx")
        ax2.plot(fx[1], color="#3498db", alpha=0.5, linewidth=0.5, label="B1 Fx")
        ax2.set_xlabel("Step"); ax2.set_ylabel("Fx (mean)")
        ax2.set_title(f"Force Output — r(Fx)={r_fx}")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(OUTPUT_DIR, "exp25_pca_comparison.png")
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    print(f"\nFigure: {fp}")


if __name__ == "__main__":
    print("="*70)
    print("EXP 25: PCA MANIFOLD COMPARISON")
    print("="*70)
    t_total = time.time()

    NSTEPS=600; GAP=0.5; PSZ=200; NGENS=300
    gx,gy,gz,sp = 10,5,4,0.35
    data = build_bodies_2(gx,gy,gz,sp,GAP)
    mass_conditions = [
        ("1:1", (1.0, 1.0)),
        ("3:1", (3.0, 1.0)),
        ("10:1", (10.0, 1.0)),
    ]
    results = {}
    for label, mrs in mass_conditions:
        print(f"\n--- Mass {label} ---")
        torch.manual_seed(42); np.random.seed(42)
        genes = evolve_2body(data, NSTEPS, NGENS, PSZ, f"pca_{label}", mrs)
        h_act, r_fx, fx = replay_pca(genes, data, NSTEPS, mrs)
        dist = np.linalg.norm(h_act[0] - h_act[1], axis=1)
        results[label] = {
            "h_act": h_act, "r_fx": r_fx, "fx": fx,
            "neural_dist_mean": float(dist.mean()),
            "neural_dist_std": float(dist.std()),
        }
        print(f"  r(Fx)={r_fx} | neural_dist={dist.mean():.3f}±{dist.std():.3f}")

    make_comparison_figure(results)

    # Save log (without numpy arrays)
    log = {}
    for k, v in results.items():
        log[k] = {kk: vv for kk, vv in v.items() if kk not in ("h_act", "fx")}
    log_path = os.path.join(RESULTS_DIR, "exp25_pca_comparison_log.json")
    with open(log_path, "w") as f: json.dump(log, f, indent=2)
    print(f"Log: {log_path}")

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*70}")
    print(f"EXP 25 COMPLETE ({total_min:.0f} min)")
    print(f"{'='*70}")
    for k, v in results.items():
        print(f"  {k:>5s}: r(Fx)={v['r_fx']:+.3f} | dist={v['neural_dist_mean']:.3f}")

    try:
        import winsound
        for _ in range(3): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
