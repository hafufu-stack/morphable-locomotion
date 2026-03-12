"""
Exp 24: Reciprocal Altruism — Can energy observation induce tit-for-tat?
========================================================================
Hypothesis: Parasite's Dilemma produces permanent freeriders because
bodies cannot observe each other's energy expenditure. Adding opponent
energy as an NN input may enable reciprocal altruism (tit-for-tat).

Design:
  A) Baseline (INPUT_DIM=7): standard Parasite at alpha=3 (replication)
  B) Observable (INPUT_DIM=8): add opponent's cumulative energy % as 8th input
  C) Observable alpha=5: higher penalty to amplify the effect

Each condition: 5 seeds x 300 gen, co-evolutionary GA.
"""
import numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os, time, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures"); RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN=32

def build_bodies(gx,gy,gz,sp,gap):
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
def sim_reciprocal(g0,g1,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,alpha,
                   observable=False, input_dim=7, return_temporal=False):
    """Simulate with optional opponent energy observation."""
    OUT=3; NW1=input_dim*HIDDEN; B=g0.shape[0]
    pos=rp.unsqueeze(0).expand(B,-1,-1).clone(); vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0; b1m=bit==1
    b0i=b0m.nonzero(as_tuple=True)[0]; b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item(); n1=b1m.sum().item()
    # Decode genomes
    gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,input_dim,HIDDEN);gi+=NW1
    b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs()
    gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,input_dim,HIDDEN);gi+=NW1
    b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    e0=torch.zeros(B,device=DEVICE); e1=torch.zeros(B,device=DEVICE)
    # For temporal tracking
    e0_temporal=[]; e1_temporal=[]
    for step in range(nsteps):
        t=step*DT
        if not cd and step%10==0:
            p0=pos[0,b0i];p1=pos[0,b1i];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb=torch.ones(B,1,1,device=DEVICE);cd=True
        # Build NN inputs
        st0=torch.sin(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        ct0=torch.cos(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        if observable:
            # Normalize cumulative energy to [0,1] range
            me_norm = N*nsteps*(BASE_AMP*1.5)**2*3
            # Body 0 sees Body 1's energy, and vice versa
            opp_e1 = (e1 / (me_norm+1e-8) * 100).clamp(0,1).reshape(B,1,1).expand(B,n0,1)
            opp_e0 = (e0 / (me_norm+1e-8) * 100).clamp(0,1).reshape(B,1,1).expand(B,n1,1)
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1),opp_e1],dim=2)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1),opp_e0],dim=2)
        else:
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
        o=torch.zeros(B,N,OUT,device=DEVICE);o[:,b0m]=o0;o[:,b1m]=o1
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        step_e0=(ext[:,b0m]**2).sum(dim=(1,2))
        step_e1=(ext[:,b1m]**2).sum(dim=(1,2))
        e0+=step_e0; e1+=step_e1
        if return_temporal and B==1:
            e0_temporal.append(step_e0.item())
            e1_temporal.append(step_e1.item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float();f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx
    me=N*nsteps*(BASE_AMP*1.5)**2*3
    ep0=alpha*(e0/me)*100; ep1=alpha*(e1/me)*100
    if return_temporal:
        return disp-ep0, disp-ep1, disp, e0/me*100, e1/me*100, e0_temporal, e1_temporal
    return disp-ep0, disp-ep1, disp, e0/me*100, e1/me*100


def run_condition(data, alpha, observable, n_seeds, label):
    """Run one condition with multiple seeds."""
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3
    input_dim = 8 if observable else 7
    NW1=input_dim*HIDDEN; NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(input_dim+HIDDEN))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    PSZ=200; NGENS=300; NSTEPS=600
    gaps=[];f0s=[];f1s=[];e0s=[];e1s=[];best_pairs=[]
    for seed in range(n_seeds):
        torch.manual_seed(seed*61+17); np.random.seed(seed*61+17)
        pop0=torch.randn(PSZ,NG,device=DEVICE)*0.3
        pop1=torch.randn(PSZ,NG,device=DEVICE)*0.3
        pop0[:,:NW1]*=s1/0.3;pop0[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
        pop1[:,:NW1]*=s1/0.3;pop1[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
        pf0=torch.full((PSZ,),float('-inf'),device=DEVICE)
        pf1=torch.full((PSZ,),float('-inf'),device=DEVICE)
        t0=time.time()
        print(f"\n  [{label}] Seed {seed+1}/{n_seeds}")
        for gen in range(NGENS):
            nd=(pf0==float('-inf'))|(pf1==float('-inf'))
            if nd.any():
                ix=nd.nonzero(as_tuple=True)[0]
                f0,f1,_,_,_=sim_reciprocal(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,
                                           N,nper,NSTEPS,alpha,observable,input_dim)
                f0=torch.where(torch.isnan(f0),torch.tensor(-9999.0,device=DEVICE),f0)
                f1=torch.where(torch.isnan(f1),torch.tensor(-9999.0,device=DEVICE),f1)
                pf0[ix]=f0;pf1[ix]=f1
            o0=pf0.argsort(descending=True);pop0=pop0[o0];pf0=pf0[o0];pop1=pop1[o0];pf1=pf1[o0]
            o1=pf1.argsort(descending=True);pop1=pop1[o1];pf1=pf1[o1];pop0=pop0[o1];pf0=pf0[o1]
            if gen%100==0 or gen==NGENS-1:
                print(f"    Gen {gen}: f0={pf0[0].item():+.1f} f1={pf1[0].item():+.1f} ({(time.time()-t0)/60:.1f}min)")
            for pop,pf in [(pop0,pf0),(pop1,pf1)]:
                ne=max(2,int(PSZ*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
                nfr=2;fr=torch.randn(nfr,NG,device=DEVICE)*0.3
                fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
                np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
                nc=PSZ-np2.shape[0]
                t1=torch.randint(PSZ,(nc,5),device=DEVICE);p1_=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
                t2=torch.randint(PSZ,(nc,5),device=DEVICE);p2_=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
                mk=torch.rand(nc,NG,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1_],pop[p2_])
                mt=torch.rand(nc,NG,device=DEVICE)<0.15;ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
                np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
                pop.copy_(np2);pf.copy_(nf2)
        # Replay best pair with temporal tracking
        bg0=pop0[0:1];bg1=pop1[0:1]
        f0_,f1_,disp_,en0_,en1_,et0,et1=sim_reciprocal(
            bg0,bg1,rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha,
            observable,input_dim,return_temporal=True)
        gap=f1_.item()-f0_.item()
        gaps.append(gap);f0s.append(f0_.item());f1s.append(f1_.item())
        e0s.append(en0_.item());e1s.append(en1_.item())
        best_pairs.append((bg0.cpu().numpy(),bg1.cpu().numpy(),et0,et1))
        print(f"    f0={f0_.item():+.1f} f1={f1_.item():+.1f} gap={gap:+.1f} E0={en0_.item():.1f} E1={en1_.item():.1f}")
    return {
        "gaps":gaps,"f0":f0s,"f1":f1s,"e0":e0s,"e1":e1s,
        "mean_gap":np.mean(gaps),"std_gap":np.std(gaps),
        "mean_f0":np.mean(f0s),"mean_f1":np.mean(f1s),
        "mean_e0":np.mean(e0s),"mean_e1":np.mean(e1s),
        "best_pairs":best_pairs
    }


def make_figure(results):
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    fig.suptitle("Exp 24: Reciprocal Altruism — Can Energy Observation Induce Tit-for-Tat?",
                fontsize=13,fontweight="bold")

    # Panel 1: Fitness gap comparison
    ax=axes[0]
    conds=list(results.keys())
    gaps_m=[results[c]["mean_gap"] for c in conds]
    gaps_s=[results[c]["std_gap"] for c in conds]
    colors=["#e74c3c","#3498db","#2ecc71","#9b59b6"]
    bars=ax.bar(range(len(conds)),gaps_m,yerr=gaps_s,capsize=8,
                color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for g in results[c]["gaps"]:
            ax.scatter(i,g,color="black",alpha=0.3,s=15,zorder=5)
        ax.text(i,gaps_m[i]+gaps_s[i]+1,f"{gaps_m[i]:+.1f}±{gaps_s[i]:.1f}",
                ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=8)
    ax.set_ylabel("Fitness Gap (f1 - f0)")
    ax.set_title("Fitness Gap\n(>0 = freeriding)")
    ax.axhline(y=0,color="gray",linestyle="--",alpha=0.5)
    ax.grid(alpha=0.3,axis="y")

    # Panel 2: Energy expenditure
    ax=axes[1]
    x=np.arange(len(conds));w=0.35
    e0s=[results[c]["mean_e0"] for c in conds]
    e1s=[results[c]["mean_e1"] for c in conds]
    ax.bar(x-w/2,e0s,w,label="Body 0 (worker?)",color="#e74c3c",alpha=0.8)
    ax.bar(x+w/2,e1s,w,label="Body 1 (freeloader?)",color="#3498db",alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=8)
    ax.set_ylabel("Energy (%)")
    ax.set_title("Energy Expenditure\n(gap = effort asymmetry)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3,axis="y")

    # Panel 3: Temporal energy for best seed of each condition
    ax=axes[2]
    window=20
    for i,c in enumerate(conds):
        _,_,et0,et1=results[c]["best_pairs"][0]
        # Rolling mean
        if len(et0)>window:
            e0_smooth=np.convolve(et0,np.ones(window)/window,mode='valid')
            e1_smooth=np.convolve(et1,np.ones(window)/window,mode='valid')
        else:
            e0_smooth=et0; e1_smooth=et1
        if i==0:
            ax.plot(e0_smooth,color="#e74c3c",alpha=0.6,linewidth=1,label=f"{c} B0")
            ax.plot(e1_smooth,color="#3498db",alpha=0.6,linewidth=1,label=f"{c} B1")
        else:
            ax.plot(e0_smooth,color="#2ecc71",alpha=0.8,linewidth=1.5,linestyle="--",label=f"{c} B0")
            ax.plot(e1_smooth,color="#9b59b6",alpha=0.8,linewidth=1.5,linestyle="--",label=f"{c} B1")
    ax.set_xlabel("Step")
    ax.set_ylabel("Instantaneous Energy (smoothed)")
    ax.set_title("Temporal Energy Pattern\n(alternating = tit-for-tat)")
    ax.legend(fontsize=7,ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    fp=os.path.join(OUTPUT_DIR,"exp24_reciprocal_altruism.png")
    plt.savefig(fp,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fp}")


if __name__=="__main__":
    t_total=time.time()
    gx,gy,gz,sp=10,5,4,0.35; GAP=0.5; N_SEEDS=5
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    print("="*70)
    print("EXP 24: RECIPROCAL ALTRUISM")
    print("="*70)

    # A) Baseline: no observation, alpha=3
    print("\n--- Condition A: Baseline (blind, alpha=3) ---")
    results["blind_a3"]=run_condition(data, alpha=3.0, observable=False, n_seeds=N_SEEDS, label="blind_a3")

    # B) Observable: can see opponent's energy, alpha=3
    print("\n--- Condition B: Observable (alpha=3) ---")
    results["obs_a3"]=run_condition(data, alpha=3.0, observable=True, n_seeds=N_SEEDS, label="obs_a3")

    # C) Observable: alpha=5 (higher pressure)
    print("\n--- Condition C: Observable (alpha=5) ---")
    results["obs_a5"]=run_condition(data, alpha=5.0, observable=True, n_seeds=N_SEEDS, label="obs_a5")

    # Serialize (without tensors)
    save_results = {}
    for k,v in results.items():
        save_results[k] = {kk:vv for kk,vv in v.items() if kk != "best_pairs"}
    log_path=os.path.join(RESULTS_DIR,"exp24_log.json")
    with open(log_path,"w") as f: json.dump(save_results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    make_figure(results)

    total_min=(time.time()-t_total)/60
    print(f"\n{'='*70}")
    print(f"EXP 24 COMPLETE ({total_min:.0f} min)")
    print(f"{'='*70}")
    for c in results:
        r=results[c]
        print(f"  {c:>12s}: gap={r['mean_gap']:+.1f}+/-{r['std_gap']:.1f} | E0={r['mean_e0']:.1f} E1={r['mean_e1']:.1f}")

    # Verdict
    blind_gap=results["blind_a3"]["mean_gap"]
    obs3_gap=results["obs_a3"]["mean_gap"]
    print(f"\n  Blind gap: {blind_gap:+.1f}")
    print(f"  Observable gap: {obs3_gap:+.1f}")
    if abs(obs3_gap) < abs(blind_gap) * 0.5:
        print("  VERDICT: Observation REDUCES freeriding! Reciprocal altruism may emerge.")
    elif abs(obs3_gap) < abs(blind_gap) * 0.8:
        print("  VERDICT: Observation partially reduces freeriding.")
    else:
        print("  VERDICT: Observation does NOT reduce freeriding. Structural constraint confirmed.")

    try:
        import winsound
        for _ in range(3): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
