"""
Exp 26: The Evolution of Divorce — Can Punishment Break Structural Constraint?
===============================================================================
Combines:
  - Parasite's Dilemma (individual energy costs, alpha=3)
  - Observable energy (INPUT_DIM=8, opponent energy % visible)
  - Topology Control (4th output controls spring cutting)

Hypothesis: When a body can both SEE its partner freeloading AND PUNISH by
cutting springs (divorce), evolution will discover divorce as a sanction.

Conditions:
  A) Baseline: Parasite α=3, blind, no spring control (replication of Exp 21)
  B) Observable + No spring control (replication of Exp 24)
  C) Observable + Spring control (DIVORCE enabled)
  D) Blind + Spring control (control: can divorce but can't see why)

Each: 5 seeds × 300 gen.
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
    # Track which springs are intra-body vs combine
    n_intra = len(sa)  # all springs built so far are intra-body
    return ap,np_,bi,np.array(sa),np.array(sb),np.array(rl),nper,nt,n_intra


@torch.no_grad()
def sim_divorce(g0,g1,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,alpha,
                observable=False, spring_control=False, n_intra=0,
                return_detailed=False):
    """Simulate with optional opponent energy observation and spring cutting."""
    input_dim = 8 if observable else 7
    OUT = 4 if spring_control else 3
    NW1=input_dim*HIDDEN; B=g0.shape[0]
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
    # Track combine spring indices
    combine_start = n_intra  # combine springs start at this index
    combine_active = False
    n_separations = 0; n_recombinations = 0
    was_combined = False
    for step in range(nsteps):
        t=step*DT
        # Combine check (when not currently combined)
        if step%10==0:
            p0=pos[0,b0i];p1=pos[0,b1i];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if not cd and cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb=torch.ones(B,1,1,device=DEVICE);cd=True
                combine_active=True
                if was_combined: n_recombinations+=1
            elif cd and not combine_active and cl.shape[0]>0:
                # Recombine after divorce
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                combine_active=True; n_recombinations+=1
        # Build NN inputs
        st0=torch.sin(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        ct0=torch.cos(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        comb_val = torch.ones(B,1,1,device=DEVICE) if combine_active else torch.zeros(B,1,1,device=DEVICE)
        if observable:
            me_norm = N*nsteps*(BASE_AMP*1.5)**2*3
            opp_e1 = (e1 / (me_norm+1e-8) * 100).clamp(0,1).reshape(B,1,1).expand(B,n0,1)
            opp_e0 = (e0 / (me_norm+1e-8) * 100).clamp(0,1).reshape(B,1,1).expand(B,n1,1)
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb_val.expand(B,n0,1),opp_e1],dim=2)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb_val.expand(B,n1,1),opp_e0],dim=2)
        else:
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb_val.expand(B,n0,1)],dim=2)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb_val.expand(B,n1,1)],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
        # Spring control: 4th output = detach signal
        if spring_control and combine_active:
            detach0=o0[:,:,3].mean(dim=1)  # mean detach signal from body 0
            detach1=o1[:,:,3].mean(dim=1)  # mean detach signal from body 1
            # Either body can trigger divorce
            if B==1:  # only track for single replay
                detach_max = max(detach0.item(), detach1.item())
                if detach_max > 0.3:  # threshold for divorce
                    # Cut combine springs
                    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
                    combine_active=False; was_combined=True; n_separations+=1
        o_move=torch.zeros(B,N,3,device=DEVICE)
        o_move[:,b0m]=o0[:,:,:3]; o_move[:,b1m]=o1[:,:,:3]
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o_move[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o_move[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o_move[:,:,2]*gc*0.5
        step_e0=(ext[:,b0m]**2).sum(dim=(1,2))
        step_e1=(ext[:,b1m]**2).sum(dim=(1,2))
        e0+=step_e0; e1+=step_e1
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

    # Fitness: displacement per body minus energy cost
    disp0 = pos[:,b0i,0].mean(dim=1) - rp[b0i,0].mean()
    disp1 = pos[:,b1i,0].mean(dim=1) - rp[b1i,0].mean()
    me=N*nsteps*(BASE_AMP*1.5)**2*3
    ep0=alpha*(e0/me)*100; ep1=alpha*(e1/me)*100
    f0 = disp0 - ep0; f1 = disp1 - ep1
    if return_detailed:
        return f0, f1, disp0, disp1, e0/me*100, e1/me*100, n_separations, n_recombinations
    return f0, f1, e0/me*100, e1/me*100


def run_condition(data, alpha, observable, spring_control, n_seeds, label):
    """Run one condition with multiple seeds."""
    ap,np_,bi,sa,sb,rl,nper,nt,n_intra=data; N=nt
    input_dim = 8 if observable else 7
    OUT = 4 if spring_control else 3
    NW1=input_dim*HIDDEN; NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(input_dim+HIDDEN))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    PSZ=200; NGENS=300; NSTEPS=600
    gaps=[];f0s=[];f1s=[];e0s=[];e1s=[];seps=[];recoms=[]
    for seed in range(n_seeds):
        torch.manual_seed(seed*71+13); np.random.seed(seed*71+13)
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
                f0,f1,en0,en1=sim_divorce(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,
                                          N,nper,NSTEPS,alpha,observable,spring_control,n_intra)
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
        # Replay best pair with detailed tracking
        bg0=pop0[0:1];bg1=pop1[0:1]
        f0_,f1_,d0_,d1_,en0_,en1_,nsep,nrec=sim_divorce(
            bg0,bg1,rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha,
            observable,spring_control,n_intra,return_detailed=True)
        gap=f1_.item()-f0_.item()
        gaps.append(gap);f0s.append(f0_.item());f1s.append(f1_.item())
        e0s.append(en0_.item());e1s.append(en1_.item())
        seps.append(nsep);recoms.append(nrec)
        print(f"    f0={f0_.item():+.1f} f1={f1_.item():+.1f} gap={gap:+.1f} "
              f"E0={en0_.item():.1f} E1={en1_.item():.1f} sep={nsep} reco={nrec}")
    return {
        "gaps":gaps,"f0":f0s,"f1":f1s,"e0":e0s,"e1":e1s,
        "separations":seps,"recombinations":recoms,
        "mean_gap":np.mean(gaps),"std_gap":np.std(gaps),
        "mean_f0":np.mean(f0s),"mean_f1":np.mean(f1s),
        "mean_e0":np.mean(e0s),"mean_e1":np.mean(e1s),
        "mean_sep":np.mean(seps),"mean_reco":np.mean(recoms),
    }


def make_figure(results):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Exp 26: The Evolution of Divorce — Can Punishment Break Structural Constraint?",
                fontsize=13, fontweight="bold")
    conds = list(results.keys())
    colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6"]

    # Panel 1: Fitness gap
    ax=axes[0]
    gaps_m=[results[c]["mean_gap"] for c in conds]
    gaps_s=[results[c]["std_gap"] for c in conds]
    bars=ax.bar(range(len(conds)),gaps_m,yerr=gaps_s,capsize=8,
                color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for g in results[c]["gaps"]:
            ax.scatter(i,g,color="black",alpha=0.3,s=15,zorder=5)
        ax.text(i,gaps_m[i]+gaps_s[i]+1,f"{gaps_m[i]:+.1f}±{gaps_s[i]:.1f}",
                ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
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
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("Energy (%)")
    ax.set_title("Energy Expenditure")
    ax.legend(fontsize=7); ax.grid(alpha=0.3,axis="y")

    # Panel 3: Separations count
    ax=axes[2]
    sep_m=[results[c]["mean_sep"] for c in conds]
    ax.bar(range(len(conds)),sep_m,color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for s in results[c]["separations"]:
            ax.scatter(i,s,color="black",alpha=0.3,s=15,zorder=5)
        ax.text(i,sep_m[i]+0.5,f"{sep_m[i]:.1f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("# Divorces")
    ax.set_title("Divorce Events\n(>0 = punishment discovered)")
    ax.grid(alpha=0.3,axis="y")

    # Panel 4: Individual fitness
    ax=axes[3]
    f0s=[results[c]["mean_f0"] for c in conds]
    f1s=[results[c]["mean_f1"] for c in conds]
    ax.bar(x-w/2,f0s,w,label="Body 0",color="#e74c3c",alpha=0.8)
    ax.bar(x+w/2,f1s,w,label="Body 1",color="#3498db",alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("Fitness")
    ax.set_title("Individual Fitness\n(divorce may equalize)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3,axis="y")

    plt.tight_layout()
    fp=os.path.join(OUTPUT_DIR,"exp26_divorce.png")
    plt.savefig(fp,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fp}")


if __name__=="__main__":
    t_total=time.time()
    gx,gy,gz,sp=10,5,4,0.35; GAP=0.5; N_SEEDS=5; ALPHA=3.0
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    print("="*70)
    print("EXP 26: THE EVOLUTION OF DIVORCE")
    print("="*70)

    # A) Baseline: blind, no spring control
    print("\n--- A: Baseline (blind, no spring ctrl) ---")
    results["blind"]=run_condition(data,ALPHA,observable=False,spring_control=False,n_seeds=N_SEEDS,label="blind")

    # B) Observable, no spring control (replication of Exp 24)
    print("\n--- B: Observable (no spring ctrl) ---")
    results["obs_only"]=run_condition(data,ALPHA,observable=True,spring_control=False,n_seeds=N_SEEDS,label="obs_only")

    # C) Observable + Spring control (DIVORCE!)
    print("\n--- C: Observable + Spring Control (DIVORCE) ---")
    results["DIVORCE"]=run_condition(data,ALPHA,observable=True,spring_control=True,n_seeds=N_SEEDS,label="DIVORCE")

    # D) Blind + Spring control (control)
    print("\n--- D: Blind + Spring Control (control) ---")
    results["blind_spring"]=run_condition(data,ALPHA,observable=False,spring_control=True,n_seeds=N_SEEDS,label="blind_spring")

    # Save results
    log_path=os.path.join(RESULTS_DIR,"exp26_divorce_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    make_figure(results)

    total_min=(time.time()-t_total)/60
    print(f"\n{'='*70}")
    print(f"EXP 26 COMPLETE ({total_min:.0f} min)")
    print(f"{'='*70}")
    for c in results:
        r=results[c]
        print(f"  {c:>14s}: gap={r['mean_gap']:+.1f}±{r['std_gap']:.1f} | "
              f"E0={r['mean_e0']:.1f} E1={r['mean_e1']:.1f} | "
              f"sep={r['mean_sep']:.1f} reco={r['mean_reco']:.1f}")

    # Verdict
    blind_gap=results["blind"]["mean_gap"]
    divorce_gap=results["DIVORCE"]["mean_gap"]
    divorce_sep=results["DIVORCE"]["mean_sep"]
    print(f"\n  VERDICT:")
    print(f"    Blind gap:   {blind_gap:+.1f}")
    print(f"    Divorce gap: {divorce_gap:+.1f}")
    print(f"    Divorces:    {divorce_sep:.1f}")
    if divorce_sep > 0 and abs(divorce_gap) < abs(blind_gap) * 0.5:
        print("    ==> DIVORCE BREAKS STRUCTURAL CONSTRAINT! Evolution discovered punishment!")
    elif divorce_sep > 0:
        print("    ==> Divorce occurs but doesn't eliminate freeriding gap.")
    else:
        print("    ==> No divorce discovered. Structural constraint remains absolute.")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
