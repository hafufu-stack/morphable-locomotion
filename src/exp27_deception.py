"""
Exp 27: The Evolution of Deception — Can Liars Survive Divorce?
================================================================
Builds on Exp 26 (Divorce). Instead of receiving the opponent's TRUE energy,
each body receives a SIGNAL that the opponent's NN outputs (5th output).
This signal can be honest or deceptive.

Conditions (all α=3, N=5 seeds, 300 gen):
  A) DIVORCE_HONEST: True energy observable + spring control (Exp 26 replication)
  B) DIVORCE_DECEPTION: Signal replaces true energy + spring control
  C) DECEPTION_NO_DIVORCE: Signal visible but no spring control (can lie, can't punish)

Key question: When divorce exists as punishment, does evolution discover deception
(fake energy signals) to avoid being divorced? Or does the threat of divorce
force honest signaling (Zahavian Handicap)?
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
    n_intra = len(sa)
    return ap,np_,bi,np.array(sa),np.array(sb),np.array(rl),nper,nt,n_intra


@torch.no_grad()
def sim_deception(g0,g1,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,alpha,
                  spring_control=True, deceptive=False, n_intra=0,
                  return_detailed=False):
    """
    deceptive=False: opponent receives TRUE energy (honest)
    deceptive=True: opponent receives NN signal output (can lie)
    """
    INPUT_DIM = 8  # always 8: sin,cos,nx,ny,nz,bid,comb,opponent_info
    # Output: 3 movement + (1 spring_ctrl if enabled) + (1 signal if deceptive)
    OUT = 3
    if spring_control: OUT += 1
    if deceptive: OUT += 1
    SIGNAL_IDX = OUT - 1 if deceptive else -1
    SPRING_IDX = 3 if spring_control else -1

    NW1=INPUT_DIM*HIDDEN; B=g0.shape[0]
    pos=rp.unsqueeze(0).expand(B,-1,-1).clone(); vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0; b1m=bit==1
    b0i=b0m.nonzero(as_tuple=True)[0]; b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item(); n1=b1m.sum().item()
    gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs()
    gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    e0=torch.zeros(B,device=DEVICE); e1=torch.zeros(B,device=DEVICE)
    combine_active=False; was_combined=False
    n_separations=0; n_recombinations=0
    # Track signals vs true energy for deception analysis
    signal_0_hist=[]; signal_1_hist=[]; true_e0_hist=[]; true_e1_hist=[]
    prev_signal_0=torch.zeros(B,1,1,device=DEVICE)
    prev_signal_1=torch.zeros(B,1,1,device=DEVICE)

    for step in range(nsteps):
        t=step*DT
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
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                combine_active=True; n_recombinations+=1

        st0=torch.sin(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        ct0=torch.cos(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        comb_val=torch.ones(B,1,1,device=DEVICE) if combine_active else torch.zeros(B,1,1,device=DEVICE)

        me_norm=N*nsteps*(BASE_AMP*1.5)**2*3
        if deceptive:
            # Opponent info = other body's SIGNAL (can be dishonest)
            opp_for_0 = prev_signal_1.expand(B,n0,1)
            opp_for_1 = prev_signal_0.expand(B,n1,1)
        else:
            # Opponent info = TRUE energy (honest)
            opp_e1=(e1/(me_norm+1e-8)*100).clamp(0,1).reshape(B,1,1).expand(B,n0,1)
            opp_e0=(e0/(me_norm+1e-8)*100).clamp(0,1).reshape(B,1,1).expand(B,n1,1)
            opp_for_0=opp_e1; opp_for_1=opp_e0

        in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb_val.expand(B,n0,1),opp_for_0],dim=2)
        in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb_val.expand(B,n1,1),opp_for_1],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)

        # Extract signals if deceptive
        if deceptive:
            prev_signal_0 = o0[:,:,SIGNAL_IDX].mean(dim=1,keepdim=True).unsqueeze(2)
            prev_signal_1 = o1[:,:,SIGNAL_IDX].mean(dim=1,keepdim=True).unsqueeze(2)
            if B==1:
                signal_0_hist.append(prev_signal_0.item())
                signal_1_hist.append(prev_signal_1.item())

        # Spring control
        if spring_control and combine_active and B==1:
            detach0=o0[0,:,SPRING_IDX].mean().item()
            detach1=o1[0,:,SPRING_IDX].mean().item()
            if max(detach0,detach1)>0.3:
                csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
                combine_active=False;was_combined=True;n_separations+=1

        o_move=torch.zeros(B,N,3,device=DEVICE)
        o_move[:,b0m]=o0[:,:,:3]; o_move[:,b1m]=o1[:,:,:3]
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o_move[:,:,0]*gc
        ext[:,:,1]=BASE_AMP*torch.clamp(o_move[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o_move[:,:,2]*gc*0.5
        step_e0=(ext[:,b0m]**2).sum(dim=(1,2))
        step_e1=(ext[:,b1m]**2).sum(dim=(1,2))
        e0+=step_e0; e1+=step_e1
        if B==1:
            true_e0_hist.append((e0/(me_norm+1e-8)*100).item())
            true_e1_hist.append((e1/(me_norm+1e-8)*100).item())

        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT

    disp0=pos[:,b0i,0].mean(dim=1)-rp[b0i,0].mean()
    disp1=pos[:,b1i,0].mean(dim=1)-rp[b1i,0].mean()
    me=N*nsteps*(BASE_AMP*1.5)**2*3
    ep0=alpha*(e0/me)*100; ep1=alpha*(e1/me)*100
    f0=disp0-ep0; f1=disp1-ep1
    if return_detailed:
        return (f0,f1,e0/me*100,e1/me*100,n_separations,n_recombinations,
                signal_0_hist,signal_1_hist,true_e0_hist,true_e1_hist)
    return f0,f1,e0/me*100,e1/me*100


def run_condition(data,alpha,spring_control,deceptive,n_seeds,label):
    ap,np_,bi,sa,sb,rl,nper,nt,n_intra=data; N=nt
    INPUT_DIM=8; OUT=3
    if spring_control: OUT+=1
    if deceptive: OUT+=1
    NW1=INPUT_DIM*HIDDEN; NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    PSZ=200;NGENS=300;NSTEPS=600
    gaps=[];f0s=[];f1s=[];e0s=[];e1s=[];seps=[];sig_corrs=[]
    for seed in range(n_seeds):
        torch.manual_seed(seed*71+13);np.random.seed(seed*71+13)
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
                ff0,ff1,_,_=sim_deception(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,
                                          N,nper,NSTEPS,alpha,spring_control,deceptive,n_intra)
                ff0=torch.where(torch.isnan(ff0),torch.tensor(-9999.0,device=DEVICE),ff0)
                ff1=torch.where(torch.isnan(ff1),torch.tensor(-9999.0,device=DEVICE),ff1)
                pf0[ix]=ff0;pf1[ix]=ff1
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
        bg0=pop0[0:1];bg1=pop1[0:1]
        res=sim_deception(bg0,bg1,rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha,
                          spring_control,deceptive,n_intra,return_detailed=True)
        f0_,f1_,en0_,en1_,nsep,nrec,sig0,sig1,te0,te1=res
        gap=f1_.item()-f0_.item()
        gaps.append(gap);f0s.append(f0_.item());f1s.append(f1_.item())
        e0s.append(en0_.item());e1s.append(en1_.item());seps.append(nsep)
        # Signal-truth correlation (deception metric)
        sc = 0.0
        if deceptive and len(sig1)>10 and len(te1)>10:
            from scipy.stats import pearsonr
            # Does body 1's signal correlate with its TRUE energy?
            s1_arr=np.array(sig1[:len(te1)]); t1_arr=np.array(te1[:len(sig1)])
            if np.std(s1_arr)>1e-10 and np.std(t1_arr)>1e-10:
                sc,_=pearsonr(s1_arr,t1_arr)
        sig_corrs.append(sc)
        print(f"    f0={f0_.item():+.1f} f1={f1_.item():+.1f} gap={gap:+.1f} "
              f"E0={en0_.item():.1f} E1={en1_.item():.1f} sep={nsep} sig_corr={sc:.3f}")
    return {
        "gaps":gaps,"f0":f0s,"f1":f1s,"e0":e0s,"e1":e1s,"separations":seps,
        "sig_corrs":sig_corrs,
        "mean_gap":float(np.mean(gaps)),"std_gap":float(np.std(gaps)),
        "mean_f0":float(np.mean(f0s)),"mean_f1":float(np.mean(f1s)),
        "mean_e0":float(np.mean(e0s)),"mean_e1":float(np.mean(e1s)),
        "mean_sep":float(np.mean(seps)),"mean_sig_corr":float(np.mean(sig_corrs)),
    }


def make_figure(results):
    fig,axes=plt.subplots(1,4,figsize=(24,6))
    fig.suptitle("Exp 27: The Evolution of Deception — Can Liars Survive Divorce?",
                fontsize=13,fontweight="bold")
    conds=list(results.keys())
    colors=["#2ecc71","#e74c3c","#9b59b6"]

    # Panel 1: Fitness gap
    ax=axes[0]
    gm=[results[c]["mean_gap"] for c in conds]
    gs=[results[c]["std_gap"] for c in conds]
    bars=ax.bar(range(len(conds)),gm,yerr=gs,capsize=8,color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for g in results[c]["gaps"]:
            ax.scatter(i,g,color="black",alpha=0.3,s=15,zorder=5)
        ax.text(i,gm[i]+gs[i]+1,f"{gm[i]:+.1f}±{gs[i]:.1f}",ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("Fitness Gap (f1 - f0)"); ax.set_title("Fitness Gap\n(>0 = freeriding)")
    ax.axhline(y=0,color="gray",linestyle="--",alpha=0.5); ax.grid(alpha=0.3,axis="y")

    # Panel 2: Divorces
    ax=axes[1]
    sm=[results[c]["mean_sep"] for c in conds]
    ax.bar(range(len(conds)),sm,color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for s in results[c]["separations"]:
            ax.scatter(i,s,color="black",alpha=0.3,s=15,zorder=5)
        ax.text(i,sm[i]+0.5,f"{sm[i]:.1f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("# Divorces"); ax.set_title("Divorce Events"); ax.grid(alpha=0.3,axis="y")

    # Panel 3: Signal-truth correlation
    ax=axes[2]
    sc=[results[c]["mean_sig_corr"] for c in conds]
    ax.bar(range(len(conds)),sc,color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for s in results[c]["sig_corrs"]:
            ax.scatter(i,s,color="black",alpha=0.3,s=15,zorder=5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("r(signal, true energy)")
    ax.set_title("Signal Honesty\n(r≈1 = honest, r≈0 = deceptive)")
    ax.axhline(y=0,color="gray",linestyle="--",alpha=0.5); ax.grid(alpha=0.3,axis="y")

    # Panel 4: Energy expenditure
    ax=axes[3]
    x=np.arange(len(conds));w=0.35
    e0s=[results[c]["mean_e0"] for c in conds]
    e1s=[results[c]["mean_e1"] for c in conds]
    ax.bar(x-w/2,e0s,w,label="Body 0",color="#e74c3c",alpha=0.8)
    ax.bar(x+w/2,e1s,w,label="Body 1",color="#3498db",alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_","\n") for c in conds],fontsize=7)
    ax.set_ylabel("Energy (%)"); ax.set_title("Energy Expenditure")
    ax.legend(fontsize=7); ax.grid(alpha=0.3,axis="y")

    plt.tight_layout()
    fp=os.path.join(OUTPUT_DIR,"exp27_deception.png")
    plt.savefig(fp,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fp}")


if __name__=="__main__":
    t_total=time.time()
    gx,gy,gz,sp=10,5,4,0.35;GAP=0.5;N_SEEDS=5;ALPHA=3.0
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    print("="*70)
    print("EXP 27: THE EVOLUTION OF DECEPTION")
    print("="*70)

    # A) DIVORCE_HONEST (replication of Exp 26 best condition)
    print("\n--- A: DIVORCE_HONEST (true energy + spring ctrl) ---")
    results["HONEST"]=run_condition(data,ALPHA,spring_control=True,deceptive=False,
                                    n_seeds=N_SEEDS,label="HONEST")

    # B) DIVORCE_DECEPTION (signal replaces true energy + spring ctrl)
    print("\n--- B: DIVORCE_DECEPTION (signal + spring ctrl) ---")
    results["DECEPTION"]=run_condition(data,ALPHA,spring_control=True,deceptive=True,
                                       n_seeds=N_SEEDS,label="DECEPTION")

    # C) DECEPTION_NO_DIVORCE (signal visible, no spring control)
    print("\n--- C: DECEPTION_NO_DIVORCE (signal, no spring ctrl) ---")
    results["DECEPTION\nNO_DIVORCE"]=run_condition(data,ALPHA,spring_control=False,deceptive=True,
                                                    n_seeds=N_SEEDS,label="DECEP_NO_DIV")

    log_path=os.path.join(RESULTS_DIR,"exp27_deception_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    make_figure(results)

    total_min=(time.time()-t_total)/60
    print(f"\n{'='*70}")
    print(f"EXP 27 COMPLETE ({total_min:.0f} min)")
    print(f"{'='*70}")
    for c in results:
        r=results[c]
        print(f"  {c:>20s}: gap={r['mean_gap']:+.1f}±{r['std_gap']:.1f} | "
              f"sep={r['mean_sep']:.1f} | sig_corr={r['mean_sig_corr']:.3f}")

    try:
        import winsound
        for _ in range(3): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
