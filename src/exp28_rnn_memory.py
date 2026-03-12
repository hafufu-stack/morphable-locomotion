"""
Exp 28: RNN Controllers — Does Memory Break the Structural Constraint?
======================================================================
Replaces feedforward NN with a simple RNN (Elman network with persistent
hidden state across timesteps) in the Parasite's Dilemma.

Hypothesis: If Structural Constraint is due to lack of temporal memory
(FFN cannot learn tit-for-tat), then RNN should enable reciprocal altruism.
If constraint persists, it confirms Physical Irreducibility.

Conditions (all α=3, N=5 seeds, 300 gen):
  A) FFN baseline (replication, INPUT_DIM=7)
  B) RNN baseline (INPUT_DIM=7, HIDDEN=16, persistent state)
  C) RNN + Observable (INPUT_DIM=8, persistent state + opponent energy)
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
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; RNN_HIDDEN=16

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
def sim_rnn(g0,g1,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,alpha,
            input_dim=7, use_rnn=True, observable=False):
    """Simulate with RNN or FFN controllers."""
    OUT=3; H=RNN_HIDDEN
    B=g0.shape[0]
    b0m=bit==0;b1m=bit==1
    b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item();n1=b1m.sum().item()

    if use_rnn:
        # RNN genome: W_ih(input_dim x H) + W_hh(H x H) + b_h(H) + W_out(H x OUT) + b_out(OUT) + freq
        NW_ih=input_dim*H; NW_hh=H*H; NB_h=H; NW_out=H*OUT; NB_out=OUT
        NG=NW_ih+NW_hh+NB_h+NW_out+NB_out+1
        # Decode body 0
        gi=0
        W_ih0=g0[:,gi:gi+NW_ih].reshape(B,input_dim,H);gi+=NW_ih
        W_hh0=g0[:,gi:gi+NW_hh].reshape(B,H,H);gi+=NW_hh
        bh0=g0[:,gi:gi+NB_h].unsqueeze(1);gi+=NB_h
        W_out0=g0[:,gi:gi+NW_out].reshape(B,H,OUT);gi+=NW_out
        bout0=g0[:,gi:gi+NB_out].unsqueeze(1);gi+=NB_out
        freq0=g0[:,gi].abs()
        # Decode body 1
        gi=0
        W_ih1=g1[:,gi:gi+NW_ih].reshape(B,input_dim,H);gi+=NW_ih
        W_hh1=g1[:,gi:gi+NW_hh].reshape(B,H,H);gi+=NW_hh
        bh1=g1[:,gi:gi+NB_h].unsqueeze(1);gi+=NB_h
        W_out1=g1[:,gi:gi+NW_out].reshape(B,H,OUT);gi+=NW_out
        bout1=g1[:,gi:gi+NB_out].unsqueeze(1);gi+=NB_out
        freq1=g1[:,gi].abs()
        # Initialize hidden states (persistent across timesteps!)
        h_state0=torch.zeros(B,n0,H,device=DEVICE)
        h_state1=torch.zeros(B,n1,H,device=DEVICE)
    else:
        # FFN genome (same as exp21b)
        HIDDEN_FFN=32; NW1=input_dim*HIDDEN_FFN
        NG=NW1+HIDDEN_FFN+HIDDEN_FFN*OUT+OUT+1
        gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,input_dim,HIDDEN_FFN);gi+=NW1
        b1_0=g0[:,gi:gi+HIDDEN_FFN].unsqueeze(1);gi+=HIDDEN_FFN
        W2_0=g0[:,gi:gi+HIDDEN_FFN*OUT].reshape(B,HIDDEN_FFN,OUT);gi+=HIDDEN_FFN*OUT
        b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs()
        gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,input_dim,HIDDEN_FFN);gi+=NW1
        b1_1=g1[:,gi:gi+HIDDEN_FFN].unsqueeze(1);gi+=HIDDEN_FFN
        W2_1=g1[:,gi:gi+HIDDEN_FFN*OUT].reshape(B,HIDDEN_FFN,OUT);gi+=HIDDEN_FFN*OUT
        b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs()

    pos=rp.unsqueeze(0).expand(B,-1,-1).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    e0=torch.zeros(B,device=DEVICE);e1=torch.zeros(B,device=DEVICE)
    e0_trace=[];e1_trace=[]

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

        st0=torch.sin(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        ct0=torch.cos(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)

        if observable:
            me_norm=N*nsteps*(BASE_AMP*1.5)**2*3
            opp_e1=(e1/(me_norm+1e-8)*100).clamp(0,1).reshape(B,1,1).expand(B,n0,1)
            opp_e0=(e0/(me_norm+1e-8)*100).clamp(0,1).reshape(B,1,1).expand(B,n1,1)
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1),opp_e1],dim=2)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1),opp_e0],dim=2)
        else:
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)

        if use_rnn:
            # Elman RNN: h_t = tanh(x*W_ih + h_{t-1}*W_hh + b_h)
            h_state0 = torch.tanh(torch.bmm(in0,W_ih0) + torch.bmm(h_state0,W_hh0) + bh0)
            o0 = torch.tanh(torch.bmm(h_state0,W_out0) + bout0)
            h_state1 = torch.tanh(torch.bmm(in1,W_ih1) + torch.bmm(h_state1,W_hh1) + bh1)
            o1 = torch.tanh(torch.bmm(h_state1,W_out1) + bout1)
        else:
            h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
            h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)

        o_all=torch.zeros(B,N,OUT,device=DEVICE);o_all[:,b0m]=o0;o_all[:,b1m]=o1
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o_all[:,:,0]*gc
        ext[:,:,1]=BASE_AMP*torch.clamp(o_all[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o_all[:,:,2]*gc*0.5
        step_e0=(ext[:,b0m]**2).sum(dim=(1,2))
        step_e1=(ext[:,b1m]**2).sum(dim=(1,2))
        e0+=step_e0;e1+=step_e1
        if B==1:
            me_norm_=N*nsteps*(BASE_AMP*1.5)**2*3
            e0_trace.append((step_e0/(me_norm_+1e-8)*100*nsteps).item())
            e1_trace.append((step_e1/(me_norm_+1e-8)*100*nsteps).item())

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

    disp=pos[:,:,0].mean(dim=1)-sx
    me=N*nsteps*(BASE_AMP*1.5)**2*3
    ep0=alpha*(e0/me)*100; ep1=alpha*(e1/me)*100
    f0_fit=disp-ep0; f1_fit=disp-ep1
    return f0_fit, f1_fit, e0/me*100, e1/me*100, e0_trace, e1_trace


def run_condition(data, alpha, use_rnn, observable, n_seeds, label):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3
    input_dim = 8 if observable else 7

    if use_rnn:
        H=RNN_HIDDEN
        NG=input_dim*H + H*H + H + H*OUT + OUT + 1
        s1=np.sqrt(2.0/(input_dim+H))
    else:
        HIDDEN_FFN=32
        NW1=input_dim*HIDDEN_FFN
        NG=NW1+HIDDEN_FFN+HIDDEN_FFN*OUT+OUT+1
        s1=np.sqrt(2.0/(input_dim+HIDDEN_FFN))

    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    PSZ=200;NGENS=300;NSTEPS=600
    gaps=[];f0s=[];f1s=[];e0s=[];e1s=[];alternation_scores=[]

    for seed in range(n_seeds):
        torch.manual_seed(seed*71+13);np.random.seed(seed*71+13)
        pop0=torch.randn(PSZ,NG,device=DEVICE)*0.3
        pop1=torch.randn(PSZ,NG,device=DEVICE)*0.3
        pop0[:,:input_dim*(RNN_HIDDEN if use_rnn else 32)]*=s1/0.3
        pop1[:,:input_dim*(RNN_HIDDEN if use_rnn else 32)]*=s1/0.3
        pop0[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
        pop1[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
        pf0=torch.full((PSZ,),float('-inf'),device=DEVICE)
        pf1=torch.full((PSZ,),float('-inf'),device=DEVICE)
        t0=time.time()
        print(f"\n  [{label}] Seed {seed+1}/{n_seeds}")
        for gen in range(NGENS):
            nd=(pf0==float('-inf'))|(pf1==float('-inf'))
            if nd.any():
                ix=nd.nonzero(as_tuple=True)[0]
                ff0,ff1,_,_,_,_=sim_rnn(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,
                                        N,nper,NSTEPS,alpha,input_dim,use_rnn,observable)
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
                fr[:,:input_dim*(RNN_HIDDEN if use_rnn else 32)]*=s1/0.3
                fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
                np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
                nc=PSZ-np2.shape[0]
                t1=torch.randint(PSZ,(nc,5),device=DEVICE);p1_=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
                t2=torch.randint(PSZ,(nc,5),device=DEVICE);p2_=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
                mk=torch.rand(nc,NG,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1_],pop[p2_])
                mt=torch.rand(nc,NG,device=DEVICE)<0.15;ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
                np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
                pop.copy_(np2);pf.copy_(nf2)
        # Replay best
        bg0=pop0[0:1];bg1=pop1[0:1]
        f0_,f1_,en0_,en1_,et0,et1=sim_rnn(bg0,bg1,rp,npt,bit,sat,sbt,rlt,
                                            N,nper,NSTEPS,alpha,input_dim,use_rnn,observable)
        gap=f1_.item()-f0_.item()
        gaps.append(gap);f0s.append(f0_.item());f1s.append(f1_.item())
        e0s.append(en0_.item());e1s.append(en1_.item())
        # Alternation score: does energy expenditure alternate between bodies?
        if len(et0)>100:
            et0a=np.array(et0);et1a=np.array(et1)
            # Window-based: split into 6 windows, check if leader alternates
            w=100;alts=0
            for i in range(0,len(et0a)-w,w):
                e0w=np.sum(et0a[i:i+w]);e1w=np.sum(et1a[i:i+w])
                if i>0:
                    if (e0w>e1w)!=(prev_leader==0): alts+=1
                prev_leader=0 if e0w>e1w else 1
            alt_score=alts/max(1,(len(et0a)//w-1))
        else:
            alt_score=0
        alternation_scores.append(alt_score)
        print(f"    f0={f0_.item():+.1f} f1={f1_.item():+.1f} gap={gap:+.1f} "
              f"E0={en0_.item():.1f} E1={en1_.item():.1f} alt={alt_score:.2f}")
    return {
        "gaps":gaps,"f0":f0s,"f1":f1s,"e0":e0s,"e1":e1s,
        "alternation":alternation_scores,
        "mean_gap":float(np.mean(gaps)),"std_gap":float(np.std(gaps)),
        "mean_f0":float(np.mean(f0s)),"mean_f1":float(np.mean(f1s)),
        "mean_e0":float(np.mean(e0s)),"mean_e1":float(np.mean(e1s)),
        "mean_alt":float(np.mean(alternation_scores)),
    }


def make_figure(results):
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    fig.suptitle("Exp 28: RNN Controllers — Does Memory Break the Structural Constraint?",
                fontsize=13,fontweight="bold")
    conds=list(results.keys())
    colors=["#e74c3c","#3498db","#2ecc71"]

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
    ax.set_xticklabels(conds,fontsize=8); ax.set_ylabel("Fitness Gap (f1 - f0)")
    ax.set_title("Fitness Gap\n(>0 = freeriding)"); ax.axhline(y=0,color="gray",linestyle="--",alpha=0.5)
    ax.grid(alpha=0.3,axis="y")

    # Panel 2: Alternation score
    ax=axes[1]
    am=[results[c]["mean_alt"] for c in conds]
    ax.bar(range(len(conds)),am,color=colors[:len(conds)],alpha=0.8)
    for i,c in enumerate(conds):
        for a in results[c]["alternation"]:
            ax.scatter(i,a,color="black",alpha=0.3,s=15,zorder=5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds,fontsize=8); ax.set_ylabel("Alternation Score")
    ax.set_title("Role Alternation\n(>0 = tit-for-tat behavior)")
    ax.grid(alpha=0.3,axis="y")

    # Panel 3: Energy expenditure
    ax=axes[2]
    x=np.arange(len(conds));w=0.35
    e0s=[results[c]["mean_e0"] for c in conds]
    e1s=[results[c]["mean_e1"] for c in conds]
    ax.bar(x-w/2,e0s,w,label="Body 0 (worker?)",color="#e74c3c",alpha=0.8)
    ax.bar(x+w/2,e1s,w,label="Body 1 (freeloader?)",color="#3498db",alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(conds,fontsize=8)
    ax.set_ylabel("Energy (%)"); ax.set_title("Energy Expenditure")
    ax.legend(fontsize=7); ax.grid(alpha=0.3,axis="y")

    plt.tight_layout()
    fp=os.path.join(OUTPUT_DIR,"exp28_rnn_memory.png")
    plt.savefig(fp,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fp}")


if __name__=="__main__":
    t_total=time.time()
    gx,gy,gz,sp=10,5,4,0.35;GAP=0.5;N_SEEDS=5;ALPHA=3.0
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    print("="*70)
    print("EXP 28: RNN CONTROLLERS — MEMORY VS STRUCTURAL CONSTRAINT")
    print("="*70)

    # A) FFN baseline
    print("\n--- A: FFN Baseline ---")
    results["FFN"]=run_condition(data,ALPHA,use_rnn=False,observable=False,
                                 n_seeds=N_SEEDS,label="FFN")

    # B) RNN baseline (memory but no observation)
    print("\n--- B: RNN (memory, blind) ---")
    results["RNN"]=run_condition(data,ALPHA,use_rnn=True,observable=False,
                                  n_seeds=N_SEEDS,label="RNN")

    # C) RNN + Observable
    print("\n--- C: RNN + Observable ---")
    results["RNN+Obs"]=run_condition(data,ALPHA,use_rnn=True,observable=True,
                                      n_seeds=N_SEEDS,label="RNN+Obs")

    log_path=os.path.join(RESULTS_DIR,"exp28_rnn_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    make_figure(results)

    total_min=(time.time()-t_total)/60
    print(f"\n{'='*70}")
    print(f"EXP 28 COMPLETE ({total_min:.0f} min)")
    print(f"{'='*70}")
    for c in results:
        r=results[c]
        print(f"  {c:>10s}: gap={r['mean_gap']:+.1f}±{r['std_gap']:.1f} | "
              f"E0={r['mean_e0']:.1f} E1={r['mean_e1']:.1f} | alt={r['mean_alt']:.2f}")

    # Verdict
    ffn_gap=results["FFN"]["mean_gap"]
    rnn_gap=results["RNN"]["mean_gap"]
    rnn_obs_gap=results["RNN+Obs"]["mean_gap"]
    print(f"\n  VERDICT:")
    print(f"    FFN gap:     {ffn_gap:+.1f}")
    print(f"    RNN gap:     {rnn_gap:+.1f}")
    print(f"    RNN+Obs gap: {rnn_obs_gap:+.1f}")
    if abs(rnn_obs_gap) < abs(ffn_gap) * 0.3:
        print("    ==> Memory BREAKS Structural Constraint! Tit-for-tat emerged!")
    else:
        print("    ==> Structural Constraint is PHYSICALLY IRREDUCIBLE. Memory doesn't help.")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
