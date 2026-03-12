"""
Statistical Validation: Multi-seed error bars for key experiments
================================================================
Part 1: Phantom Sync x20 seeds (~40min)
Part 2: No-Cost Differentiation x20 seeds (~30min)
Part 3: Parasite alpha=3 x10 seeds (~100min)
Total: ~3 hours.
"""
import numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.stats import pearsonr
import os, time, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures"); RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN=32; INPUT_DIM=7

def build_2body(gx,gy,gz,sp,gap):
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

def build_3body(gx,gy,gz,sp,gap):
    nper=gx*gy*gz; nt=nper*3; ap=np.zeros((nt,3)); bi=np.zeros(nt,dtype=np.int64)
    bw=(gx-1)*sp; offsets=[-(gap+bw),0.0,(gap+bw)]; idx=0
    for b in range(3):
        cx=offsets[b]
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp=cx+(x-(gx-1)/2)*sp
                    ap[idx]=[xp,2.0+y*sp,z*sp-(gz-1)*sp/2]; bi[idx]=b; idx+=1
    sa,sb,rl=[],[],[]
    for b in range(3):
        m=np.where(bi==b)[0]; bp=ap[m]; tri=Delaunay(bp); edges=set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a);sb.append(bb);rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_=np.zeros_like(ap)
    for b in range(3):
        m=bi==b
        for d in range(3):
            vn,vx=ap[m,d].min(),ap[m,d].max(); np_[m,d]=2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap,np_,bi,np.array(sa),np.array(sb),np.array(rl),nper,nt

# ================================================================
# EVOLVE GENERIC (shared NN, 2-body)
# ================================================================
def evolve_shared(data,nsteps,ngens,psz,per_body_mass=None,friction_map=None):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3; NW1=INPUT_DIM*HIDDEN
    NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN)); s2=np.sqrt(2.0/(HIDDEN+OUT))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    pop[:,:NW1]*=s1/0.3; pop[:,NW1:NW1+HIDDEN]=0
    i2=NW1+HIDDEN; pop[:,i2:i2+HIDDEN*OUT]*=s2/0.3; pop[:,i2+HIDDEN*OUT:i2+HIDDEN*OUT+OUT]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    mass=torch.ones(1,N,device=DEVICE)
    if per_body_mass: mass[:,b0m]=per_body_mass[0]; mass[:,b1m]=per_body_mass[1]
    fric_default=3.0; fric=torch.full((N,),fric_default,device=DEVICE)
    if friction_map: fric[b0m]=friction_map[0]; fric[b1m]=friction_map[1]
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]; BZ=ix.shape[0]
            g=pop[ix]; pos=rp.unsqueeze(0).expand(BZ,-1,-1).clone()
            vel=torch.zeros(BZ,N,3,device=DEVICE)
            gi=0;W1=g[:,gi:gi+NW1].reshape(BZ,INPUT_DIM,HIDDEN);gi+=NW1
            b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
            W2=g[:,gi:gi+HIDDEN*OUT].reshape(BZ,HIDDEN,OUT);gi+=HIDDEN*OUT
            b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
            freq=g[:,gi].abs()
            sx=pos[:,:,0].mean(dim=1)
            bid=bit.float().unsqueeze(0).unsqueeze(2).expand(BZ,N,1)
            ni=npt.unsqueeze(0).expand(BZ,-1,-1)
            csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
            comb=torch.zeros(BZ,1,1,device=DEVICE);te=torch.zeros(BZ,device=DEVICE);cd=False
            mass_e=mass.expand(BZ,N)
            for step in range(nsteps):
                t=step*DT
                if not cd and step%10==0:
                    p0=pos[0,b0i];p1=pos[0,b1i];ds=torch.cdist(p0,p1)
                    cl=(ds<1.2).nonzero(as_tuple=False)
                    if cl.shape[0]>0:
                        nn_=min(cl.shape[0],500)
                        csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                        crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                        comb=torch.ones(BZ,1,1,device=DEVICE);cd=True
                st=torch.sin(2*np.pi*freq*t).reshape(BZ,1,1).expand(BZ,N,1)
                ct=torch.cos(2*np.pi*freq*t).reshape(BZ,1,1).expand(BZ,N,1)
                nn_in=torch.cat([st,ct,ni,bid,comb.expand(BZ,N,1)],dim=2)
                h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
                og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
                ext=torch.zeros(BZ,N,3,device=DEVICE)
                ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
                ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
                f=torch.zeros(BZ,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass_e
                pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
                dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
                rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
                ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
                f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(BZ,-1,3),ft_)
                f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(BZ,-1,3),-ft_)
                pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
                bl=(pos[:,:,1]<GROUND_Y).float()
                f[:,:,0]-=fric.unsqueeze(0)*vel[:,:,0]*bl;f[:,:,2]-=fric.unsqueeze(0)*vel[:,:,2]*bl
                f-=DRAG*vel;f+=ext
                acc=f/(mass_e.clamp(min=0.01).unsqueeze(2))
                vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
            disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
            sp_=pos.max(dim=1).values-pos.min(dim=1).values
            spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
            me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
            c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
            coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
            fit=disp-dz-spp-bw-ep+coh
            fit=torch.where(torch.isnan(fit),torch.tensor(-9999.0,device=DEVICE),fit)
            pf[ix]=fit
        o_=pf.argsort(descending=True);pop=pop[o_];pf=pf[o_]
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,NG,device=DEVICE)*0.3
        fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE);p1_=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE);p2_=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,NG,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1_],pop[p2_])
        mt=torch.rand(nc,NG,device=DEVICE)<0.15;ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    return pop[0].cpu().numpy(), pf[0].item()

def replay_rfx(genes,data,nsteps,per_body_mass=None,friction_map=None):
    """Replay and return r(Fx) correlation."""
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3; NW1=INPUT_DIM*HIDDEN
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    g=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gi=0;W1=g[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2=g[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
    fv=g[:,gi].abs().item()
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    mass=torch.ones(B,N,device=DEVICE)
    if per_body_mass: mass[:,b0m]=per_body_mass[0]; mass[:,b1m]=per_body_mass[1]
    fric_default=3.0; fric=torch.full((N,),fric_default,device=DEVICE)
    if friction_map: fric[b0m]=friction_map[0]; fric[b1m]=friction_map[1]
    fx0,fx1=[],[]
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
        sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
        st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0.append(ext[0,b0m,0].mean().item());fx1.append(ext[0,b1m,0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=fric.unsqueeze(0)*vel[:,:,0]*bl;f[:,:,2]-=fric.unsqueeze(0)*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        acc=f/(mass.clamp(min=0.01).unsqueeze(2))
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0)<1e-10 or np.std(fx1)<1e-10: return 0.0
    r_,_=pearsonr(fx0,fx1); return round(r_,4)

# ================================================================
# PART 1: PHANTOM SYNC x20 seeds
# ================================================================
def run_phantom_sync(n_seeds=20):
    from season6_experiments import evolve_3body, phantom_replay, build_bodies as build_bodies_s6
    print("\n"+"="*70)
    print(f"PART 1: PHANTOM SYNCHRONIZATION x{n_seeds} SEEDS")
    print("="*70)
    NSTEPS=600;GAP=0.5;PSZ=200;NGENS=300
    gx,gy,gz,sp=10,5,4,0.35
    mrs=(5.0,1.0,5.0)
    r02_nocut_all=[];r02_cut300_all=[];disp_all=[]
    for seed in range(n_seeds):
        torch.manual_seed(seed*42+7); np.random.seed(seed*42+7)
        data3=build_bodies_s6(gx,gy,gz,sp,GAP,3)
        print(f"\n  Seed {seed+1}/{n_seeds}")
        best=evolve_3body(data3,NSTEPS,NGENS,PSZ,f"ps_s{seed}",mrs)
        r_nc,d_nc,_=phantom_replay(best,data3,NSTEPS,mrs,cut_step=None)
        r_c3,d_c3,_=phantom_replay(best,data3,NSTEPS,mrs,cut_step=300)
        r02_nocut_all.append(r_nc['r02']);r02_cut300_all.append(r_c3['r02'])
        disp_all.append(d_nc)
        print(f"    r02_nocut={r_nc['r02']:.3f} r02_cut300={r_c3['r02']:.3f} disp={d_nc}")
    return {
        "r02_nocut": r02_nocut_all, "r02_cut300": r02_cut300_all, "disp": disp_all,
        "mean_nocut": np.mean(r02_nocut_all), "std_nocut": np.std(r02_nocut_all),
        "mean_cut300": np.mean(r02_cut300_all), "std_cut300": np.std(r02_cut300_all),
    }

# ================================================================
# PART 2: NO-COST DIFFERENTIATION x20 seeds
# ================================================================
def run_nocost_diff(n_seeds=20):
    print("\n"+"="*70)
    print(f"PART 2: NO-COST DIFFERENTIATION (Friction 0.1 vs 5.0) x{n_seeds} SEEDS")
    print("="*70)
    NSTEPS=600;GAP=0.5;PSZ=200;NGENS=150
    gx,gy,gz,sp=10,5,4,0.35
    rfx_sym_all=[];fit_sym_all=[];rfx_fric_all=[];fit_fric_all=[]
    for seed in range(n_seeds):
        torch.manual_seed(seed*37+13); np.random.seed(seed*37+13)
        data=build_2body(gx,gy,gz,sp,GAP)
        print(f"\n  Seed {seed+1}/{n_seeds}")
        # Symmetric baseline
        bg_s,fit_s=evolve_shared(data,NSTEPS,NGENS,PSZ)
        r_s=replay_rfx(bg_s,data,NSTEPS)
        rfx_sym_all.append(r_s);fit_sym_all.append(fit_s)
        # Friction asymmetry
        bg_f,fit_f=evolve_shared(data,NSTEPS,NGENS,PSZ,friction_map=(0.1,5.0))
        r_f=replay_rfx(bg_f,data,NSTEPS,friction_map=(0.1,5.0))
        rfx_fric_all.append(r_f);fit_fric_all.append(fit_f)
        print(f"    Sym: r={r_s:.3f} fit={fit_s:+.1f} | Fric: r={r_f:.3f} fit={fit_f:+.1f}")
    return {
        "rfx_sym": rfx_sym_all, "fit_sym": fit_sym_all,
        "rfx_fric": rfx_fric_all, "fit_fric": fit_fric_all,
        "mean_sym_r": np.mean(rfx_sym_all), "std_sym_r": np.std(rfx_sym_all),
        "mean_fric_r": np.mean(rfx_fric_all), "std_fric_r": np.std(rfx_fric_all),
        "mean_sym_fit": np.mean(fit_sym_all), "mean_fric_fit": np.mean(fit_fric_all),
    }

# ================================================================
# PART 3: PARASITE alpha=3 x10 seeds
# ================================================================
def run_parasite(n_seeds=10, alpha=3.0):
    from exp21b_parasite_sweep import sim_parasite, build_bodies as build_parasite
    print("\n"+"="*70)
    print(f"PART 3: PARASITE'S DILEMMA (alpha={alpha}) x{n_seeds} SEEDS")
    print("="*70)
    NSTEPS=600;GAP=0.5;PSZ=200;NGENS=300
    gx,gy,gz,sp=10,5,4,0.35
    OUT=3;NW1=INPUT_DIM*HIDDEN;NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN))
    gaps_all=[];f0_all=[];f1_all=[];e0_all=[];e1_all=[]
    for seed in range(n_seeds):
        torch.manual_seed(seed*53+19); np.random.seed(seed*53+19)
        data=build_parasite(gx,gy,gz,sp,GAP)
        ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt
        rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
        npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
        bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
        sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
        sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
        rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
        print(f"\n  Seed {seed+1}/{n_seeds}")
        pop0=torch.randn(PSZ,NG1,device=DEVICE)*0.3
        pop1=torch.randn(PSZ,NG1,device=DEVICE)*0.3
        pop0[:,:NW1]*=s1/0.3;pop0[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
        pop1[:,:NW1]*=s1/0.3;pop1[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
        pf0=torch.full((PSZ,),float('-inf'),device=DEVICE)
        pf1=torch.full((PSZ,),float('-inf'),device=DEVICE)
        t0=time.time()
        for gen in range(NGENS):
            nd=(pf0==float('-inf'))|(pf1==float('-inf'))
            if nd.any():
                ix=nd.nonzero(as_tuple=True)[0]
                f0,f1,_,_,_=sim_parasite(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha)
                f0=torch.where(torch.isnan(f0),torch.tensor(-9999.0,device=DEVICE),f0)
                f1=torch.where(torch.isnan(f1),torch.tensor(-9999.0,device=DEVICE),f1)
                pf0[ix]=f0;pf1[ix]=f1
            o0=pf0.argsort(descending=True);pop0=pop0[o0];pf0=pf0[o0];pop1=pop1[o0];pf1=pf1[o0]
            o1=pf1.argsort(descending=True);pop1=pop1[o1];pf1=pf1[o1];pop0=pop0[o1];pf0=pf0[o1]
            if gen%100==0 or gen==NGENS-1:
                print(f"    Gen {gen}: f0={pf0[0].item():+.1f} f1={pf1[0].item():+.1f} ({(time.time()-t0)/60:.1f}min)")
            for pop,pf in [(pop0,pf0),(pop1,pf1)]:
                ne=max(2,int(PSZ*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
                nfr=2;fr=torch.randn(nfr,NG1,device=DEVICE)*0.3
                fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
                np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
                nc=PSZ-np2.shape[0]
                t1=torch.randint(PSZ,(nc,5),device=DEVICE);p1_=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
                t2=torch.randint(PSZ,(nc,5),device=DEVICE);p2_=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
                mk=torch.rand(nc,NG1,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1_],pop[p2_])
                mt=torch.rand(nc,NG1,device=DEVICE)<0.15;ch+=torch.randn(nc,NG1,device=DEVICE)*0.3*mt.float()
                np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
                pop.copy_(np2);pf.copy_(nf2)
        bg0=pop0[0:1];bg1=pop1[0:1]
        f0_,f1_,disp_,en0_,en1_=sim_parasite(bg0,bg1,rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha)
        gap_v=f1_.item()-f0_.item()
        gaps_all.append(gap_v);f0_all.append(f0_.item());f1_all.append(f1_.item())
        e0_all.append(en0_.item());e1_all.append(en1_.item())
        print(f"    f0={f0_.item():+.1f} f1={f1_.item():+.1f} gap={gap_v:+.1f} E0={en0_.item():.1f} E1={en1_.item():.1f}")
    return {
        "gaps": gaps_all, "f0": f0_all, "f1": f1_all, "e0": e0_all, "e1": e1_all,
        "mean_gap": np.mean(gaps_all), "std_gap": np.std(gaps_all),
        "mean_f0": np.mean(f0_all), "mean_f1": np.mean(f1_all),
    }

# ================================================================
# FIGURE
# ================================================================
def make_figure(ps_res, nc_res, pa_res):
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    fig.suptitle("Statistical Validation: Multi-Seed Error Bars",fontsize=14,fontweight="bold")

    # Panel 1: Phantom Sync
    ax=axes[0]
    conditions=["No cut","Cut@300"]
    means=[ps_res["mean_nocut"],ps_res["mean_cut300"]]
    stds=[ps_res["std_nocut"],ps_res["std_cut300"]]
    bars=ax.bar(conditions,means,yerr=stds,capsize=8,color=["#3498db","#e74c3c"],alpha=0.8)
    for i,b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()/2,means[i]+stds[i]+0.02,
                f"{means[i]:.3f}±{stds[i]:.3f}",ha="center",fontsize=9,fontweight="bold")
    n=len(ps_res["r02_nocut"])
    ax.scatter([0]*n,ps_res["r02_nocut"],color="navy",alpha=0.4,zorder=5,s=20)
    ax.scatter([1]*n,ps_res["r02_cut300"],color="darkred",alpha=0.4,zorder=5,s=20)
    ax.set_ylabel("r(Fx) mirror sync r02")
    ax.set_title(f"Phantom Sync (5:1:5, N={n})\nIs neural sync robust?")
    ax.grid(alpha=0.3,axis="y");ax.set_ylim(-0.5,1.2)

    # Panel 2: No-Cost Differentiation
    ax=axes[1]
    conditions2=["Symmetric","Friction\n(0.1 vs 5.0)"]
    means2=[nc_res["mean_sym_r"],nc_res["mean_fric_r"]]
    stds2=[nc_res["std_sym_r"],nc_res["std_fric_r"]]
    bars2=ax.bar(conditions2,means2,yerr=stds2,capsize=8,color=["#3498db","#e74c3c"],alpha=0.8)
    for i,b in enumerate(bars2):
        ax.text(b.get_x()+b.get_width()/2,means2[i]+stds2[i]+0.02,
                f"r={means2[i]:.3f}±{stds2[i]:.3f}",ha="center",fontsize=9,fontweight="bold")
    n2=len(nc_res["rfx_sym"])
    ax.scatter([0]*n2,nc_res["rfx_sym"],color="navy",alpha=0.4,zorder=5,s=20)
    ax.scatter([1]*n2,nc_res["rfx_fric"],color="darkred",alpha=0.4,zorder=5,s=20)
    ax.axhline(y=0.3,color="gray",linestyle="--",alpha=0.4,label="Diff threshold")
    ax.set_ylabel("r(Fx)")
    fit_s=nc_res["mean_sym_fit"];fit_f=nc_res["mean_fric_fit"]
    ax.set_title(f"No-Cost Diff (N={n2})\nSym fit={fit_s:+.0f} | Fric fit={fit_f:+.0f}")
    ax.grid(alpha=0.3,axis="y");ax.set_ylim(-0.5,1.2);ax.legend(fontsize=8)

    # Panel 3: Parasite
    ax=axes[2]
    n3=len(pa_res["gaps"])
    ax.bar(["Body 0\n(worker)","Body 1\n(freeloader)"],
           [pa_res["mean_f0"],pa_res["mean_f1"]],
           yerr=[np.std(pa_res["f0"]),np.std(pa_res["f1"])],
           capsize=8,color=["#e74c3c","#3498db"],alpha=0.8)
    ax.scatter([0]*n3,pa_res["f0"],color="darkred",alpha=0.4,zorder=5,s=20)
    ax.scatter([1]*n3,pa_res["f1"],color="navy",alpha=0.4,zorder=5,s=20)
    gap_m=pa_res["mean_gap"];gap_s=pa_res["std_gap"]
    ax.set_ylabel("Fitness")
    ax.set_title(f"Parasite α=3 (N={n3})\nGap={gap_m:+.1f}±{gap_s:.1f}")
    ax.grid(alpha=0.3,axis="y")

    plt.tight_layout()
    fig_path=os.path.join(OUTPUT_DIR,"stat_validation.png")
    plt.savefig(fig_path,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fig_path}")

# ================================================================
# MAIN
# ================================================================
if __name__=="__main__":
    t_total=time.time()
    all_results={}

    ps_res=run_phantom_sync(n_seeds=20)
    all_results["phantom_sync"]=ps_res
    print(f"\n  PHANTOM SYNC: r02_nocut={ps_res['mean_nocut']:.3f}+/-{ps_res['std_nocut']:.3f}")
    print(f"                r02_cut300={ps_res['mean_cut300']:.3f}+/-{ps_res['std_cut300']:.3f}")

    nc_res=run_nocost_diff(n_seeds=20)
    all_results["nocost_diff"]=nc_res
    print(f"\n  NO-COST DIFF: sym r={nc_res['mean_sym_r']:.3f}+/-{nc_res['std_sym_r']:.3f}")
    print(f"                fric r={nc_res['mean_fric_r']:.3f}+/-{nc_res['std_fric_r']:.3f}")

    pa_res=run_parasite(n_seeds=10, alpha=3.0)
    all_results["parasite"]=pa_res
    print(f"\n  PARASITE: gap={pa_res['mean_gap']:+.1f}+/-{pa_res['std_gap']:.1f}")

    make_figure(ps_res, nc_res, pa_res)

    log_path=os.path.join(RESULTS_DIR,"stat_validation_log.json")
    with open(log_path,"w") as f: json.dump(all_results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    total_min=(time.time()-t_total)/60
    print(f"\n{'='*70}")
    print(f"ALL STATISTICAL VALIDATION COMPLETE ({total_min:.0f} min)")
    print(f"{'='*70}")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800,300); time.sleep(0.2)
    except: pass
    print("\nDone!")
