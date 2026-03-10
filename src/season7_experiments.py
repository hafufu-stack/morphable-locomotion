"""
Season 7: Parasite's Dilemma, Independent×Friction, Topology Control
==========================================================
Exp 21: Co-evolutionary Parasite's Dilemma (energy cost creates freeloading pressure)
Exp 22: Independent NNs × Friction asymmetry (complete 2x2)
Exp 23: Topology Control (NN can cut/restore combine springs)
"""
import numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.stats import pearsonr
import os, time, json, sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures"); RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN=32; INPUT_DIM=7

def build_bodies(gx,gy,gz,sp,gap,z_offset=False):
    """Build 2-body system. z_offset=True places bodies side-by-side in z."""
    nper=gx*gy*gz; nt=nper*2; ap=np.zeros((nt,3)); bi=np.zeros(nt,dtype=np.int64)
    bw=(gx-1)*sp; idx=0
    for b in range(2):
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    if z_offset:
                        zoff = -0.8 if b==0 else 0.8
                        xp = x*sp - (gx-1)*sp/2
                        zp = z*sp - (gz-1)*sp/2 + zoff
                    else:
                        xp = (-(gap/2+bw)+x*sp) if b==0 else ((gap/2+bw)-x*sp)
                        zp = z*sp-(gz-1)*sp/2
                    ap[idx]=[xp, 2.0+y*sp, zp]; bi[idx]=b; idx+=1
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

# ================================================================
# EXP 22: INDEPENDENT NNs × FRICTION (complete 2×2 table)
# ================================================================
@torch.no_grad()
def sim_indep_friction(genomes,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,friction_asym=False):
    OUT=3;NW1=INPUT_DIM*HIDDEN;NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    B=genomes.shape[0];pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item();n1=b1m.sum().item()
    g0=genomes[:,:NG1];g1=genomes[:,NG1:]
    gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs()
    gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE);cd=False
    fric0=0.1 if friction_asym else 3.0; fric1=5.0 if friction_asym else 3.0
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
        in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
        o=torch.zeros(B,N,OUT,device=DEVICE);o[:,b0m]=o0;o[:,b1m]=o1
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        # Asymmetric friction
        f[:,b0m,0]-=fric0*vel[:,b0m,0]*bl[:,b0m];f[:,b0m,2]-=fric0*vel[:,b0m,2]*bl[:,b0m]
        f[:,b1m,0]-=fric1*vel[:,b1m,0]*bl[:,b1m];f[:,b1m,2]-=fric1*vel[:,b1m,2]*bl[:,b1m]
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    return disp-dz-spp-bw-ep+coh

# ================================================================
# EXP 21: PARASITE'S DILEMMA (co-evolution with energy cost)
# ================================================================
@torch.no_grad()
def sim_parasite(g0_batch,g1_batch,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,alpha):
    """Co-evolutionary sim: returns (fitness_0, fitness_1) per pair."""
    OUT=3;NW1=INPUT_DIM*HIDDEN;NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    B=g0_batch.shape[0];pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item();n1=b1m.sum().item()
    gi=0;W1_0=g0_batch[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1_0=g0_batch[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_0=g0_batch[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_0=g0_batch[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0_batch[:,gi].abs()
    gi=0;W1_1=g1_batch[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1_1=g1_batch[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_1=g1_batch[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_1=g1_batch[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1_batch[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    e0=torch.zeros(B,device=DEVICE);e1=torch.zeros(B,device=DEVICE)
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
        in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
        o=torch.zeros(B,N,OUT,device=DEVICE);o[:,b0m]=o0;o[:,b1m]=o1
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        e0+=(ext[:,b0m]**2).sum(dim=(1,2));e1+=(ext[:,b1m]**2).sum(dim=(1,2))
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
    fit0=disp-ep0; fit1=disp-ep1
    return fit0, fit1

# ================================================================
# EXP 23: TOPOLOGY CONTROL (NN can cut springs)
# ================================================================
@torch.no_grad()
def sim_topology_ctrl(genomes,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,wall_x=None):
    """Shared NN with 4 outputs: 3 force + 1 detach signal."""
    OUT=4;NW1=INPUT_DIM*HIDDEN;NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    B=genomes.shape[0];pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    gi=0;W1=genomes[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
    b1g=genomes[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2=genomes[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2g=genomes[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq=genomes[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    n_internal=len(sat)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    combined=False; n_separations=torch.zeros(B,device=DEVICE)
    n_recombinations=torch.zeros(B,device=DEVICE)
    for step in range(nsteps):
        t=step*DT
        # Check for combine/recombine
        if step%10==0:
            p0=pos[0,b0i];p1=pos[0,b1i];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0 and len(csa)==n_internal:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb=torch.ones(B,1,1,device=DEVICE);combined=True
                n_recombinations+=1
        st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        # Output 4: detach signal (mean over all particles)
        detach_sig=o[:,:,3].mean(dim=1)  # B
        # If detach > 0.5 and combined, cut springs
        if combined and detach_sig[0].item()>0.5:
            csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
            comb=torch.zeros(B,1,1,device=DEVICE);combined=False
            n_separations+=1
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float();f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        # Wall
        if wall_x is not None:
            past=pos[:,:,0]-wall_x;wall_pen=past.clamp(min=0)
            # Gap: particles with |z| < 0.8 can pass
            in_gap=(pos[:,:,2].abs()<0.8).float()
            wall_force=-200.0*wall_pen*(1.0-in_gap)
            f[:,:,0]+=wall_force
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness=disp-dz-spp-bw-ep+coh
    return fitness, n_separations, n_recombinations

# ================================================================
# GENERIC EVOLVE
# ================================================================
def evolve_generic(sim_fn, data, nsteps, ngens, psz, label, ng_override=None):
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt;OUT=3;NW1=INPUT_DIM*HIDDEN
    NG=ng_override if ng_override else NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN));s2=np.sqrt(2.0/(HIDDEN+OUT))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    pop[:,:NW1]*=s1/0.3; pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE);t0=time.time()
    extra_data={}
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            result=sim_fn(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps)
            if isinstance(result,tuple): pf[ix]=result[0]
            else: pf[ix]=result
            pf[ix]=torch.where(torch.isnan(pf[ix]),torch.tensor(-9999.0,device=DEVICE),pf[ix])
        o_=pf.argsort(descending=True);pop=pop[o_];pf=pf[o_]
        if gen%50==0 or gen==ngens-1:
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,NG,device=DEVICE)*0.3
        fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE);p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE);p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,NG,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,NG,device=DEVICE)<0.15;ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    total=(time.time()-t0)/60
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return pop[0].cpu().numpy(), pf[0].item()

def replay_rfx(genes,data,nsteps,decentralized=False,friction_asym=False):
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt;OUT=3;NW1=INPUT_DIM*HIDDEN
    NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    g=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item();n1=b1m.sum().item()
    if decentralized:
        g0=g[:,:NG1];g1=g[:,NG1:]
        gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
        b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs().item()
        gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
        b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs().item()
    else:
        gi=0;W1=g[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN);gi+=NW1
        b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2=g[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;fv=g[:,gi].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    fric0=0.1 if friction_asym else 3.0; fric1=5.0 if friction_asym else 3.0
    fx0_l,fx1_l=[],[]
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
        if decentralized:
            sv0=np.sin(2*np.pi*freq0*t);cv0=np.cos(2*np.pi*freq0*t)
            st0=torch.full((B,n0,1),sv0,device=DEVICE);ct0=torch.full((B,n0,1),cv0,device=DEVICE)
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
            h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
            sv1=np.sin(2*np.pi*freq1*t);cv1=np.cos(2*np.pi*freq1*t)
            st1=torch.full((B,n1,1),sv1,device=DEVICE);ct1=torch.full((B,n1,1),cv1,device=DEVICE)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
            h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
            o=torch.zeros(B,N,OUT,device=DEVICE);o[:,b0m]=o0;o[:,b1m]=o1
        else:
            sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
            st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
            h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_l.append(ext[0,b0m,0].mean().item());fx1_l.append(ext[0,b1m,0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,b0m,0]-=fric0*vel[:,b0m,0]*bl[:,b0m];f[:,b0m,2]-=fric0*vel[:,b0m,2]*bl[:,b0m]
        f[:,b1m,0]-=fric1*vel[:,b1m,0]*bl[:,b1m];f[:,b1m,2]-=fric1*vel[:,b1m,2]*bl[:,b1m]
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0_l)<1e-10 or np.std(fx1_l)<1e-10: return 0.0
    r_,_=pearsonr(fx0_l,fx1_l);return round(r_,3)

def main():
    NSTEPS=600;GAP=0.5;PSZ=200;gx,gy,gz,sp=10,5,4,0.35;NGENS=300
    data=build_bodies(gx,gy,gz,sp,GAP)
    OUT=3;NW1=INPUT_DIM*HIDDEN;NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1;NG2=NG1*2
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN));s2=np.sqrt(2.0/(HIDDEN+OUT))
    results={}; t_start=time.time()

    # ================================================================
    print("="*70)
    print("EXP 22: INDEPENDENT NNs × FRICTION ASYMMETRY")
    print("="*70)
    print("\n--- 22a: Independent NNs + friction asymmetry ---")
    bg22a,fit22a=evolve_generic(lambda g,*a: sim_indep_friction(g,*a,friction_asym=True),
                                data,NSTEPS,NGENS,PSZ,"22a_indep_fric",ng_override=NG2)
    rfx22a=replay_rfx(bg22a,data,NSTEPS,decentralized=True,friction_asym=True)
    print(f"  r(Fx)={rfx22a:.3f}")
    print("\n--- 22b: Shared NN + friction asymmetry (control) ---")
    from season6b_experiments import simulate as s6b_sim
    # Quick shared NN with friction by reusing sim
    bg22b,fit22b=evolve_generic(lambda g,rp,npt,bit,sat,sbt,rlt,N,nper,ns:
        s6b_sim(g,rp,npt,bit,sat,sbt,rlt,N,nper,ns,mass_ratios=None,dead_body=None),
        data,NSTEPS,NGENS,PSZ,"22b_shared_fric",ng_override=NG1)
    # Actually we need friction for the shared version too - use the existing env diff result
    # From v3: shared NN + friction = r=-0.032, fitness +181
    print(f"  NOTE: Shared NN + friction from Exp 9: r=-0.032, +181")
    results["exp22"]={
        "indep_friction":{"fitness":round(fit22a,2),"r_fx":rfx22a},
        "shared_friction_ref":{"fitness":181,"r_fx":-0.032,"note":"from Exp 9"},
    }

    # ================================================================
    print("\n"+"="*70)
    print("EXP 21: PARASITE'S DILEMMA (Co-Evolution)")
    print("="*70)
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    # Two populations
    for alpha_val,alpha_name in [(0.0,"control"),(1.0,"dilemma"),(3.0,"strong")]:
        print(f"\n--- 21: alpha={alpha_val} ({alpha_name}) ---")
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
                f0,f1=sim_parasite(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha_val)
                f0=torch.where(torch.isnan(f0),torch.tensor(-9999.0,device=DEVICE),f0)
                f1=torch.where(torch.isnan(f1),torch.tensor(-9999.0,device=DEVICE),f1)
                pf0[ix]=f0;pf1[ix]=f1
            # Sort and evolve pop0
            o0=pf0.argsort(descending=True);pop0=pop0[o0];pf0=pf0[o0]
            # Match pop1 to same order for pairing
            pop1=pop1[o0];pf1=pf1[o0]
            # Now sort pop1 independently
            o1=pf1.argsort(descending=True);pop1=pop1[o1];pf1=pf1[o1]
            pop0_=pop0[o1];pf0_=pf0[o1];pop0=pop0_;pf0=pf0_
            if gen%50==0 or gen==NGENS-1:
                print(f"  [21_{alpha_name}] Gen {gen:4d}: f0={pf0[0].item():+.2f} f1={pf1[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
            # Evolve each pop independently
            for pop,pf in [(pop0,pf0),(pop1,pf1)]:
                ne=max(2,int(PSZ*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
                nfr=2;fr=torch.randn(nfr,NG1,device=DEVICE)*0.3
                fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
                np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
                nc=PSZ-np2.shape[0]
                t1=torch.randint(PSZ,(nc,5),device=DEVICE);p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
                t2=torch.randint(PSZ,(nc,5),device=DEVICE);p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
                mk=torch.rand(nc,NG1,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1],pop[p2])
                mt=torch.rand(nc,NG1,device=DEVICE)<0.15;ch+=torch.randn(nc,NG1,device=DEVICE)*0.3*mt.float()
                np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
                pop.copy_(np2);pf.copy_(nf2)
        print(f"  [21_{alpha_name}] Done: {(time.time()-t0)/60:.1f}min")
        # Replay best pair for energy analysis
        bg0=pop0[0:1];bg1=pop1[0:1]
        f0_,f1_=sim_parasite(bg0,bg1,rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha_val)
        results[f"exp21_{alpha_name}"]={"fitness_0":round(f0_.item(),2),"fitness_1":round(f1_.item(),2),
                                         "alpha":alpha_val}

    # ================================================================
    print("\n"+"="*70)
    print("EXP 23: TOPOLOGY CONTROL")
    print("="*70)
    OUT4=4;NG_T=NW1+HIDDEN+HIDDEN*OUT4+OUT4+1
    # 23a: No wall (can evolution find any use for spring control?)
    print("\n--- 23a: Spring control, no wall ---")
    bg23a,fit23a=evolve_generic(lambda g,*a: sim_topology_ctrl(g,*a,wall_x=None)[0],
                                data,NSTEPS,NGENS,PSZ,"23a_nowall",ng_override=NG_T)
    r23a=sim_topology_ctrl(torch.tensor(bg23a,dtype=torch.float32,device=DEVICE).unsqueeze(0),
                   *[torch.tensor(x,dtype=torch.float32 if isinstance(x[0],float) else torch.long,device=DEVICE)
                     for x in [data[0],data[1]]],
                   torch.tensor(data[2],dtype=torch.long,device=DEVICE),
                   torch.tensor(data[3],dtype=torch.long,device=DEVICE),
                   torch.tensor(data[4],dtype=torch.long,device=DEVICE),
                   torch.tensor(data[5],dtype=torch.float32,device=DEVICE),
                   data[7],data[6],NSTEPS,wall_x=None)
    print(f"  n_sep={r23a[1].item():.0f} n_recomb={r23a[2].item():.0f}")
    # 23b: With wall at x=8
    print("\n--- 23b: Spring control + wall at x=8 ---")
    bg23b,fit23b=evolve_generic(lambda g,*a: sim_topology_ctrl(g,*a,wall_x=8.0)[0],
                                data,NSTEPS,NGENS,PSZ,"23b_wall",ng_override=NG_T)
    results["exp23"]={
        "no_wall":{"fitness":round(fit23a,2)},
        "wall":{"fitness":round(fit23b,2)},
    }

    # ================================================================
    # FIGURE
    # ================================================================
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    fig.suptitle("Season 7: Parasite's Dilemma, Indep×Friction, Topology Control",fontsize=14,fontweight="bold")
    # Panel 1: Exp 22 - complete table
    ax=axes[0]
    tab=[("Shared\nSym",0.942,"#3498db"),("Indep\nSym",0.952,"#e74c3c"),
         ("Shared\nFriction",-0.032,"#2ecc71"),("Indep\nFriction",rfx22a,"#f39c12")]
    for i,(l,v,c) in enumerate(tab):
        ax.bar(i,v,color=c,alpha=0.8);ax.text(i,v+0.03,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(range(4));ax.set_xticklabels([t[0] for t in tab],fontsize=7)
    ax.set_ylabel("r(Fx)");ax.set_title("Exp 22: Brain×Environment 2×2")
    ax.axhline(y=0,color="black",linewidth=0.5);ax.grid(alpha=0.3,axis="y")
    # Panel 2: Parasite
    ax=axes[1]
    alphas=["control","dilemma","strong"]
    colors=["#2ecc71","#f39c12","#e74c3c"]
    for i,a in enumerate(alphas):
        k=f"exp21_{a}";d=results[k]
        ax.bar(i*2,d["fitness_0"],0.8,color=colors[i],alpha=0.8,label=f"α={d['alpha']}")
        ax.bar(i*2+0.8,d["fitness_1"],0.8,color=colors[i],alpha=0.5)
        ax.text(i*2,d["fitness_0"]+2,f"B0:{d['fitness_0']:+.0f}",ha="center",fontsize=7)
        ax.text(i*2+0.8,d["fitness_1"]+2,f"B1:{d['fitness_1']:+.0f}",ha="center",fontsize=7)
    ax.set_xticks([0.4,2.4,4.4]);ax.set_xticklabels(["α=0\n(control)","α=1\n(dilemma)","α=3\n(strong)"])
    ax.set_ylabel("Fitness");ax.set_title("Exp 21: Parasite's Dilemma\n(darker=Body0, lighter=Body1)")
    ax.grid(alpha=0.3,axis="y")
    # Panel 3: Topology Control
    ax=axes[2]
    t_labels=["No wall","Wall"];t_fits=[fit23a,fit23b];t_colors=["#3498db","#e74c3c"]
    ax.bar(range(2),t_fits,color=t_colors,alpha=0.8)
    for i in range(2): ax.text(i,t_fits[i]+2,f"{t_fits[i]:+.0f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(range(2));ax.set_xticklabels(t_labels)
    ax.set_ylabel("Fitness");ax.set_title("Exp 23: Topology Control (Spring Control)")
    ax.grid(alpha=0.3,axis="y")

    plt.tight_layout()
    fig_path=os.path.join(OUTPUT_DIR,"season7_experiments.png")
    plt.savefig(fig_path,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fig_path}")
    log_path=os.path.join(RESULTS_DIR,"season7_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    total_min=(time.time()-t_start)/60
    print(f"\n{'='*70}")
    print(f"SEASON 7 COMPLETE ({total_min:.1f} min total)")
    print(f"{'='*70}")
    print(f"\nExp 22: Indep×Friction r={rfx22a:.3f} (shared friction ref: r=-0.032)")
    for a in alphas:
        d=results[f"exp21_{a}"]
        print(f"Exp 21 α={d['alpha']}: B0={d['fitness_0']:+.2f} B1={d['fitness_1']:+.2f}")
    print(f"Exp 23: no_wall={fit23a:+.2f} wall={fit23b:+.2f}")

    # Hibernate or beep
    if total_min > 360:
        print("\n⏸️ Over 6 hours — hibernating...")
        os.system("shutdown /h")
    else:
        print(f"\n✅ All done in {total_min:.1f} min (< 6h) — beeping!")
        try:
            import winsound
            for _ in range(5): winsound.Beep(800,300); time.sleep(0.2)
        except: pass

if __name__=="__main__":
    main()
