"""
Season 6B: Decentralized×Asymmetric + Dead Body
================================================
Exp 19: Independent brains + 3:1 mass (complete 2x2 table)
Exp 20: Dead Body (one NN frozen, other must drag)
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

# ================================================================
# SIMULATE: supports decentralized, mass_ratios, dead_body
# ================================================================
@torch.no_grad()
def simulate(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
             decentralized=False, mass_ratios=None, dead_body=None):
    """
    decentralized: if True, genome has 2 NNs concatenated
    mass_ratios: (m0, m1) tuple
    dead_body: 0 or 1 to freeze that body's forces to zero
    """
    OUT=3; NW1=INPUT_SIZE*HIDDEN; NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    B=genomes.shape[0]; pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0; b1m=bit==1; b0i=b0m.nonzero(as_tuple=True)[0]; b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item(); n1=b1m.sum().item()

    if decentralized:
        g0=genomes[:,:NG1]; g1=genomes[:,NG1:]
        gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
        b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs()
        gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
        b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs()
    else:
        gi=0;W1=genomes[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
        b1g=genomes[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2=genomes[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2g=genomes[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq=genomes[:,gi].abs()

    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE);cd=False
    mass=torch.ones(B,N,device=DEVICE)
    if mass_ratios:
        mass[:,b0m]=mass_ratios[0]; mass[:,b1m]=mass_ratios[1]

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
            st0=torch.sin(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
            ct0=torch.cos(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
            in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
            h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
            st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
            ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
            in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
            h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
            o=torch.zeros(B,N,OUT,device=DEVICE);o[:,b0m]=o0;o[:,b1m]=o1
        else:
            st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
            ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
            h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)

        # Dead body: zero out forces for one body
        if dead_body is not None:
            if dead_body==0: o[:,b0m]=0
            else: o[:,b1m]=0

        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
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
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT

    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness=disp-dz-spp-bw-ep+coh
    fitness=torch.where(torch.isnan(fitness),torch.tensor(-9999.0,device=DEVICE),fitness)
    return fitness

def evolve(data, nsteps, ngens, psz, label, decentralized=False, mass_ratios=None, dead_body=None):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3; NW1=INPUT_SIZE*HIDDEN
    NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    NG=NG1*2 if decentralized else NG1
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN));s2=np.sqrt(2.0/(HIDDEN+OUT))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    if decentralized:
        for offset in [0,NG1]:
            pop[:,offset:offset+NW1]*=s1/0.3;pop[:,offset+NW1:offset+NW1+HIDDEN]=0
            i2=offset+NW1+HIDDEN;pop[:,i2:i2+HIDDEN*OUT]*=s2/0.3
            pop[:,i2+HIDDEN*OUT:i2+HIDDEN*OUT+OUT]=0
            pop[:,offset+NG1-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    else:
        pop[:,:NW1]*=s1/0.3;pop[:,NW1:NW1+HIDDEN]=0
        i2=NW1+HIDDEN;pop[:,i2:i2+HIDDEN*OUT]*=s2/0.3
        pop[:,i2+HIDDEN*OUT:i2+HIDDEN*OUT+OUT]=0
        pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE);t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f=simulate(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,decentralized,mass_ratios,dead_body)
            pf[ix]=f
        o_=pf.argsort(descending=True);pop=pop[o_];pf=pf[o_]
        if gen%50==0 or gen==ngens-1:
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,NG,device=DEVICE)*0.3
        if decentralized:
            for offset in [0,NG1]:
                fr[:,offset:offset+NW1]*=s1/0.3
                fr[:,offset+NG1-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        else:
            fr[:,:NW1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE);p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE);p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,NG,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,NG,device=DEVICE)<0.15;ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    best=pop[0].cpu().numpy();total=(time.time()-t0)/60
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best, pf[0].item()

def replay_rfx(genes, data, nsteps, decentralized=False, mass_ratios=None, dead_body=None):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3; NW1=INPUT_SIZE*HIDDEN
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
        gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
        b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs().item()
        gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
        b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs().item()
    else:
        gi=0;W1=g[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
        b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
        W2=g[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
        b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;fv=g[:,gi].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    mass=torch.ones(B,N,device=DEVICE)
    if mass_ratios: mass[:,b0m]=mass_ratios[0];mass[:,b1m]=mass_ratios[1]
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
        if dead_body is not None:
            if dead_body==0: o[:,b0m]=0
            else: o[:,b1m]=0
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_l.append(ext[0,b0m,0].mean().item());fx1_l.append(ext[0,b1m,0].mean().item())
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
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0_l)<1e-10 or np.std(fx1_l)<1e-10: return 0.0
    r_,_=pearsonr(fx0_l,fx1_l);return round(r_,3)

def main():
    NSTEPS=600;GAP=0.5;PSZ=200;gx,gy,gz,sp=10,5,4,0.35;NGENS=300
    data=build_bodies_2(gx,gy,gz,sp,GAP)
    results={}

    # ================================================================
    print("="*70)
    print("EXP 19: DECENTRALIZED BRAINS × MASS ASYMMETRY (2×2 Table)")
    print("="*70)
    # 19a: Decentralized + 3:1 mass
    print("\n--- 19a: Decentralized (2 NNs) + mass 3:1 ---")
    bg19a,fit19a=evolve(data,NSTEPS,NGENS,PSZ,"19a_dec_31",decentralized=True,mass_ratios=(3.0,1.0))
    rfx19a=replay_rfx(bg19a,data,NSTEPS,decentralized=True,mass_ratios=(3.0,1.0))
    print(f"  r(Fx)={rfx19a:.3f}")
    # 19b: Shared + 3:1 mass (control)
    print("\n--- 19b: Shared NN + mass 3:1 (control) ---")
    bg19b,fit19b=evolve(data,NSTEPS,NGENS,PSZ,"19b_shared_31",decentralized=False,mass_ratios=(3.0,1.0))
    rfx19b=replay_rfx(bg19b,data,NSTEPS,decentralized=False,mass_ratios=(3.0,1.0))
    print(f"  r(Fx)={rfx19b:.3f}")
    results["exp19"]={
        "decentralized_31":{"fitness":round(fit19a,2),"r_fx":rfx19a},
        "shared_31":{"fitness":round(fit19b,2),"r_fx":rfx19b},
    }

    # ================================================================
    print("\n"+"="*70)
    print("EXP 20: DEAD BODY (Parasite/Host Test)")
    print("="*70)
    # 20a: Body 1 is dead (mass=1, no forces), Body 0 must drag
    print("\n--- 20a: Body 1 dead (sym mass) ---")
    bg20a,fit20a=evolve(data,NSTEPS,NGENS,PSZ,"20a_dead1_sym",dead_body=1)
    rfx20a=replay_rfx(bg20a,data,NSTEPS,dead_body=1)
    print(f"  fit={fit20a:+.2f} r(Fx)={rfx20a:.3f}")
    # 20b: Body 0 is dead, Body 1 drags (light drags heavy? no, both mass=1)
    print("\n--- 20b: Body 0 dead (sym mass) ---")
    bg20b,fit20b=evolve(data,NSTEPS,NGENS,PSZ,"20b_dead0_sym",dead_body=0)
    rfx20b=replay_rfx(bg20b,data,NSTEPS,dead_body=0)
    print(f"  fit={fit20b:+.2f} r(Fx)={rfx20b:.3f}")
    # 20c: Both alive (baseline for comparison)
    print("\n--- 20c: Both alive (control, sym mass) ---")
    bg20c,fit20c=evolve(data,NSTEPS,NGENS,PSZ,"20c_alive_sym")
    rfx20c=replay_rfx(bg20c,data,NSTEPS)
    print(f"  fit={fit20c:+.2f} r(Fx)={rfx20c:.3f}")
    results["exp20"]={
        "dead1_sym":{"fitness":round(fit20a,2),"r_fx":rfx20a},
        "dead0_sym":{"fitness":round(fit20b,2),"r_fx":rfx20b},
        "alive_sym":{"fitness":round(fit20c,2),"r_fx":rfx20c},
    }

    # ================================================================
    # FIGURE
    # ================================================================
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    fig.suptitle("Season 6B: Decentralized×Asymmetric & Dead Body",fontsize=14,fontweight="bold")

    # Panel 1: 2×2 table
    ax=axes[0]
    # Include Season 6 data for comparison
    table_data=[
        ("Shared\n1:1", 0.942, "#3498db"),
        ("Decentralized\n1:1", 0.952, "#e74c3c"),
        ("Shared\n3:1", rfx19b, "#2ecc71"),
        ("Decentralized\n3:1", rfx19a, "#f39c12"),
    ]
    labels=[d[0] for d in table_data]
    vals=[d[1] for d in table_data]
    colors=[d[2] for d in table_data]
    bars=ax.bar(range(len(table_data)),vals,color=colors,alpha=0.8)
    for i,d in enumerate(table_data):
        ax.text(i,vals[i]+0.02,f"r={vals[i]:.3f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(range(len(table_data)));ax.set_xticklabels(labels,fontsize=8)
    ax.set_ylabel("r(Fx)");ax.set_title("Exp 19: Brain Architecture × Mass Asymmetry\n(Complete 2×2 Factorial)")
    ax.grid(alpha=0.3,axis="y");ax.axhline(y=0,color="black",linewidth=0.5)
    ax.set_ylim(-0.2,1.2)

    # Panel 2: Dead Body
    ax=axes[1]
    dead_labels=["Body 1\ndead","Body 0\ndead","Both\nalive"]
    dead_fits=[fit20a,fit20b,fit20c]
    dead_colors=["#e74c3c","#e67e22","#2ecc71"]
    bars=ax.bar(range(3),dead_fits,color=dead_colors,alpha=0.8)
    for i in range(3):
        ax.text(i,dead_fits[i]+2,f"{dead_fits[i]:+.0f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(range(3));ax.set_xticklabels(dead_labels)
    ax.set_ylabel("Fitness");ax.set_title("Exp 20: Dead Body (Parasite/Host)\nCan one body drag the other?")
    ax.grid(alpha=0.3,axis="y")

    plt.tight_layout()
    fig_path=os.path.join(OUTPUT_DIR,"season6b_experiments.png")
    plt.savefig(fig_path,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fig_path}")
    log_path=os.path.join(RESULTS_DIR,"season6b_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    # Summary
    print("\n"+"="*70)
    print("SEASON 6B SUMMARY")
    print("="*70)
    print("\nExp 19: 2×2 Brain×Mass Table")
    print(f"  Shared    1:1  r=0.942  (from S6)")
    print(f"  Decentral 1:1  r=0.952  (from S6)")
    print(f"  Shared    3:1  r={rfx19b:.3f}  fit={fit19b:+.2f}")
    print(f"  Decentral 3:1  r={rfx19a:.3f}  fit={fit19a:+.2f}")
    if rfx19a < 0.4:
        print(f"\n  🔥 Independent brains ALSO differentiate under mass asymmetry!")
        print(f"     → Differentiation is a PHYSICAL inevitability, not a neural strategy")
    else:
        print(f"\n  🧠 Independent brains do NOT differentiate under mass asymmetry")
        print(f"     → Shared brain is REQUIRED for translating mass signals into behavior")
    print(f"\nExp 20: Dead Body")
    print(f"  Dead Body 1: fit={fit20a:+.2f}")
    print(f"  Dead Body 0: fit={fit20b:+.2f}")
    print(f"  Both alive:  fit={fit20c:+.2f}")
    drag_cost=1.0-max(fit20a,fit20b)/fit20c if fit20c>0 else 0
    print(f"  Drag cost: {drag_cost*100:.0f}% fitness reduction")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800,300);time.sleep(0.2)
    except: pass
    print("\nAll Season 6B experiments complete!")

if __name__=="__main__":
    main()
