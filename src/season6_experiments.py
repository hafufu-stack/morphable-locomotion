"""
Season 6: Phantom Synchronization & Decentralized Brains
=========================================================
Exp 17: Cut combine springs mid-sim in 5:1:5 3-body → neural or embodied sync?
Exp 18: Two independent NNs (no weight sharing) for 2-body → emergent coordination?
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

def build_bodies(gx,gy,gz,sp,gap,n_bodies=2):
    nper=gx*gy*gz; nt=nper*n_bodies; ap=np.zeros((nt,3)); bi=np.zeros(nt,dtype=np.int64)
    bw=(gx-1)*sp
    if n_bodies==2:
        offsets=[-(gap/2+bw/2), (gap/2+bw/2)]
    else:
        offsets=[-(gap+bw), 0.0, (gap+bw)]
    idx=0
    for b in range(n_bodies):
        cx=offsets[b]
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp=cx+(x-(gx-1)/2)*sp
                    ap[idx]=[xp,2.0+y*sp,z*sp-(gz-1)*sp/2]; bi[idx]=b; idx+=1
    sa,sb,rl=[],[],[]
    for b in range(n_bodies):
        m=np.where(bi==b)[0]; bp=ap[m]; tri=Delaunay(bp); edges=set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a);sb.append(bb);rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_=np.zeros_like(ap)
    for b in range(n_bodies):
        m=bi==b
        for d in range(3):
            vn,vx=ap[m,d].min(),ap[m,d].max(); np_[m,d]=2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap,np_,bi,np.array(sa),np.array(sb),np.array(rl),nper,nt

# ================================================================
# EXP 17: PHANTOM SYNCHRONIZATION
# ================================================================
@torch.no_grad()
def phantom_replay(genes,data,nsteps,mass_ratios,cut_step=None):
    """Replay 3-body, optionally cutting combine springs at cut_step."""
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3; NW1=INPUT_SIZE*HIDDEN
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    n_internal=len(sa)  # internal springs count
    g=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gi=0;W1=g[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
    b1g=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2=g[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2g=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
    fv=g[:,gi].abs().item()
    bid=(bit.float()/(max(mass_ratios)-1 if max(mass_ratios)>1 else 1)).unsqueeze(0).unsqueeze(2).expand(B,N,1)
    bid_vals=bit.float()/2.0; bid=bid_vals.unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    masks=[(bit==b) for b in range(3)]; idxs=[masks[b].nonzero(as_tuple=True)[0] for b in range(3)]
    cd01=False;cd12=False;comb=torch.zeros(B,1,1,device=DEVICE)
    mass=torch.ones(B,N,device=DEVICE)
    for b in range(3): mass[:,masks[b]]=mass_ratios[b]
    fx={0:[],1:[],2:[]}
    for step in range(nsteps):
        t=step*DT
        # Combination logic
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
        # CUT SPRINGS at cut_step
        if cut_step is not None and step==cut_step:
            csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
        sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
        st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
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
    pairs={}
    for(a,b)in[(0,1),(1,2),(0,2)]:
        if np.std(fx[a])<1e-10 or np.std(fx[b])<1e-10: pairs[f"r{a}{b}"]=0.0
        else: r_,_=pearsonr(fx[a],fx[b]);pairs[f"r{a}{b}"]=round(r_,3)
    disp=pos[:,:,0].mean(dim=1).item()-rp[:,0].mean().item()
    return pairs, round(disp,2), fx

def evolve_3body(data,nsteps,ngens,psz,label,mass_ratios):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=3; NW1=INPUT_SIZE*HIDDEN
    NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN));s2=np.sqrt(2.0/(HIDDEN+OUT))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    masks=[(bit==b) for b in range(3)];idxs=[masks[b].nonzero(as_tuple=True)[0] for b in range(3)]
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    pop[:,:NW1]*=s1/0.3;pop[:,NW1:NW1+HIDDEN]=0
    i2=NW1+HIDDEN;pop[:,i2:i2+HIDDEN*OUT]*=s2/0.3
    pop[:,i2+HIDDEN*OUT:i2+HIDDEN*OUT+OUT]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    t0=time.time();mass_t=torch.ones(1,N,device=DEVICE)
    for b in range(3): mass_t[:,masks[b]]=mass_ratios[b]
    bid_vals=bit.float()/2.0;bid=bid_vals.unsqueeze(0).unsqueeze(2)
    ni=npt.unsqueeze(0)
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]; BZ=ix.shape[0]
            g=pop[ix];pos_=rp.unsqueeze(0).expand(BZ,-1,-1).clone()
            vel_=torch.zeros(BZ,N,3,device=DEVICE)
            gi=0;W1=g[:,gi:gi+NW1].reshape(BZ,INPUT_SIZE,HIDDEN);gi+=NW1
            b1=g[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
            W2=g[:,gi:gi+HIDDEN*OUT].reshape(BZ,HIDDEN,OUT);gi+=HIDDEN*OUT
            b2=g[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
            freq=g[:,gi].abs()
            sx_=pos_[:,:,0].mean(dim=1);bid_e=bid.expand(BZ,N,1);ni_e=ni.expand(BZ,-1,-1)
            csa_=sat.clone();csb_=sbt.clone();crl_=rlt.clone()
            comb_=torch.zeros(BZ,1,1,device=DEVICE);te=torch.zeros(BZ,device=DEVICE)
            cd01_=False;cd12_=False;mass_e=mass_t.expand(BZ,N)
            for step in range(nsteps):
                t=step*DT
                if not cd01_ and step%10==0:
                    p0=pos_[0,idxs[0]];p1=pos_[0,idxs[1]];ds=torch.cdist(p0,p1)
                    cl=(ds<1.2).nonzero(as_tuple=False)
                    if cl.shape[0]>0:
                        nn_=min(cl.shape[0],500)
                        csa_=torch.cat([csa_,idxs[0][cl[:nn_,0]]]);csb_=torch.cat([csb_,idxs[1][cl[:nn_,1]]])
                        crl_=torch.cat([crl_,ds[cl[:nn_,0],cl[:nn_,1]]]);cd01_=True
                if not cd12_ and step%10==0:
                    p1=pos_[0,idxs[1]];p2=pos_[0,idxs[2]];ds=torch.cdist(p1,p2)
                    cl=(ds<1.2).nonzero(as_tuple=False)
                    if cl.shape[0]>0:
                        nn_=min(cl.shape[0],500)
                        csa_=torch.cat([csa_,idxs[1][cl[:nn_,0]]]);csb_=torch.cat([csb_,idxs[2][cl[:nn_,1]]])
                        crl_=torch.cat([crl_,ds[cl[:nn_,0],cl[:nn_,1]]]);cd12_=True
                if cd01_ or cd12_: comb_=torch.ones(BZ,1,1,device=DEVICE)
                st=torch.sin(2*np.pi*freq*t).reshape(BZ,1,1).expand(BZ,N,1)
                ct=torch.cos(2*np.pi*freq*t).reshape(BZ,1,1).expand(BZ,N,1)
                nn_in=torch.cat([st,ct,ni_e,bid_e,comb_.expand(BZ,N,1)],dim=2)
                h=torch.tanh(torch.bmm(nn_in,W1)+b1);o=torch.tanh(torch.bmm(h,W2)+b2)
                og=(pos_[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
                ext=torch.zeros(BZ,N,3,device=DEVICE)
                ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
                ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
                f=torch.zeros(BZ,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass_e
                pa=pos_[:,csa_];pb=pos_[:,csb_];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
                dr=d_/di;r=crl_.unsqueeze(0).unsqueeze(2);s=di-r
                rv=vel_[:,csb_]-vel_[:,csa_];va=(rv*dr).sum(dim=2,keepdim=True)
                ft__=SPRING_K*s*dr+SPRING_DAMP*va*dr
                f.scatter_add_(1,csa_.unsqueeze(0).unsqueeze(2).expand(BZ,-1,3),ft__)
                f.scatter_add_(1,csb_.unsqueeze(0).unsqueeze(2).expand(BZ,-1,3),-ft__)
                pen=(GROUND_Y-pos_[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
                bl=(pos_[:,:,1]<GROUND_Y).float()
                f[:,:,0]-=3.0*vel_[:,:,0]*bl;f[:,:,2]-=3.0*vel_[:,:,2]*bl
                f-=DRAG*vel_;f+=ext
                acc=f/(mass_e.clamp(min=0.01).unsqueeze(2))
                vel_+=acc*DT;vel_.clamp_(-50,50);pos_+=vel_*DT
            disp=pos_[:,:,0].mean(dim=1)-sx_
            dz=pos_[:,:,2].mean(dim=1).abs()
            sp_=pos_.max(dim=1).values-pos_.min(dim=1).values
            spp=((sp_-10.0).clamp(min=0)*1.5).sum(dim=1)
            bw=(pos_[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
            me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
            c=[pos_[:,masks[b]].mean(dim=1) for b in range(3)]
            d01=torch.norm(c[0]-c[1],dim=1);d12=torch.norm(c[1]-c[2],dim=1)
            coh=torch.clamp(3.0-d01,min=0)*1.5+torch.clamp(3.0-d12,min=0)*1.5
            fit=disp-dz-spp-bw-ep+coh
            fit=torch.where(torch.isnan(fit),torch.tensor(-9999.0,device=DEVICE),fit)
            pf[ix]=fit
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
    best=pop[0].cpu().numpy();total=(time.time()-t0)/60
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best

# ================================================================
# EXP 18: DECENTRALIZED BRAINS
# ================================================================
@torch.no_grad()
def simulate_decentralized(genomes,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps):
    """Two independent NNs: genome[:NG1] for body0, genome[NG1:] for body1."""
    OUT=3;NW1=INPUT_SIZE*HIDDEN;NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    B=genomes.shape[0];pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    # Unpack two NNs
    g0=genomes[:,:NG1];g1=genomes[:,NG1:]
    gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
    b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
    freq0=g0[:,gi].abs()
    gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
    b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT
    freq1=g1[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False;n0=b0m.sum().item();n1=b1m.sum().item()
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
        # Body 0 NN
        st0=torch.sin(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        ct0=torch.cos(2*np.pi*freq0*t).reshape(B,1,1).expand(B,n0,1)
        in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        # Body 1 NN
        st1=torch.sin(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        ct1=torch.cos(2*np.pi*freq1*t).reshape(B,1,1).expand(B,n1,1)
        in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
        # Merge outputs
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
        f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness=disp-dz-spp-bw-ep+coh
    fitness=torch.where(torch.isnan(fitness),torch.tensor(-9999.0,device=DEVICE),fitness)
    return fitness

def evolve_decentralized(data,nsteps,ngens,psz,label):
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt;OUT=3;NW1=INPUT_SIZE*HIDDEN
    NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1; NG=NG1*2  # two NNs
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN));s2=np.sqrt(2.0/(HIDDEN+OUT))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    for offset in [0,NG1]:
        pop[:,offset:offset+NW1]*=s1/0.3;pop[:,offset+NW1:offset+NW1+HIDDEN]=0
        i2=offset+NW1+HIDDEN;pop[:,i2:i2+HIDDEN*OUT]*=s2/0.3
        pop[:,i2+HIDDEN*OUT:i2+HIDDEN*OUT+OUT]=0
        pop[:,offset+NG1-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f=simulate_decentralized(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps)
            pf[ix]=f
        o_=pf.argsort(descending=True);pop=pop[o_];pf=pf[o_]
        if gen%50==0 or gen==ngens-1:
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,NG,device=DEVICE)*0.3
        for offset in [0,NG1]:
            fr[:,offset:offset+NW1]*=s1/0.3
            fr[:,offset+NG1-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
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
    return best

def replay_rfx_decentralized(genes,data,nsteps):
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt;OUT=3;NW1=INPUT_SIZE*HIDDEN
    NG1=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    g=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    g0=g[:,:NG1];g1=g[:,NG1:]
    gi=0;W1_0=g0[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
    b1_0=g0[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_0=g0[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_0=g0[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq0=g0[:,gi].abs().item()
    gi=0;W1_1=g1[:,gi:gi+NW1].reshape(B,INPUT_SIZE,HIDDEN);gi+=NW1
    b1_1=g1[:,gi:gi+HIDDEN].unsqueeze(1);gi+=HIDDEN
    W2_1=g1[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT);gi+=HIDDEN*OUT
    b2_1=g1[:,gi:gi+OUT].unsqueeze(1);gi+=OUT;freq1=g1[:,gi].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False;n0=b0m.sum().item();n1=b1m.sum().item()
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
        sv0=np.sin(2*np.pi*freq0*t);cv0=np.cos(2*np.pi*freq0*t)
        st0=torch.full((B,n0,1),sv0,device=DEVICE);ct0=torch.full((B,n0,1),cv0,device=DEVICE)
        in0=torch.cat([st0,ct0,ni[:,b0m],bid[:,b0m],comb.expand(B,n0,1)],dim=2)
        h0=torch.tanh(torch.bmm(in0,W1_0)+b1_0);o0=torch.tanh(torch.bmm(h0,W2_0)+b2_0)
        sv1=np.sin(2*np.pi*freq1*t);cv1=np.cos(2*np.pi*freq1*t)
        st1=torch.full((B,n1,1),sv1,device=DEVICE);ct1=torch.full((B,n1,1),cv1,device=DEVICE)
        in1=torch.cat([st1,ct1,ni[:,b1m],bid[:,b1m],comb.expand(B,n1,1)],dim=2)
        h1=torch.tanh(torch.bmm(in1,W1_1)+b1_1);o1=torch.tanh(torch.bmm(h1,W2_1)+b2_1)
        o=torch.zeros(B,N,OUT,device=DEVICE);o[:,b0m]=o0;o[:,b1m]=o1
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
        f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext;vel+=f*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0_l)<1e-10 or np.std(fx1_l)<1e-10: return 0.0
    r_,_=pearsonr(fx0_l,fx1_l);return round(r_,3)

def main():
    NSTEPS=600;GAP=0.5;PSZ=200;gx,gy,gz,sp=10,5,4,0.35
    results={}

    # ================================================================
    print("="*70)
    print("EXP 17: PHANTOM SYNCHRONIZATION (Spring Cutting Ablation)")
    print("="*70)
    # First evolve a 5:1:5 3-body
    data3=build_bodies(gx,gy,gz,sp,GAP,3)
    mrs=(5.0,1.0,5.0)
    print(f"\n--- Evolving 5:1:5 3-body (300 gen) ---")
    best515=evolve_3body(data3,NSTEPS,300,PSZ,"17_515",mrs)
    # Replay: no cut
    print("\n--- Replay: no cut (control) ---")
    r_nocut,disp_nocut,fx_nocut=phantom_replay(best515,data3,NSTEPS,mrs,cut_step=None)
    print(f"  No cut: r01={r_nocut['r01']:.3f} r12={r_nocut['r12']:.3f} r02={r_nocut['r02']:.3f} disp={disp_nocut}")
    # Replay: cut at step 200 (early)
    print("--- Replay: cut at step 200 ---")
    r_cut200,disp_200,fx_200=phantom_replay(best515,data3,NSTEPS,mrs,cut_step=200)
    print(f"  Cut@200: r01={r_cut200['r01']:.3f} r12={r_cut200['r12']:.3f} r02={r_cut200['r02']:.3f} disp={disp_200}")
    # Replay: cut at step 300 (mid)
    print("--- Replay: cut at step 300 ---")
    r_cut300,disp_300,fx_300=phantom_replay(best515,data3,NSTEPS,mrs,cut_step=300)
    print(f"  Cut@300: r01={r_cut300['r01']:.3f} r12={r_cut300['r12']:.3f} r02={r_cut300['r02']:.3f} disp={disp_300}")
    # Replay: cut at step 400 (late)
    print("--- Replay: cut at step 400 ---")
    r_cut400,disp_400,fx_400=phantom_replay(best515,data3,NSTEPS,mrs,cut_step=400)
    print(f"  Cut@400: r01={r_cut400['r01']:.3f} r12={r_cut400['r12']:.3f} r02={r_cut400['r02']:.3f} disp={disp_400}")
    # Also test 3:1:1 for comparison
    print("\n--- Evolving 3:1:1 3-body (300 gen) ---")
    mrs311=(3.0,1.0,1.0)
    data3_311=build_bodies(gx,gy,gz,sp,GAP,3)
    best311=evolve_3body(data3_311,NSTEPS,300,PSZ,"17_311",mrs311)
    print("--- 3:1:1 Replay: no cut ---")
    r311_nc,d311_nc,_=phantom_replay(best311,data3_311,NSTEPS,mrs311,cut_step=None)
    print(f"  No cut: {r311_nc}")
    print("--- 3:1:1 Replay: cut at 300 ---")
    r311_cut,d311_cut,_=phantom_replay(best311,data3_311,NSTEPS,mrs311,cut_step=300)
    print(f"  Cut@300: {r311_cut}")

    results["exp17_phantom"]={
        "515_nocut":{"r":r_nocut,"disp":disp_nocut},
        "515_cut200":{"r":r_cut200,"disp":disp_200},
        "515_cut300":{"r":r_cut300,"disp":disp_300},
        "515_cut400":{"r":r_cut400,"disp":disp_400},
        "311_nocut":{"r":r311_nc,"disp":d311_nc},
        "311_cut300":{"r":r311_cut,"disp":d311_cut},
    }

    # ================================================================
    print("\n"+"="*70)
    print("EXP 18: DECENTRALIZED BRAINS (Independent NNs)")
    print("="*70)
    data2=build_bodies(gx,gy,gz,sp,GAP,2)
    exp18_results=[]
    # 18a: Decentralized, symmetric
    print("\n--- 18a: Decentralized (2 independent NNs), symmetric ---")
    bg=evolve_decentralized(data2,NSTEPS,300,PSZ,"18a_decentral")
    rfx=replay_rfx_decentralized(bg,data2,NSTEPS)
    exp18_results.append({"label":"decentralized_sym","r_fx":rfx,"n_brains":2})
    print(f"  r(Fx)={rfx:.3f}")
    # 18b: Shared NN baseline for comparison (reuse from season5)
    print("\n--- 18b: Shared NN (control), symmetric ---")
    # Quick evolve shared NN for fair comparison
    from season5b_experiments import evolve_2body, replay_rfx_2body, build_bodies_2
    data2b=build_bodies_2(gx,gy,gz,sp,GAP)
    bg_sh,_,fl_sh,_=evolve_2body(data2b,NSTEPS,300,PSZ,INPUT_SIZE,0,"18b_shared")
    rfx_sh=replay_rfx_2body(bg_sh,data2b,NSTEPS,INPUT_SIZE,0)
    exp18_results.append({"label":"shared_sym","r_fx":rfx_sh,"n_brains":1})
    print(f"  r(Fx)={rfx_sh:.3f}")

    results["exp18_decentralized"]=exp18_results

    # ================================================================
    # FIGURE
    # ================================================================
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    fig.suptitle("Season 6: Phantom Synchronization & Decentralized Brains",fontsize=14,fontweight="bold")
    # Panel 1: Phantom Sync
    ax=axes[0]
    cuts=["No cut","Cut@200","Cut@300","Cut@400"]
    r02_vals=[r_nocut['r02'],r_cut200['r02'],r_cut300['r02'],r_cut400['r02']]
    r01_vals=[r_nocut['r01'],r_cut200['r01'],r_cut300['r01'],r_cut400['r01']]
    r12_vals=[r_nocut['r12'],r_cut200['r12'],r_cut300['r12'],r_cut400['r12']]
    x=np.arange(len(cuts));w=0.25
    ax.bar(x-w,r02_vals,w,color="#e74c3c",alpha=0.8,label="r(0-2) mirror")
    ax.bar(x,r01_vals,w,color="#3498db",alpha=0.8,label="r(0-1)")
    ax.bar(x+w,r12_vals,w,color="#2ecc71",alpha=0.8,label="r(1-2)")
    for i in range(len(cuts)):
        ax.text(i-w,r02_vals[i]+0.02,f"{r02_vals[i]:.3f}",ha="center",fontsize=7,color="#e74c3c",fontweight="bold")
    ax.set_xticks(x);ax.set_xticklabels(cuts);ax.set_ylabel("r(Fx)")
    ax.set_title("Exp 17: Phantom Synchronization (5:1:5)\nDo bodies sync via springs or NN?")
    ax.legend(fontsize=7);ax.grid(alpha=0.3,axis="y");ax.axhline(y=0,color="black",linewidth=0.5)
    # Panel 2: Decentralized
    ax=axes[1]
    labels_d=[r["label"].replace("_","\n") for r in exp18_results]
    rfx_d=[r["r_fx"] for r in exp18_results]
    colors_d=["#e74c3c","#3498db"]
    ax.bar(range(len(exp18_results)),rfx_d,color=colors_d,alpha=0.8)
    for i,r in enumerate(exp18_results):
        ax.text(i,rfx_d[i]+0.02,f"r={rfx_d[i]:.3f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(range(len(exp18_results)));ax.set_xticklabels(labels_d)
    ax.set_ylabel("r(Fx)");ax.set_title("Exp 18: Decentralized vs Shared Brain\nCan independent NNs coordinate?")
    ax.grid(alpha=0.3,axis="y");ax.axhline(y=0,color="black",linewidth=0.5)
    plt.tight_layout()
    fig_path=os.path.join(OUTPUT_DIR,"season6_experiments.png")
    plt.savefig(fig_path,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fig_path}")
    log_path=os.path.join(RESULTS_DIR,"season6_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    # Summary
    print("\n"+"="*70)
    print("SEASON 6 SUMMARY")
    print("="*70)
    print("\nExp 17: Phantom Synchronization (5:1:5)")
    print(f"  No cut:  r02={r_nocut['r02']:.3f}")
    print(f"  Cut@200: r02={r_cut200['r02']:.3f}")
    print(f"  Cut@300: r02={r_cut300['r02']:.3f}")
    print(f"  Cut@400: r02={r_cut400['r02']:.3f}")
    if abs(r_cut300['r02']-r_nocut['r02'])<0.1:
        print(f"\n  🧠 NEURAL SYNC! Cutting springs barely changes r02 → sync is in the NN, not the body")
    else:
        print(f"\n  🦴 EMBODIED SYNC! Cutting springs destroys r02 → sync requires physical coupling")
    print(f"\n  3:1:1 control: nocut r01={r311_nc['r01']:.3f}, cut@300 r01={r311_cut['r01']:.3f}")
    print(f"\nExp 18: Decentralized Brains")
    for r in exp18_results:
        print(f"  {r['label']:>20s}: r(Fx)={r['r_fx']:.3f}")
    if exp18_results[0]["r_fx"]<0.3:
        print(f"\n  🔥 DIFFERENTIATION! Independent brains spontaneously specialize via physical coupling alone!")
    else:
        print(f"\n  🔒 SYNCHRONIZATION. Independent brains converge on similar strategies.")
    try:
        import winsound
        for _ in range(5): winsound.Beep(800,300);time.sleep(0.2)
    except: pass
    print("\nAll Season 6 experiments complete!")

if __name__=="__main__":
    main()
