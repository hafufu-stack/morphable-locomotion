"""
Perdim Verification: Reproduce Season 3's DoF Trap with EXACT implementation
vs Season 4B's simplified perdim, to explain the discrepancy.

Key differences being tested:
- Season 3: INPUT_SIZE=8 (mass feedback), body-wise softmax(d*3.0), EMA(alpha=0.05)
- Season 4B: INPUT_SIZE=7 (no mass feedback), global softmax(d), instant update
"""

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.stats import pearsonr
import os, time, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
FRICTION=3.0; BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5
HIDDEN_SIZE=32

def build_bodies(gx,gy,gz,sp,gap):
    nper=gx*gy*gz;nt=nper*2;ap=np.zeros((nt,3));bi=np.zeros(nt,dtype=np.int64)
    bw=(gx-1)*sp;idx=0
    for b in range(2):
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp=(-(gap/2+bw)+x*sp) if b==0 else ((gap/2+bw)-x*sp)
                    ap[idx]=[xp,2.0+y*sp,z*sp-(gz-1)*sp/2];bi[idx]=b;idx+=1
    sa,sb,rl=[],[],[]
    for b in range(2):
        m=np.where(bi==b)[0];bp=ap[m];tri=Delaunay(bp);edges=set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4):edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges:sa.append(a);sb.append(bb);rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_=np.zeros_like(ap)
    for b in range(2):
        m=bi==b
        for d in range(3):
            vn,vx=ap[m,d].min(),ap[m,d].max();np_[m,d]=2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap,np_,bi,np.array(sa),np.array(sb),np.array(rl),nper,nt


@torch.no_grad()
def simulate_perdim(genomes,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                    input_size,has_mass_fb,bodywise_softmax,ema_alpha):
    """Flexible perdim simulation with configurable implementation details."""
    OUTPUT_SIZE=4; N_W1=input_size*HIDDEN_SIZE
    N_GENES=input_size*HIDDEN_SIZE+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE+1
    B=genomes.shape[0];pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    idx=0;W1=genomes[:,idx:idx+N_W1].reshape(B,input_size,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+HIDDEN_SIZE].unsqueeze(1);idx+=HIDDEN_SIZE
    W2=genomes[:,idx:idx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2=genomes[:,idx:idx+OUTPUT_SIZE].unsqueeze(1);idx+=OUTPUT_SIZE
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False;mass=torch.ones(B,N,device=DEVICE)
    total_mass_b0=mass[:,b0m].sum(dim=1,keepdim=True)
    total_mass_b1=mass[:,b1m].sum(dim=1,keepdim=True)

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
        st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        if has_mass_fb:
            mass_input=torch.log(mass+0.1).unsqueeze(2)
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1),mass_input],dim=2)
        else:
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1);o=torch.tanh(torch.bmm(h,W2)+b2)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og

        # Mass redistribution
        mass_desire=o[:,:,3]
        if bodywise_softmax:
            d0=mass_desire[:,b0m];d1=mass_desire[:,b1m]
            w0=torch.softmax(d0*3.0,dim=1);w1=torch.softmax(d1*3.0,dim=1)
            t0_mass=w0*total_mass_b0*nper;t1_mass=w1*total_mass_b1*nper
            t0_mass=t0_mass.clamp(0.1,10.0);t1_mass=t1_mass.clamp(0.1,10.0)
            t0_mass=t0_mass/t0_mass.sum(dim=1,keepdim=True)*total_mass_b0*nper
            t1_mass=t1_mass/t1_mass.sum(dim=1,keepdim=True)*total_mass_b1*nper
            if ema_alpha<1.0:
                mass[:,b0m]=(1-ema_alpha)*mass[:,b0m]+ema_alpha*t0_mass
                mass[:,b1m]=(1-ema_alpha)*mass[:,b1m]+ema_alpha*t1_mass
            else:
                mass[:,b0m]=t0_mass;mass[:,b1m]=t1_mass
        else:
            mass_soft=torch.softmax(mass_desire,dim=1)*N*1.0
            mass=mass_soft.clamp(min=0.01)

        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl;f[:,:,2]-=FRICTION*vel[:,:,2]*bl
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
    return fitness,disp


def evolve_perdim(data,nsteps,ngens,psz,input_size,has_mass_fb,bodywise_softmax,ema_alpha,label):
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt
    OUTPUT_SIZE=4;N_W1=input_size*HIDDEN_SIZE
    N_GENES=input_size*HIDDEN_SIZE+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE+1
    s1=np.sqrt(2.0/(input_size+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+HIDDEN_SIZE]=0
    pop[:,N_W1+HIDDEN_SIZE:N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE]*=s2/0.3
    pop[:,N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE:N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    gen_log,fitness_log=[],[];t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate_perdim(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                                input_size,has_mass_fb,bodywise_softmax,ema_alpha)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0 or gen==ngens-1:
            gen_log.append(gen);fitness_log.append(pf[0].item())
            elapsed=time.time()-t0
            print(f"  [{label}] Gen {gen:4d}: fit={pf[0].item():+.2f} ({elapsed/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,N_GENES,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES,device=DEVICE)<0.5
        ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES,device=DEVICE)<0.15
        ch+=torch.randn(nc,N_GENES,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    total=(time.time()-t0)/60;best_genes=pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best_genes,gen_log,fitness_log,total


def main():
    NSTEPS=600;GAP=0.5;PSZ=200;NGENS=300
    gx,gy,gz,sp=10,5,4,0.35
    data=build_bodies(gx,gy,gz,sp,GAP)

    configs = [
        # (input_size, has_mass_fb, bodywise_softmax, ema_alpha, label)
        (8, True,  True,  0.05, "S3_exact"),       # Season 3 exact reproduction
        (7, False, False, 1.0,  "S4B_simplified"),  # Season 4B implementation
        (7, False, True,  0.05, "hybrid_bodywise"), # S4B inputs + S3 mass mechanics
    ]

    results = []
    for inp_sz, mass_fb, bw_sm, ema, label in configs:
        print(f"\n{'='*60}")
        print(f"  {label}: input={inp_sz}, mass_fb={mass_fb}, bodywise={bw_sm}, ema={ema}")
        print(f"{'='*60}")
        _,gl,fl,elapsed = evolve_perdim(data,NSTEPS,NGENS,PSZ,inp_sz,mass_fb,bw_sm,ema,label)
        results.append({"label":label,"input_size":inp_sz,"mass_fb":mass_fb,
                        "bodywise":bw_sm,"ema":ema,"fitness":round(fl[-1],2),
                        "elapsed":round(elapsed,1),"gen_log":gl,"fitness_log":fl})
        print(f"  Final fit={fl[-1]:+.2f}")

    # Summary
    print("\n" + "="*60)
    print("PERDIM VERIFICATION SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['label']:>20s}  inp={r['input_size']}  mass_fb={r['mass_fb']}  "
              f"bodywise={r['bodywise']}  ema={r['ema']}  fit={r['fitness']:+.2f}")

    # Save
    log_path = os.path.join(RESULTS_DIR, "perdim_verification_log.json")
    save_results = [{k:v for k,v in r.items()} for r in results]
    with open(log_path,"w") as f:
        json.dump(save_results,f,indent=2,default=str)
    print(f"\nLog saved: {log_path}")

    try:
        import winsound
        for _ in range(3): winsound.Beep(800,300); time.sleep(0.2)
    except: pass
    print("Done!")


if __name__=="__main__":
    main()
