"""
Exp 21c: Parasite × Mass Asymmetry (complete 2×2 table)
α=3 + mass 3:1 — does the heavy body become worker or freeloader?
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
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN=32; INPUT_DIM=7

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
def sim_parasite_mass(g0_batch,g1_batch,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,alpha,mass_ratios):
    OUT=3;NW1=INPUT_DIM*HIDDEN;B=g0_batch.shape[0]
    pos=rp.unsqueeze(0).expand(B,-1,-1).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    n0=b0m.sum().item();n1=b1m.sum().item()
    mass=torch.ones(B,N,device=DEVICE)
    mass[:,b0m]=mass_ratios[0]; mass[:,b1m]=mass_ratios[1]
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
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float();f[:,:,0]-=3.0*vel[:,:,0]*bl;f[:,:,2]-=3.0*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx
    me=N*nsteps*(BASE_AMP*1.5)**2*3
    ep0=alpha*(e0/me)*100; ep1=alpha*(e1/me)*100
    return disp-ep0, disp-ep1, disp, e0/me*100, e1/me*100

def main():
    NSTEPS=600;GAP=0.5;PSZ=200;gx,gy,gz,sp=10,5,4,0.35;NGENS=300
    data=build_bodies(gx,gy,gz,sp,GAP)
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt
    NW1=INPUT_DIM*HIDDEN;NG1=NW1+HIDDEN+HIDDEN*3+3+1
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    results={}; t_start=time.time()
    # Conditions: (alpha, mass_ratios, label)
    conditions = [
        (3.0, (1.0,1.0), "a3_sym"),
        (3.0, (3.0,1.0), "a3_31"),
        (3.0, (1.0,3.0), "a3_13"),  # flip: B0 light, B1 heavy
    ]
    print("="*70)
    print("EXP 21c: PARASITE × MASS ASYMMETRY")
    print("="*70)
    for alpha_val,mr,label in conditions:
        print(f"\n--- {label}: α={alpha_val} mass={mr[0]}:{mr[1]} ---")
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
                f0,f1,_,_,_=sim_parasite_mass(pop0[ix],pop1[ix],rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha_val,mr)
                f0=torch.where(torch.isnan(f0),torch.tensor(-9999.0,device=DEVICE),f0)
                f1=torch.where(torch.isnan(f1),torch.tensor(-9999.0,device=DEVICE),f1)
                pf0[ix]=f0;pf1[ix]=f1
            o0=pf0.argsort(descending=True);pop0=pop0[o0];pf0=pf0[o0]
            pop1=pop1[o0];pf1=pf1[o0]
            o1=pf1.argsort(descending=True);pop1=pop1[o1];pf1=pf1[o1]
            pop0=pop0[o1];pf0=pf0[o1]
            if gen%100==0 or gen==NGENS-1:
                print(f"  [{label}] Gen {gen:4d}: f0={pf0[0].item():+.2f} f1={pf1[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
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
        bg0=pop0[0:1];bg1=pop1[0:1]
        f0_,f1_,disp_,en0_,en1_=sim_parasite_mass(bg0,bg1,rp,npt,bit,sat,sbt,rlt,N,nper,NSTEPS,alpha_val,mr)
        results[label]={
            "alpha":alpha_val,"mass":f"{mr[0]}:{mr[1]}",
            "fitness_0":round(f0_.item(),2),"fitness_1":round(f1_.item(),2),
            "displacement":round(disp_.item(),2),
            "energy_0":round(en0_.item(),2),"energy_1":round(en1_.item(),2),
            "fitness_gap":round(f1_.item()-f0_.item(),2),
        }
        d=results[label]
        who_works="B0(heavy)" if mr[0]>mr[1] and d['energy_0']>d['energy_1'] else \
                  "B1(heavy)" if mr[1]>mr[0] and d['energy_1']>d['energy_0'] else \
                  "B0(light)" if mr[0]<mr[1] and d['energy_0']>d['energy_1'] else \
                  "B1(light)" if mr[1]<mr[0] and d['energy_1']>d['energy_0'] else "symmetric"
        print(f"  Done: f0={d['fitness_0']:+.2f} f1={d['fitness_1']:+.2f} gap={d['fitness_gap']:+.2f} E0={d['energy_0']:.1f} E1={d['energy_1']:.1f} → Worker={who_works}")

    total=(time.time()-t_start)/60
    log_path=os.path.join(RESULTS_DIR,"exp21c_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"\nLog: {log_path}")
    print(f"\n{'='*70}")
    print(f"EXP 21c COMPLETE ({total:.1f} min)")
    print(f"{'='*70}")
    for label in ["a3_sym","a3_31","a3_13"]:
        d=results[label]
        print(f"  {label}: f0={d['fitness_0']:+.2f} f1={d['fitness_1']:+.2f} gap={d['fitness_gap']:+.2f} E0={d['energy_0']:.1f} E1={d['energy_1']:.1f}")
    try:
        import winsound
        for _ in range(5): winsound.Beep(800,300); time.sleep(0.2)
    except: pass

if __name__=="__main__":
    main()
