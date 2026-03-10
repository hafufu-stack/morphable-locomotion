"""
Exp 23: Topology Control (standalone rerun)
NN has 4th output to control combine spring cutting.
23a: no wall (baseline), 23b: wall at x=8
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
def sim_topology_ctrl(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps, wall_x=None):
    OUT=4; NW1=INPUT_DIM*HIDDEN; B=genomes.shape[0]
    pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    b0m=bit==0; b1m=bit==1; b0i=b0m.nonzero(as_tuple=True)[0]; b1i=b1m.nonzero(as_tuple=True)[0]
    gi=0; W1=genomes[:,gi:gi+NW1].reshape(B,INPUT_DIM,HIDDEN); gi+=NW1
    b1g=genomes[:,gi:gi+HIDDEN].unsqueeze(1); gi+=HIDDEN
    W2=genomes[:,gi:gi+HIDDEN*OUT].reshape(B,HIDDEN,OUT); gi+=HIDDEN*OUT
    b2g=genomes[:,gi:gi+OUT].unsqueeze(1); gi+=OUT; freq=genomes[:,gi].abs()
    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    n_internal=len(sat)
    csa=sat.clone(); csb=sbt.clone(); crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE); te=torch.zeros(B,device=DEVICE)
    combined=False; n_sep=0; n_recomb=0
    for step in range(nsteps):
        t=step*DT
        if step%10==0:
            p0=pos[0,b0i]; p1=pos[0,b1i]; ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0 and len(csa)==n_internal:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]); csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb=torch.ones(B,1,1,device=DEVICE); combined=True; n_recomb+=1
        st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g); o=torch.tanh(torch.bmm(h,W2)+b2g)
        detach_sig=o[:,:,3].mean(dim=1)
        if combined and detach_sig[0].item()>0.5:
            csa=sat.clone(); csb=sbt.clone(); crl=rlt.clone()
            comb=torch.zeros(B,1,1,device=DEVICE); combined=False; n_sep+=1
        og=(pos[:,:,1]<GROUND_Y+0.3).float(); gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc; ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5; te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE); f[:,:,1]+=GRAVITY
        pa=pos[:,csa]; pb=pos[:,csb]; d_=pb-pa
        di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di; r=crl.unsqueeze(0).unsqueeze(2); s=di-r
        rv=vel[:,csb]-vel[:,csa]; va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0); f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=3.0*vel[:,:,0]*bl; f[:,:,2]-=3.0*vel[:,:,2]*bl
        if wall_x is not None:
            past=(pos[:,:,0]-wall_x).clamp(min=0)
            in_gap=(pos[:,:,2].abs()<0.8).float()
            f[:,:,0]+=-200.0*past*(1.0-in_gap)
        f-=DRAG*vel; f+=ext; vel+=f*DT; vel.clamp_(-50,50); pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx; dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1)
    bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3; ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1); c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness=disp-dz-spp-bw-ep+coh
    fitness=torch.where(torch.isnan(fitness),torch.tensor(-9999.0,device=DEVICE),fitness)
    return fitness, n_sep, n_recomb

def evolve_topology_ctrl(data, nsteps, ngens, psz, label, wall_x=None):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUT=4; NW1=INPUT_DIM*HIDDEN
    NG=NW1+HIDDEN+HIDDEN*OUT+OUT+1
    s1=np.sqrt(2.0/(INPUT_DIM+HIDDEN)); s2=np.sqrt(2.0/(HIDDEN+OUT))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,NG,device=DEVICE)*0.3
    pop[:,:NW1]*=s1/0.3; pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE); t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_,_=sim_topology_ctrl(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,wall_x)
            pf[ix]=f
        o_=pf.argsort(descending=True); pop=pop[o_]; pf=pf[o_]
        if gen%50==0 or gen==ngens-1:
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05)); np2=pop[:ne].clone(); nf2=pf[:ne].clone()
        nfr=2; fr=torch.randn(nfr,NG,device=DEVICE)*0.3
        fr[:,:NW1]*=s1/0.3; fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]); nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,NG,device=DEVICE)<0.5; ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,NG,device=DEVICE)<0.15; ch+=torch.randn(nc,NG,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]); nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2; pf=nf2
    total=(time.time()-t0)/60
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    # Replay best for separation count
    best=pop[0:1]
    _,ns,nr=sim_topology_ctrl(best,rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,wall_x)
    print(f"  Separations: {ns}, Recombinations: {nr}")
    return pop[0].cpu().numpy(), pf[0].item(), ns, nr

def main():
    NSTEPS=600; GAP=0.5; PSZ=200; gx,gy,gz,sp=10,5,4,0.35; NGENS=300
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}; t0=time.time()

    print("="*70)
    print("EXP 23: TOPOLOGY CONTROL — RERUN")
    print("="*70)

    print("\n--- 23a: Spring control, no wall ---")
    bg23a,fit23a,sep_a,rec_a=evolve_topology_ctrl(data,NSTEPS,NGENS,PSZ,"23a_nowall",wall_x=None)
    results["23a"]={"fitness":round(fit23a,2),"separations":sep_a,"recombinations":rec_a}

    print("\n--- 23b: Spring control + wall at x=8 ---")
    bg23b,fit23b,sep_b,rec_b=evolve_topology_ctrl(data,NSTEPS,NGENS,PSZ,"23b_wall",wall_x=8.0)
    results["23b"]={"fitness":round(fit23b,2),"separations":sep_b,"recombinations":rec_b}

    # 23c: No spring control baseline (standard 3-output NN, always combined)
    print("\n--- 23c: Fixed combine, no wall (baseline) ---")
    # Reuse standard evolve from season6b
    from season6b_experiments import evolve, build_bodies_2
    data2=build_bodies_2(gx,gy,gz,sp,GAP)
    bg23c,fit23c=evolve(data2,NSTEPS,NGENS,PSZ,"23c_baseline")
    results["23c"]={"fitness":round(fit23c,2)}

    total=(time.time()-t0)/60
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    labels=["Spring Ctrl\n(no wall)","Spring Ctrl\n(wall@x=8)","Fixed Combine\n(baseline)"]
    fits=[fit23a,fit23b,fit23c]
    colors=["#3498db","#e74c3c","#2ecc71"]
    bars=ax.bar(range(3),fits,color=colors,alpha=0.8)
    for i in range(3):
        ax.text(i,fits[i]+2,f"{fits[i]:+.0f}",ha="center",fontsize=11,fontweight="bold")
    # Add separation info
    ax.text(0,fits[0]-10,f"sep={sep_a} recomb={rec_a}",ha="center",fontsize=8,color="white")
    ax.text(1,fits[1]-10,f"sep={sep_b} recomb={rec_b}",ha="center",fontsize=8,color="white")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("Fitness"); ax.set_title("Exp 23: Topology Control\nCan evolution discover useful separation?")
    ax.grid(alpha=0.3,axis="y")
    plt.tight_layout()
    fig_path=os.path.join(OUTPUT_DIR,"exp23_topology_control.png")
    plt.savefig(fig_path,dpi=200,bbox_inches="tight")
    print(f"\nFigure: {fig_path}")
    log_path=os.path.join(RESULTS_DIR,"exp23_log.json")
    with open(log_path,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Log: {log_path}")

    print(f"\n{'='*70}")
    print(f"EXP 23 COMPLETE ({total:.1f} min)")
    print(f"{'='*70}")
    print(f"  23a (ctrl, no wall): {fit23a:+.2f} | sep={sep_a} recomb={rec_a}")
    print(f"  23b (ctrl, wall):    {fit23b:+.2f} | sep={sep_b} recomb={rec_b}")
    print(f"  23c (fixed, base):   {fit23c:+.2f}")
    if sep_a>0 or sep_b>0:
        print(f"\n  🤖 TOPOLOGY SEPARATION DISCOVERED!")
    else:
        print(f"\n  🔒 No separation discovered (evolution keeps bodies combined)")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800,300); time.sleep(0.2)
    except: pass

if __name__=="__main__":
    main()
