"""
ЭЛЕКТРОН-3c: РЕШАЮЩИЙ тест — sigma_eff(t) на длинной шкале.
warm = 4k, 8k, 16k, 32k, 64k. Затухает к нулю или плато?
Если затухает: закон (экспонента? степенной?) — тогда для стационара нужна накачка.
N=3000 (меньше ради скорости), deg=6, K=1, 2 seeds.
"""
import numpy as np

def build(N, deg, seed, box=30.0):
    rng=np.random.RandomState(seed)
    pos=rng.uniform(0,box,(N,3))
    cell=box/8; grid={}
    for i,p in enumerate(pos):
        grid.setdefault(tuple((p//cell).astype(int)),[]).append(i)
    nbrs=[]
    for i,p in enumerate(pos):
        key=(p//cell).astype(int); cand=[]
        for a in(-1,0,1):
            for b in(-1,0,1):
                for cc in(-1,0,1):
                    cand+=grid.get((key[0]+a,key[1]+b,key[2]+cc),[])
        cand=[j for j in cand if j!=i]
        d=np.sqrt(((pos[cand]-p)**2).sum(1))
        order=np.argsort(d)[:deg]
        nbrs.append([cand[k] for k in order])
    maxd=max(len(n) for n in nbrs)
    na=np.full((N,maxd),-1)
    for i,n in enumerate(nbrs): na[i,:len(n)]=n
    valid=na>=0; cnt=valid.sum(1); cnt[cnt==0]=1
    return na,valid,cnt

def run_seed(seed, N=3000, deg=6, K=1.0, dt=0.02, window=1500):
    na,valid,cnt=build(N,deg,seed)
    rng=np.random.RandomState(seed+500)
    th=rng.uniform(0,2*np.pi,N)
    om=np.ones(N)
    def step(th):
        tn=np.where(valid,th[na],0.0)
        coup=np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
        return th+dt*(om+K*coup)
    checkpoints=[4000,8000,16000,32000,64000]
    out=[]
    t_done=0
    for cp in checkpoints:
        while t_done<cp:
            th=step(th); t_done+=1
        th0=th.copy()
        for _ in range(window): th=step(th)
        t_done+=window
        v=(th-th0)/(window*dt)
        out.append((cp, v.std()))
    return out

print("=== sigma_eff(t): затухание или плато? (N=3000, deg=6, 2 seeds) ===")
allruns=[]
for seed in [0,1]:
    res=run_seed(seed)
    allruns.append(res)
    print(f"seed {seed}: " + "  ".join(f"t={cp}: {s:.6f}" for cp,s in res))
# закон затухания: log-log
cps=np.array([r[0] for r in allruns[0]],dtype=float)
sig=np.mean([[s for _,s in run] for run in allruns],axis=0)
print()
print(f"{'t':>7} {'sigma_mean':>11}")
for c,s in zip(cps,sig): print(f"{int(c):>7} {s:>11.6f}")
A=np.vstack([np.log(cps),np.ones(len(cps))]).T
coef,_,_,_=np.linalg.lstsq(A,np.log(sig),rcond=None)
slope,inter=coef
pred=A@coef
r2=1-((np.log(sig)-pred)**2).sum()/((np.log(sig)-np.log(sig).mean())**2).sum()
print(f"степенной фит: sigma ~ t^{slope:.2f}, R²={r2:.4f}")
print("slope<0 и R² высокий => затухание степенное, статического остатка не видно => нужна накачка")
print("выход на плато (последние точки ровные) => статический остаток есть")
