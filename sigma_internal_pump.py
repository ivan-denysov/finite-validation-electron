"""
ЭЛЕКТРОН-3e: ВНУТРЕННЯЯ накачка — честный уровень.
Вместо rng-пинков: по сети ходят ДВИЖУЩИЕСЯ ВОЗМУЩЕНИЯ (наши же "частицы"-драйверы из
прогонов трубы): M узлов-носителей с повышенным темпом, каждый перепрыгивает на случайного
соседа каждые hop шагов (движение = пересборка). Трафик = M носителей.
Вопрос: (а) плато sigma_eff фоновых узлов? (б) растёт с M (трафиком)?
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
    return na,valid,cnt,nbrs

def run(M, seed=0, N=3000, deg=6, K=1.0, dt=0.02, drive=2.0, hop=50):
    na,valid,cnt,nbrs=build(N,deg,seed)
    rng=np.random.RandomState(seed+700)
    th=rng.uniform(0,2*np.pi,N)
    carriers=rng.choice(N,M,replace=False).tolist()
    def step(th,carriers,t):
        om=np.ones(N); om[carriers]=drive
        tn=np.where(valid,th[na],0.0)
        coup=np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
        th=th+dt*(om+K*coup)
        if t%hop==0:
            carriers=[rng.choice(nbrs[c]) if nbrs[c] else c for c in carriers]
        return th,carriers
    for t in range(20000):
        th,carriers=step(th,carriers,t)
    window=1500
    sigs=[]
    for rep in range(3):
        th0=th.copy(); cs0=set(carriers)
        for tt in range(window):
            th,carriers=step(th,carriers,tt)
        v=(th-th0)/(window*dt)
        mask=np.ones(N,bool); mask[list(set(carriers)|cs0)]=False  # фон: не носители
        sigs.append(v[mask].std())
    return sigs

print("=== Внутренняя накачка: трафик M движущихся возмущений ===")
print(f"{'M':>4} {'три замера':>28} {'sigma_stat':>11}")
res=[]
for M in [0, 5, 15, 45]:
    sigs=run(M)
    s=np.mean(sigs)
    res.append((M,s))
    print(f"{M:>4} {'  '.join(f'{x:.5f}' for x in sigs):>28} {s:>11.5f}")
print()
ms=np.array([r[0] for r in res[1:]],dtype=float); ss=np.array([r[1] for r in res[1:]])
A=np.vstack([np.log(ms),np.ones(len(ms))]).T
coef,_,_,_=np.linalg.lstsq(A,np.log(ss),rcond=None)
slope,inter=coef
pred=A@coef
r2=1-((np.log(ss)-pred)**2).sum()/((np.log(ss)-np.log(ss).mean())**2).sum()
print(f"sigma_stat ~ M^{slope:.2f} (R²={r2:.4f}); M=0 фон: {res[0][1]:.5f}")
print("Плато + рост с M => Λ управляется ТРАФИКОМ самой сети (внутренняя накачка)")
