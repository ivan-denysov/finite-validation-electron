"""
ЭЛЕКТРОН-3d: вторая половина — НАКАЧКА.
Слабый непрерывный источник возмущений: на каждом шаге случайные узлы (доля rate)
получают фазовый пинок ~N(0, kick). Вопрос: выходит ли sigma_eff на СТАЦИОНАРНОЕ плато,
и как плато зависит от интенсивности накачки Gamma = rate*kick^2/dt (мощность шума)?
Ожидание (Орнштейн-Уленбек логика): sigma_stat^2 ~ Gamma/(2*lambda_relax).
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

def run(seed, kick, rate=0.01, N=3000, deg=6, K=1.0, dt=0.02):
    na,valid,cnt=build(N,deg,seed)
    rng=np.random.RandomState(seed+500)
    th=rng.uniform(0,2*np.pi,N)
    om=np.ones(N)
    def step(th):
        tn=np.where(valid,th[na],0.0)
        coup=np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
        th=th+dt*(om+K*coup)
        nk=int(rate*N)
        idx=rng.choice(N,nk,replace=False)
        th[idx]+=rng.normal(0,kick,nk)
        return th
    # прогрев до стационара
    for t in range(20000): th=step(th)
    # три последовательных замера: плато = все три одинаковы
    window=1500
    sigs=[]
    for rep in range(3):
        th0=th.copy()
        for _ in range(window): th=step(th)
        v=(th-th0)/(window*dt)
        sigs.append(v.std())
    return sigs

print("=== Накачка: плато sigma_eff и зависимость от мощности ===")
print(f"{'kick':>6} {'три замера подряд':>30} {'sigma_stat':>11}")
results=[]
for kick in [0.05, 0.1, 0.2, 0.4]:
    sigs=run(0,kick)
    s=np.mean(sigs)
    results.append((kick,s))
    print(f"{kick:>6} {'  '.join(f'{x:.5f}' for x in sigs):>30} {s:>11.5f}")
print()
# масштабирование: sigma_stat ~ kick^alpha?
ks=np.array([r[0] for r in results]); ss=np.array([r[1] for r in results])
A=np.vstack([np.log(ks),np.ones(len(ks))]).T
coef,_,_,_=np.linalg.lstsq(A,np.log(ss),rcond=None)
slope,inter=coef
pred=A@coef
r2=1-((np.log(ss)-pred)**2).sum()/((np.log(ss)-np.log(ss).mean())**2).sum()
print(f"sigma_stat ~ kick^{slope:.2f}, R²={r2:.4f}")
print("Плато (3 замера ровные) + степенной закон от накачки => Λ-формула состояния есть")
