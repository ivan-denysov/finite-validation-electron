"""
ЭЛЕКТРОН-3b: два контроля шероховатости.
(1) Прогрев x4: sigma_eff стационарна или это переходник?
(2) Скейлинг по блокам: sigma(средней частоты блока) от размера блока — закон 1/sqrt(N_блока)?
   Если да: малость Λ = статистика усреднения локальной шероховатости по объёму.
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
    return pos,na,valid,cnt

def measure(N,deg,seed,K=1.0,warm=4000,dt=0.02,window=2000):
    pos,na,valid,cnt=build(N,deg,seed)
    rng=np.random.RandomState(seed+500)
    th=rng.uniform(0,2*np.pi,N)
    om=np.ones(N)
    def step(th):
        tn=np.where(valid,th[na],0.0)
        coup=np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
        return th+dt*(om+K*coup)
    for t in range(warm): th=step(th)
    th0=th.copy()
    for t in range(window): th=step(th)
    v=(th-th0)/(window*dt)
    return pos,v

print("=== (1) Стационарность: sigma_eff при прогреве 4k vs 16k (deg=6, N=4000, seed=0) ===")
_,v1=measure(4000,6,0,warm=4000)
_,v2=measure(4000,6,0,warm=16000)
print(f"warm= 4000: sigma_eff={v1.std():.6f}")
print(f"warm=16000: sigma_eff={v2.std():.6f}")
print("(упало сильно => мерили переходник; стоит => стационарная шероховатость)")
print()
print("=== (2) Скейлинг по блокам (warm=16000): sigma средней частоты блока от N_блока ===")
pos,v=measure(8000,6,0,warm=16000)
box=30.0
for nb in [2,3,4,6,8]:
    cell=box/nb
    blocks={}
    for i,p in enumerate(pos):
        key=tuple((p//cell).astype(int))
        blocks.setdefault(key,[]).append(i)
    means=[v[idx].mean() for idx in blocks.values() if len(idx)>=3]
    sizes=[len(idx) for idx in blocks.values() if len(idx)>=3]
    Nb=np.mean(sizes)
    s=np.std(means)
    print(f"блоков {len(means):4}, ср.узлов/блок {Nb:7.1f}: sigma_блока = {s:.6f}, sigma*sqrt(Nb) = {s*np.sqrt(Nb):.5f}")
print()
print("Если sigma*sqrt(Nb) ~ const => закон 1/sqrt(N): шероховатость усредняется статистически")
