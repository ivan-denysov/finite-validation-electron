"""
ЭЛЕКТРОН-3: sigma как геометрическая шероховатость.
Аморфная 3D сеть, ВСЕ узлы с ОДИНАКОВЫМ собственным темпом omega0=1 (шаблон один!).
Вопрос: каков разброс ЭФФЕКТИВНЫХ частот узлов из-за разной конфигурации соседей?
Эффективная частота узла в синхроне: omega_eff_i = omega0 + K*<sin(theta_j-theta_i)>_сосед
(в идеальной решётке все слагаемые сократились бы; в аморфной — остаток от геометрии).
Меряем: sigma_eff = std(средних фазовых скоростей узлов) на стационаре, 
скан по плотности (deg) и по размеру усреднения.
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

def sigma_eff(N,deg,seed,K=1.0,steps=4000,dt=0.02):
    na,valid,cnt=build(N,deg,seed)
    rng=np.random.RandomState(seed+500)
    th=rng.uniform(0,2*np.pi,N)
    om=np.ones(N)  # ШАБЛОН ОДИН: все темпы = 1.0 точно
    # прогрев
    for t in range(steps):
        tn=np.where(valid,th[na],0.0)
        coup=np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
        th=th+dt*(om+K*coup)
    # измерение: средняя фазовая скорость каждого узла за окно
    window=2000
    th0=th.copy()
    for t in range(window):
        tn=np.where(valid,th[na],0.0)
        coup=np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
        th=th+dt*(om+K*coup)
    v=(th-th0)/(window*dt)   # эффективная частота узла
    return v.std(), v.mean()

print("=== sigma_eff от плотности связей (N=4000, 3 seeds) ===")
print(f"{'deg':>4} {'sigma_eff':>12} {'mean':>8}")
for deg in [4,6,10,16]:
    ss=[]
    for seed in range(3):
        s,m=sigma_eff(4000,deg,seed)
        ss.append(s)
    print(f"{deg:>4} {np.mean(ss):>12.6f} {m:>8.4f}   (по seeds: {' '.join(f'{x:.5f}' for x in ss)})")
print()
print("Если sigma_eff>0 при omega одинаковых => шероховатость реальна (геометрия даёт разброс)")
print("Если sigma_eff падает с deg => плотная сеть глаже => механизм малости Λ")
