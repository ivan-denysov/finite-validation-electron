"""
ЭЛЕКТРОН-4b: ДИНАМИЧЕСКИЙ тест двузначности пары.
2D фазовая сеть (Kuramoto-решётка). Пара = узлы A,B с сильной внутренней связью.
ОРИЕНТИРОВАННАЯ пара: связь несимметрична (A ведёт B: J_AB != J_BA).
Поворот: позиции A,B вращаются адиабатически вокруг общего центра (ось диполя
поворачивается в решётке; связи с окружением пересчитываются по близости).
Мерим внутреннюю разность delta = theta_B - theta_A (mod 2pi) на углах 0, 2pi, 4pi.
КОНТРОЛЬ: симметричная пара (J_AB = J_BA) — обязана вернуться при 2pi.
"""
import numpy as np

N=40  # решётка N x N фоновых узлов
K=1.0; dt=0.05
Jpair_lead=3.0   # A->B (ведущий)
Jpair_back=0.6   # B->A (ведомый) — асимметрия = ориентация
omega0=1.0

yy,xx=np.mgrid[0:N,0:N]
pos_bg=np.stack([xx.ravel(),yy.ravel()],1).astype(float)
Nbg=len(pos_bg)
cx,cy=N/2-0.5,N/2-0.5
Rax=2.2  # полудлина оси диполя

def neighbors_bg():
    # фоновая решётка: 4-соседи
    idx=lambda i,j:(i%N)*N+(j%N)
    nb=[[idx(i+1,j),idx(i-1,j),idx(i,j+1),idx(i,j-1)] for i in range(N) for j in range(N)]
    return np.array(nb)
NB=neighbors_bg()

def run(symmetric, total_angle, steps_per_rad=400, seed=0):
    rng=np.random.RandomState(seed)
    th=rng.uniform(0,2*np.pi,Nbg)
    thA=rng.uniform(0,2*np.pi); thB=rng.uniform(0,2*np.pi)
    JAB=Jpair_lead; JBA=Jpair_lead if symmetric else Jpair_back
    # прогрев на угле 0
    def pairpos(ang):
        ax=np.array([np.cos(ang),np.sin(ang)])*Rax
        return np.array([cx,cy])-ax, np.array([cx,cy])+ax
    def couple_bg(pA,pB,thA,thB,th):
        # пара связана с 4 ближайшими фоновыми узлами каждая, вес ~ exp(-d)
        out=np.zeros(Nbg); inA=0.0; inB=0.0
        for p,thp,which in [(pA,thA,'A'),(pB,thB,'B')]:
            d2=((pos_bg-p)**2).sum(1)
            nearest=np.argsort(d2)[:4]
            w=np.exp(-np.sqrt(d2[nearest]))
            for n_,w_ in zip(nearest,w):
                out[n_]+=w_*np.sin(thp-th[n_])
                if which=='A': inA+=w_*np.sin(th[n_]-thp)
                else: inB+=w_*np.sin(th[n_]-thp)
        return out,inA,inB
    def step(th,thA,thB,ang):
        pA,pB=pairpos(ang)
        nbsum=np.sin(th[NB]-th[:,None]).sum(1)/4
        out,inA,inB=couple_bg(pA,pB,thA,thB,th)
        th2=th+dt*(omega0+K*nbsum+0.5*out)
        dA=omega0+0.5*inA+JBA*np.sin(thB-thA)
        dB=omega0+0.5*inB+JAB*np.sin(thA-thB)
        return th2, thA+dt*dA, thB+dt*dB
    # прогрев
    for t in range(3000):
        th,thA,thB=step(th,thA,thB,0.0)
    d0=np.angle(np.exp(1j*(thB-thA)))
    # адиабатический поворот
    total_steps=int(total_angle*steps_per_rad)
    for t in range(total_steps):
        ang=total_angle*t/total_steps
        th,thA,thB=step(th,thA,thB,ang)
    # дорелаксация на конечном угле
    for t in range(3000):
        th,thA,thB=step(th,thA,thB,total_angle)
    d1=np.angle(np.exp(1j*(thB-thA)))
    return d0,d1

print("=== Ориентированная пара (A ведёт B) ===")
for ang,label in [(2*np.pi,"2pi"),(4*np.pi,"4pi")]:
    d0,d1=run(False,ang)
    shift=np.angle(np.exp(1j*(d1-d0)))
    print(f"оборот {label}: delta до={d0:+.4f}, после={d1:+.4f}, сдвиг={shift:+.4f} рад ({shift/np.pi:+.3f} pi)")
print()
print("=== КОНТРОЛЬ: симметричная пара ===")
for ang,label in [(2*np.pi,"2pi"),(4*np.pi,"4pi")]:
    d0,d1=run(True,ang)
    shift=np.angle(np.exp(1j*(d1-d0)))
    print(f"оборот {label}: delta до={d0:+.4f}, после={d1:+.4f}, сдвиг={shift:+.4f} рад ({shift/np.pi:+.3f} pi)")
print()
print("Спинорная сигнатура: ориентированная пара сдвиг ~pi на 2pi и ~0 на 4pi;")
print("контроль: ~0 на 2pi. Иначе — классическая однозначность (честный open debt).")
