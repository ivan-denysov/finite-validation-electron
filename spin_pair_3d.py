"""
ЭЛЕКТРОН-4c: тест двузначности пары в 3D (pi1(SO(3))=Z2 — там живёт спин!).
3D кубическая фоновая решётка N^3, пара-диполь в центре, ось диполя вращается
адиабатически в плоскости xz (вокруг оси y) на 2pi и 4pi.
Ориентированная (J_AB != J_BA) vs симметричная пара.
Метрика: внутренняя разность delta=theta_B-theta_A до/после + дорелаксация.
"""
import numpy as np

N=16  # 16^3=4096 фоновых узлов
K=1.0; dt=0.05
Jlead=3.0; Jback=0.6
omega0=1.0

g=np.mgrid[0:N,0:N,0:N]
pos_bg=np.stack([g[0].ravel(),g[1].ravel(),g[2].ravel()],1).astype(float)
Nbg=len(pos_bg)
center=np.array([N/2-0.5]*3)
Rax=2.2

idx=lambda i,j,k:((i%N)*N+(j%N))*N+(k%N)
NB=np.array([[idx(i+1,j,k),idx(i-1,j,k),idx(i,j+1,k),idx(i,j-1,k),idx(i,j,k+1),idx(i,j,k-1)]
             for i in range(N) for j in range(N) for k in range(N)])

def pairpos(ang):
    ax=np.array([np.cos(ang),0.0,np.sin(ang)])*Rax  # вращение в xz вокруг y
    return center-ax, center+ax

def run(symmetric, total_angle, steps_per_rad=300, seed=0):
    rng=np.random.RandomState(seed)
    th=rng.uniform(0,2*np.pi,Nbg)
    thA=rng.uniform(0,2*np.pi); thB=rng.uniform(0,2*np.pi)
    JAB=Jlead; JBA=Jlead if symmetric else Jback
    def step(th,thA,thB,ang):
        pA,pB=pairpos(ang)
        nbsum=np.sin(th[NB]-th[:,None]).sum(1)/6
        out=np.zeros(Nbg); inA=0.0; inB=0.0
        for p,thp,which in [(pA,thA,'A'),(pB,thB,'B')]:
            d2=((pos_bg-p)**2).sum(1)
            nearest=np.argpartition(d2,6)[:6]
            w=np.exp(-np.sqrt(d2[nearest]))
            sin_out=w*np.sin(thp-th[nearest])
            out[nearest]+=sin_out
            s_in=(w*np.sin(th[nearest]-thp)).sum()
            if which=='A': inA=s_in
            else: inB=s_in
        th2=th+dt*(omega0+K*nbsum+0.5*out)
        dA=omega0+0.5*inA+JBA*np.sin(thB-thA)
        dB=omega0+0.5*inB+JAB*np.sin(thA-thB)
        return th2, thA+dt*dA, thB+dt*dB
    for t in range(2500):
        th,thA,thB=step(th,thA,thB,0.0)
    d0=np.angle(np.exp(1j*(thB-thA)))
    total_steps=int(total_angle*steps_per_rad)
    for t in range(total_steps):
        ang=total_angle*t/total_steps
        th,thA,thB=step(th,thA,thB,ang)
    for t in range(2500):
        th,thA,thB=step(th,thA,thB,total_angle)
    d1=np.angle(np.exp(1j*(thB-thA)))
    return d0,d1

print("=== 3D: ориентированная пара ===")
for ang,label in [(2*np.pi,"2pi"),(4*np.pi,"4pi")]:
    d0,d1=run(False,ang)
    sh=np.angle(np.exp(1j*(d1-d0)))
    print(f"оборот {label}: delta {d0:+.4f} -> {d1:+.4f}, сдвиг {sh:+.4f} ({sh/np.pi:+.3f} pi)")
print()
print("=== 3D: контроль (симметричная) ===")
for ang,label in [(2*np.pi,"2pi"),(4*np.pi,"4pi")]:
    d0,d1=run(True,ang)
    sh=np.angle(np.exp(1j*(d1-d0)))
    print(f"оборот {label}: delta {d0:+.4f} -> {d1:+.4f}, сдвиг {sh:+.4f} ({sh/np.pi:+.3f} pi)")
print()
print("Сигнатура спина: ~pi на 2pi И ~0 на 4pi у ориентированной; контроль ~0 на обоих.")
