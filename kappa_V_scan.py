"""
ХВОСТ-1: kappa(V) — закон показателя туннельного затухания от высоты барьера.
Линеаризация sine-Gordon в барьере: theta_tt = c² theta_xx - V theta =>
эванесцентный хвост exp(-kappa x), kappa = sqrt(V - omega²)/c для omega² < V.
Пакет k=1.2 => omega² ≈ c²k²+1 = 2.44 (дисперсия омега²=k²+m², m²=1 вне барьера).
Прогноз: kappa_theory = sqrt(V - 2.44).
Скан V = 4, 6, 9, 14, 20; фит exp по d=1..4 (сигнал>фон).
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1500
barrier_x=120
damp_width=15

damp=np.zeros((Ny,Nx))
for i in range(damp_width):
    w=(1-i/damp_width)*0.5
    damp[i,:]=np.maximum(damp[i,:],w); damp[Ny-1-i,:]=np.maximum(damp[Ny-1-i,:],w)
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w); damp[:,i]=np.maximum(damp[:,i],w)

def lap(th):
    return (np.roll(th,1,1)-2*th+np.roll(th,-1,1))/dx**2+(np.roll(th,1,0)-2*th+np.roll(th,-1,0))/dx**2

def run(d,V):
    Vf=np.ones((Ny,Nx))
    if d>0: Vf[:,barrier_x:barrier_x+d]=V
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2; A=0.8
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0))
    theta_t=-c*np.gradient(theta,dx,axis=1)
    for t in range(steps):
        a=c**2*lap(theta)-Vf*np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
    x_meas=barrier_x+d+5
    return (theta_t[:,x_meas:]**2).sum()+(np.gradient(theta[:,x_meas:],dx,axis=1)**2).sum()

floor=1.916  # фон с глухой стены (tunnel_decay_v3)
print("=== kappa(V): фит exp по d, сигнал>фон ===")
print(f"{'V':>4} {'kappa_num':>10} {'kappa_th=sqrt(V-2.44)':>22} {'ratio':>6} {'точек':>6}")
ks_num=[]; ks_th=[]
for V in [4.0,6.0,9.0,14.0,20.0]:
    ds=[1,2,3,4,5]
    pts=[]
    for d in ds:
        E=run(d,V); S=E-floor
        if S>floor: pts.append((d,S))
    if len(pts)>=3:
        dd=np.array([p[0] for p in pts],dtype=float)*dx  # толщина в ЕДИНИЦАХ ДЛИНЫ
        lnS=np.log([p[1] for p in pts])
        A_=np.vstack([dd,np.ones(len(dd))]).T
        coef,_,_,_=np.linalg.lstsq(A_,lnS,rcond=None)
        kappa=-coef[0]/2  # энергия ~ амплитуда² => E~exp(-2 kappa d)
        kth=np.sqrt(max(V-2.44,0.01))
        ks_num.append(kappa); ks_th.append(kth)
        print(f"{V:>4} {kappa:>10.3f} {kth:>22.3f} {kappa/kth:>6.2f} {len(pts):>6}")
    else:
        print(f"{V:>4}  мало точек выше фона ({len(pts)})")
if len(ks_num)>=3:
    r=np.corrcoef(ks_num,ks_th)[0,1]
    print(f"\nкорреляция kappa_num vs theory: {r:.4f}")
    print("ratio ~ const и корреляция высокая => закон sqrt(V-omega²) воспроизведён")
