"""
ЭЛЕКТРОН-1: туннелирование как просачивание распределения.
2D sine-Gordon, барьер = СПЛОШНОЙ разрыв связности толщины d (не щель!).
Пакет бьёт в барьер; мерим энергию за барьером E(d) для d = 1..6 слоёв.
Гипотеза: E(d) затухает ~экспоненциально с толщиной (туннельный закон).
Контроль: d=0 (нет барьера) — нормировка.
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1500
barrier_x=120; 
damp_width=15

damp=np.zeros((Ny,Nx))
for i in range(damp_width):
    w=(1-i/damp_width)*0.5
    damp[i,:]=np.maximum(damp[i,:],w); damp[Ny-1-i,:]=np.maximum(damp[Ny-1-i,:],w)
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w); damp[:,i]=np.maximum(damp[:,i],w)

def lap_cut(th,wall):
    tot=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax); wb=np.roll(wall,sh,ax)
        link=1.0-np.maximum(wall,wb)
        tot+=link*(nb-th)
    return tot/dx**2

def run(d):
    wall=np.zeros((Ny,Nx))
    if d>0:
        wall[:,barrier_x:barrier_x+d]=1.0   # сплошной разрыв толщины d
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2; A=0.8
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0))
    theta_t=-c*np.gradient(theta,dx,axis=1)
    for t in range(steps):
        a=c**2*lap_cut(theta,wall)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
    x_meas=barrier_x+d+5
    E_right=(theta_t[:,x_meas:]**2).sum()+(np.gradient(theta[:,x_meas:],dx,axis=1)**2).sum()
    return E_right

print("=== Туннель: энергия за сплошным барьером толщины d ===")
E0=run(0)
print(f"d=0 (нет барьера): E={E0:.2f} (нормировка)")
ds=[1,2,3,4,5,6]
Es=[]
for d in ds:
    E=run(d); Es.append(E)
    print(f"d={d}: E={E:.4f}  T=E/E0={E/E0:.2e}")
# экспонента? ln E vs d
lnE=np.log(np.array(Es))
A_=np.vstack([np.array(ds),np.ones(len(ds))]).T
coef,res,_,_=np.linalg.lstsq(A_,lnE,rcond=None)
slope,inter=coef
pred=A_@coef
r2=1-((lnE-pred)**2).sum()/((lnE-lnE.mean())**2).sum()
print()
print(f"ln E = {slope:.3f}·d + {inter:.2f};  R²={r2:.4f}")
print(f"затухание на слой: ×{np.exp(slope):.3f}")
print("Экспонента (R²>0.95) => туннельный закон в инструменте есть")
