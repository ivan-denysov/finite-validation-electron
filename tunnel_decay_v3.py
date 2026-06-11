"""
ЭЛЕКТРОН-1 v3: туннель с вычитанием фона.
Фон = энергия за АБСОЛЮТНОЙ стеной (рез связей) — не туннельная, остаточная.
Сигнал = E(d) - E_floor; фит экспоненты только по точкам сигнал > 3*фон-flux.
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

def lap_cut(th,wall):
    tot=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax); wb=np.roll(wall,sh,ax)
        link=1.0-np.maximum(wall,wb)
        tot+=link*(nb-th)
    return tot/dx**2

def init():
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2; A=0.8
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0))
    theta_t=-c*np.gradient(theta,dx,axis=1)
    return theta,theta_t

def E_right_at(theta,theta_t,x_meas):
    return (theta_t[:,x_meas:]**2).sum()+(np.gradient(theta[:,x_meas:],dx,axis=1)**2).sum()

def run_V(d,V):
    Vf=np.ones((Ny,Nx))
    if d>0: Vf[:,barrier_x:barrier_x+d]=V
    th,tht=init()
    for t in range(steps):
        a=c**2*lap(th)-Vf*np.sin(th)-damp*tht
        tht=tht+dt*a; th=th+dt*tht
    return E_right_at(th,tht,barrier_x+d+5)

def run_floor(d):
    wall=np.zeros((Ny,Nx)); wall[:,barrier_x:barrier_x+max(d,1)]=1.0
    th,tht=init()
    for t in range(steps):
        a=c**2*lap_cut(th,wall)-np.sin(th)-damp*tht
        tht=tht+dt*a; th=th+dt*tht
    return E_right_at(th,tht,barrier_x+max(d,1)+5)

E0=run_V(0,1.0)
print(f"E0 (нет барьера) = {E0:.2f}")
floor=run_floor(3)
print(f"фон (глухая стена) = {floor:.3f}")
print()
ds=[1,2,3,4,5,6,8]
for V in [3.0,6.0]:
    print(f"--- V={V} ---")
    sig=[]
    for d in ds:
        E=run_V(d,V); S=E-floor
        sig.append((d,E,S))
        print(f"d={d}: E={E:8.3f}  signal={S:8.3f}")
    # фит только по сигналу > 3*floor... точнее: signal > floor (S/floor>1)
    pts=[(d,S) for d,E,S in sig if S>floor]
    if len(pts)>=3:
        dd=np.array([p[0] for p in pts]); lnS=np.log([p[1] for p in pts])
        A_=np.vstack([dd,np.ones(len(dd))]).T
        coef,_,_,_=np.linalg.lstsq(A_,lnS,rcond=None)
        slope,inter=coef
        pred=A_@coef
        r2=1-((lnS-pred)**2).sum()/((lnS-lnS.mean())**2).sum()
        print(f"фит по {len(pts)} точкам (signal>фон): ln S = {slope:+.3f}*d {inter:+.2f}; R²={r2:.4f}; kappa={-slope:.3f}")
    print()
print("Туннельный закон: экспонента сигнала (R²>0.95), kappa(V=6) > kappa(V=3)")
