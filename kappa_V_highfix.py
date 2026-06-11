"""ХВОСТ-1c: высокие V, фон через рез связей, A=1.6."""
import numpy as np
Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1500; barrier_x=120; damp_width=15
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
def init(A):
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    th=A*gauss*np.cos(k*(X-x0))
    return th, -c*np.gradient(th,dx,axis=1)
def Eright(th,tht,xm):
    return (tht[:,xm:]**2).sum()+(np.gradient(th[:,xm:],dx,axis=1)**2).sum()
def run_V(d,V,A):
    Vf=np.ones((Ny,Nx)); Vf[:,barrier_x:barrier_x+d]=V
    th,tht=init(A)
    for t in range(steps):
        a=c**2*lap(th)-Vf*np.sin(th)-damp*tht
        tht=tht+dt*a; th=th+dt*tht
    return Eright(th,tht,barrier_x+d+5)
def run_floor(A):
    wall=np.zeros((Ny,Nx)); wall[:,barrier_x:barrier_x+3]=1.0
    th,tht=init(A)
    for t in range(steps):
        a=c**2*lap_cut(th,wall)-np.sin(th)-damp*tht
        tht=tht+dt*a; th=th+dt*tht
    return Eright(th,tht,barrier_x+8)
A=1.6
floor=run_floor(A)
print(f"фон (рез связей, A={A}): {floor:.3f}")
print(f"{'V':>4} {'kappa_num':>10} {'kappa_th':>9} {'ratio':>6} {'точек':>6}")
for V in [9.0,14.0,20.0]:
    pts=[]
    for d in [1,2,3,4]:
        E=run_V(d,V,A); S=E-floor
        if S>floor: pts.append((d,S))
    if len(pts)>=3:
        dd=np.array([p[0] for p in pts],float)*dx
        lnS=np.log([p[1] for p in pts])
        Am=np.vstack([dd,np.ones(len(dd))]).T
        coef,_,_,_=np.linalg.lstsq(Am,lnS,rcond=None)
        kappa=-coef[0]/2; kth=np.sqrt(V-2.44)
        print(f"{V:>4} {kappa:>10.3f} {kth:>9.3f} {kappa/kth:>6.2f} {len(pts):>6}")
    else:
        det='; '.join(f'd={d}:S={S:.2f}' for d,S in pts)
        print(f"{V:>4}  точек выше фона: {len(pts)}  {det}")
