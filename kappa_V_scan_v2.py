"""ХВОСТ-1b: высокие V с усиленным пакетом A=1.6 (сигнал x4)."""
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
def run(d,V,A):
    Vf=np.ones((Ny,Nx))
    if d>0: Vf[:,barrier_x:barrier_x+d]=V
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0)); theta_t=-c*np.gradient(theta,dx,axis=1)
    for t in range(steps):
        a=c**2*lap(theta)-Vf*np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
    x_meas=barrier_x+d+5
    return (theta_t[:,x_meas:]**2).sum()+(np.gradient(theta[:,x_meas:],dx,axis=1)**2).sum()
A=1.6
floor=run(60,1e6,A)  # фон для A=1.6: непробиваемый барьер
print(f"фон (A=1.6): {floor:.3f}")
print(f"{'V':>4} {'kappa_num':>10} {'kappa_th':>9} {'ratio':>6} {'точек':>6}")
for V in [9.0,14.0,20.0]:
    pts=[]
    for d in [1,2,3,4]:
        E=run(d,V,A); S=E-floor
        if S>floor: pts.append((d,S))
    if len(pts)>=3:
        dd=np.array([p[0] for p in pts],float)*dx
        lnS=np.log([p[1] for p in pts])
        Am=np.vstack([dd,np.ones(len(dd))]).T
        coef,_,_,_=np.linalg.lstsq(Am,lnS,rcond=None)
        kappa=-coef[0]/2; kth=np.sqrt(V-2.44)
        print(f"{V:>4} {kappa:>10.3f} {kth:>9.3f} {kappa/kth:>6.2f} {len(pts):>6}")
    else:
        print(f"{V:>4}  точек выше фона: {len(pts)}")
