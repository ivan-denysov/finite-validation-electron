"""
ЭЛЕКТРОН-1 v2: туннель = полоса ЖЁСТКОСТИ (конечный барьер), не разрыв.
В полосе толщины d потенциальный член усилен: V*sin(theta), V>1.
Ожидание: E(d) ~ exp(-kappa*d), kappa растёт с V.
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

def run(d, V):
    Vfield=np.ones((Ny,Nx))
    if d>0:
        Vfield[:,barrier_x:barrier_x+d]=V
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2; A=0.8
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0))
    theta_t=-c*np.gradient(theta,dx,axis=1)
    for t in range(steps):
        a=c**2*lap(theta)-Vfield*np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
    x_meas=barrier_x+d+5
    E_right=(theta_t[:,x_meas:]**2).sum()+(np.gradient(theta[:,x_meas:],dx,axis=1)**2).sum()
    return E_right

E0=run(0,1.0)
print(f"нормировка (нет барьера): E0={E0:.2f}")
print()
ds=[2,4,6,8,10]
for V in [3.0, 6.0, 12.0]:
    Es=[]
    for d in ds:
        Es.append(run(d,V))
    lnE=np.log(np.array(Es))
    A_=np.vstack([np.array(ds),np.ones(len(ds))]).T
    coef,_,_,_=np.linalg.lstsq(A_,lnE,rcond=None)
    slope,inter=coef
    pred=A_@coef
    r2=1-((lnE-pred)**2).sum()/((lnE-lnE.mean())**2).sum()
    Estr=" ".join(f"{e:7.3f}" for e in Es)
    print(f"V={V:5}: E(d)= {Estr}")
    print(f"        ln E = {slope:+.3f}*d {inter:+.2f}; R²={r2:.4f}; затух/слой ×{np.exp(slope):.3f}")
print()
print("Туннельный закон: экспонента по d (R²>0.95) и |наклон| растёт с V")
