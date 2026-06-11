"""
ЭЛЕКТРОН-2 v5: лапласиан БЕЗ roll (честный Нейман) — снят шов 0|4pi.
Статический тест знака силы, обе пары из покоя.
"""
import numpy as np

L=300; dx=0.25; Nx=int(L/dx); c=1.0; dt=0.2*dx/c
steps=24000
x=np.arange(Nx)*dx

def prof(x0,s): return 4*np.arctan(np.exp(s*(x-x0)))

def lap_neumann(th):
    l=np.empty_like(th)
    l[1:-1]=(th[2:]-2*th[1:-1]+th[:-2])/dx**2
    l[0]=(th[1]-th[0])/dx**2*2*0+ (th[1]-2*th[0]+th[1])/dx**2  # ghost=th[1]
    l[-1]=(th[-2]-2*th[-1]+th[-2])/dx**2
    return l

def centers(th):
    g=np.abs(np.gradient(th,dx))
    idx=[i for i in range(2,Nx-2) if g[i]>g[i-1] and g[i]>g[i+1] and g[i]>0.3*g.max()]
    idx=sorted(idx,key=lambda i:-g[i])[:2]
    return sorted(x[i] for i in idx)

def energy(th,tht):
    gx=np.gradient(th,dx)
    return (0.5*tht**2+0.5*c**2*gx**2+(1-np.cos(th))).sum()*dx

def run(s2,sep0,label):
    x1=L/2-sep0/2; x2=L/2+sep0/2
    th=prof(x1,+1)+prof(x2,s2)
    tht=np.zeros_like(th)
    E0=energy(th,tht)
    print(f"--- {label}, sep0={sep0} ---")
    for t in range(steps):
        a=c**2*lap_neumann(th)-np.sin(th)
        tht=tht+dt*a; th=th+dt*tht
        if t%4000==0 or t==steps-1:
            cs=centers(th)
            sep=round(cs[1]-cs[0],2) if len(cs)==2 else 0.0
            print(f"t={t:6} sep={sep:7} E/E0={energy(th,tht)/E0:.4f}")

run(-1, 14, "kink-ANTIkink (ожид: сближение)")
print()
run(+1, 14, "kink-KINK (ожид: разъезд)")
