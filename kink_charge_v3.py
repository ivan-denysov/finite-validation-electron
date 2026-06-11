"""
ЭЛЕКТРОН-2 v3: kink-kink с корректными начальными скоростями.
Точная производная: theta(x,t)=4 arctan(exp(s*(x-x0-v t)*g)) =>
theta_t = -v*g*s * 4 e^{u}/(1+e^{2u}) = -v*g*s*2/cosh(u), u=s*(x-x0)g.
Для кинка s=+1; антикинка s=-1. ВАЖНО: множитель s в theta_t.
(в v2 для kink с v=-0.2 знак s не учтён -> кривые скорости -> взрыв)
"""
import numpy as np

L=400; dx=0.5; Nx=int(L/dx); c=1.0; dt=0.2*dx/c
steps=8000
x=np.arange(Nx)*dx

def kink_pair(x0,v,s):
    g=1/np.sqrt(1-v**2)
    u=s*(x-x0)*g
    th=4*np.arctan(np.exp(u))
    tht=-v*g*s*2/np.cosh(u)
    return th,tht

def lap(th): return (np.roll(th,1)-2*th+np.roll(th,-1))/dx**2

def centers(th):
    g=np.abs(np.gradient(th,dx))
    idx=[i for i in range(2,Nx-2) if g[i]>g[i-1] and g[i]>g[i+1] and g[i]>0.3*g.max()]
    idx=sorted(idx,key=lambda i:-g[i])[:2]
    return sorted(x[i] for i in idx)

def energy(th,tht):
    gx=np.gradient(th,dx)
    return (0.5*tht**2+0.5*c**2*gx**2+(1-np.cos(th))).sum()*dx

def run(s2, v=0.2, x1=160, x2=240, label=""):
    th1,tht1=kink_pair(x1,+v,+1)       # кинк вправо
    th2,tht2=kink_pair(x2,-v,s2)       # второй: кинк(s=+1) или анти(s=-1), влево
    th=th1+th2; tht=tht1+tht2
    print(f"--- {label} ---")
    E0=energy(th,tht)
    for t in range(steps):
        a=c**2*lap(th)-np.sin(th)
        tht=tht+dt*a; th=th+dt*tht
        th[0]=th[1]; th[-1]=th[-2]
        if t%1000==0 or t==steps-1:
            cs=centers(th)
            sep=round(cs[1]-cs[0],1) if len(cs)==2 else 0.0
            Q=round((th[-1]-th[0])/(2*np.pi),2)
            E=energy(th,tht)
            print(f"t={t:5} sep={sep:7} Q={Q:5} E/E0={E/E0:.3f}")

print("=== (б-fix) кинк+КИНК, встречные ===")
run(+1, label="kink-kink")
print()
print("=== (а-контроль) кинк+АНТИкинк той же постановкой ===")
run(-1, label="kink-antikink")
