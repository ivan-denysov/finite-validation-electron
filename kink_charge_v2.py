"""
ЭЛЕКТРОН-2 v2: столкновение. Ближе старт (sep=80), дольше прогон (8000), лог чаще у встречи.
Ключ: что ПОСЛЕ встречи — прошли/аннигилировали (анти) vs отскочили (одноимённые).
"""
import numpy as np

L=400; dx=0.5; Nx=int(L/dx); c=1.0; dt=0.2*dx/c
steps=8000
x=np.arange(Nx)*dx

def kink(x0,v,anti=False):
    g=1/np.sqrt(1-v**2)
    sgn=-1 if anti else 1
    th=4*np.arctan(np.exp(sgn*(x-x0)*g))
    tht=-v*g*2/np.cosh((x-x0)*g)
    if anti: tht=-tht
    return th,tht

def lap(th): return (np.roll(th,1)-2*th+np.roll(th,-1))/dx**2

def centers(th):
    g=np.abs(np.gradient(th,dx))
    idx=[i for i in range(2,Nx-2) if g[i]>g[i-1] and g[i]>g[i+1] and g[i]>0.3*g.max()]
    idx=sorted(idx,key=lambda i:-g[i])[:2]
    return sorted(x[i] for i in idx)

def run(anti_second, v=0.2, x1=160, x2=240, label=""):
    th1,tht1=kink(x1,+v)
    th2,tht2=kink(x2,-v,anti=anti_second)
    th=th1+th2; tht=tht1+tht2
    print(f"--- {label} ---")
    log=[]
    for t in range(steps):
        a=c**2*lap(th)-np.sin(th)
        tht=tht+dt*a; th=th+dt*tht
        th[0]=th[1]; th[-1]=th[-2]
        if t%800==0 or t==steps-1:
            cs=centers(th)
            sep=round(cs[1]-cs[0],1) if len(cs)==2 else 0.0
            Q=round((th[-1]-th[0])/(2*np.pi),2)
            Emax=round(np.abs(tht).max(),2)
            log.append((t,sep,Q,Emax))
    for row in log: print(f"t={row[0]:5} sep={row[1]:7} Q={row[2]:5} |θt|max={row[3]}")
    return log

print("=== (а) кинк+АНТИкинк, старт sep=80 ===")
run(True, label="kink-antikink")
print()
print("=== (б) кинк+КИНК ===")
run(False, label="kink-kink")
