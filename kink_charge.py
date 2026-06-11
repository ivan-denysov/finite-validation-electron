"""
ЭЛЕКТРОН-2: заряд как топологический знак.
1D sine-Gordon: кинк (+2pi, "заряд +1") и антикинк (-2pi, "заряд -1").
(а) кинк-антикинк на встречных: притяжение/аннигиляция/прохождение?
(б) кинк-кинк (одноимённые): отталкивание?
Метрики: топологический заряд Q = (theta[конец]-theta[начало])/2pi; расстояние между центрами.
"""
import numpy as np

L=400; dx=0.5; Nx=int(L/dx); c=1.0; dt=0.2*dx/c
steps=3000
x=np.arange(Nx)*dx

def kink(x0,v,anti=False):
    g=1/np.sqrt(1-v**2); sgn=-1 if anti else 1
    th=4*np.arctan(np.exp(sgn*(x-x0)*g))
    tht=-v*g*2*sgn/np.cosh((x-x0)*g)*sgn  # производная по t
    # точная: d/dt 4 arctan(e^{sgn (x-vt) g}) = -v g sgn * 2 sech(sgn(x-x0)g)... sech чётна
    tht=-v*g*2/np.cosh((x-x0)*g)
    if anti: tht=-tht
    return th,tht

def lap(th): return (np.roll(th,1)-2*th+np.roll(th,-1))/dx**2

def centers(th):
    # центры кинков: пересечения уровня pi (кинк) и 3pi->pi (анти) — найду через градиент |th'| пики
    g=np.abs(np.gradient(th,dx))
    # два крупнейших локальных максимума градиента
    idx=[i for i in range(2,Nx-2) if g[i]>g[i-1] and g[i]>g[i+1] and g[i]>0.3*g.max()]
    idx=sorted(idx,key=lambda i:-g[i])[:2]
    return sorted(x[i] for i in idx)

def run(anti_second, v=0.2, x1=120, x2=280, label=""):
    th1,tht1=kink(x1,+v,anti=False)
    th2,tht2=kink(x2,-v,anti=anti_second)
    th=th1+th2-(0 if anti_second else 0)  # суперпозиция; для kink+kink th уходит 0->4pi
    tht=tht1+tht2
    Q0=(th[-1]-th[0])/(2*np.pi)
    print(f"--- {label}: Q_нач={Q0:.2f} ---")
    seps=[]
    for t in range(steps):
        a=c**2*lap(th)-np.sin(th)
        tht=tht+dt*a; th=th+dt*tht
        th[0]=th[1]; th[-1]=th[-2]
        if t%500==0 or t==steps-1:
            cs=centers(th)
            sep=cs[1]-cs[0] if len(cs)==2 else 0.0
            Q=(th[-1]-th[0])/(2*np.pi)
            Emax=np.abs(tht).max()
            seps.append((t,round(sep,1),round(Q,2)))
    print(f"(t, расстояние, Q): {seps}")
    return seps

print("=== (а) кинк + АНТИкинк, встречные v=0.2 ===")
s1=run(True, label="kink-antikink")
print()
print("=== (б) кинк + КИНК (одноимённые), встречные v=0.2 ===")
s2=run(False, label="kink-kink")
print()
print("Чтение: (а) расстояние ->0 и Q сохраняется 0 (аннигиляция/прохождение);")
print("(б) расстояние сжимается, затем РАСТЁТ, Q=2 сохраняется (отталкивание/отскок)")
