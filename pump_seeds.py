"""ХВОСТ-2: статистика плато внутренней накачки — 3 seeds x M-скан."""
import numpy as np
exec(open('sigma_internal_pump.py').read().split('print("=== Внутренняя')[0])
print("=== Внутренняя накачка: 3 seeds ===")
print(f"{'M':>4} {'sigma по seeds':>30} {'mean±std':>16}")
res=[]
for M in [5,15,45]:
    ss=[]
    for seed in [0,1,2]:
        sigs=run(M,seed=seed)
        ss.append(np.mean(sigs))
    res.append((M,np.mean(ss),np.std(ss)))
    print(f"{M:>4} {'  '.join(f'{x:.5f}' for x in ss):>30} {np.mean(ss):.5f}±{np.std(ss):.5f}")
ms=np.array([r[0] for r in res],float); sm=np.array([r[1] for r in res])
A=np.vstack([np.log(ms),np.ones(3)]).T
coef,_,_,_=np.linalg.lstsq(A,np.log(sm),rcond=None)
print(f"\nsigma_stat ~ M^{coef[0]:.2f} (3 seeds)")
