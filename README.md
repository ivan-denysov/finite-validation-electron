# Numerical Companion: Tunnelling, Charge Sign, Vacuum Traffic, and the Spin Carrier in Finite-Validation Models

*Ivan Denysov, Lead Researcher, United Field Initiative (UFI). Companion to: "Physics from Finite Validation: The Electron as a Mode of the Vacuum Network" [DOI to be assigned after Zenodo deposit]. Code: https://github.com/ivan-denysov/finite-validation-electron*

## Purpose and Status Discipline

This companion contains every numerical run supporting the main article, with full code, parameters, seeds, and per-result status labels. The discipline follows the series: **confirmed in-model**, **observation**, **structural result**, **rejected** (tested and failed), **lesson/artefact** (a result traced to the setup, documented). Runs are exploratory instruments, not measurements of nature: "confirmed" always means confirmed *within the stated model*. A systematic feature of the instrument, named in the main article (§3): across the programme, runs reproduce the *form* of each law at R² > 0.98 while coefficients come to 0.8–0.85 of the analytic prediction, consistent with finite packet/band width — the form is the claim, the coefficient is instrument-limited.

Two model regimes are used, inherited from the preceding work (DOI 10.5281/zenodo.20634290): a Kuramoto-type phase network (traffic, σ-state) and a 2D/1D sine-Gordon field (tunnelling, charge, spin tests). Their continuum bridge remains the programme's load-bearing open debt.

All scripts are standalone Python 3 (numpy only). One pairing: `pump_seeds.py` executes `sigma_internal_pump.py` via exec — keep the two files together.

---

## Block A — Tunnelling (mass as delay; main article §3)

**A.1 The infinite-barrier lesson [lesson].**
- `tunnel_decay.py` — barrier as a *break of connectivity* (cut links): energy beyond it is independent of thickness (E ≈ 1.91 = background at every d). A cut-link barrier is an **infinite** barrier; nothing tunnels through it — correct behaviour, wrong model of a finite barrier. Lesson preserved.

**A.2 The exponential law [confirmed in-model, linear regime].**
- `tunnel_decay_v2.py` — finite barrier as a stiffness band (V·sinθ): structure visible but contaminated by the background floor (~1.97). Intermediate step, preserved.
- `tunnel_decay_v3.py` — background subtraction (floor 1.916 measured against a solid cut-link wall): transmitted signal decays exponentially, ln S = −2κd; V=3: seven thicknesses on one line, R² = 0.989, κ = 0.488; V=6: κ = 1.641, R² = 0.9993 (three points above floor — noted). Both tunnelling signatures: exponential in thickness, exponent grows with height.

**A.3 The κ(V) law [confirmed in-model in form; coefficient instrument-limited].**
- `kappa_V_scan.py` — linear regime (A=0.8): κ/κ_th = 0.83 (V=3), 0.87 (V=6) against κ_th = √(V−ω²), ω²≈2.44. Stable coefficient ~0.85. Higher V unreachable in the linear regime (signal below floor) — a setup limit, named.
- `kappa_V_scan_v2.py` — attempted A=1.6 with V=10⁶ floor: CFL blow-up (effective mass √V breaks the time step). Lesson: measure the floor with cut links, not with giant V.
- `kappa_V_highfix.py` — correct floor (cut links, A=1.6): at V=9, κ = 0.456 against theory 2.561 (ratio 0.18) — **nonlinearity enhances seepage** (harmonic transfer beyond linear evanescence). Status: observation; it bounds the validity of A.2 to the linear regime (main article §3, §8.6).

---

## Block B — Charge (topological sign; main article §4)

**B.1 Setup lessons [artefacts, documented].**
- `kink_charge.py` — counter-propagating pair, run too short: no collision reached. Preserved.
- `kink_charge_v2.py` — sign error in the time derivative of the second kink → energy blow-up before contact. Preserved as a trap.
- `kink_charge_v3.py` — corrected velocities: kink–antikink passes through cleanly (E/E₀ = 0.999 throughout); kink–kink still blows up (E/E₀ = 1.78 at the first step) — traced to the periodic seam of the roll-based Laplacian tearing on the 0|4π configuration (the 2π|2π seam of kink–antikink is accidentally clean). Preserved as the decisive diagnostic.

**B.2 The sign of the force [confirmed in-model].**
- `kink_force_static.py` — static start (zero velocities); kink–antikink branch clean (attraction), kink–kink still torn by the seam.
- `kink_force_static_v2.py` — honest Neumann Laplacian (no roll): **kink–antikink approach on their own, 14 → 9.5** (then oscillate ~9.5–11 — a hint of a bound state); **kink–kink separate on their own, 14 → 17**; E/E₀ = 1.0000 in both branches. Opposite charges attract, like charges repel, from pure statics, no fitting. Known sine-Gordon behaviour (U ~ ±32e^(−sep)) — the carrier of charge is validated in the instrument, not discovered. The force is short-ranged; Coulomb 1/r² is a named debt (main article §8.1).

---

## Block C — σ as a State Property; Λ as an Equation of State (main article §5–6)

**C.1 Geometric roughness exists; densification does not smooth it [confirmed in-model].**
- `sigma_geometry.py` — amorphous 3D network, all intrinsic tempos strictly equal (one template): σ_eff ≈ 0.0085 > 0 from geometry alone; flat in deg = 4..16 (3 seeds each).

**C.2 The rest state decays; block averaging is sub-√N [confirmed in-model].**
- `sigma_scaling.py` — warm-up ×4 drops σ_eff threefold (0.0076 → 0.0026): the early value is mostly transient. Block averaging works but slower than 1/√N (σ·√N drifts 0.011 → 0.029): nodes are spatially correlated.
- `sigma_longtime.py` — the decisive rest test: σ_eff(t) decays as ~t^−1.4 over two orders (2 seeds), no plateau. **A vacuum at rest synchronises to zero: Λ(rest) = 0.**

**C.3 Pumping produces a stationary plateau [confirmed in-model].**
- `sigma_pumped.py` — external random kicks: plateau at each power level (three consecutive measurements coincide); σ_stat ~ kick^0.77 (R²=0.989). External source is a crutch — superseded by C.4.
- `sigma_internal_pump.py` — **internal pumping**: M moving disturbance-carriers hopping between neighbours (reassemblies). M=0 control decays; M = 5/15/45 give plateaus with σ_stat ~ M^0.51 (single seed).
- `pump_seeds.py` — statistics: 3 seeds × M-scan: σ_stat = 0.00787±0.00091 / 0.01198±0.00092 / 0.02616±0.00243; exponent 0.55. Consistent with σ² ∝ M (CLT) within the precision of three realisations; an exponent through three values of M is weakly constrained; 0.55-vs-½ unresolved (main article §8.4). Through ε = σ²/2K²: **vacuum energy ∝ traffic density** — the equation-of-state structure of Λ.

---

## Block D — Spin (the carrier; main article §7)

**D.1 The dynamical hypothesis — killed [rejected, documented].**
- `spin_pair_rotation.py` — 2D: oriented pair (J_AB=3.0, J_BA=0.6) under adiabatic rotation; internal phase returns with shift +0.022π (2π) and +0.024π (4π) — no π-jump; symmetric control clean.
- `spin_pair_3d.py` — 3D (π₁(SO(3))=Z₂ — the group where spin lives): shifts −0.025π at both turns, control −0.003π. **No two-valuedness in 2D or 3D. The hypothesis "fermionicity = orientation of the link" is rejected by dynamical test.** Reinterpreted as a test of the condition: links were recomputed at every step (no path memory) — spin lives in the memory of the links.

**D.2 The geometric carrier [structural result + known mathematics].**
- `spin_loop_test.py` — setup sketch only (an arctan2 tautology; NOT evidence; preserved as scaffolding, labelled as such in the code).
- `spin_belt_geometry.py` — the Z₂ structure explicit: 0/2π/4π/6π/8π → identity/non-identity/identity/non-identity/identity. Quaternion control (the double cover SU(2)→SO(3)): endpoint sign +1/−1/+1/−1/+1 by turns — the spinor alternation. **Honest labels:** the quaternionic part is known mathematics, cited as support; the "isotopy = mod 4π" step is the known belt-trick relation, *built in*, not derived by enumerating ribbon moves — an illustration, marked. Our result is the **choice of carrier**: of the full geometric enumeration of two-node configurations, only the pair-with-ties carries Z₂ — and ties are constitutive in the validation ontology (a node without links does not exist). Spin ½ as the twist parity of validation ties: a property of an object that cannot be isolated. Dynamics with link memory, and the spin–statistics connection, are named debts (main article §8.2–8.3).

---

## Reproducibility

Python 3.10+, numpy only. Every script runs standalone (exception: `pump_seeds.py` requires `sigma_internal_pump.py` alongside). Seeds are written in the code; per-seed outputs are printed by the scripts. Runtime: seconds to ~minutes per script; the longest are `sigma_longtime.py` (~30 min) and `pump_seeds.py` (~45 min) on a laptop.

## File → article map

| Script | Article section | Status |
|---|---|---|
| tunnel_decay.py | §3 | lesson (infinite barrier) |
| tunnel_decay_v2.py | §3 | intermediate (floor contamination) |
| tunnel_decay_v3.py | §3 | confirmed in-model (exponential law, linear regime) |
| kappa_V_scan.py | §3 | confirmed in-model (κ~√(V−ω²) in form, coeff. 0.85) |
| kappa_V_scan_v2.py | §3 | lesson (CFL blow-up at giant V) |
| kappa_V_highfix.py | §3 | observation (nonlinear enhancement; validity boundary) |
| kink_charge.py | §4 | intermediate (no collision reached) |
| kink_charge_v2.py | §4 | rejected setup (velocity sign error) |
| kink_charge_v3.py | §4 | diagnostic (periodic-seam tear on 0\|4π) |
| kink_force_static.py | §4 | intermediate (seam in kink–kink branch) |
| kink_force_static_v2.py | §4 | confirmed in-model (sign of the force, statics) |
| sigma_geometry.py | §5–6 | confirmed in-model (roughness from geometry) |
| sigma_scaling.py | §6 | confirmed in-model (transient; sub-√N averaging) |
| sigma_longtime.py | §6 | confirmed in-model (rest decays; Λ(rest)=0) |
| sigma_pumped.py | §6 | confirmed in-model (plateau; external pump — superseded) |
| sigma_internal_pump.py | §6 | confirmed in-model (traffic plateau) |
| pump_seeds.py | §6 | confirmed in-model (M^0.55, 3 seeds) |
| spin_loop_test.py | §7 | scaffolding only (not evidence) |
| spin_pair_rotation.py | §7 | rejected hypothesis (2D, documented) |
| spin_pair_3d.py | §7 | rejected hypothesis (3D, documented) |
| spin_belt_geometry.py | §7 | structural result + known mathematics (labelled) |

## License
Code MIT; text and figures CC BY 4.0. © United Field Initiative.

## Research Programme
This document is one output of a broader, staged research programme of the United Field Initiative on cumulative complexity and validation dynamics. Details: ufi.observer/research-programme.
