# DtoD
finding drug AI

## Autonomous proof engine v2 — disease & drug mathematical representations

`autonomous_proof_engine_v2.py` is an autonomous formula-discovery and proof
engine (strict non-settlement: it certifies only gap-free, machine-verified
derivations). A disease/drug layer declares diseases and drugs purely as
constraints — progression dynamics, organ capacity, infusion and elimination,
exposure — and the engine autonomously derives their mathematical
representations. No solution form is ever wired in.

Run it:

```bash
pip install sympy
python3 autonomous_proof_engine_v2.py
```

Certified representations (from the latest run, exit 0, 52/52 checks):

| Problem | Declared constraints | Certified representation |
| --- | --- | --- |
| `disease_growth` | `D' = k D`, `D(0) = D0` | `D(t) = D0 · e^{k t}` (proven family) |
| `disease_logistic` | `D' = k D (1 - D/K)`, `D(0) = D0` | `D(t) = D0 K e^{kt} / (D0 e^{kt} - D0 + K)`, saturates at `K` (proven family) |
| `drug_pk` | `C' = R - k C`, `C(0) = 0` | `C(t) = (R/k)(1 - e^{-kt})`, plateau `R/k` (PROVEN_EXACT) |
| `drug_disease` | `D' = (g - k c) D`, `D(0) = D0` | `D(t) = D0 · e^{(g - k c) t}` (proven family) |
| `drug_disease_pinned` | same, `D(0) = 1` | `D(t) = e^{(g - k c) t}` (PROVEN_EXACT) |

The cure law falls out of the certified interaction representation: the
disease load decays iff the drug exposure beats endogenous progression,
`k · c > g`; below that threshold the certified solution grows and the engine
still refuses to certify a cure. Models that the engine cannot settle
(Riemann, Collatz, Navier–Stokes) remain honest refusals with the missing
machinery listed.

Artifacts: `proof_run_report.txt` (full derivation traces, candidates,
eliminated branches, gaps) and `certificates/<problem>.json` (machine
checkable certificates with proof obligations for a formal kernel).
NOT clinical advice.

## Disease → mathematics: full formal representations (`disease_math.py`)

`disease_math.py` turns a disease itself into a complete mathematical object,
built and certified with the engine — designed as the **input of a
mathematical formula**:

```text
R(d) = (chi_D, g(d), Law_d, In_d)          representation of a DISEASE
  chi_D  in GF(2)^23   feature vector (6 etiology / 10 pathway / 7 organ axes)
  g(d)   in Q_>0       endogenous progression rate
  Law_d  engine-certified untreated law  D' = g(d) D,  D(0) = D0
         with certified family  D(t) = D0 · e^{g(d) t}
  In_d   exact formula-input bundle (chi_D, g, law, family)

R(m) = (chi_m, k(m), c(m))                 representation of a DRUG
  chi_m  in GF(2)^23   mechanism vector; k(m) kill rate; c(m) exposure
```

These objects are the only inputs of the formulas:

```text
coverage(d,m) = |chi_D^PW ∧ chi_m^PW| / |chi_D^PW|
k_eff(d,m)    = k(m) · c(m) · coverage(d,m)
cure law      B(d,m) = 1  <=>  coverage > 0  AND  k_eff(d,m) > g(d)
treated law   D' = (g(d) − k_eff(d,m)) · D,   rho(d,m,t) = D0 · e^{(g − k_eff)t}
```

Example (exact arithmetic): hypertension `g = 1/10` × ramipril
`k_eff = 9/10` → `B = 1`, treated load `D0·e^{−4t/5}` (strict decay);
ramipril at 1/20 exposure → `k_eff = 9/200 ≤ g` → `B = 0`, cure refused.

Run it (exit 0 ⇔ every theorem proved):

```bash
python3 disease_math.py
```

### The formulas (shared)

All formulas consume only the representations `R(d)`, `R(m)` — nothing else:

| Formula | Definition |
| --- | --- |
| coverage | `coverage(d,m) = \|χ_D^PW ∧ χ_m^PW\| / \|χ_D^PW\|` |
| effective kill | `k_eff(d,m) = k(m) · c(m) · coverage(d,m)` |
| cure law | `B(d,m) = 1 ⟺ coverage(d,m) > 0 AND k_eff(d,m) > g(d)` |
| treated law | `D'(t) = (g(d) − k_eff(d,m)) · D(t)` |
| treated family | `ρ(d,m,t) = D0 · e^{(g(d) − k_eff(d,m))·t}` |

### What the input puts on the drug (derived, always true)

The engine derives from the disease input `R(d)` alone, by complete branch
enumeration over `s = |χ_D^PW ∧ χ_m^PW|` with every step machine-checked
exactly, what ANY drug must satisfy:

```text
Drug(d) = { m : s(d,m) > 0  AND  k(m)·c(m) > g(d)·n/s(d,m) }
                     (equivalently c(m) > c*(d,m) = g(d)·n / (s(d,m)·k(m)))
```

Universal theorems (hold for ALL inputs, not just corpus instances):
- **M7** `B(d,m) = 1 ⟺ s(d,m) > 0 AND c(m) > c*(d,m)` — the `s = 0` branch is
  eliminated for every `k > 0, c ≥ 0` (no pathway coverage → no cure, ever)
- **M8** single-crossing derivation: `dk_eff/dc = k·s/n > 0`, unique crossing
  `c* = g·n/(s·k)` (exact residual 0), `f(0) = −g < 0` ⇒ `B = 1 ⟺ c > c*`
- **M9** retrieval: the derived requirement reproduces `Drug(d)` for every
  disease, always contains the declared therapy, and refuses every decoy —
  verified over all 120 corpus pairs

Emergent result: for `BREAST_CANCER` (pathway axes `{PW_TUBULIN, PW_CELL_CYCLE}`)
the derived `Drug(d) = {paclitaxel, imatinib}` — the formula itself surfaces
imatinib as a candidate through the cell-cycle axis.

Proof battery — 69/69 theorems PROVED:
- **M1** each of 10 diseases' untreated laws engine-certified (exact residual 0)
- **M2** encoding injectivity (diseases, therapeutic drugs, `(χ_m, c)` pairs)
- **M3/M6** every declared therapy satisfies the cure law and strictly decays
- **M4** every decoy (wrong pathway / insufficient exposure) is refused with its recorded reason
- **M5** treated family solves the treated ODE exactly
- **M7–M9** the derived drug requirement: universal cure-law biconditional,
  single-crossing derivation, corpus-wide retrieval

Artifacts: `disease_math_out/representations.json` (all representations in
pluggable form), `disease_math_out/disease_math_report.txt`,
`disease_math_out/disease_math_certificate.json`. NOT clinical advice.

## Structure → structure: disease → drug by direct formula (`structure_math.py`)

`structure_math.py` removes the last data dependence: no real names, no data
matching. The chemical structure of a molecule IS its identity, represented
as mathematics:

```text
R(d) = (T(d), g(d), Law_d)        DISEASE = its target structure
  T(d) ⊆ GF(2)^12 structural fragments (amine, sulfonamide, aromatic ring, …)
  g(d) progression rate; Law_d engine-certified

R(m) = (χ_m, c(m))                DRUG = its molecular structure
  χ_m ⊆ GF(2)^12 the molecule's fragments; c(m) its exposure
```

The direct formula — disease structure → drug structure, pure algebra, no
lookup table, no corpus scan:

```text
rho(d,m)  = |T(d) ∧ χ_m| / |T(d)|                     structural affinity
B(d,m) = 1 ⟺ rho(d,m) = 1 AND K0·c(m) > g(d)         cure law
Drug(d) = { m : T(d) ⊆ χ_m  AND  c(m) > g(d)/K0 }     THE DIRECT FORMULA
m*(d)  = T(d)  at c ≥ 2·g(d)/K0                       direct construction
```

The only declared axiom is binding completeness (a cure requires full
structural coverage of the target site); everything else is derived:
- **S2 (universal)** `rho = 1 ⟺ T(d) ⊆ χ_m` for all structure pairs
- **S3 (universal)** the full cure-law biconditional, branch enumeration over
  `rho ∈ {<1, =1}`
- **S4** `m*(d) = T(d)` is a fixed point of the cure law and minimal (every
  `B = 1` molecule contains `T(d)`)
- **S5 (name-freeness)** the whole derivation is invariant under a label
  permutation — `Drug(d)`, thresholds, and `B` are keyed by structure address
  and reproduce bit-for-bit; an AST scan confirms the derivation path contains
  no label-keyed constants
- **S6/S7** treated family exact residual 0, strict decay; per-disease direct
  derivation trace

Proof battery — **57/57 theorems PROVED**. Run:

```bash
python3 structure_math.py
```

Artifacts: `structure_math_out/representations.json`,
`structure_math_out/structure_math_report.txt`,
`structure_math_out/structure_math_certificate.json`. NOT clinical advice.
