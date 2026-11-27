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
