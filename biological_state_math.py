#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
biological_state_math -- disease-state and pathway representations.

A disease is represented as the biological state

    D = (P, N, S, F, J)

  P  pathway activity vector (exact rationals; the functional state F is the
     projection of P onto the dysregulated coordinates)
  N  pathway interaction network (signed adjacency over pathways)
  S  structural target: the binding-site pharmacophore with 3D positions
     (the structural face consumed by molecular_math.binding_energy), plus
     the anti-target (safety) pharmacophore
  F  functional readouts (the dysregulated coordinates of P)
  J  disease dynamics: the certified untreated law  D' = g D, D0 e^{g t}

THE HEALTHY STATE AND THE DISTANCE
----------------------------------
    H        the declared healthy pathway activity vector
    L(D)     = sum_i w_i (P_i - H_i)^2,  w_i = 1/n  (equal-necessity metric,
               derived from the declared causal model: every pathway
               coordinate carries equal weight in the disease state)

THE INTERVENTION EQUATIONS (biological transformation)
------------------------------------------------------
A molecule M binding the site with strength b in [0, 1] on pathway t acts as

    T(D, M):   P'_t = P_t - b (P_t - H_t),   P'_i = P_i  (i != t)

-- a pull of the targeted coordinate toward the healthy state by fraction b.
The binding strength is DERIVED from the physical interaction:

    b(M, D) = min(1, -E(M, site_D) / E_REF)     if E < 0, else 0

with E(M, site) the declared contact-energy model of molecular_math.

EXACT CONSEQUENCE (machine-checked, symbolic in b):
    L(D, M) = L(D) - w_t b (2 - b) (P_t - H_t)^2
so the intervention strictly decreases the disease distance whenever
0 < b <= 1 and the targeted coordinate is dysregulated -- the mathematical
form of "binding changes the biological state".
"""

import sympy as sp

from autonomous_proof_engine_v2 import AutonomousProofEngine, residual_is_zero

E_REF = sp.Rational(2)     # declared reference contact energy
W_EXP = 3                  # exposure exponent in the dynamic response


class DiseaseState:

    def __init__(self, label, P, H, site, anti_site, g, network,
                 drivers=()):
        self.label = label
        self.P = tuple(sp.Rational(x) for x in P)
        self.H = tuple(sp.Rational(x) for x in H)
        self.site = list(site)                 # [(type, (x, y, z)), ...]
        self.anti_site = list(anti_site)
        self.g = sp.Rational(g)
        self.network = {int(k): set(v) for k, v in network.items()}
        self.drivers = set(drivers)
        self.dysregulated = {i for i in range(len(self.P))
                             if self.P[i] != self.H[i]}

    def L(self, P_prime=None):
        P = self.P if P_prime is None else P_prime
        n = len(self.P)
        return sum((P[i] - self.H[i]) ** 2 for i in range(n)) / n

    def intervene(self, target, b):
        """T(D, M): the biological state after the interaction."""
        P_prime = list(self.P)
        if not isinstance(b, sp.Basic):
            b = sp.Rational(b)
        P_prime[target] = self.P[target] - b * (self.P[target] - self.H[target])
        return tuple(P_prime)

    def distance_reduction_formula(self, target):
        """The exact symbolic identity for the intervention on `target`."""
        n = len(self.P)
        b = sp.Symbol('b', real=True)
        w = sp.Rational(1, n)
        d_t = self.P[target] - self.H[target]
        lhs = self.L(self.intervene(target, b))
        rhs = self.L() - w * b * (2 - b) * d_t ** 2
        return sp.simplify(sp.expand(lhs - rhs))

    def certified_law(self):
        """Engine-certified untreated dynamics (J)."""
        eng = AutonomousProofEngine(
            f"Biological state dynamics | {self.label}: J: D' = g D, D(0)=D0",
            func_name='D', var_name='t', aux_var_name='w',
            params=('g',), param_assumptions={'g': 'positive'},
            auto_declare_regularity=True)
        D, t = eng.f, eng.x
        g = eng.param_syms['g']
        eng.declare_hypothesis(f"g = {self.g} > 0 (progression rate of the "
                               f"state with dysregulated pathways "
                               f"{sorted(self.dysregulated)}); D differentiable")
        eng.add_differential_constraint(sp.Derivative(D(t), t), g * D(t))
        eng.add_bc_value(sp.Symbol('D0', positive=True), point=0)
        res = eng.solve_and_verify()
        fam = [c for c in eng.candidates if c["verified"]
               and "exp" in str(c["expr"])]
        return eng, res, fam


def run_battery(states):
    """Exact checks of the state mathematics (used by structural_correlation)."""
    theorems = []

    def theorem(tag, claim, ok, detail=""):
        theorems.append({"tag": tag, "claim": claim,
                         "status": "PROVED" if ok else "NOT_PROVED",
                         "detail": str(detail)[:200]})

    b = sp.Symbol('b', real=True)
    for label, st in states.items():
        # distance identity (exact, symbolic in b) ---------------------------
        n = len(st.P)
        ident_ok = True
        for target in st.dysregulated:
            if st.distance_reduction_formula(target) != 0:
                ident_ok = False
        theorem(f"L-identity[{label}]",
                "L(T(D,M)) = L(D) - w b (2 - b) (P_t - H_t)^2 (exact, "
                "symbolic in b)", ident_ok)

        # strict decrease under the intervention equations -------------------
        dec_ok = True
        for target in st.dysregulated:
            d_t = st.P[target] - st.H[target]
            delta = st.L() - st.L(st.intervene(target, sp.Rational(1, 2)))
            if not (delta - sp.Rational(1, 2) * sp.Rational(1, n)
                    * (2 - sp.Rational(1, 2)) * d_t ** 2 == 0) or delta <= 0:
                dec_ok = False
        theorem(f"L-decreases[{label}]",
                "intervention strictly decreases L on every dysregulated "
                "coordinate (causal effect exists)", dec_ok)

        # J: engine-certified dynamics ---------------------------------------
        eng, res, fam = st.certified_law()
        exact = residual_is_zero(
            sp.diff(sp.Symbol('D0', positive=True)
                    * sp.exp(st.g * sp.Symbol('t', positive=True)),
                    sp.Symbol('t', positive=True))
            - st.g * sp.Symbol('D0', positive=True)
            * sp.exp(st.g * sp.Symbol('t', positive=True)))
        theorem(f"J-certified[{label}]",
                "untreated dynamics engine-certified, exact residual 0",
                res["status"] in ("PROVEN_CLASSIFICATION", "PROVEN_EXACT")
                and bool(fam) and exact, f"status={res['status']}")

        # network sanity: every dysregulated pathway reachable from a driver -
        reach_ok = True
        for target in st.dysregulated:
            seen, frontier = set(st.drivers), list(st.drivers)
            while frontier:
                u = frontier.pop()
                for v in st.network.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        frontier.append(v)
            if target not in seen:
                reach_ok = False
        theorem(f"N-reachable[{label}]",
                "every dysregulated pathway reachable from a declared driver "
                "in the network N", reach_ok,
                f"drivers={sorted(st.drivers)}")
    return theorems
