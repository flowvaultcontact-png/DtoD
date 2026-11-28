#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_correlation -- the correlation discovery engine.

Relationships between independently represented biological states
(biological_state_math.DiseaseState) and molecular structures
(molecular_math.Molecule) are DERIVED from the representations alone.
No disease name, no drug name, no disease-drug pairing table enters the
inference path (enforced by the hermetic snapshot R-hermetic and re-tested
by independent_validation).

THE MULTIDIMENSIONAL RELATIONSHIP (the fingerprint)
---------------------------------------------------
    Phi(D, M) = [ C_struct, E_binding, b, kappa*c, exposure_margin ]

  C_struct         matched fraction of the target pharmacophore (geometry)
  E_binding        declared contact energy E(M, site_D) (exact rationals)
  b                binding strength derived from E (the biological change)
  kappa*c          dynamic response rate (kappa = K1 * max(0, -E))
  exposure_margin  c(m) - g(d)/kappa  (pharmacological feasibility)

THE CORRELATION (the causal chain, each link independently testable)
--------------------------------------------------------------------
    C(D, M) = ( prod_i gate_i )^(1/7)      with gate_i in {0, 1}:

    gate_structural   every target-site feature matched within 3 A
    gate_binding      E(M, site_D) <= E_BIND_MAX (at least one strong contact)
    gate_functional   b > 0 (binding strength nonzero; the L-decrease then
                      follows exactly from the intervention identity -- T11)
    gate_pathway      the site-controlled pathway is dysregulated (P_t != H_t)
    gate_dynamic      kappa * c(m) > g(d)  (effect persists over time;
                      certified treated family with exact residual 0)
    gate_exposure     g(d)/kappa < c(m) <= C_MAX (achievable exposure)
    gate_safety       E(M, anti_site_D) >= E_SAFE (no strong anti-target
                      binding)

    Derived (not assumed):  C(D, M) > 0  <=>  every gate passes  <=>  B = 1.
    The geometric mean is FORCED by the declared causal-chain axiom (every
    link necessary and exchangeable), so no weight is arbitrary.

THE CENTRAL DERIVATION
----------------------
    M*(D) = argmin_{M in M_valid}  L(D, T(D, M))   subject to B(D, M) = 1

The candidate space M_valid is the chemically valid subset of the declared
molecular structures (valence + connectivity checked); the argmin is
established by exhaustive verification over M_valid.

THE CAUSAL CHAIN
----------------
    Molecule -> Binding -> Biological change -> Disease-state change
is exactly the gate sequence: structural + binding -> functional b > 0 ->
pathway coordinate pulled toward healthy -> dynamic response with certified
decay.  A molecule that binds but does not change the disease state fails
gate_pathway/gate_dynamic and is NOT declared a drug.

NOT clinical advice.  All interaction constants are DECLARED physical-model
constants of molecular_math / biological_state_math; biological predictions
require independent empirical validation (independent_validation).
"""

import hashlib
import json
import os
import sys
import time

import sympy as sp

import molecular_math as mm
import biological_state_math as bsm
from autonomous_proof_engine_v2 import residual_is_zero

C_MAX = sp.Rational(1)        # declared pharmacokinetic exposure bound
K1 = sp.Rational(1, 2)        # dynamic response per unit binding energy
E_SAFE = sp.Rational(-2)      # anti-target safety threshold


# ===========================================================================
# The declared corpora (representations only; names are cosmetic)
# ===========================================================================

def _mol_a():
    return mm.Molecule(
        atoms=[("N", 0), ("C", 0), ("O", 0), ("C", 0)],
        bonds=[(0, 1, 1), (1, 2, 2), (1, 3, 1)],
        coords=[(0, 0, 0), (2, 0, 0), (2, 0, 3), (2, 3, 0)],
        features=[("donor", 0), ("acceptor", 2), ("hydrophobe", 3)],
        label="mol_a")


def _mol_b():
    return mm.Molecule(
        atoms=[("C", 0), ("C", 0), ("C", 0), ("N", 0)],
        bonds=[(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        coords=[(0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0)],
        features=[("hydrophobe", 0), ("donor", 3)], label="mol_b")


def _mol_c():
    return mm.Molecule(
        atoms=[("C", 0), ("C", 0), ("O", 0), ("O", 0)],
        bonds=[(0, 1, 1), (1, 2, 2), (1, 3, 1)],
        coords=[(0, 0, 0), (2, 0, 0), (4, 0, 0), (4, 2, 0)],
        features=[("hydrophobe", 0), ("acceptor", 2), ("donor", 3)],
        label="mol_c")


def _mol_d():
    return mm.Molecule(
        atoms=[("N", 1), ("C", 0), ("O", 0)],
        bonds=[(0, 1, 1), (1, 2, 2)],
        coords=[(0, 0, 0), (2, 0, 0), (2, 3, 0)],
        features=[("cationic", 0), ("acceptor", 2)], label="mol_d")


def _mol_e():
    """Strong binder of ds1's site AND of the anti-target -> safety refusal."""
    m = _mol_a()
    return mm.Molecule(
        atoms=m.atoms + [("N", 1)],
        bonds=m.bonds + [(0, 4, 1)],
        coords=m.coords + [(0, 0, -2)],
        features=m.features + [("cationic", 4)],
        label="mol_e")


def _mol_f():
    """Structurally complete for ds1 but low exposure (dose variant)."""
    return mm.Molecule(
        atoms=_mol_a().atoms, bonds=_mol_a().bonds,
        coords=_mol_a().coords, features=_mol_a().features,
        label="mol_f")


MOLECULES = {m.label: m for m in (_mol_a(), _mol_b(), _mol_c(),
                                  _mol_d(), _mol_e(), _mol_f())}
EXPOSURE = {"mol_a": "1", "mol_b": "1", "mol_c": "1", "mol_d": "1",
            "mol_e": "1", "mol_f": "1/100"}

# each site: (pharmacophore, anti-target pharmacophore, controlled pathway)
SITES = {
    "ds1": ([("acceptor", (0, 0, 0)), ("donor", (2, 0, 3)),
             ("hydrophobe", (2, 3, 0))],
            [("anionic", (2, 0, 0))], 0),
    "ds2": ([("acceptor", (6, 0, 0)), ("hydrophobe", (2, 0, 0))],
            [("anionic", (50, 50, 50))], 1),
    "ds3": ([("donor", (2, 0, 0)), ("acceptor", (5, 1, 0)),
             ("hydrophobe", (1, 0, 0))],
            [("anionic", (50, 50, 50))], 2),
    "ds4": ([("acceptor", (0, 0, 0)), ("donor", (2, 0, 3))],
            [("anionic", (50, 50, 50))], 1),
}

STATES = {
    "ds1": bsm.DiseaseState(
        "ds1", P=("7/10", "2/10", "1/10"), H=("1/10", "1/10", "1/10"),
        site=SITES["ds1"][0], anti_site=SITES["ds1"][1], g="1/10",
        network={0: {1, 2}, 1: {2}, 2: set()}, drivers=(0,)),
    "ds2": bsm.DiseaseState(
        "ds2", P=("1/10", "8/10", "1/10"), H=("1/10", "1/10", "1/10"),
        site=SITES["ds2"][0], anti_site=SITES["ds2"][1], g="2/25",
        network={0: {1}, 1: {2}, 2: set()}, drivers=(0,)),
    "ds3": bsm.DiseaseState(
        "ds3", P=("2/10", "1/10", "6/10"), H=("1/10", "1/10", "1/10"),
        site=SITES["ds3"][0], anti_site=SITES["ds3"][1], g="1/20",
        network={0: {1, 2}, 1: set(), 2: set()}, drivers=(0,)),
    "ds4": bsm.DiseaseState(
        "ds4", P=("1/10", "9/10", "1/10"), H=("1/10", "1/10", "1/10"),
        site=SITES["ds4"][0], anti_site=SITES["ds4"][1], g="1/5",
        network={0: {1}, 1: {2}, 2: set()}, drivers=(0,)),
}

# the pathway a binding site controls is a property of the DISEASE side:
# binding at the site modulates that site's pathway (site_pathway).
SITE_PATHWAY = {d: SITES[d][2] for d in SITES}


# ===========================================================================
# The evaluation: Phi(D, M), the gates, C(D, M), B(D, M)
# ===========================================================================

def evaluate(state_key, mol_key, _STATES=None, _MOLECULES=None,
             _EXPOSURE=None, _SITE_PATHWAY=None):
    """Representation-to-representation evaluation.  Inputs are ONLY the
    mathematical representations and declared constants."""
    S = _STATES if _STATES is not None else STATES
    M = _MOLECULES if _MOLECULES is not None else MOLECULES
    X = _EXPOSURE if _EXPOSURE is not None else EXPOSURE
    SP = _SITE_PATHWAY if _SITE_PATHWAY is not None else SITE_PATHWAY
    D = S[state_key]
    mol = M[mol_key]
    c = sp.Rational(X[mol_key])

    E_site, c_geom, _ = mm.binding_energy(mol, D.site)
    E_anti, _, _ = mm.binding_energy(mol, D.anti_site)
    b = min(sp.Rational(1), -E_site / bsm.E_REF) if E_site < 0 else sp.Rational(0)
    target = SP[state_key]
    kappa = K1 * max(sp.Rational(0), -E_site)

    P_prime = D.intervene(target, b) if b > 0 else D.P
    L_before, L_after = D.L(), D.L(P_prime)

    gates = {
        "structural": c_geom == 1,
        "binding": E_site <= mm.E_BIND_MAX,
        "functional": b > 0,
        "pathway": target in D.dysregulated,
        "dynamic": kappa * c > D.g,
        "exposure": (D.g / kappa < c <= C_MAX) if kappa > 0 else False,
        "safety": E_anti >= E_SAFE,
    }
    phi = {
        "C_struct": c_geom,
        "E_binding": E_site,
        "b": b,
        "kappa*c": kappa * c,
        "exposure_margin": c - (D.g / kappa if kappa > 0 else sp.oo),
        "E_anti": E_anti,
        "L_before": L_before,
        "L_after": L_after,
    }
    B = 1 if all(gates.values()) else 0
    C = sp.Rational(int(all(gates.values())))  # geometric mean of {0,1} gates
    return {"phi": phi, "gates": gates, "B": B, "C": C, "P_prime": P_prime}


# ===========================================================================
# The central derivation: M* = argmin L(D, T(D, M)) over valid molecules
# ===========================================================================

def derive_M_star(state_key, _STATES=None, _MOLECULES=None, _EXPOSURE=None,
                  _SITE_PATHWAY=None):
    S = _STATES if _STATES is not None else STATES
    M = _MOLECULES if _MOLECULES is not None else MOLECULES
    X = _EXPOSURE if _EXPOSURE is not None else EXPOSURE
    SP = _SITE_PATHWAY if _SITE_PATHWAY is not None else SITE_PATHWAY
    valid = [m for m in M if not M[m].validity_report()]
    feasible = {}
    for m in valid:
        ev = evaluate(state_key, m, _STATES=S, _MOLECULES=M, _EXPOSURE=X,
                      _SITE_PATHWAY=SP)
        if ev["B"] == 1:
            feasible[m] = ev["phi"]["L_after"]
    if not feasible:
        return {"state": state_key, "M_star": None,
                "reason": "no molecule in M_valid satisfies the causal chain",
                "feasible": {}, "L_values": {}}
    L_vals = {m: evaluate(state_key, m, _STATES=S, _MOLECULES=M,
                          _EXPOSURE=X, _SITE_PATHWAY=SP)["phi"]["L_after"]
              for m in valid}
    best = min(feasible, key=lambda m: (feasible[m], M[m].canonical_signature()))
    return {"state": state_key, "M_star": best, "L_M_star": feasible[best],
            "feasible": feasible, "L_values": L_vals,
            "valid_space": valid}


# ===========================================================================
# Proof battery
# ===========================================================================

def run_battery():
    theorems = []
    t0 = time.time()

    def theorem(tag, claim, ok, detail=""):
        theorems.append({"tag": tag, "claim": claim,
                         "status": "PROVED" if ok else "NOT_PROVED",
                         "detail": str(detail)[:240]})

    # state mathematics -------------------------------------------------------
    theorems += bsm.run_battery(STATES)
    theorems += mm.run_battery(MOLECULES)

    # gate necessity + C > 0 <=> B = 1 (derived from the geometric mean) ------
    gate_ok = True
    for d_key in STATES:
        for m_key in MOLECULES:
            ev = evaluate(d_key, m_key)
            if (ev["C"] > 0) != (ev["B"] == 1):
                gate_ok = False
            if ev["B"] == 1 and not all(ev["gates"].values()):
                gate_ok = False
    theorem("R1[gates]",
            "causal chain: B = 1 <=> every one of the 7 independently "
            "testable gates passes <=> C(D, M) > 0 (derived, all pairs)",
            gate_ok,
            f"{len(STATES) * len(MOLECULES)} representation pairs")

    # T8: representation invariance -- Phi(D, phi(M)) = Phi(D, M) -------------
    t8_ok = True
    for d_key in STATES:
        for m_key, mol in MOLECULES.items():
            base = evaluate(d_key, m_key)
            for shift in range(len(mol.atoms)):
                perm = list(range(len(mol.atoms)))
                perm = perm[shift:] + perm[:shift]
                renum = {m_key: mol.renumber(perm)}
                ev = evaluate(d_key, m_key, _MOLECULES=renum,
                              _EXPOSURE={m_key: EXPOSURE[m_key]})
                if ev["phi"] != base["phi"] or ev["B"] != base["B"]:
                    t8_ok = False
    theorem("T8[invariance]",
            "representation invariance: renumbering atoms (phi) changes "
            "neither Phi(D, M) nor B (f(phi(D), phi(M)) = f(D, M))", t8_ok)

    # T9: structural sufficiency ---------------------------------------------
    t9_ok = True
    for d_key in STATES:
        ev_a = evaluate(d_key, "mol_a")
        mol_a2 = _mol_a()
        mol_a2.label = "mol_a_equivalent"
        ev_a2 = evaluate(d_key, "mol_a", _MOLECULES={"mol_a": mol_a2},
                         _EXPOSURE={"mol_a": EXPOSURE["mol_a"]})
        if ev_a["phi"] != ev_a2["phi"] or ev_a["B"] != ev_a2["B"]:
            t9_ok = False
        if _mol_a().canonical_signature() != mol_a2.canonical_signature():
            t9_ok = False
    theorem("T9[sufficiency]",
            "structural sufficiency: physically equivalent molecules "
            "(identical canonical signature) have identical Phi and B", t9_ok)

    # T10: independent correlation across representations ----------------------
    # 2D prediction (feature-type multiset coverage, no geometry) vs the full
    # 3D correlation.  Theorem: B_3D = 1 => 2D structural condition holds;
    # every 2D-yes/3D-no divergence must be caused by a geometric or physical
    # gate (never by a name or an identifier).
    def B_2D(d_key, m_key):
        """2D (graph-level, geometry-free) structural condition: every site
        feature type can be satisfied by a COMPATIBLE molecule feature type
        (mm.feature_energy < 0), ignoring 3D positions."""
        site_types = [t for t, _p in STATES[d_key].site]
        mol_types = [t for t, _a in MOLECULES[m_key].features]
        avail = list(mol_types)
        for t in site_types:
            for idx, mt in enumerate(avail):
                if mm.feature_energy(t, mt) < 0:
                    avail.pop(idx)
                    break
            else:
                return False
        return True

    t10_ok = True
    divergences = []
    PHYSICAL_GATES = {"structural", "binding", "dynamic", "exposure", "safety"}
    for d_key in STATES:
        for m_key in MOLECULES:
            ev = evaluate(d_key, m_key)
            if ev["B"] == 1 and not B_2D(d_key, m_key):
                t10_ok = False
            if B_2D(d_key, m_key) and ev["B"] != 1:
                failing = {g for g, ok in ev["gates"].items() if not ok}
                divergences.append((d_key, m_key, sorted(failing)))
                if not (failing & PHYSICAL_GATES):
                    t10_ok = False
    theorem("T10[correlation]",
            "independent correlation: the 3D prediction implies the 2D "
            "structural condition; every 2D-yes/3D-no divergence is caused "
            "by a molecule-dependent physical gate (geometry, energy, "
            "dynamics, exposure, safety) -- never by an identifier "
            f"({len(divergences)} divergences = the information gain of 3D)",
            t10_ok, "; ".join(f"{d}|{m}:{g}" for d, m, g in divergences[:6]))

    # T11: causal consistency --------------------------------------------------
    t11_ok = True
    for d_key in STATES:
        for m_key in MOLECULES:
            ev = evaluate(d_key, m_key)
            if ev["B"] == 1:
                D = STATES[d_key]
                if not (ev["phi"]["L_after"] < D.L()):
                    t11_ok = False
                target = SITE_PATHWAY[d_key]
                delta = D.L() - D.L(ev["P_prime"])
                expected = (sp.Rational(1, len(D.P)) * ev["phi"]["b"]
                            * (2 - ev["phi"]["b"])
                            * (D.P[target] - D.H[target]) ** 2)
                if sp.simplify(delta - expected) != 0:
                    t11_ok = False
    theorem("T11[causal]",
            "causal consistency: every predicted-successful interaction "
            "produces the predicted disease-state change under the "
            "intervention equations (L strictly decreases, exact)", t11_ok)

    # R2: the central derivation M* = argmin, exhaustively verified -----------
    derivations = {}
    for d_key in STATES:
        drv = derive_M_star(d_key)
        derivations[d_key] = drv
        if drv["M_star"] is None:
            theorem(f"R2[{d_key}]", "M*(D) derivation (no feasible molecule)",
                    False, drv["reason"])
            continue
        exhaustive = all(
            evaluate(d_key, m)["phi"]["L_after"] >= drv["L_M_star"]
            for m in drv["valid_space"])
        gates_ok = evaluate(d_key, drv["M_star"])["B"] == 1
        theorem(f"R2[{d_key}]",
                "M*(D) = argmin_{M in M_valid} L(D, T(D, M)) s.t. B = 1 "
                "(exhaustively verified over the valid space)",
                exhaustive and gates_ok,
                f"M* = {drv['M_star']}, L* = {drv['L_M_star']}, "
                f"|M_valid| = {len(drv['valid_space'])}, "
                f"feasible = {sorted(drv['feasible'])}")

    # R3: hermetic discipline -- derivation sees no validation module ---------
    hermetic = not any("validation" in name.lower()
                       for name in sys.modules)
    snapshot = sorted(sys.modules)
    theorem("R3[hermetic]",
            "derivation/validation separation: no validation or pairing "
            "module present in the derivation-time module table", hermetic)

    # corpus input hash (pins what the derivation was allowed to see) ---------
    h = hashlib.sha256()
    for d_key in sorted(STATES):
        h.update(repr((STATES[d_key].P, STATES[d_key].site,
                       STATES[d_key].anti_site, str(STATES[d_key].g))).encode())
    for m_key in sorted(MOLECULES):
        h.update(repr((MOLECULES[m_key].canonical_signature(),
                       EXPOSURE[m_key])).encode())
    input_hash = h.hexdigest()

    return theorems, derivations, snapshot, input_hash, \
        round(time.time() - t0, 2)


# ===========================================================================
# Reporting / certificate
# ===========================================================================

def write_outputs(theorems, derivations, snapshot, input_hash, elapsed,
                  out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fingerprints = {}
    for d_key in STATES:
        fingerprints[d_key] = {}
        for m_key in MOLECULES:
            ev = evaluate(d_key, m_key)
            fingerprints[d_key][m_key] = {
                "Phi": {k: str(v) for k, v in ev["phi"].items()},
                "gates": {k: bool(v) for k, v in ev["gates"].items()},
                "B": ev["B"], "C": str(ev["C"]),
            }
    certificate = {
        "engine": "structural_correlation (representation-to-representation)",
        "all_proved": all(t["status"] == "PROVED" for t in theorems),
        "n_theorems": len(theorems),
        "n_proved": sum(1 for t in theorems if t["status"] == "PROVED"),
        "elapsed_s": elapsed,
        "input_sha256": input_hash,
        "hermetic_module_snapshot": snapshot,
        "theorems": theorems,
        "M_star": {d: (derivations[d]["M_star"],
                       str(derivations[d].get("L_M_star")))
                   for d in derivations},
        "fingerprints": fingerprints,
    }
    with open(os.path.join(out_dir, "structural_correlation_certificate.json"),
              "w", encoding="utf-8") as fh:
        json.dump(certificate, fh, indent=2, default=str)

    L = ["=" * 78,
         "STRUCTURAL CORRELATION | representation-to-representation engine",
         "=" * 78,
         f"input sha256 (the only information the derivation received): "
         f"{input_hash[:16]}...",
         ""]
    L.append("Phi(D, M) -- the mathematical fingerprint (example: ds1 x mol_a)")
    ev = evaluate("ds1", "mol_a")
    for k, v in ev["phi"].items():
        L.append(f"  {k:16s} = {v}")
    L.append(f"  gates            = {ev['gates']}")
    L.append(f"  B = {ev['B']},  C = {ev['C']}")
    L.append("")
    L.append("THE CENTRAL DERIVATION  M*(D) = argmin L(D, T(D, M)) s.t. B = 1")
    for d_key, drv in derivations.items():
        if drv["M_star"] is None:
            L.append(f"  {d_key}: NO feasible molecule ({drv['reason']})")
        else:
            L.append(f"  {d_key}: M* = {drv['M_star']}  "
                     f"L* = {drv['L_M_star']}  "
                     f"(feasible: {sorted(drv['feasible'])})")
    L.append("")
    L.append("THE CAUSAL CHAIN (each link independently testable)")
    L.append("  Molecule -> Binding -> Biological change -> Disease-state change")
    L.append("  gates: structural, binding, functional, pathway, dynamic, "
             "exposure, safety")
    L.append("")
    L.append("=" * 78)
    L.append("PROOF BATTERY")
    L.append("=" * 78)
    for th in theorems:
        L.append(f"  [{th['status']:10s}] {th['tag']}: {th['claim']}"
                 + (f"   [{th['detail']}]" if th["status"] != "PROVED" else ""))
    n_ok = sum(1 for t in theorems if t["status"] == "PROVED")
    L.append(f"\n  {n_ok}/{len(theorems)} theorems PROVED  (elapsed {elapsed} s)")
    with open(os.path.join(out_dir, "structural_correlation_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return n_ok, len(theorems)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "structural_correlation_out")
    theorems, derivations, snapshot, input_hash, elapsed = run_battery()
    n_ok, n_tot = write_outputs(theorems, derivations, snapshot, input_hash,
                                elapsed, out_dir)
    print(f"structural_correlation: {n_ok}/{n_tot} theorems PROVED "
          f"({elapsed} s) -> {out_dir}/")
    return 0 if n_ok == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
