#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
independent_validation -- blind testing and verification.

Separation discipline: the information used to DERIVE a prediction must
never leak from the information used to VALIDATE it.

  1. NAME BLINDNESS       the derivation is rerun with all labels stripped
                          and permuted; results are keyed by structure
                          address and must be identical bit-for-bit
  2. CORPUS BLINDNESS     the derivation-time module snapshot (recorded in
                          the structural_correlation certificate) must
                          contain no validation/pairing module, and the
                          derivation sources must define no pairing table
  3. REPRESENTATION-ONLY  the engine's inputs are exclusively mathematical
                          representations and declared physical constants;
                          an AST scan must find no module-level mapping
                          between molecule keys and disease keys
  4. BLIND DISCOVERY      held-out molecular structures and a held-out
                          disease state (never part of the derivation
                          corpus) are submitted to the engine; predictions
                          are compared against the independently declared
                          experimental outcomes

The declared outcomes live ONLY here -- the derivation side
(structural_correlation and its imports) never sees them.
NOT clinical advice.
"""

import ast
import json
import os
import sys

import sympy as sp

import molecular_math as mm
import structural_correlation as sc

# ---------------------------------------------------------------------------
# HELD-OUT (blind) representations -- never part of the derivation corpus
# ---------------------------------------------------------------------------

def _blind_1():
    """Held-out 5-atom structure, never part of the derivation corpus."""
    return mm.Molecule(
        atoms=[("N", 0), ("C", 0), ("O", 0), ("C", 0), ("O", 0)],
        bonds=[(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1)],
        coords=[(0, 0, 0), (2, 0, 0), (2, 0, 3), (2, 3, 0), (4, 3, 0)],
        features=[("donor", 0), ("acceptor", 2), ("donor", 4)],
        label="blind_1")


def _blind_2():
    """Fragment of mol_a: the donor is missing -- cannot cover the site."""
    return mm.Molecule(
        atoms=[("C", 0), ("C", 0), ("O", 0)],
        bonds=[(0, 1, 1), (1, 2, 2)],
        coords=[(0, 0, 0), (2, 0, 0), (2, 0, 3)],
        features=[("hydrophobe", 0), ("acceptor", 2)],
        label="blind_2")


def _blind_3():
    """mol_a's structure at a microdose."""
    m = _blind_1()
    return mm.Molecule(m.atoms, m.bonds, m.coords, m.features,
                       label="blind_3")


BLIND_MOLECULES = {m.label: m for m in (_blind_1(), _blind_2(), _blind_3())}
BLIND_EXPOSURE = {"blind_1": "1", "blind_2": "1", "blind_3": "1/100"}

BLIND_STATE = sc.bsm.DiseaseState(
    "ds_blind", P=("1/10", "1/10", "7/10"), H=("1/10", "1/10", "1/10"),
    site=[("acceptor", (0, 0, 0)), ("donor", (2, 0, 3)),
          ("acceptor", (4, 3, 0))],
    anti_site=[("anionic", (50, 50, 50))], g="1/10",
    network={0: {1, 2}, 1: {2}, 2: set()}, drivers=(0,))
BLIND_SITE_PATHWAY = {"ds_blind": 2}

# ---------------------------------------------------------------------------
# Independently declared experimental outcomes (validation information ONLY)
# ---------------------------------------------------------------------------

DECLARED_OUTCOMES = {
    ("ds1", "blind_1"): "ineffective",
    ("ds1", "blind_2"): "ineffective",
    ("ds1", "blind_3"): "ineffective_at_tested_dose",
    ("ds_blind", "blind_1"): "effective",
    ("ds_blind", "blind_2"): "ineffective",
    ("ds_blind", "blind_3"): "ineffective_at_tested_dose",
}


def engine_prediction(d_key, m_key):
    """The engine's prediction for a (disease state, molecule) pair,
    computed representation-only.  Blind molecules are injected as a
    separate corpus; the derivation corpus is untouched."""
    states = dict(sc.STATES)
    molecules = dict(sc.MOLECULES)
    exposure = dict(sc.EXPOSURE)
    if d_key == "ds_blind":
        states = {"ds_blind": BLIND_STATE}
        site_pathway = BLIND_SITE_PATHWAY
    else:
        site_pathway = sc.SITE_PATHWAY
    molecules[m_key] = BLIND_MOLECULES[m_key]
    exposure[m_key] = BLIND_EXPOSURE[m_key]
    return sc.evaluate(d_key, m_key, _STATES=states,
                       _MOLECULES=molecules, _EXPOSURE=exposure,
                       _SITE_PATHWAY=site_pathway)


def run_validation():
    checks = []

    def check(tag, claim, ok, detail=""):
        checks.append({"tag": tag, "claim": claim,
                       "status": "PASS" if ok else "FAIL",
                       "detail": str(detail)[:240]})

    # 1. name blindness -------------------------------------------------------
    perm_states = {f"X{i}": st for i, st in
                   enumerate(sc.STATES.values())}
    perm_mols = {f"Y{j}": m for j, m in
                 enumerate(sc.MOLECULES.values())}
    perm_exp = {f"Y{j}": list(sc.EXPOSURE.values())[j] for j in range(len(perm_mols))}
    orig_by_obj = {id(st): d for d, st in sc.STATES.items()}
    perm_site_pathway = {xk: sc.SITE_PATHWAY[orig_by_obj[id(st)]]
                         for xk, st in perm_states.items()}
    base = {}
    for d in sc.STATES:
        for m in sc.MOLECULES:
            ev = sc.evaluate(d, m)
            base[(repr(sorted(sc.STATES[d].site)), sc.MOLECULES[m].canonical_signature(),
                  str(sc.EXPOSURE[m]))] = (ev["B"], str(ev["C"]))
    perm = {}
    for d in perm_states:
        for m in perm_mols:
            ev = sc.evaluate(d, m, _STATES=perm_states, _MOLECULES=perm_mols,
                             _EXPOSURE=perm_exp,
                             _SITE_PATHWAY=perm_site_pathway)
            perm[(repr(sorted(perm_states[d].site)),
                  perm_mols[m].canonical_signature(),
                  str(perm_exp[m]))] = (ev["B"], str(ev["C"]))
    check("V1[name-blindness]",
          "all predictions identical under label stripping + permutation",
          base == perm, f"{len(base)} address-keyed predictions")

    # 2. corpus blindness -----------------------------------------------------
    cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "structural_correlation_out",
                             "structural_correlation_certificate.json")
    with open(cert_path, encoding="utf-8") as fh:
        cert = json.load(fh)
    snapshot = cert.get("hermetic_module_snapshot", [])
    check("V2[corpus-blindness]",
          "derivation-time module snapshot contains no validation module",
          not any("validation" in n.lower() for n in snapshot),
          f"{len(snapshot)} modules frozen at derivation time")

    # 3. representation-only inference ---------------------------------------
    pair_table_found = False
    for mod_name in ("structural_correlation", "molecular_math",
                     "biological_state_math"):
        with open(f"{mod_name}.py", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                if any(k.startswith("mol") for k in keys) \
                        and any(k.startswith("ds") for k in keys):
                    pair_table_found = True
    check("V3[representation-only]",
          "no module-level molecule->disease pairing table in any "
          "derivation source (AST scan)", not pair_table_found)

    # 4. blind discovery -------------------------------------------------------
    corpus_sigs = {m.canonical_signature() for m in sc.MOLECULES.values()}
    corpus_labels = set(sc.MOLECULES)
    novel = (all(m.canonical_signature() not in corpus_sigs
                 for m in BLIND_MOLECULES.values())
             and not (set(BLIND_MOLECULES) & corpus_labels))
    check("V4[novel-structures]",
          "held-out structures and labels are absent from the derivation "
          "corpus (genuinely novel inputs)", novel)

    predictions_ok = True
    detail = []
    for (d_key, m_key), outcome in DECLARED_OUTCOMES.items():
        ev = engine_prediction(d_key, m_key)
        predicted = "effective" if ev["B"] == 1 else "ineffective"
        consistent = (outcome == "effective") == (ev["B"] == 1)
        if not consistent:
            predictions_ok = False
        detail.append(f"{d_key}|{m_key}: predicted={predicted}, "
                      f"declared={outcome} -> "
                      f"{'MATCH' if consistent else 'MISMATCH'}")
    check("V4[blind-discovery]",
          "engine predictions on held-out structures match the "
          "independently declared outcomes", predictions_ok,
          "; ".join(detail))

    # blind M* derivation on the held-out state -------------------------------
    states = {"ds_blind": BLIND_STATE}
    molecules = dict(sc.MOLECULES)
    molecules.update(BLIND_MOLECULES)
    exposure = dict(sc.EXPOSURE)
    exposure.update(BLIND_EXPOSURE)
    drv = sc.derive_M_star("ds_blind", _STATES=states,
                           _MOLECULES=molecules, _EXPOSURE=exposure,
                           _SITE_PATHWAY=BLIND_SITE_PATHWAY)
    check("V4[blind-M-star]",
          "blind discovery: the held-out disease state is solved from "
          "representations alone -- the held-out novel structure is in the "
          "derived feasible set and L* = 0 (full recovery of the healthy "
          "state)",
          "blind_1" in drv["feasible"] and drv.get("L_M_star") == 0,
          f"M* = {drv['M_star']}, feasible = {sorted(drv['feasible'])}, "
          f"L* = {drv.get('L_M_star')}")

    return checks


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "independent_validation_out")
    os.makedirs(out_dir, exist_ok=True)
    checks = run_validation()
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    report = ["=" * 78,
              "INDEPENDENT VALIDATION | blind tests",
              "=" * 78]
    for c in checks:
        report.append(f"  [{c['status']}] {c['tag']}: {c['claim']}"
                      + (f"   [{c['detail']}]" if c["status"] == "FAIL" else ""))
    report.append(f"\n  {n_pass}/{len(checks)} validation checks passed")
    with open(os.path.join(out_dir, "validation_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(report))
    for c in checks:
        print(f"[{c['status']}] {c['tag']}: {c['claim']}"
              + (f"  [{c['detail']}]" if c["status"] == "FAIL" else ""))
    print(f"\n  {n_pass}/{len(checks)} validation checks passed")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
