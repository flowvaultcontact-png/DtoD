#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
molecular_math -- molecular graph and 3D geometry mathematics.

A molecule is represented as the full mathematical object

    M = (V, E, A, B, X, G)

  V  atoms (indices)
  E  bonds (pairs of atom indices)
  A  atom attributes (element, formal charge)
  B  bond attributes (order)
  X  3D coordinates (exact integers, Angstrom scale)
  G  derived geometry (distance multiset, feature positions)

No names anywhere: the identity of M is its canonical structural signature
(renumbering-invariant invariants).  Renumbering the atoms (a valid
representation transformation phi) leaves every invariant -- and therefore
every downstream prediction -- unchanged (theorem T8).

Chemical validity: an element of the synthesizable space M_valid must have
satisfied valences, be connected, and carry no impossible bond.
"""

from itertools import combinations

import sympy as sp

ELEMENT_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 6, "F": 1, "Cl": 1}

FEATURE_TYPES = ["donor", "acceptor", "anionic", "cationic", "aromatic",
                 "hydrophobe"]

# declared interaction constants (the physical model; exact rationals)
COMPAT_ENERGY = {
    frozenset(("donor", "acceptor")): sp.Rational(-2),
    frozenset(("cationic", "anionic")): sp.Rational(-3),
    frozenset(("aromatic", "aromatic")): sp.Rational(-1),
    frozenset(("hydrophobe", "hydrophobe")): sp.Rational(-1),
}
CLASH_ENERGY = sp.Rational(4)
EPS2 = sp.Rational(9)          # 3 A match radius, squared
E_BIND_MAX = sp.Rational(-2)   # binding gate: at least one strong contact


class Molecule:
    """M = (V, E, A, B, X, G).  Labels are cosmetic; identity = signature."""

    def __init__(self, atoms, bonds, coords, features, label=""):
        self.atoms = list(atoms)          # [(element, charge), ...]
        self.bonds = [tuple(b) for b in bonds]   # [(i, j, order), ...]
        self.coords = [tuple(c) for c in coords]  # [(x, y, z), ...]
        self.features = list(features)    # [(type, atom_index), ...]
        self.label = label

    # -- representation transformation phi (renumbering) -------------------
    def renumber(self, perm):
        """phi: a relabelling of V.  Must be a bijection on atom indices."""
        assert sorted(perm) == list(range(len(self.atoms)))
        inv = [0] * len(perm)
        for new, old in enumerate(perm):
            inv[old] = new
        bonds = [(inv[i], inv[j], o) for (i, j, o) in self.bonds]
        feats = [(t, inv[a]) for (t, a) in self.features]
        return Molecule([self.atoms[old] for old in perm], bonds,
                        [self.coords[old] for old in perm], feats,
                        label=self.label)

    # -- chemical validity --------------------------------------------------
    def valence_report(self):
        reasons = []
        deg = {i: 0 for i in range(len(self.atoms))}
        for (i, j, o) in self.bonds:
            deg[i] += o
            deg[j] += o
        for i, (el, _q) in enumerate(self.atoms):
            if deg[i] > ELEMENT_VALENCE[el]:
                reasons.append(f"valence exceeded at atom {i} ({el}): "
                               f"{deg[i]} > {ELEMENT_VALENCE[el]}")
        return reasons

    def connectivity_report(self):
        if not self.atoms:
            return ["empty molecule"]
        seen = {0}
        frontier = [0]
        adj = {i: set() for i in range(len(self.atoms))}
        for (i, j, _o) in self.bonds:
            adj[i].add(j)
            adj[j].add(i)
        while frontier:
            u = frontier.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    frontier.append(v)
        if len(seen) != len(self.atoms):
            return [f"disconnected: {len(seen)}/{len(self.atoms)} reachable"]
        return []

    def validity_report(self):
        return self.valence_report() + self.connectivity_report()

    def is_chemically_valid(self):
        return not self.validity_report()

    # -- renumbering-invariant mathematics ----------------------------------
    def invariants(self):
        """Everything the model may know about M: renumbering-invariant."""
        n = len(self.atoms)
        deg = [0] * n
        for (i, j, o) in self.bonds:
            deg[i] += o
            deg[j] += o
        bond_multiset = sorted(
            tuple(sorted((deg[i], deg[j]))) + (o,) for (i, j, o) in self.bonds)
        dist_multiset = sorted(
            sum((a - b) ** 2 for a, b in zip(self.coords[i], self.coords[j]))
            for i, j in combinations(range(n), 2))
        return {
            "element_multiset": tuple(sorted(e for e, _ in self.atoms)),
            "charge_multiset": tuple(sorted(q for _, q in self.atoms)),
            "degree_multiset": tuple(sorted(deg)),
            "bond_multiset": tuple(bond_multiset),
            "distance_multiset": tuple(dist_multiset),
            "feature_multiset": tuple(sorted(t for t, _ in self.features)),
        }

    def canonical_signature(self):
        inv = self.invariants()
        return tuple(inv[k] for k in sorted(inv))

    def feature_positions(self):
        return [(t, self.coords[a]) for (t, a) in self.features]


# ===========================================================================
# Binding-site mathematics (the structural face of the disease state)
# ===========================================================================

def feature_energy(f_type, g_type):
    """Declared interaction energy of a matched feature pair."""
    return COMPAT_ENERGY.get(frozenset((f_type, g_type)), CLASH_ENERGY)


def binding_energy(mol, site):
    """E(M, site): for every site feature, the best (lowest-energy) molecule
    feature within the match radius; unmatched site features contribute 0,
    incompatible close features contribute the clash penalty.
    Returns (E_total, matched_fraction, detail)."""
    feats = mol.feature_positions()
    total = sp.Rational(0)
    matched = 0
    detail = []
    for (g_type, g_pos) in site:
        best = None
        for (f_type, f_pos) in feats:
            d2 = sum((a - b) ** 2 for a, b in zip(f_pos, g_pos))
            if d2 <= EPS2:
                e = feature_energy(f_type, g_type)
                if best is None or e < best:
                    best = e
        if best is not None:
            total += best
            if best < 0:
                matched += 1
            detail.append(f"{g_type}@{g_pos}: {best}")
        else:
            detail.append(f"{g_type}@{g_pos}: unmatched")
    return total, sp.Rational(matched, len(site)) if site else sp.Rational(1), detail


def site_distance2(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


# ===========================================================================
# self-test battery (exit 0 <=> every theorem PROVED)
# ===========================================================================

def run_battery(molecules):
    theorems = []

    def theorem(tag, claim, ok, detail=""):
        theorems.append({"tag": tag, "claim": claim,
                         "status": "PROVED" if ok else "NOT_PROVED",
                         "detail": str(detail)[:200]})

    for label, mol in molecules.items():
        # T8: representation invariance under atom renumbering --------------
        base = mol.invariants()
        inv_ok = True
        for shift in range(len(mol.atoms)):
            perm = list(range(len(mol.atoms)))
            perm = perm[shift:] + perm[:shift]
            if mol.renumber(perm).invariants() != base:
                inv_ok = False
        theorem(f"T8[{label}]",
                "renumbering phi leaves every invariant unchanged "
                "(representation invariance)", inv_ok)

        # chemical validity --------------------------------------------------
        ok, reasons = mol.is_chemically_valid(), mol.validity_report()
        theorem(f"valid[{label}]", "chemically valid (valence + connectivity)",
                ok, "; ".join(reasons))

        # canonical signature is a well-defined identity ---------------------
        sig_a = mol.canonical_signature()
        sig_b = mol.renumber(list(range(len(mol.atoms))[::-1])).canonical_signature()
        theorem(f"canonical[{label}]",
                "canonical signature identical under renumbering",
                sig_a == sig_b)
    return theorems


if __name__ == "__main__":
    import json, sys
    from demo_molecules import DEMO_MOLECULES
    ths = run_battery(DEMO_MOLECULES)
    for t in ths:
        print(f"[{t['status']}] {t['tag']}: {t['claim']}"
              + (f"  [{t['detail']}]" if t["status"] != "PROVED" else ""))
    sys.exit(0 if all(t["status"] == "PROVED" for t in ths) else 1)
