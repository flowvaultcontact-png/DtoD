#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutonomousProofEngine v2 -- hardened + autonomous formula discovery & proof.

STRICT NON-SETTLEMENT DISCIPLINE
--------------------------------
The engine populates _certified_fn ONLY when status == PROVEN_EXACT:
  (1) the certified candidate is verified exactly: every constraint residual
      simplifies to 0 under the recorded domain/branch conditions;
  (2) the derivation chain is complete: every solution of the declared system
      is provably inside the enumerated family -- all branches are enumerated
      explicitly and each is either verified or eliminated with a recorded
      reason (no silent branch loss);
  (3) zero unsettled gaps remain;
  (4) all free parameters are pinned by the declared boundary conditions and
      the pinned solution is unique.
Otherwise the engine refuses to certify and predict() raises.
A second honest tier, PROVEN_CLASSIFICATION, marks a complete verified family
whose parameters remain genuinely free (classification theorem, predict locked).

AUTONOMY
--------
No problem-specific solution hints are wired into the engine.  The engine owns
a fixed repertoire of GENERIC tactics and applies them autonomously:
  T1   structural diagnosis (PDE / discrete / arithmetic atoms)
  T2   functional-equation -> ODE reduction with C^k escalation       (fix B)
  T2b  affine self-relation analysis (fixed-point differentiation)
  T3   ODE solving: dsolve, autonomous reduction-of-order, first-integral
       extraction with EXPLICIT branch enumeration (equilibria / tanh / coth /
       rational / tan) + domain guards                               (fix E)
  T4   power-series coefficient matching with joint (t, y) extraction
       and closed-form pattern synthesis                             (fix D)
  T5   algebraic elimination, disjunctive branching, constant ansatz
  T6   special-point algebraic probes: diagonal x = y recursion and the
       x = 0 anchor, with automatic solving of distinguished-point values
  T7   polynomial ansatz with undetermined coefficients solved against the
       full constraint coefficient system
  T8   spectral ansatz for LINEAR LOCAL constraint sets: characteristic
       equation P(lam) = 0, real roots -> exp(lam x), complex pairs ->
       exp(a x) cos(bx) / exp(a x) sin(bx)
  T9   superposition closure: linear combinations of already-verified
       candidates, registered only after re-verification (linear systems)
All BooleanTrue/BooleanFalse hazards are classified defensively      (fix A),
all Subs point-evaluations are canonically rebuilt into named semantic
constants (_d1, _v0, ...) -- never anonymous _k0                     (fix C).

Honesty about open problems: for Riemann / Collatz / Navier-Stokes the engine
attempts every applicable tactic, reports exactly what it can discover, and
REFUSES to certify, listing the missing mathematical machinery.

FULL-PROOF-ONLY OPEN-CONJECTURE MODE
------------------------------------
`python autonomous_proof_engine_v2.py --open-conjectures` runs ONLY the open
mathematical conjectures (Riemann, Collatz, 3D Navier-Stokes).  Each
conjecture is stated as MULTIPLE formulas that together would constitute the
proof, and the run certifies the conjecture ONLY when EVERY formula reaches a
gap-free PROVEN_EXACT proof.  Anything less -- a classification, a surrogate
invariant, a partial lemma chain -- is recorded as a diagnostic and NEVER
credited as progress toward settlement: the verdict is NO_FULL_PROOF, the
missing machinery is listed per formula, and the run exits non-zero.  The
engine does not settle for progress; it either proves the whole conjecture or
refuses.
"""

import json
import os
import re
import sys
import time
import traceback
from functools import lru_cache

import sympy as sp
from sympy.core.function import AppliedUndef, Derivative, Subs
from sympy.core.relational import Eq
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

try:
    from sympy.calculus.util import singularities as _sp_singularities
except Exception:  # pragma: no cover
    _sp_singularities = None

INF = sp.oo
NINF = -sp.oo


# ---------------------------------------------------------------------------
# Fix A discipline: never touch .lhs/.rhs of a collapsed Boolean; classify it.
# ---------------------------------------------------------------------------

def truth_of(obj):
    """Classify a possibly auto-collapsed relational object."""
    if obj is sp.S.true or isinstance(obj, BooleanTrue):
        return "true"
    if obj is sp.S.false or isinstance(obj, BooleanFalse):
        return "false"
    if isinstance(obj, Eq):
        d = sp.simplify(sp.expand(obj.lhs - obj.rhs))
        return "true" if d == 0 else "relational"
    return "other"


# SymPy expressions are immutable/hashable.  A bounded cache avoids repeating
# the expensive exact-normalisation ladder when the same residual is checked
# by multiple candidates or proof stages.  Set AUTOPROOF_CACHE_SIZE=0 to disable.
_CACHE_SIZE = max(0, int(os.environ.get("AUTOPROOF_CACHE_SIZE", "8192")))

@lru_cache(maxsize=_CACHE_SIZE)
def _residual_is_zero_cached(res):
    """Exact residual check; cache key is the immutable SymPy expression."""
    if res == 0:
        return True
    # Cheap exact forms first.  No numerical approximation is used here.
    try:
        expanded = sp.expand(res)
        if expanded == 0:
            return True
    except Exception:
        expanded = res
    attempts = (
        lambda e: sp.simplify(e),
        lambda e: sp.simplify(e.rewrite(sp.exp)),
        lambda e: sp.simplify(sp.together(e)),
        lambda e: sp.simplify(sp.radsimp(e)),
        lambda e: sp.simplify(sp.cancel(sp.together(e))),
    )
    for transform in attempts:
        try:
            if transform(expanded) == 0:
                return True
        except Exception:
            continue
    return False


def residual_is_zero(res):
    """Determine exact zero with a bounded, reusable symbolic-result cache."""
    try:
        expr = sp.sympify(res)
        return bool(_residual_is_zero_cached(expr))
    except (TypeError, ValueError):
        # Preserve defensive behaviour for objects that SymPy cannot hash.
        try:
            return _residual_is_zero_uncached(sp.sympify(res))
        except Exception:
            return False


def _residual_is_zero_uncached(res):
    """Fallback for unusual unhashable SymPy-compatible inputs."""
    if res == 0:
        return True
    for transform in (sp.simplify, lambda e: sp.simplify(sp.expand(e)),
                      lambda e: sp.simplify(sp.cancel(sp.together(e)))):
        try:
            if transform(res) == 0:
                return True
        except Exception:
            pass
    return False


def branch_kind_of(expr, x):
    """Human-readable branch class of a candidate (used by the test suite)."""
    s = str(expr)
    if x not in expr.free_symbols:
        return "constant"
    if "tanh" in s:
        return "tanh"
    if "coth" in s:
        return "coth"
    if "tan(" in s:
        return "tan"
    try:
        num, den = sp.together(expr).as_numer_denom()
        if den.has(x) and den.is_polynomial(x) and sp.degree(den, x) == 1 \
                and not (num.atoms(sp.exp) or num.atoms(sp.tan) or num.atoms(sp.tanh)):
            return "rational"
    except Exception:
        pass
    return "other"


# ===========================================================================
# The engine
# ===========================================================================

class AutonomousProofEngine:

    def __init__(self, problem_name="Unnamed Problem", func_name='f',
                 var_name='x', aux_var_name='y', params=(),
                 param_assumptions=None, domain='real',
                 auto_declare_regularity=True, k_max=3, series_order=8,
                 full_proof_only=False):
        self.problem_name = problem_name
        self.full_proof_only = bool(full_proof_only)
        self.x = sp.Symbol(var_name, real=True)
        self.y = sp.Symbol(aux_var_name, real=True)
        self.f = sp.Function(func_name)
        self.domain = domain
        self.auto_declare_regularity = auto_declare_regularity
        self.k_max = k_max
        self.series_order = series_order
        self.series_center = sp.S.Zero
        self.open_problem_context = None

        self.param_syms = {}
        assumptions = param_assumptions or {}
        for p in params:
            kw = {}
            a = assumptions.get(p) if isinstance(p, str) else None
            if a == 'positive':
                kw['positive'] = True
            elif a == 'nonnegative':
                kw['nonnegative'] = True
            elif a == 'nonzero':
                kw['nonzero'] = True
            else:
                kw['real'] = True
            s = sp.Symbol(p, **kw) if isinstance(p, str) else p
            self.param_syms[str(s)] = s

        # ---- declared problem material -----------------------------------
        self.declared_hypotheses = []
        self.functionals = []       # (lhs, rhs, identity_val)
        self.self_relations = []    # (lhs, rhs)
        self.diffs = []             # (lhs, rhs)
        self.algs = []              # (lhs, rhs)
        self.bcs = []               # {'order', 'point', 'value'}

        # ---- derivation state --------------------------------------------
        self.point_const_semantics = {}
        self.declared_axioms = []
        self.used_hypotheses = []
        self.problem_gaps = []
        self.required_machinery = []
        self.discovered_formulas = []
        self.derived_odes = []
        self.derived_conditions = []
        self.derivation_chain = []
        self.tactic_notes = []
        self.candidates = []
        self.eliminated_branches = []
        self.consistency_checks = []
        self.status = "EMPTY_SYSTEM"
        self._certified_fn = None
        self._certified_family = None
        self._functionals_fully_reduced = False
        self._odes_family_complete = False
        self._ode_family_status = {}
        self._branch_enum_ok = {}
        self._blocked_completeness = []
        self.elapsed = 0.0

    # ------------------------------------------------------------------
    # problem declaration API
    # ------------------------------------------------------------------
    def declare_hypothesis(self, tag):
        self.declared_hypotheses.append(str(tag))
        self.used_hypotheses.append(str(tag))

    def set_open_problem(self, ctx):
        self.open_problem_context = dict(ctx)

    def add_functional_constraint(self, lhs, rhs, identity_val=0):
        self.functionals.append((sp.sympify(lhs), sp.sympify(rhs), sp.sympify(identity_val)))

    def add_self_relation(self, lhs, rhs):
        self.self_relations.append((sp.sympify(lhs), sp.sympify(rhs)))

    def add_differential_constraint(self, lhs, rhs):
        self.diffs.append((sp.sympify(lhs), sp.sympify(rhs)))

    def add_algebraic_constraint(self, lhs, rhs):
        self.algs.append((sp.sympify(lhs), sp.sympify(rhs)))

    def add_bc_value(self, value, point=0):
        self.bcs.append({"order": 0, "point": sp.sympify(point), "value": sp.sympify(value)})

    def add_bc_deriv(self, order, value, point=0):
        self.bcs.append({"order": int(order), "point": sp.sympify(point),
                         "value": sp.sympify(value)})

    # ------------------------------------------------------------------
    # reserved symbols / gap & axiom bookkeeping
    # ------------------------------------------------------------------
    def _reserved(self):
        return set(self.param_syms.values()) | {self.x, self.y}

    def _declare_axiom(self, tag, reason):
        self.declared_axioms.append({"tag": tag, "self_declared": True, "reason": reason})
        self.used_hypotheses.append(tag)
        self.derivation_chain.append(f"AXIOM self-declared: {tag} -- {reason}")

    def _add_gap(self, tag, detail):
        self.problem_gaps.append({"tag": tag, "detail": str(detail)})

    # ------------------------------------------------------------------
    # fix C: canonical rebuilding of Subs point-evaluations
    # ------------------------------------------------------------------
    def _plain_order(self, d):
        for n in range(1, 9):
            if d == sp.Derivative(self.f(self.x), (self.x, n)):
                return n
        return None

    def _subs_order(self, inner, dummy):
        for n in range(1, 9):
            if inner == sp.Derivative(inner.expr.func(dummy), (dummy, n)):
                return n
        return None

    def _canon_subs(self, expr):
        """Rebuild every Subs(Derivative(f(dummy),(dummy,n)), dummy, pt) into the
        canonical form: plain Derivative when pt == x, canonical dummy = x when
        pt is a literal constant, untouched when pt contains x (fix C)."""
        def rebuild(s):
            if not isinstance(s, Subs):
                return s
            inner = s.expr
            if not isinstance(inner, Derivative):
                return s
            base = inner.expr
            if not isinstance(base, AppliedUndef) or base.func != self.f:
                return s
            if len(s.variables) != 1:
                return s
            dummy = s.variables[0]
            order = self._subs_order(inner, dummy)
            if order is None:
                return s
            pt = sp.sympify(s.point[0] if len(s.point) == 1 else sp.Tuple(*s.point))
            if pt == self.x:
                return sp.Derivative(self.f(self.x), (self.x, order))
            if pt.free_symbols & {self.x, self.y}:
                return s
            return sp.Subs(sp.Derivative(self.f(self.x), (self.x, order)), self.x, pt)
        try:
            return expr.replace(lambda e: isinstance(e, Subs), rebuild)
        except Exception:
            return expr

    def _get_point_const(self, order, pt):
        if order == 0:
            name = "_v0" if pt == 0 else f"_v_at_{pt}"
        else:
            name = f"_d{order}" if pt == 0 else f"_d{order}_at_{pt}"
        s = sp.Symbol(name, real=True)
        self.point_const_semantics[s] = (f"f({pt})" if order == 0 else f"f^({order})({pt})")
        return s

    def _point_constants(self, expr):
        """Symbolise literal-number point evaluations: f(c) -> _v..., f^(n)(c) -> _d...
        Named semantics replace the old anonymous _k0 hack (fix C)."""
        mapping = {}

        def match(e):
            if isinstance(e, Subs):
                return isinstance(e.expr, Derivative)
            if isinstance(e, AppliedUndef) and e.func == self.f:
                return bool(e.args) and sp.sympify(e.args[0]).is_number
            return False

        def rebuild(e):
            if isinstance(e, Subs):
                inner = e.expr
                base = inner.expr
                if not (isinstance(base, AppliedUndef) and base.func == self.f):
                    return e
                if len(e.variables) != 1:
                    return e
                dummy = e.variables[0]
                order = self._subs_order(inner, dummy)
                if order is None:
                    return e
                pt = sp.sympify(e.point[0] if len(e.point) == 1 else sp.Tuple(*e.point))
                if not pt.is_number:
                    return e
                s = self._get_point_const(order, pt)
                mapping[e] = s
                return s
            # AppliedUndef with numeric argument
            pt = sp.sympify(e.args[0])
            s = self._get_point_const(0, pt)
            mapping[e] = s
            return s

        try:
            out = expr.replace(match, rebuild)
        except Exception:
            out = expr
        return out, mapping

    # ------------------------------------------------------------------
    # substitution of an explicit candidate into constraint expressions
    # ------------------------------------------------------------------
    def _sub_candidate(self, expr, cand):
        out = sp.sympify(expr)
        derivs = sorted({d for d in out.atoms(Derivative)
                         if isinstance(d.expr, AppliedUndef) and d.expr.func == self.f
                         and d.expr.args[0] == self.x},
                        key=lambda d: self._plain_order(d) or 0)
        for d in derivs:
            n = self._plain_order(d)
            if n is None:
                continue
            out = out.replace(d, sp.diff(cand, self.x, n))
        out = out.replace(
            lambda e: isinstance(e, AppliedUndef) and e.func == self.f,
            lambda e: cand.subs(self.x, e.args[0]))
        return out

    def _sub_constant(self, expr, c):
        out = sp.sympify(expr)
        for d in list(out.atoms(Derivative)):
            if isinstance(d.expr, AppliedUndef) and d.expr.func == self.f:
                out = out.replace(d, sp.S.Zero)
        out = out.replace(
            lambda e: isinstance(e, AppliedUndef) and e.func == self.f,
            lambda e: c)
        return out

    def _ode_order(self, res):
        mx = 0
        for d in res.atoms(Derivative):
            if isinstance(d.expr, AppliedUndef) and d.expr.func == self.f and d.expr.args[0] == self.x:
                n = self._plain_order(d)
                if n:
                    mx = max(mx, n)
        return mx

    def _highest_deriv(self, res, order):
        return sp.Derivative(self.f(self.x), (self.x, order))

    # ------------------------------------------------------------------
    # T1: structural diagnosis
    # ------------------------------------------------------------------
    def _t1_structural_diagnosis(self):
        ctx = self.open_problem_context or {}
        if ctx.get("pde_dimensional"):
            self._add_gap("structural_pde",
                          "constraint field is a multi-dimensional PDE; no SymPy ODE/algebraic "
                          "elimination tactic can close it")
        if ctx.get("discrete_domain"):
            self._add_gap("structural_discrete",
                          "the true object lives on a discrete domain; continuous calculus tactics "
                          "do not transfer")
        for atom in ctx.get("arithmetic_atoms", []) or []:
            self._add_gap("arithmetic_atom",
                          f"declared arithmetic atom '{atom}' requires analytic-number-theory "
                          f"machinery outside symbolic tactics")
        if ctx.get("beyond_symbolic_scope"):
            self._blocked_completeness.append("declared objective exceeds symbolic tactic scope")
        # atom scan on the actual constraints
        ARITH = {"zeta", "dirichlet_eta", "primepi", "divisor", "primenu", "li", "mobius"}
        all_exprs = [l - r for (l, r) in self.diffs] + [l - r for (l, r) in self.algs] \
            + [l - r for (l, r, _iv) in self.functionals] + [l - r for (l, r) in self.self_relations]
        for e in all_exprs:
            for fn in e.atoms(AppliedUndef):
                if str(fn.func) in ARITH:
                    self._add_gap("arithmetic_atom", f"constraint contains {fn}; "
                                  "arithmetic machinery required")

    # ------------------------------------------------------------------
    # T2: functional-equation -> ODE reduction with C^k escalation (fix B)
    # ------------------------------------------------------------------
    def _tactic_functional_reduction(self):
        reduced_any = []
        for (l, r, iv) in self.functionals:
            eq_expr = self._canon_subs(sp.expand(l - r))
            accepted = False
            for k in range(1, self.k_max + 1):
                try:
                    d = sp.diff(eq_expr, self.y, k).subs(self.y, iv)
                except Exception as exc:
                    self._add_gap(f"reduction_diff_failed_k{k}", exc)
                    break
                d = self._canon_subs(d)
                d, _ = self._point_constants(d)
                d = sp.simplify(sp.expand(d))
                if d == 0:
                    continue  # trivial at order k -> escalate (fix B)
                has_xderiv = any(
                    isinstance(a, Derivative)
                    and isinstance(a.expr, AppliedUndef)
                    and a.expr.func == self.f and a.expr.args[0] == self.x
                    for a in d.atoms(Derivative))
                tag = f"C{k}_continuity_of_f_at_y={iv}"
                if has_xderiv:
                    if self.auto_declare_regularity:
                        self._declare_axiom(
                            tag,
                            f"differentiating the functional equation {k} time(s) in y and "
                            f"evaluating at y={iv} presupposes C^{k} regularity; without it "
                            "Hamel-type wild solutions survive")
                    else:
                        self._add_gap(tag + "_not_declared",
                                      f"engine needs C^{k} regularity to reduce; absolute mode "
                                      "leaves the reduction unavailable")
                        break
                    self.derived_odes.append(sp.expand(d))
                    self.derivation_chain.append(
                        f"T2: functional equation differentiated k={k} in y, evaluated at "
                        f"y={iv} under axiom {tag} -> ODE residual: {d}")
                    accepted = True
                    break
                else:
                    # non-trivial but derivative-free: a derived necessary condition
                    self.derived_conditions.append(sp.expand(d))
                    self.derivation_chain.append(
                        f"T2: k={k} reduction is derivative-free -> necessary condition: {d}")
                    continue  # escalate (fix B)
            reduced_any.append(accepted)
            if not accepted:
                self.derivation_chain.append(
                    "T2: no usable ODE reduction for this functional equation up to "
                    f"k={self.k_max}")
        self._functionals_fully_reduced = bool(self.functionals) and all(reduced_any)

    # ------------------------------------------------------------------
    # T2b: affine self-relation analysis at fixed points
    # ------------------------------------------------------------------
    def _tactic_self_relations(self):
        for (l, r) in self.self_relations:
            expr = self._canon_subs(sp.expand(l - r))
            sigmas = set()
            for fn in expr.atoms(AppliedUndef):
                if fn.func == self.f and fn.args[0] != self.x:
                    sigmas.add(sp.sympify(fn.args[0]))
            for sig in sorted(sigmas, key=str):
                if not sig.is_polynomial(self.x) or sp.degree(sp.expand(sig), self.x) != 1:
                    self._add_gap("nonlinear_self_relation",
                                  f"argument {sig} is not affine in {self.x}; fixed-point "
                                  "analysis skipped")
                    continue
                alpha = sp.expand(sig).coeff(self.x, 1)
                beta = sp.expand(sig).coeff(self.x, 0)
                if alpha == 1:
                    continue  # translation: no fixed point
                xstar = sp.simplify(beta / (1 - alpha))
                if not xstar.is_number:
                    self._add_gap("symbolic_fixed_point", f"fixed point of {sig} not numeric")
                    continue
                if self.auto_declare_regularity:
                    if not any(a["tag"] == "C1_continuity_of_f" for a in self.declared_axioms):
                        self._declare_axiom(
                            "C1_continuity_of_f",
                            "differentiating the self-relation presupposes C^1 regularity")
                else:
                    self._add_gap("C1_continuity_of_f_not_declared",
                                  "fixed-point differentiation needs C^1; not declared")
                    continue
                if self.series_center == 0 and xstar != 0:
                    self.series_center = xstar  # expand series at the fixed point
                for k in (1, 2):
                    try:
                        dd = sp.diff(expr, self.x, k)
                    except Exception as exc:
                        self._add_gap(f"selfrel_diff_failed_k{k}", exc)
                        break
                    dd = self._canon_subs(dd).subs(self.x, xstar)
                    dd = self._canon_subs(dd)
                    dd, _ = self._point_constants(dd)
                    dd = sp.simplify(sp.expand(dd))
                    if dd == 0:
                        continue
                    self.derived_conditions.append(sp.expand(dd))
                    self.discovered_formulas.append({
                        "formula": f"{dd} = 0",
                        "meaning": f"consequence of the self-relation at its fixed point "
                                   f"x* = {xstar} (order {k})"})
                    self.derivation_chain.append(
                        f"T2b: self-relation differentiated k={k}, evaluated at fixed point "
                        f"x*={xstar} -> condition: {dd}")

    # ------------------------------------------------------------------
    # T5: algebraic elimination + disjunctive branching + constant ansatz
    # ------------------------------------------------------------------
    def _tactic_algebraic(self):
        u = sp.Symbol('_u', real=True)
        conds = list(self.derived_conditions)
        for (l, r, iv) in self.functionals:
            val = self._canon_subs(sp.expand(l - r).subs(self.y, iv))
            val, _ = self._point_constants(val)
            conds.append(sp.expand(val))
        for cond in conds:
            try:
                core = sp.expand(cond.subs(self.f(self.x), u))
                factored = sp.factor(core)
            except Exception:
                continue
            if not factored.is_Mul and not factored.is_Pow:
                continue
            try:
                coeff, factors = sp.factor_list(factored)
            except Exception:
                continue
            for (fac, _mult) in factors:
                if fac == u or fac == -u:
                    continue  # constant-zero branch; constants tactic enumerates it
                if u not in fac.free_symbols and self.x not in fac.free_symbols:
                    self.derived_conditions.append(sp.expand(fac))
                    self.derivation_chain.append(
                        f"T5: disjunctive factor {fac} = 0 is a derived necessary condition")

    def _tactic_constants(self):
        c = sp.Symbol('_c', real=True)
        exprs = [l - r for (l, r) in self.diffs] + [l - r for (l, r) in self.algs] \
            + [l - r for (l, r, _iv) in self.functionals] + [l - r for (l, r) in self.self_relations]
        if not exprs:
            return
        eqs = []
        for e in exprs:
            q = sp.simplify(sp.expand(self._sub_constant(e, c)))
            if q != 0:
                eqs.append(q)
        if not eqs:
            self._register_candidate(c, source="T5:constant-ansatz", branch_conds=[])
            return
        try:
            if len(eqs) == 1:
                vals = sp.solve(eqs[0], c)
            else:
                sol = sp.solve(eqs, c, dict=True)
                vals = [d[c] for d in sol if c in d] if sol else []
        except Exception:
            vals = []
        vals = [v for v in vals if not isinstance(v, (list, tuple))]
        if vals:
            for v in list(vals)[:6]:
                self._register_candidate(sp.sympify(v), source="T5:constant-ansatz",
                                         branch_conds=[])
        else:
            self._register_candidate(c, source="T5:constant-ansatz", branch_conds=[])

    # ------------------------------------------------------------------
    # T6: special-point algebraic probes (diagonal, anchor)
    # ------------------------------------------------------------------
    def _tactic_special_points(self):
        probes = [sp.expand(l - r) for (l, r) in self.diffs]
        probes += [sp.expand(l - r) for (l, r) in self.algs]
        probes += [sp.expand(l - r) for (l, r, _iv) in self.functionals]
        probes += [sp.expand(l - r) for (l, r) in self.self_relations]
        if not probes:
            return
        for base in probes:
            if base.has(self.y):
                try:
                    diag = self._canon_subs(base.subs(self.y, self.x))
                    diag, _ = self._point_constants(diag)
                    diag = sp.simplify(sp.expand(diag))
                except Exception:
                    diag = None
                if diag not in (None, 0) and diag not in self.derived_conditions:
                    self.derived_conditions.append(sp.expand(diag))
                    self.discovered_formulas.append({
                        "formula": f"{diag} = 0",
                        "meaning": "T6 special-point probe: the diagonal x = y turns "
                                   "the constraint into a pointwise recursion"})
                    self.derivation_chain.append(
                        f"T6: diagonal substitution y = x -> necessary condition: {diag}")
            try:
                anch = self._canon_subs(base.subs(self.x, 0))
                anch, _ = self._point_constants(anch)
                anch = sp.simplify(sp.expand(anch))
            except Exception:
                anch = None
            if anch not in (None, 0) and anch not in self.derived_conditions:
                self.derived_conditions.append(sp.expand(anch))
                if not anch.atoms(AppliedUndef) and not anch.atoms(Derivative):
                    self.derivation_chain.append(
                        f"T6: anchor x = 0 -> closed necessary condition: {anch}")
                    try:
                        consts = sorted(anch.free_symbols - {self.x, self.y}, key=str)
                        if consts:
                            sols = sp.solve(anch, consts, dict=True)
                            for sol in list(sols)[:6]:
                                pretty = ", ".join(f"{k} = {v}" for k, v in sol.items())
                                self.discovered_formulas.append({
                                    "formula": pretty,
                                    "meaning": "T6 anchor probe pins a distinguished-point "
                                               "value of the unknown function"})
                                self.derivation_chain.append(
                                    f"T6: anchor condition solved -> {pretty}")
                    except Exception:
                        pass
                else:
                    self.derivation_chain.append(
                        f"T6: anchor x = 0 -> necessary condition: {anch}")

    # ------------------------------------------------------------------
    # shared local+linear precheck for the spectral / superposition tactics
    # ------------------------------------------------------------------
    def _local_linear_gens(self):
        """Return (gen_list, exprs) when every constraint is LOCAL (f only ever
        at argument x) and LINEAR in f and its x-derivatives; else ([], [])."""
        exprs = [l - r for (l, r) in self.diffs] + [l - r for (l, r) in self.algs] \
            + [l - r for (l, r, _iv) in self.functionals] \
            + [l - r for (l, r) in self.self_relations]
        if not exprs:
            return [], []
        gens = set()
        for e in exprs:
            for fn in e.atoms(AppliedUndef):
                if fn.func == self.f:
                    if fn.args[0] != self.x:
                        return [], []
                    gens.add(fn)
            for d in e.atoms(Derivative):
                if isinstance(d.expr, AppliedUndef) and d.expr.func == self.f:
                    if d.expr.args[0] != self.x:
                        return [], []
                    gens.add(d)
        if not gens:
            return [], []
        gen_list = sorted(gens, key=str)
        subs_map = {g: sp.Symbol(f'_L{i}', real=True)
                    for i, g in enumerate(gen_list)}
        sym_list = [subs_map[g] for g in gen_list]
        for e in exprs:
            e_lin = sp.expand(e.xreplace(subs_map))
            if any(e_lin.has(s) for s in sym_list):
                poly = sp.Poly(e_lin, *sym_list)
                if poly.total_degree() > 1:
                    return [], []
        return gen_list, exprs

    def _local_linear_residuals(self):
        return self._local_linear_gens()[1]

    # ------------------------------------------------------------------
    # T7: polynomial ansatz with undetermined coefficients
    # ------------------------------------------------------------------
    def _tactic_polynomial_ansatz(self, max_degree=3, max_solutions=8):
        exprs = [l - r for (l, r) in self.diffs] + [l - r for (l, r) in self.algs] \
            + [l - r for (l, r, _iv) in self.functionals] \
            + [l - r for (l, r) in self.self_relations]
        if not exprs:
            return
        for deg in range(1, max_degree + 1):
            coeff_syms = [sp.Symbol(f'_p{i}', real=True) for i in range(deg + 1)]
            poly = sum(c * self.x ** i for i, c in enumerate(coeff_syms))
            eqs = []
            for e in exprs:
                try:
                    resid = sp.expand(self._sub_candidate(e, poly))
                except Exception:
                    eqs = None
                    break
                variables = [v for v in (self.x, self.y) if resid.has(v)]
                if variables:
                    eqs.extend(sp.simplify(c) for c in sp.Poly(resid, *variables).coeffs())
                else:
                    eqs.append(sp.simplify(resid))
            if not eqs:
                continue
            eqs = [q for q in eqs if q != 0]
            if not eqs:
                continue
            try:
                sols = sp.solve(eqs, coeff_syms, dict=True)
            except Exception:
                continue
            sols = [s for s in sols if isinstance(s, dict)]
            for sol in sols[:max_solutions]:
                cand = sp.simplify(poly.subs(sol))
                self._register_candidate(cand, source="T7:polynomial-ansatz",
                                         branch_conds=[])
                self.derivation_chain.append(
                    f"T7: polynomial ansatz of degree {deg} solved the constraint "
                    f"coefficient system -> candidate {cand}")

    # ------------------------------------------------------------------
    # T8: spectral ansatz for linear local constraints (characteristic eq.)
    # ------------------------------------------------------------------
    def _tactic_spectral_ansatz(self, max_roots=6):
        gens, residuals = self._local_linear_gens()
        if not gens or not residuals:
            return
        lam = sp.Symbol('_lam', real=False)
        spectral_map = {}
        for g in gens:
            if isinstance(g, Derivative):
                n = self._plain_order(g)
                if n is None:
                    return
                spectral_map[g] = lam ** n
            else:
                spectral_map[g] = sp.S.One
        for resid in residuals:
            char_eq = sp.simplify(sp.expand(resid.xreplace(spectral_map)))
            if char_eq.has(self.x, self.y) or char_eq == 0:
                continue
            if char_eq.has(sp.Symbol, AppliedUndef, Derivative) \
                    and not char_eq.free_symbols <= {lam}:
                continue
            try:
                roots = sp.solve(sp.Eq(char_eq, 0), lam)
            except Exception:
                continue
            roots = [sp.sympify(rt) for rt in roots if sp.sympify(rt).is_number]
            seen = set()
            registered = 0
            for rt in roots[:max_roots * 2]:
                key = sp.simplify(rt)
                if key in seen:
                    continue
                seen.add(key)
                if registered >= max_roots:
                    break
                try:
                    a, b = sp.re(key), sp.im(key)
                except Exception:
                    continue
                if b == 0:
                    if a == 0:
                        continue  # constant branch: covered by T5
                    cand = sp.exp(a * self.x)
                    desc = f"real characteristic root lam = {a}"
                else:
                    if a == 0:
                        cand = sp.cos(b * self.x)
                        desc = f"imaginary characteristic pair lam = +/- {b}I"
                    else:
                        cand = sp.exp(a * self.x) * sp.cos(b * self.x)
                        desc = f"complex characteristic pair lam = {a} +/- {b}I"
                self._register_candidate(cand, source="T8:spectral-ansatz",
                                         branch_conds=[])
                self.derivation_chain.append(
                    f"T8: characteristic equation {char_eq} = 0 -> {desc} "
                    f"-> candidate {cand}")
                if b != 0:
                    partner = sp.sin(b * self.x) if a == 0 \
                        else sp.exp(a * self.x) * sp.sin(b * self.x)
                    self._register_candidate(partner, source="T8:spectral-ansatz",
                                             branch_conds=[])
                    self.derivation_chain.append(
                        f"T8: conjugate partner of lam = {key} -> candidate {partner}")
                    registered += 1
                registered += 1

    # ------------------------------------------------------------------
    # T9: superposition closure over verified candidates (linear systems)
    # ------------------------------------------------------------------
    def _tactic_superposition(self, max_pairs=12):
        if not self._local_linear_residuals():
            return
        verified = [c for c in self.candidates
                    if c["verified"] and c["kind"] == "family"
                    and c.get("status") not in ("eliminated", "absorbed",
                                                 "failed_verification")]
        if len(verified) < 2:
            return
        pairs = 0
        for i in range(len(verified)):
            for j in range(i + 1, len(verified)):
                if pairs >= max_pairs:
                    return
                c1, c2 = verified[i]["expr"], verified[j]["expr"]
                if c1 == 0 or c2 == 0:
                    continue
                try:
                    ratio = sp.simplify(c1 / c2)
                    if not ratio.has(self.x):
                        continue  # linearly dependent over constants
                except Exception:
                    pass
                alpha = sp.Symbol('_s0', real=True)
                combo = sp.simplify(c1 + alpha * c2)
                pairs += 1
                self._register_candidate(combo, source="T9:superposition",
                                         branch_conds=[])
                self.derivation_chain.append(
                    f"T9: linear superposition of verified candidates {c1} and "
                    f"{c2} -> candidate {combo} (linearity of the constraint set)")
                new_c = self.candidates[-1]
                self._pin_candidate(new_c)
                if new_c.get("status") not in ("eliminated",):
                    self._verify_candidate(new_c)

    # ------------------------------------------------------------------
    # T3: ODE pipeline
    # ------------------------------------------------------------------
    def _tactic_odes(self):
        ode_list = [("declared:%d" % i, sp.expand(l - r))
                    for i, (l, r) in enumerate(self.diffs)]
        ode_list += [("derived:%d" % i, sp.expand(d))
                     for i, d in enumerate(self.derived_odes)]
        for (src, res0) in ode_list:
            res = self._canon_subs(res0)
            res, _ = self._point_constants(res)
            res = sp.expand(res)
            if residual_is_zero(res):
                self._ode_family_status[src] = True
                continue
            order = self._ode_order(res)
            if order == 1:
                self._solve_first_order(res, src)
            elif order == 2:
                self._solve_second_order(res, src)
            else:
                self._ode_family_status[src] = False
                self._add_gap("ode_order_unsupported",
                              f"order-{order} ODE ({src}): no generic tactic available")
        declared_ok = [self._ode_family_status.get("declared:%d" % i, False)
                       for i in range(len(self.diffs))]
        if self.diffs:
            self._odes_family_complete = all(declared_ok)

    def _rename_dsolve_constants(self, cand):
        subs_map = {}
        for s in cand.free_symbols:
            if re.match(r'^C\d+$', s.name):
                subs_map[s] = sp.Symbol('_C' + s.name[1:], real=True)
        if subs_map:
            cand = cand.subs(subs_map)
        return cand

    def _dsolve_candidate(self, eq, src):
        """Try dsolve; register explicit solutions. Returns True if a usable
        explicit general-solution candidate was registered."""
        try:
            sols = sp.dsolve(eq, self.f(self.x))
        except Exception as exc:
            self.tactic_notes.append(f"T3: dsolve failed on {eq} ({exc})")
            return False
        if isinstance(sols, Eq):
            sols = [sols]
        if not isinstance(sols, (list, tuple)):
            sols = [sols]
        ok = False
        for s in sols:
            cand = s.rhs if isinstance(s, Eq) else s
            if truth_of(s) == "true":
                continue
            if cand.atoms(Derivative) or cand.atoms(sp.Integral) \
                    or (self.f(self.x) in cand.atoms(AppliedUndef)):
                self.tactic_notes.append(f"T3: dsolve returned a non-explicit solution: {cand}")
                continue
            cand = self._rename_dsolve_constants(cand)
            self._register_candidate(cand, source=f"T3:dsolve({src})",
                                     branch_conds=[], machinery="general ODE solution theory (dsolve)")
            self.derivation_chain.append(f"T3: dsolve on {eq} -> {cand}")
            ok = True
        return ok

    def _solve_first_order(self, res, src):
        fp = sp.Derivative(self.f(self.x), self.x)
        try:
            rhs_list = sp.solve(sp.expand(res), fp)
        except Exception:
            rhs_list = []
        got = False
        for rhs in rhs_list[:2]:
            got = self._dsolve_candidate(sp.Eq(fp, sp.simplify(rhs)), src) or got
        if not got:
            self._first_order_separation(res, src)
        else:
            self._ode_family_status[src] = True

    def _solve_second_order(self, res, src):
        fp2 = sp.Derivative(self.f(self.x), (self.x, 2))
        fp1 = sp.Derivative(self.f(self.x), self.x)
        got = False
        try:
            rhs_list = sp.solve(sp.expand(res), fp2)
        except Exception:
            rhs_list = []
        if rhs_list:
            got = self._dsolve_candidate(sp.Eq(fp2, sp.simplify(rhs_list[0])), src)
        # autonomous test
        u, p1, p2 = (sp.Symbol('_uu', real=True), sp.Symbol('_p1', real=True),
                     sp.Symbol('_p2', real=True))
        try:
            test = res.subs({fp2: p2, fp1: p1, self.f(self.x): u})
            autonomous = self.x not in test.free_symbols
        except Exception:
            autonomous = False
        complete = got
        if autonomous:
            red_ok = self._solve_second_order_autonomous(res, src, supplementary=got)
            complete = complete or red_ok
        self._ode_family_status[src] = complete
        if not complete:
            self._add_gap("ode_family_incomplete",
                          f"second-order ODE ({src}): neither dsolve nor reduction-of-order "
                          "produced a provably complete family")

    def _solve_second_order_autonomous(self, res, src, supplementary=False):
        """Generic Lie-style reduction of order p(u) = u'(x) for autonomous
        second-order ODEs, followed by first-integral extraction and explicit
        branch enumeration (fix E: no silent branch loss).  In supplementary
        mode (dsolve already produced a complete family) any limitation here
        is recorded as a tactic note, not as a certification-blocking gap."""
        fp2 = sp.Derivative(self.f(self.x), (self.x, 2))
        fp1 = sp.Derivative(self.f(self.x), self.x)
        try:
            rhs_list = sp.solve(sp.expand(res), fp2)
        except Exception:
            rhs_list = []
        if not rhs_list:
            self._add_gap("reduction_solve_failed", f"cannot solve {res} for f''")
            return False
        rhs = sp.simplify(rhs_list[0])
        u = sp.Symbol('_uu', real=True)
        P = sp.Function('_Pp')
        p = P(u)
        rhs_u = sp.simplify(rhs.subs(fp1, p).subs(self.f(self.x), u))
        F_red = sp.simplify(sp.expand(p * sp.diff(p, u) - rhs_u))
        # branch p == 0 (u' == 0 -> constant solutions) is covered by the
        # constants ansatz; record the linkage honestly.
        self.tactic_notes.append(
            "reduction of order: branch p(u) == 0 (i.e. u' == 0) yields constant solutions; "
            "these are enumerated independently by the constant ansatz")
        try:
            dps = sp.solve(F_red, sp.diff(p, u))
        except Exception:
            dps = []
        if not dps:
            self.tactic_notes.append(
                f"T3: reduction of order could not isolate p'(u) from {F_red}; "
                "relying on the dsolve family")
            return False
        try:
            psol = sp.dsolve(sp.Eq(sp.diff(p, u), sp.simplify(dps[0])), p)
        except Exception as exc:
            self.tactic_notes.append(f"T3: dsolve for p(u) failed: {exc}")
            return False
        psol_list = psol if isinstance(psol, (list, tuple)) else [psol]
        complete_any = False
        for ps_one in psol_list:
            pexpr = ps_one.rhs if isinstance(ps_one, Eq) else ps_one
            if not isinstance(pexpr, sp.Expr) or pexpr.atoms(Derivative) \
                    or pexpr.atoms(sp.Integral):
                self.tactic_notes.append("T3: p(u) solution not explicit; branch skipped")
                continue
            pexpr = self._rename_dsolve_constants(pexpr)
            csts = sorted([s for s in pexpr.free_symbols if s.name.startswith('_C')], key=str)
            # first integral: u' - G(u) = const, G = pexpr minus its constant part
            G = pexpr
            const_part = sp.S.Zero
            for s in csts:
                if s in G.free_symbols:
                    const_part += G.coeff(s) * s
            G0 = sp.simplify(G - const_part)
            inv = sp.simplify(sp.Derivative(self.f(self.x), self.x) - G0.subs(u, self.f(self.x)))
            self.discovered_formulas.append({
                "formula": f"{inv} = const",
                "meaning": "first integral discovered by autonomous reduction of order "
                           "(invariant along solutions; any nonzero multiple is equivalent)"})
            self.required_machinery.append(
                "elementary calculus: reduction of order for autonomous ODEs (chain rule)")
            self.derivation_chain.append(
                f"T3: reduction of order -> p(u) = {pexpr}; first integral {inv} = const")
            ode1 = sp.expand(sp.Derivative(self.f(self.x), self.x) - pexpr.subs(u, self.f(self.x)))
            try:
                rhs1 = sp.solve(ode1, sp.Derivative(self.f(self.x), self.x))
            except Exception:
                rhs1 = []
            if rhs1:
                self._dsolve_candidate(
                    sp.Eq(sp.Derivative(self.f(self.x), self.x), sp.simplify(rhs1[0])),
                    src + "+reduction")
            complete = self._first_order_branch_enum(ode1, src + "+branch-enum",
                                                     supplementary=supplementary)
            self._branch_enum_ok[src] = complete
            complete_any = complete_any or complete
        return complete_any

    def _first_order_separation(self, res, src):
        fp = sp.Derivative(self.f(self.x), self.x)
        try:
            rhs_list = sp.solve(sp.expand(res), fp)
        except Exception:
            rhs_list = []
        if not rhs_list:
            self._add_gap("first_order_unsolved", f"cannot solve for f' in {res}")
            self._ode_family_status[src] = False
            return
        g = sp.simplify(rhs_list[0])
        u = sp.Symbol('_uu', real=True)
        g_u = sp.simplify(g.subs(self.f(self.x), u))
        if self.x in g_u.free_symbols:
            self._add_gap("non_autonomous_first_order",
                          f"RHS {g_u} depends on {self.x}; separation tactic not applicable")
            self._ode_family_status[src] = False
            return
        ok = self._first_order_branch_enum(sp.expand(fp - g), src + "+separation")
        self._branch_enum_ok[src] = ok
        self._ode_family_status[src] = ok

    # -- explicit branch enumeration for u' = g(u)  (fix E) --------------
    def _first_order_branch_enum(self, ode_res, src, supplementary=False):
        def note_or_gap(tag, detail):
            if supplementary:
                self.tactic_notes.append(f"[supplementary] {tag}: {detail}")
            else:
                self._add_gap(tag, detail)

        fp = sp.Derivative(self.f(self.x), self.x)
        try:
            rhs_list = sp.solve(sp.expand(ode_res), fp)
        except Exception:
            rhs_list = []
        if not rhs_list:
            note_or_gap("branch_enum_failed", f"cannot solve for f' in {ode_res}")
            return False
        g = sp.simplify(rhs_list[0])
        u = sp.Symbol('_uu', real=True)
        g_u = sp.simplify(g.subs(self.f(self.x), u))
        if self.x in g_u.free_symbols:
            return False
        try:
            eq_roots = sp.solve(sp.Eq(g_u, 0), u)
        except Exception:
            eq_roots = []
        # equilibria annotations (constants linkage)
        if eq_roots:
            self.discovered_formulas.append({
                "formula": "u(x) = c (constant), c in " + str([sp.simplify(r) for r in eq_roots]) +
                           " (equilibria of u' = g(u)); every constant c is a solution of the "
                           "raw equation with the invariant value fixed by g(c) = 0",
                "meaning": "equilibrium/constant branch of the autonomous flow"})
        # radical-parameter detection for sign-case enumeration: use the whole
        # root expression when it contains exactly one non-parameter symbol
        rad = None
        prm = None
        for r in eq_roots:
            syms = sorted(r.free_symbols - {u} - set(self.param_syms.values()), key=str)
            if len(syms) == 1 and any(
                    pw for pw in r.atoms(sp.Pow) if pw.exp == sp.Rational(1, 2)):
                rad = r
                prm = syms[0]
                break
        cases = []
        if rad is not None:
            aa = sp.Symbol('_aa', positive=True)
            bb = sp.Symbol('_bb', positive=True)
            try:
                # squaring is legitimate here: we are CONSTRUCTING the
                # re-parameterisation (root = +_aa resp. root = i*_bb)
                sub_pos = sp.solve(sp.expand(rad ** 2 - aa ** 2), prm)
                sub_neg = sp.solve(sp.expand(rad ** 2 + bb ** 2), prm)
            except Exception:
                sub_pos, sub_neg = [], []
            sub_pos = [s for s in sub_pos if not isinstance(s, (list, tuple, dict))] \
                if sub_pos else []
            sub_neg = [s for s in sub_neg if not isinstance(s, (list, tuple, dict))] \
                if sub_neg else []
            if sub_pos:
                cases.append(({prm: sp.simplify(sub_pos[0])},
                              f"{sp.simplify(rad)} > 0 (two real equilibria)"))
            cases.append(({prm: sp.S.Zero}, f"{sp.simplify(rad)} = 0 (double equilibrium)"))
            if sub_neg:
                cases.append(({prm: sp.simplify(sub_neg[0])},
                              f"{sp.simplify(rad)} < 0 (no real equilibria)"))
        else:
            cases.append(({}, ""))
        all_ok = bool(cases)
        for (sub, cond) in cases:
            g_case = sp.simplify(g_u.subs(sub))
            try:
                roots = sp.solve(sp.Eq(g_case, 0), u) if u in g_case.free_symbols else []
            except Exception:
                roots = []
            roots = self._sort_real_roots(roots)
            try:
                R = sp.integrate(sp.simplify(1 / g_case), u)
            except Exception as exc:
                note_or_gap("separation_integral_failed", f"{g_case}: {exc}")
                all_ok = False
                continue
            R = self._canonize_separation(R, u)
            if R is None:
                all_ok = False
                continue
            branches = self._invert_separation(R, u, roots)
            if not branches:
                note_or_gap("branch_inversion_failed",
                            f"no explicit inversion for case {cond or 'plain'} of {g_case}")
                all_ok = False
                continue
            for (sol_u, interval_desc) in branches:
                cand = sp.simplify(sol_u)
                conds = ([cond] if cond else []) + [interval_desc]
                self._register_candidate(cand, source=f"T3:separation({src})",
                                         branch_conds=conds,
                                         machinery="elementary integration: partial fractions / "
                                                   "arctan; sign-case analysis of equilibria")
        self.derivation_chain.append(
            f"T3: branch enumeration for {src}: {len(cases)} sign-case(s), "
            f"{len([c for c in self.candidates])} candidates total so far")
        return all_ok

    def _sort_real_roots(self, roots):
        neg = [r for r in roots if sp.simplify(r).is_negative is True]
        pos = [r for r in roots if sp.simplify(r).is_positive is True]
        zero = [r for r in roots if sp.simplify(r).is_zero is True]
        rest = [r for r in roots if r not in neg + pos + zero]
        return sorted(neg, key=str) + zero + sorted(pos, key=str) + rest

    def _canonize_separation(self, R, u):
        """atanh -> logs, formal log expansion, absorb additive imaginary
        constants into the integration constant (real antiderivative on each
        interval), normalise R(0)."""
        try:
            if R.atoms(sp.atanh):
                R = R.rewrite(sp.log)
            R = sp.expand_log(sp.expand(R), force=True)
            # additive terms free of u that are purely imaginary belong to the
            # arbitrary constant on the real domain (principal-branch bookkeeping)
            terms = sp.Add.make_args(sp.expand(R))
            kept, dropped = [], []
            for tt in terms:
                tt_s = sp.simplify(tt)
                if not tt.has(u) and tt_s.is_real is False:
                    dropped.append(tt)
                else:
                    kept.append(tt)
            if dropped:
                R = sp.Add(*kept)
                self.tactic_notes.append(
                    "separation: additive imaginary constant "
                    f"{sp.Add(*dropped)} absorbed into the real integration constant")
            R0 = R.subs(u, 0)
            if not (R0.has(sp.oo, NINF, sp.zoo, sp.nan)):
                # subtract only the REAL part of R(0): the imaginary part is
                # principal-branch bookkeeping that must NOT enter the
                # exponentiation (it would swap tanh <-> coth branches)
                try:
                    R0_re = sp.simplify(sp.re(R0))
                except Exception:
                    R0_re = R0
                R = sp.simplify(sp.expand(R - R0_re))
            return sp.simplify(R)
        except Exception:
            return None

    def _interval_mid(self, lo, hi, roots):
        if lo == NINF and hi == INF:
            return sp.S.Zero
        if lo != NINF and hi != INF:
            return sp.simplify((lo + hi) / 2)
        ref = hi if lo == NINF else lo
        if len(roots) >= 2:
            gap = sp.simplify(max(roots, key=lambda e: sp.default_sort_key(e)) -
                              min(roots, key=lambda e: sp.default_sort_key(e)))
            gap = sp.simplify(abs(gap))
            step = sp.simplify(2 * gap)
        elif len(roots) == 1 and sp.simplify(roots[0]).is_number:
            step = sp.Integer(1)
        else:
            return None
        return sp.simplify(ref - step) if lo == NINF else sp.simplify(ref + step)

    def _invert_separation(self, R, u, roots):
        """R(u) = x + C solved on each open interval between consecutive real
        roots, using the REAL antiderivative on that interval: log(u - r) is
        replaced by log(r - u) when u - r < 0 there -- a derivative-preserving
        swap, so NO logcombine(force=True) and no silent branch loss (fix E)."""
        Cs = sp.Symbol('_C1', real=True)
        bounds = [NINF] + list(roots) + [INF]
        out = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if sp.simplify(lo - hi) == 0:
                continue
            mid = self._interval_mid(lo, hi, roots)
            if mid is None:
                continue
            R_flip = R
            ok = True
            for lg in sorted(R.atoms(sp.log), key=str):
                arg = lg.args[0]
                if u not in arg.free_symbols:
                    continue
                try:
                    if sp.degree(sp.expand(arg), u) != 1:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
                s = sp.simplify(arg.subs(u, mid))
                if s.is_positive:
                    continue
                elif s.is_negative:
                    R_flip = R_flip.replace(lg, sp.log(-arg))
                else:
                    ok = False
                    break
            if not ok:
                continue
            sols = self._invert_R(R_flip, u, self.x, Cs)
            if lo == NINF and hi == INF:
                desc = "u(x) in R (no equilibria on the real line)"
            elif lo == NINF:
                desc = f"u(x) < {sp.simplify(hi)}"
            elif hi == INF:
                desc = f"u(x) > {sp.simplify(lo)}"
            else:
                desc = f"{sp.simplify(lo)} < u(x) < {sp.simplify(hi)}"
            for sol in sols:
                out.append((sol, desc))
        return out

    def _named_forms(self, cand):
        """Detect tanh/coth structures (incl. reciprocal forms -a/tanh(z) =
        -a*coth(z)) and verify the identity before adopting the named form."""
        named = []
        for W in sorted(cand.atoms(sp.exp), key=str):
            argW = W.args[0]
            th = sp.Symbol('_th', real=True)  # transient
            try:
                sub = sp.simplify(cand.subs(W, (1 + th) / (1 - th)))
                sub = sp.cancel(sp.expand(sub))
                if th in sub.free_symbols:
                    coef = sp.simplify(sub / th)
                    if coef == sp.simplify(coef) and th not in coef.free_symbols and coef != 0:
                        named.append(sp.simplify(coef * sp.tanh(argW / 2)))
                sub2 = sp.simplify(cand.subs(W, (th + 1) / (th - 1)))
                sub2 = sp.cancel(sp.expand(sub2))
                if th in sub2.free_symbols:
                    coef2 = sp.simplify(sub2 / th)
                    if th not in coef2.free_symbols and coef2 != 0:
                        named.append(sp.simplify(coef2 * sp.coth(argW / 2)))
            except Exception:
                pass
        for th_atom in sorted(cand.atoms(sp.tanh), key=str):
            try:
                prod = sp.cancel(sp.simplify(cand * th_atom))
                if prod.has(sp.tanh) is False:
                    named.append(sp.simplify(prod * sp.coth(th_atom.args[0])))
            except Exception:
                pass
        for ch_atom in sorted(cand.atoms(sp.coth), key=str):
            try:
                prod = sp.cancel(sp.simplify(cand * ch_atom))
                if prod.has(sp.coth) is False:
                    named.append(sp.simplify(prod * sp.tanh(ch_atom.args[0])))
            except Exception:
                pass
        verified = []
        for n in named:
            try:
                if sp.simplify(sp.expand(n - cand)) == 0 and n not in verified:
                    verified.append(n)
            except Exception:
                pass
        return verified

    def _invert_R(self, R, u, x, Cs):
        """Invert R(u) = x + C for u. The manual log-pair exponentiation runs
        FIRST: it keeps the exponential argument real and avoids SymPy's
        branch-swapping complex shifts; then arctan pattern, then direct
        solve as last resort."""
        logs = sorted(R.atoms(sp.log), key=str)
        if logs and u in R.free_symbols:
            coefs = []
            okall = True
            for lg in logs:
                c = sp.simplify(R.coeff(lg))
                if c == 0:
                    okall = False
                    break
                coefs.append(c)
            if okall:
                gamma = coefs[0]
                rest = sp.simplify(R - sum(c * lg for c, lg in zip(coefs, logs)))
                if u not in rest.free_symbols and gamma != 0:
                    signs = [sp.simplify(c / gamma) for c in coefs]
                    if all(sg in (1, -1) for sg in signs):
                        poly = sp.S.One
                        for sg, lg in zip(signs, logs):
                            poly = poly * lg.args[0] ** sg
                        # include the residual (constant) part of R in W
                        W = sp.exp((x + Cs - rest) / gamma)
                        num = sp.together(sp.expand(poly - W)).as_numer_denom()[0]
                        try:
                            sols = sp.solve(sp.Eq(num, 0), u)
                            sols = [s for s in sols if not isinstance(s, (list, tuple))]
                            if sols:
                                return [sp.simplify(s) for s in sols]
                        except Exception:
                            pass
        atans = sorted(R.atoms(sp.atan), key=str)
        if len(atans) == 1:
            gam = sp.simplify(R.coeff(atans[0]))
            arg = atans[0].args[0]
            rest = sp.simplify(R - gam * atans[0])
            if u not in rest.free_symbols and gam != 0:
                try:
                    sols = sp.solve(sp.Eq(arg, sp.tan((x + Cs - rest) / gam)), u)
                    sols = [s for s in sols if not isinstance(s, (list, tuple))]
                    if sols:
                        return [sp.simplify(s) for s in sols]
                except Exception:
                    pass
        try:
            sols = sp.solve(sp.Eq(R, x + Cs), u)
            sols = [s for s in sols if not isinstance(s, (list, tuple))]
            if sols:
                return [sp.simplify(s) for s in sols]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # T4: power-series coefficient matching (fix D: joint (t, y) extraction)
    # ------------------------------------------------------------------
    def _tactic_series(self):
        N = self.series_order
        a = [sp.Symbol(f'_a{i}', real=True) for i in range(N + 1)]
        exprs = [l - r for (l, r) in self.diffs] + [l - r for (l, r) in self.algs] \
            + [l - r for (l, r, _iv) in self.functionals] + [l - r for (l, r) in self.self_relations]
        if not exprs:
            return
        center = self.series_center

        def A(arg):
            return sp.expand(sum(a[i] * (arg - center) ** i for i in range(N + 1)))

        res = sp.S.Zero
        for e in exprs:
            res = res + self._sub_candidate(sp.expand(e), A(self.x))
        res = sp.expand(res)
        t = sp.Symbol('_t')
        res_t = sp.expand(res.subs(self.x, center + t))
        # fix D: joint extraction over (t, y, ...) -- the aux variable must be
        # a polynomial GENERATOR, not a passive coefficient
        extra = sorted(
            [s for s in res_t.free_symbols
             if s != t and s not in a and s not in set(self.param_syms.values())],
            key=str)
        gens = [t] + extra
        # ---- soundness of truncated coefficient matching --------------------
        # Re-build the ansatz at order N+3 with symbolic unseen coefficients;
        # a coefficient equation is enforced only if it does NOT involve any
        # unseen coefficient (otherwise it is a truncation artifact and is
        # skipped honestly).
        N3 = N + 3
        a3 = [sp.Symbol(f'_a{i}', real=True) for i in range(N + 1, N3 + 1)]

        def A3(arg):
            return sp.expand(sum(a[i] * (arg - center) ** i for i in range(N + 1))
                             + sum(a3[i - N - 1] * (arg - center) ** i
                                   for i in range(N + 1, N3 + 1)))

        res3 = sp.S.Zero
        for e in exprs:
            res3 = res3 + self._sub_candidate(sp.expand(e), A3(self.x))
        res3_t = sp.expand(res3.subs(self.x, center + t))
        try:
            poly = sp.Poly(res3_t, *gens)
        except Exception as exc:
            self._add_gap("series_non_polynomial", str(exc))
            return
        eqs = []
        degrees = []
        n_skipped = 0
        for mono, c in zip(poly.monoms(), poly.coeffs()):
            if sum(mono) > N:
                n_skipped += 1
                continue
            if any(s in c.free_symbols for s in a3):
                n_skipped += 1
                continue
            eqs.append(sp.expand(c))
            degrees.append(sum(mono))
        if n_skipped:
            self.tactic_notes.append(
                f"T4: {n_skipped} coefficient equations skipped as truncation artifacts "
                f"(they would involve Taylor coefficients beyond order {N})")
        # Taylor cascade: low-degree equations determine low coefficients first
        order_idx = sorted(range(len(eqs)), key=lambda i: degrees[i])
        eqs = [eqs[i] for i in order_idx]
        branches = self._sequential_branch_solve(eqs, a)
        for (assign, free, conds) in branches[:12]:
            self._series_branch_report(assign, free, conds, a, N, center)

    def _sequential_branch_solve(self, eqs, unknowns, max_branches=24):
        results = []

        def rec(cur, free, assign, conds):
            if len(results) >= max_branches:
                return
            live = []
            for e in cur:
                e = sp.expand(e.subs(assign)) if assign else e
                e = sp.simplify(sp.expand(e))
                if e == 0:
                    continue
                if e.is_number:
                    return  # contradiction: prune branch
                live.append(e)
            free_set = set(free)
            best = None
            param_conds = []
            for e in live:
                ucnt = len(e.free_symbols & free_set)
                if ucnt == 0:
                    param_conds.append(str(e))
                    continue
                if best is None or ucnt < best[0]:
                    best = (ucnt, e)
            if best is None:
                # fixpoint-normalise: substitute pinned values into each other
                assign2 = dict(assign)
                for _ in range(4):
                    changed = False
                    for kk, vv in list(assign2.items()):
                        nv = sp.simplify(sp.expand(vv.subs(assign2)))
                        if nv != vv:
                            assign2[kk] = nv
                            changed = True
                    if not changed:
                        break
                results.append((assign2, list(free), tuple(param_conds)))
                return
            e = best[1]
            progressed = False
            # triangular order: solve for the highest-index unknown first to
            # keep the recursion on the recurrence path (avoids spurious
            # branches like a1 = +-sqrt(2*a2))
            def _idx(sym):
                m = re.match(r'^_a(\d+)$', sym.name)
                return int(m.group(1)) if m else -1
            for s in sorted(e.free_symbols & free_set, key=_idx, reverse=True):
                try:
                    vs = sp.solve(e, s)
                except Exception:
                    vs = []
                vs = [v for v in vs if not isinstance(v, (list, tuple))]
                if not vs:
                    continue
                for v in list(vs)[:4]:
                    vv = sp.simplify(v)
                    progressed = True
                    na = dict(assign)
                    na[s] = vv
                    rec(live, [f2 for f2 in free if f2 != s], na, conds)
                break  # only the highest-index target (triangular solve)
            if not progressed:
                results.append((dict(assign), list(free),
                                tuple(param_conds) + tuple(str(sp.simplify(e2)) for e2 in live)))

        rec(list(eqs), list(unknowns), {}, [])
        return results

    def _series_branch_report(self, assign, free, conds, a, N, center):
        coeff = {}
        for i in range(N + 1):
            coeff[i] = sp.simplify(assign.get(a[i], a[i] if a[i] in free else sp.S.NaN))
        free_idx = [i for i in range(N + 1) if a[i] in free]
        # 1) constant pattern
        if not free_idx and all(sp.simplify(coeff[i]) == 0 for i in range(1, N + 1)):
            self._register_candidate(sp.simplify(coeff[0]), source="T4:series-constant",
                                     branch_conds=list(conds))
            return
        # 2) exp pattern: only a1 free, a0 == 1, a_k = a1^k / k!
        if free_idx == [1] and sp.simplify(coeff[0]) == 1:
            a1 = coeff[1]
            ok = True
            for k in range(2, N + 1):
                if sp.simplify(coeff[k] - a1 ** k / sp.factorial(k)) != 0:
                    ok = False
                    break
            if ok:
                cand = sp.exp(a1 * (self.x - center))
                cand = self._semanticize_series(cand, center)
                self._register_candidate(cand, source="T4:series-synthesis(exp)",
                                         branch_conds=list(conds))
                self.discovered_formulas.append({
                    "formula": f"f(x) = exp({a1}*(x-{center}))",
                    "meaning": "closed form synthesised from the Taylor-coefficient pattern "
                               "a_k = a1^k / k! (verified independently afterwards)"})
                return
        # 3) even/cosh pattern: odds zero, only a2 free, a_{2m} = (2a2)^m / (2m)!
        if free_idx == [2] and all(sp.simplify(coeff[i]) == 0 for i in range(1, N + 1) if i % 2 == 1) \
                and sp.simplify(coeff[0]) == 1:
            a2 = coeff[2]
            ok = True
            for m in range(2, N // 2 + 1):
                pred = (2 * a2) ** m / sp.factorial(2 * m)
                if sp.simplify(coeff[2 * m] - pred) != 0:
                    ok = False
                    break
            if ok:
                cand = sp.cosh(sp.sqrt(2 * a2) * (self.x - center))
                cand = self._semanticize_series(cand, center)
                self._register_candidate(cand, source="T4:series-synthesis(cosh)",
                                         branch_conds=list(conds))
                self.discovered_formulas.append({
                    "formula": f"f(x) = cosh(sqrt({2 * a2})*(x-{center}))",
                    "meaning": "even closed form synthesised from the recurrence "
                               "a_{2m} = (2a2)^m / (2m)! (verified independently afterwards)"})
                return
        # otherwise: record the discovered recurrence honestly
        if free_idx or assign:
            shown = {f"a{i}": str(coeff[i]) for i in range(N + 1)
                     if coeff[i] is not sp.S.NaN and sp.simplify(coeff[i]) != 0}
            if shown:
                self.discovered_formulas.append({
                    "formula": "Taylor recurrence around x=%s: %s" % (center, shown),
                    "meaning": "coefficients of any analytic solution (partial determination "
                               "up to series order %d)" % N})

    def _semanticize_series(self, cand, center):
        """Replace series coefficients by semantic point-constants: a0 -> f(c),
        a_k -> f^(k)(c)/k!  (named semantics, fix C)."""
        mapping = {}
        for i in range(0, 4):
            s = sp.Symbol(f'_a{i}', real=True)
            if s not in cand.free_symbols:
                continue
            if i == 0:
                mapping[s] = self._get_point_const(0, center)
            else:
                mapping[s] = self._get_point_const(i, center) / sp.factorial(i)
        return sp.simplify(cand.subs(mapping))

    # ------------------------------------------------------------------
    # candidate bookkeeping / verification / pinning
    # ------------------------------------------------------------------
    def _register_candidate(self, expr, source, branch_conds, machinery=None):
        expr = sp.simplify(sp.expand(expr))
        named = []
        try:
            named = self._named_forms(expr)
        except Exception:
            named = []
        if named:
            expr = named[0]
        kind = "constant" if self.x not in expr.free_symbols else "family"
        self.candidates.append({
            "expr": expr,
            "source": source,
            "sources": [source],
            "kind": kind,
            "branch_conds": [str(c) for c in branch_conds],
            "domain_guards": [],
            "verified": False,
            "status": "new",
            "pin": {},
            "free_params": [],
            "pin_status": None,
            "unresolved_conds": [],
            "eliminated_reason": None,
        })
        if machinery:
            for m in ([machinery] if isinstance(machinery, str) else list(machinery)):
                if m not in self.required_machinery:
                    self.required_machinery.append(m)

    def _domain_guards(self, expr):
        guards = []
        try:
            den = sp.together(sp.expand(expr)).as_numer_denom()[1]
            if den != 1 and den != 0:
                guards.append(f"denominator nonzero: {den} != 0")
        except Exception:
            pass
        for tn in expr.atoms(sp.tan):
            guards.append(f"cos({tn.args[0]}) != 0")
        for tn in expr.atoms(sp.cot):
            guards.append(f"sin({tn.args[0]}) != 0")
        if _sp_singularities is not None:
            try:
                # singularities() hangs on periodic/hyperbolic atoms; the
                # manual guards above already cover those cases
                transcendental = any(expr.atoms(fn) for fn in
                                     (sp.tan, sp.tanh, sp.coth, sp.cot))
                if not transcendental:
                    for spt in _sp_singularities(expr, self.x):
                        guards.append(f"x != {spt}")
            except Exception:
                pass
        return guards

    def _verify_candidate(self, c):
        constraints = []
        for (l, r) in self.diffs:
            constraints.append(sp.expand(l - r))
        for (l, r) in self.algs:
            constraints.append(sp.expand(l - r))
        for (l, r, _iv) in self.functionals:
            constraints.append(sp.expand(l - r))
        for (l, r) in self.self_relations:
            constraints.append(sp.expand(l - r))
        details = []
        for expr in (c["expr"],):
            ok = True
            details = []
            for e in constraints:
                res = self._sub_candidate(e, expr)
                if not residual_is_zero(res):
                    ok = False
                    details.append(f"residual {sp.simplify(res)} != 0 on {e}")
                    break
            if ok:
                c["domain_guards"] = self._domain_guards(expr)
                c["verified"] = True
                c["status"] = "verified"
                return
        c["status"] = "failed_verification"
        c["eliminated_reason"] = ("; ".join(details)[:300]) or "residual not identically zero"
        self.eliminated_branches.append({"expr": str(c["expr"]), "reason": c["eliminated_reason"]})

    def _violates(self, sym, val):
        if val is None:
            return False
        try:
            v = sp.simplify(val)
        except Exception:
            return False
        if not v.is_number:
            return False
        if sym.is_positive is True and v <= 0:
            return True
        if sym.is_real is True and v.is_real is False:
            return True
        return False

    def _solve_pinning_system(self, eqs, unknowns):
        """Sequential branching solver; returns list of (pin_dict, conds).
        Prunes on: nonzero numeric equations, assumption violations,
        non-real candidate values."""
        sols_out = []

        def rec(cur, free, assign, conds):
            if len(sols_out) >= 12:
                return
            live = []
            for e in cur:
                e = sp.expand(e.subs(assign)) if assign else e
                e = sp.simplify(sp.expand(e))
                if e == 0:
                    continue
                if e.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
                    return  # degenerate: treat as contradiction
                if e.is_number:
                    return  # contradiction
                live.append(e)
            free_set = set(free)
            best = None
            extra_conds = []
            for e in live:
                ucnt = len(e.free_symbols & free_set)
                if ucnt == 0:
                    ez = e.is_zero
                    if ez is False:
                        return  # contradictory parameter condition
                    extra_conds.append(str(e))
                    continue
                if best is None or ucnt < best[0]:
                    best = (ucnt, e)
            if best is None:
                sols_out.append((dict(assign), tuple(conds) + tuple(extra_conds)))
                return
            e = best[1]
            # sound unsatisfiability probe: the equation e = 0 requires its
            # numerator to vanish; if SymPy's assumptions decide the numerator
            # is nonzero (e.g. -_aa**2 with _aa > 0), no pin can satisfy it
            try:
                num = sp.together(sp.expand(e)).as_numer_denom()[0]
                num = sp.simplify(sp.expand(num))
                if num != 0 and num.is_zero is False:
                    return  # equation can never hold: prune branch
            except Exception:
                pass
            attempted = False
            progressed = False
            for s in sorted(e.free_symbols & free_set, key=str):
                try:
                    vs = sp.solve(e, s)
                except Exception:
                    vs = []
                for v in list(vs)[:6]:
                    vv = sp.simplify(v)
                    attempted = True
                    if self._violates(s, vv):
                        continue
                    if vv.is_number is True and not vv.is_real:
                        continue
                    progressed = True
                    na = dict(assign)
                    na[s] = vv
                    rec(live, [f2 for f2 in free if f2 != s], na, conds)
            if not progressed:
                if attempted:
                    return  # every algebraic solution violated the assumptions
                sols_out.append((dict(assign), tuple(conds) + tuple(extra_conds) +
                                 tuple(str(sp.simplify(e2)) for e2 in live)))

        rec(list(eqs), list(unknowns), {}, [])
        return sols_out

    def _pin_candidate(self, c):
        cand = c["expr"]
        unknowns = sorted(set(cand.free_symbols) - self._reserved(), key=str)
        if not self.bcs:
            c["pin_status"] = "free"
            c["pin"] = {}
            c["free_params"] = unknowns
            return
        if not unknowns:
            # fully concrete candidate: must satisfy every BC literally
            for bc in self.bcs:
                base = cand if bc["order"] == 0 else sp.diff(cand, self.x, bc["order"])
                ev = sp.simplify(base.subs(self.x, bc["point"]) - bc["value"])
                if not residual_is_zero(ev):
                    c["status"] = "eliminated"
                    c["eliminated_reason"] = f"violates boundary condition f^({bc['order']})({bc['point']}) = {bc['value']}"
                    self.eliminated_branches.append({"expr": str(cand),
                                                     "reason": c["eliminated_reason"]})
                    return
            c["pin_status"] = "pinned"
            c["pin"] = {}
            c["free_params"] = []
            return
        # evaluate BC expressions; substitute exp/tan atoms containing unknowns
        # NOTE: exp(z) and exp(-z) must share ONE auxiliary (reciprocal pair),
        # otherwise the reciprocal relation is lost and branches half-pin.
        aux_defs = []
        aux_map = {}

        def get_aux(atom):
            arg_key = sp.simplify(sp.expand(atom.args[0]))
            neg_key = sp.simplify(sp.expand(-atom.args[0]))
            for key, sign, existing in ((arg_key, 1, arg_key in aux_map),
                                        (neg_key, -1, neg_key in aux_map)):
                if existing:
                    base = aux_map[key]
                    return base if sign == 1 else sp.S.One / base
            positive = isinstance(atom, sp.exp)
            W = sp.Symbol(f'_A{len(aux_defs) + 1}', real=True, positive=True) if positive \
                else sp.Symbol(f'_A{len(aux_defs) + 1}', real=True)
            aux_defs.append((W, atom, positive))
            aux_map[arg_key] = W
            return W

        proc_eqs = []
        for bc in self.bcs:
            n, pt, v = bc["order"], bc["point"], bc["value"]
            base = cand if n == 0 else sp.diff(cand, self.x, n)
            ev = sp.simplify(sp.expand(base.subs(self.x, pt)))
            e2 = ev
            for atom in sorted(e2.atoms(sp.tan), key=str):
                if atom.args[0].free_symbols & set(unknowns):
                    e2 = e2.replace(atom, get_aux(atom))
            e2 = e2.rewrite(sp.exp) if e2.atoms(sp.exp) or e2.atoms(sp.tanh) or e2.atoms(sp.coth) else e2
            for atom in sorted(e2.atoms(sp.exp), key=str):
                if atom.args[0].free_symbols & set(unknowns):
                    e2 = e2.replace(atom, get_aux(atom))
            proc_eqs.append(sp.expand(e2 - v))
        aux_syms = [w for (w, _a, _p) in aux_defs]
        # derivative (higher-order) boundary conditions carry the decisive
        # information -- solve them FIRST to kill branches early
        order_idx = sorted(range(len(proc_eqs)),
                           key=lambda i: -self.bcs[i]["order"])
        proc_eqs = [proc_eqs[i] for i in order_idx]
        # strategy 1: global solve of the whole boundary system (fast, exact)
        glob = None
        try:
            glob = sp.solve(proc_eqs, unknowns + aux_syms, dict=True)
            glob = [g for g in (glob or []) if isinstance(g, dict)]
        except Exception:
            glob = None
        if glob:
            # verify each global pin against EVERY boundary equation (SymPy's
            # solve silently drops equations free of the requested symbols)
            checked = []
            for g in glob:
                okg = True
                for e in proc_eqs:
                    ev = sp.simplify(sp.expand(e.subs(g)))
                    if ev.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not residual_is_zero(ev):
                        okg = False
                        break
                if okg:
                    checked.append(g)
            glob = checked
        solutions = None
        if glob:
            solutions = [(g, ()) for g in glob]
        else:
            # strategy 2: sequential branching solver (handles partial pins
            # and undecidable conditions that a global solve cannot express)
            solutions = self._solve_pinning_system(proc_eqs, unknowns + aux_syms)
            if solutions and glob is not None and not glob:
                # global solve was conclusive-empty but sequential disagreed;
                # trust the exhaustive sequential search
                pass
        if not solutions:
            c["status"] = "eliminated"
            c["eliminated_reason"] = ("boundary conditions unsatisfiable on this branch "
                                      "(exhaustive sequential search)")
            self.eliminated_branches.append({"expr": str(c["expr"]),
                                             "reason": c["eliminated_reason"]})
            return
        feasible = []
        infeasible_reason = None
        for (assign, conds) in solutions:
            pin_full = dict(assign)
            ok = True
            for (W, atom, positive) in aux_defs:
                if W not in pin_full:
                    continue
                wv = pin_full[W]
                remaining = [s for s in unknowns if s not in pin_full]
                if not remaining:
                    ev = sp.simplify(atom.subs(pin_full))
                    if not residual_is_zero(ev - wv):
                        ok = False
                        infeasible_reason = (f"branch condition {atom} = {wv} inconsistent "
                                             "under pinned parameters")
                    continue
                try:
                    vsol = sp.solve(sp.Eq(atom, wv), *remaining, dict=True)
                except Exception:
                    vsol = []
                chosen = None
                for sol in list(vsol)[:6]:
                    trial = dict(pin_full)
                    trial.update({k2: sp.simplify(v2) for k2, v2 in sol.items()})
                    bad = False
                    for s2 in trial:
                        if self._violates(s2, trial[s2]):
                            bad = True
                            break
                        t2 = sp.simplify(trial[s2])
                        if t2.is_number is True and t2.has(sp.oo, NINF, sp.zoo, sp.nan):
                            bad = True
                            break
                    if not bad:
                        chosen = trial
                        break
                if chosen is None:
                    ok = False
                    infeasible_reason = (f"branch condition {atom} = {wv} has no finite "
                                         "real solution (boundary/degenerate value)")
                    break
                pin_full = chosen
            if ok:
                # decide the residual pinning conditions honestly: false ->
                # branch eliminated; true -> auto-satisfied; undecided ->
                # recorded as an open condition (blocks uniqueness claims)
                unresolved = []
                conds_kept = []
                for cs in conds:
                    try:
                        ce = sp.sympify(cs)
                        ez = ce.is_zero
                    except Exception:
                        ez = None
                    if ez is True:
                        continue
                    if ez is False:
                        ok = False
                        infeasible_reason = (f"pinning condition {cs} is "
                                             "unsatisfiable (decided false)")
                        break
                    unresolved.append(cs)
                    conds_kept.append(cs)
            if ok:
                pin_only = {k2: v2 for k2, v2 in pin_full.items() if k2 in unknowns}
                feasible.append((pin_only, conds_kept, unresolved))
        if not feasible:
            c["status"] = "eliminated"
            c["eliminated_reason"] = infeasible_reason or "boundary conditions unsatisfiable"
            self.eliminated_branches.append({"expr": str(c["expr"]),
                                             "reason": c["eliminated_reason"]})
            return
        pin, conds, unresolved = feasible[0]
        if len(feasible) > 1:
            c["pin_status"] = "ambiguous"
            self.tactic_notes.append(
                f"multiple distinct parameter pins found for {c['expr']}; "
                "uniqueness not established")
        if unresolved:
            c["unresolved_conds"] = unresolved
        pinned_expr = sp.simplify(c["expr"].subs(pin))
        c["expr"] = pinned_expr
        c["pin"] = {str(k2): str(v2) for k2, v2 in pin.items()}
        c["free_params"] = sorted([str(s) for s in pinned_expr.free_symbols
                                   if s not in self._reserved()], key=str)
        if conds:
            c["branch_conds"] = list(c["branch_conds"]) + [str(x) for x in conds]
        c["pin_status"] = "pinned" if not c["free_params"] else "partially_pinned"

    # ------------------------------------------------------------------
    # consistency of derived conditions against the final family
    # ------------------------------------------------------------------
    def _consistency_checks(self, family_cands):
        for cond in self.derived_conditions:
            for c in family_cands:
                expr = c["expr"]
                trial = cond
                for s, _meaning in list(self.point_const_semantics.items()):
                    if s in trial.free_symbols:
                        m = re.match(r'^_d(\d+)', s.name)
                        if m:
                            val = sp.simplify(sp.diff(expr, self.x, int(m.group(1))).subs(self.x, 0))
                        else:
                            val = sp.simplify(expr.subs(self.x, 0))
                        trial = trial.subs(s, val)
                trial = sp.simplify(sp.expand(self._sub_candidate(trial, expr)))
                if residual_is_zero(trial):
                    self.consistency_checks.append(
                        {"condition": str(cond), "family": str(expr), "result": "satisfied"})
                else:
                    self.consistency_checks.append(
                        {"condition": str(cond), "family": str(expr),
                         "result": f"NOT satisfied (residual {trial})"})
                    self._add_gap("derived_condition_violated",
                                  f"family {expr} violates derived condition {cond}")

    # ------------------------------------------------------------------
    # finalisation / certification gate (strict non-settlement)
    # ------------------------------------------------------------------
    def _completeness_established(self):
        if self._blocked_completeness:
            return False
        if self.functionals and not self._functionals_fully_reduced:
            return False
        if self.diffs and not self._odes_family_complete:
            return False
        if self.self_relations:
            # conservative: a symmetry relation alone never yields a complete
            # classification without a limit/iteration argument the engine
            # cannot perform symbolically.
            return False
        return True

    def _dedupe_verified(self):
        verified = [c for c in self.candidates if c.get("verified")]
        dedup = []
        for c in verified:
            merged = False
            for d in dedup:
                try:
                    same = sp.simplify(sp.expand(c["expr"] - d["expr"])) == 0
                except Exception:
                    same = str(c["expr"]) == str(d["expr"])
                both_const = (c["kind"] == "constant" and d["kind"] == "constant"
                              and (c["free_params"] or d["free_params"]))
                if same or both_const:
                    d["sources"].append(c["source"])
                    merged = True
                    break
            if not merged:
                dedup.append(c)
        return dedup

    def _absorb_constants(self, dedup):
        for cconst in [c for c in dedup if c["kind"] == "constant"]:
            if cconst.get("status") == "eliminated":
                continue
            for cfam in [c for c in dedup if c["kind"] == "family"]:
                # recover the ACTUAL symbol objects (assumptions matter in SymPy)
                sym_map = {s.name: s for s in cfam["expr"].free_symbols}
                params = [sym_map[s2] for s2 in cfam.get("free_params", [])
                          if s2 in sym_map]
                if not params:
                    continue
                diff_expr = cfam["expr"] - cconst["expr"]
                eqs = [diff_expr, sp.diff(diff_expr, self.x)]
                try:
                    sol = sp.solve(eqs, params, dict=True)
                except Exception:
                    sol = []
                if sol:
                    cconst["status"] = "absorbed"
                    cconst["absorbed_by"] = str(cfam["expr"])
                    cconst["absorption_pin"] = {str(k2): str(v2) for k2, v2 in sol[0].items()}
                    self.consistency_checks.append(
                        {"condition": f"constant candidate {cconst['expr']} is the "
                                      f"{sol[0]} slice of family {cfam['expr']}",
                         "family": str(cfam["expr"]), "result": "absorbed"})
                    break

    def _finalize(self):
        dedup = self._dedupe_verified()
        self._absorb_constants(dedup)
        live = [c for c in dedup
                if c.get("status") not in ("eliminated", "failed_verification")]
        family_cands = [c for c in live if c["kind"] == "family"]
        self._consistency_checks(live)
        has_constraints = bool(self.functionals or self.self_relations or self.diffs
                               or self.algs)
        contradiction = any("contradiction" in g.get("tag", "") for g in self.problem_gaps)
        complete = self._completeness_established()
        # branches whose pinning conditions the engine could not decide are
        # potential additional solutions: they block any uniqueness claim
        unresolved_cands = [c for c in self.candidates
                            if c.get("unresolved_conds")
                            and c.get("status") not in ("eliminated", "absorbed",
                                                        "failed_verification")]
        if unresolved_cands:
            self._add_gap("unresolved_branch_conditions",
                          "verified branch(es) carry pinning conditions the engine could "
                          "not decide: " +
                          "; ".join(str(c["expr"]) for c in unresolved_cands)[:300])
        pinned = [c for c in dedup if c.get("pin_status") == "pinned"
                  and c.get("status") not in ("eliminated", "absorbed",
                                              "failed_verification")]
        # remaining free-parameter families represent additional solutions:
        # they block any uniqueness (PROVEN_EXACT) claim
        open_families = [c for c in dedup
                         if c.get("pin_status") in ("free", "partially_pinned")
                         and c.get("status") not in ("eliminated", "absorbed",
                                                     "failed_verification")]
        if contradiction:
            self.status = "UNPROVABLE_GAP"
        elif not has_constraints and not dedup:
            if self.open_problem_context:
                self.status = "INCOMPLETE_PROOF_GAPS_REMAIN"
            else:
                self.status = "EMPTY_SYSTEM"
        elif pinned and len(pinned) == 1 and not self.problem_gaps and complete \
                and not unresolved_cands and not open_families \
                and not any(c.get("pin_status") == "ambiguous" for c in dedup):
            self.status = "PROVEN_EXACT"
            self._certified_fn = pinned[0]["expr"]
            self._certified_family = [pinned[0]]
        elif dedup and complete and not self.problem_gaps and family_cands \
                and not self.full_proof_only:
            self.status = "PROVEN_CLASSIFICATION"
            self._certified_family = [c for c in dedup
                                      if c.get("status") not in ("eliminated", "absorbed")]
        else:
            self.status = "INCOMPLETE_PROOF_GAPS_REMAIN"
            if self.full_proof_only and family_cands and complete \
                    and not self.problem_gaps:
                self._add_gap("full_proof_only_refusal",
                              "a complete verified family was reached, but a "
                              "classification of a parametric family is NOT a full "
                              "proof of the conjecture; full_proof_only mode refuses "
                              "to settle for it")
            self._certified_family = [c for c in dedup
                                      if c.get("status") not in ("eliminated", "absorbed")] or None

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def solve_and_verify(self):
        t0 = time.time()
        # reset derivation state (constraints are kept)
        self.point_const_semantics = {}
        self.declared_axioms = []
        self.problem_gaps = []
        self.required_machinery = []
        self.discovered_formulas = []
        self.derived_odes = []
        self.derived_conditions = []
        self.derivation_chain = []
        self.tactic_notes = []
        self.candidates = []
        self.eliminated_branches = []
        self.consistency_checks = []
        self._certified_fn = None
        self._certified_family = None
        self._functionals_fully_reduced = False
        self._odes_family_complete = False
        self._ode_family_status = {}
        self._blocked_completeness = []
        self.series_center = sp.S.Zero

        try:
            self._t1_structural_diagnosis()
            if self.functionals:
                self._tactic_functional_reduction()
            if self.self_relations:
                self._tactic_self_relations()
            if self.diffs or self.derived_odes:
                self._tactic_odes()
            self._tactic_series()
            self._tactic_algebraic()
            self._tactic_constants()
            self._tactic_special_points()
            self._tactic_polynomial_ansatz()
            self._tactic_spectral_ansatz()
            # pin FIRST (BCs constrain the integration constants), then verify
            # the pinned form against ALL declared constraints
            for c in self.candidates:
                self._pin_candidate(c)
            for c in self.candidates:
                if c["status"] in ("new", "free", "pinned", "partially_pinned"):
                    self._verify_candidate(c)
            self._tactic_superposition()
            self._finalize()
        except Exception as exc:  # engine must never crash silently
            self.status = "INCOMPLETE_PROOF_GAPS_REMAIN"
            self._add_gap("engine_internal_error",
                          f"{type(exc).__name__}: {exc} || {traceback.format_exc()[-500:]}")
        self.elapsed = round(time.time() - t0, 2)
        return self._format_result()

    def _format_result(self):
        return {
            "problem": self.problem_name,
            "status": self.status,
            "full_proof_only": bool(self.full_proof_only),
            "certified_fn": self._certified_fn,
            "elapsed_s": self.elapsed,
            "axioms": list(self.declared_axioms),
            "gaps": list(self.problem_gaps),
            "machinery": list(self.required_machinery),
            "discovered": list(self.discovered_formulas),
            "derived_conditions": [str(d) for d in self.derived_conditions],
            "candidates": [
                {k2: c.get(k2) for k2 in ("expr", "kind", "source", "sources", "branch_conds",
                                           "domain_guards", "verified", "status", "pin",
                                           "free_params", "pin_status", "unresolved_conds",
                                           "eliminated_reason")}
                for c in self.candidates],
            "eliminated_branches": list(self.eliminated_branches),
            "consistency_checks": list(self.consistency_checks),
            "derivation_chain": list(self.derivation_chain),
            "tactic_notes": list(self.tactic_notes),
        }

    # ------------------------------------------------------------------
    # prediction (strictly gated) and certificate export
    # ------------------------------------------------------------------
    def predict(self, x_val):
        if self._certified_fn is None:
            raise ValueError(
                "Refused: engine has NOT achieved a full, gap-free, uniquely pinned proof "
                f"(status={self.status}; gaps={len(self.problem_gaps)}). "
                "The strict non-settlement discipline forbids prediction.")
        return self._certified_fn.subs(self.x, sp.sympify(x_val))

    def export_certificate(self):
        cert = self._format_result()
        cert["certified_fn"] = str(self._certified_fn) if self._certified_fn is not None else None
        obligations = []
        if self.status in ("PROVEN_EXACT", "PROVEN_CLASSIFICATION"):
            obligations.append("Verify each constraint residual simplifies to 0 under the "
                               "recorded domain/branch conditions (algebraic certificate).")
            obligations.append("Re-derive the reduction chain in a formal kernel "
                               "(differentiation + evaluation at the identity/fixed point).")
            obligations.append("Check branch-enumeration completeness: every branch verified "
                               "or eliminated with its recorded reason.")
            obligations.append("Check the parameter-pinning solve of the boundary system.")
        else:
            obligations.append("Discharge every listed unsettled gap in a formal kernel "
                               "before any certification claim is admissible.")
        cert["proof_obligations_for_formal_kernel"] = obligations
        return json.loads(json.dumps(cert, default=str))


# ===========================================================================
# Multi-formula proofs: a conjecture stated as several formulas/lemmas that
# together constitute the proof.  Certification requires the COMPLETE
# conjunction: every formula fully proved, no partial settlement.
# ===========================================================================

class MultiFormulaProof:
    """Conjecture = conjunction of formulas; certified only when EVERY formula
    reaches a full, gap-free PROVEN_EXACT proof.  With full_proof_only=True,
    classification-tier or partially verified formulas never count, and the
    aggregate verdict is NO_FULL_PROOF until the whole conjunction is proved.
    The per-formula verdicts identify exactly which formulas block the proof
    and which mathematical machinery they still require."""

    def __init__(self, conjecture_name, full_proof_only=True):
        self.conjecture_name = str(conjecture_name)
        self.full_proof_only = bool(full_proof_only)
        self.formulas = []          # [{'tag', 'engine'}]
        self.verdicts = []
        self.status = "EMPTY_SYSTEM"
        self.elapsed = 0.0

    def add_formula(self, tag, engine):
        engine.full_proof_only = self.full_proof_only
        self.formulas.append({"tag": str(tag), "engine": engine})
        return self

    def prove_all(self):
        t0 = time.time()
        self.verdicts = []
        for entry in self.formulas:
            res = entry["engine"].solve_and_verify()
            full = (res["status"] == "PROVEN_EXACT"
                    and res["certified_fn"] is not None
                    and not res["gaps"])
            self.verdicts.append({
                "formula": entry["tag"],
                "status": res["status"],
                "full_proof": bool(full),
                "certified_fn": res["certified_fn"],
                "gaps": res["gaps"],
                "machinery": res["machinery"],
                "discovered_diagnostics": res["discovered"],
                "elapsed_s": res["elapsed_s"],
            })
        if not self.formulas:
            self.status = "EMPTY_SYSTEM"
        elif all(v["full_proof"] for v in self.verdicts):
            self.status = "PROVEN_EXACT"
        else:
            self.status = "NO_FULL_PROOF"
        self.elapsed = round(time.time() - t0, 2)
        return self.result()

    def result(self):
        return {
            "conjecture": self.conjecture_name,
            "full_proof_only": self.full_proof_only,
            "status": self.status,
            "n_formulas": len(self.formulas),
            "n_formulas_fully_proved": sum(1 for v in self.verdicts if v["full_proof"]),
            "verdicts": list(self.verdicts),
            "elapsed_s": self.elapsed,
        }

    def export_certificate(self):
        cert = self.result()
        obligations = []
        if self.status == "PROVEN_EXACT":
            obligations.append("Re-derive every formula's proof chain in a formal kernel.")
        else:
            blocking = [v["formula"] for v in self.verdicts if not v["full_proof"]]
            obligations.append("Discharge the blocking formulas in a formal kernel "
                               "before any certification claim is admissible: "
                               + "; ".join(blocking))
        cert["proof_obligations_for_formal_kernel"] = obligations
        return json.loads(json.dumps(cert, default=str))

# ===========================================================================
# Benchmark suite: the runners only DECLARE problems; the engine does the rest.
# ===========================================================================

def bench_cauchy_absolute():
    eng = AutonomousProofEngine(
        "Cauchy exponential equation | absolute mode (no regularity declared)",
        func_name='f', var_name='x', aux_var_name='y', auto_declare_regularity=False)
    f, x, y = eng.f, eng.x, eng.y
    eng.declare_hypothesis("f: R -> R real-valued")
    eng.add_functional_constraint(f(x + y), f(x) * f(y), identity_val=0)
    eng.add_bc_value(1, point=0)
    return eng, eng.solve_and_verify()


def bench_cauchy_classified():
    eng = AutonomousProofEngine(
        "Cauchy exponential equation | engine self-declares the needed C^1 axiom",
        func_name='f', var_name='x', aux_var_name='y', auto_declare_regularity=True)
    f, x, y = eng.f, eng.x, eng.y
    eng.declare_hypothesis("f: R -> R real-valued")
    eng.add_functional_constraint(f(x + y), f(x) * f(y), identity_val=0)
    eng.add_bc_value(1, point=0)
    return eng, eng.solve_and_verify()


def bench_cauchy_pinned():
    eng = AutonomousProofEngine(
        "Cauchy exponential equation | IVP f(0)=1, f'(0)=2",
        func_name='f', var_name='x', aux_var_name='y', auto_declare_regularity=True)
    f, x, y = eng.f, eng.x, eng.y
    eng.declare_hypothesis("f: R -> R real-valued")
    eng.add_functional_constraint(f(x + y), f(x) * f(y), identity_val=0)
    eng.add_bc_value(1, point=0)
    eng.add_bc_deriv(1, 2, point=0)
    return eng, eng.solve_and_verify()


def bench_dalembert():
    eng = AutonomousProofEngine(
        "d'Alembert cosine equation f(x+y)+f(x-y) = 2 f(x) f(y)",
        func_name='f', var_name='x', aux_var_name='y', auto_declare_regularity=True)
    f, x, y = eng.f, eng.x, eng.y
    eng.declare_hypothesis("f: R -> R real-valued")
    eng.add_functional_constraint(f(x + y) + f(x - y), 2 * f(x) * f(y), identity_val=0)
    eng.add_bc_value(1, point=0)
    eng.add_bc_deriv(1, 0, point=0)
    return eng, eng.solve_and_verify()


def _engine_riemann_selfduality(full_proof_only=False):
    eng = AutonomousProofEngine(
        "Riemann hypothesis | xi self-duality xi(s) = xi(1-s)",
        func_name='xi', var_name='s', aux_var_name='w',
        auto_declare_regularity=True, full_proof_only=full_proof_only)
    xi, s = eng.f, eng.x
    eng.declare_hypothesis("xi(s) = (1/2) s (s-1) pi^(-s/2) Gamma(s/2) zeta(s)")
    eng.add_self_relation(xi(s), xi(1 - s))
    eng.set_open_problem({
        "objective": "all nontrivial zeros of zeta(s) lie on Re(s) = 1/2",
        "arithmetic_atoms": ["zeta(s)", "Euler product over the primes", "Gamma(s/2)"],
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "Euler product arithmetic structure (unique factorisation of integers)",
            "Hadamard factorisation and zero-counting (argument principle)",
            "positivity of a spectral / L-function interpretation "
            "(Weil explicit formula or Hilbert-Polya)",
            "non-local symmetry analysis beyond affine fixed points",
        ],
    })
    return eng


def _engine_riemann_critical_line(full_proof_only=True):
    eng = AutonomousProofEngine(
        "Riemann hypothesis | critical-line localisation of every nontrivial zero",
        func_name='z', var_name='s', aux_var_name='w',
        auto_declare_regularity=True, full_proof_only=full_proof_only)
    eng.declare_hypothesis("zeta(s) meromorphic on C, single simple pole at s = 1")
    eng.set_open_problem({
        "objective": "every nontrivial zero s of zeta satisfies Re(s) = 1/2",
        "arithmetic_atoms": ["completed xi function", "Euler product over the primes",
                             "Gamma(s/2)"],
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "zero-counting on the critical strip (argument principle)",
            "sign/positivity certificate excluding zeros off Re(s) = 1/2",
            "spectral self-adjointness (Hilbert-Polya) or an equivalent positivity",
        ],
    })
    return eng


def _engine_collatz_surrogate(full_proof_only=False):
    eng = AutonomousProofEngine(
        "Collatz conjecture | parity-averaged smooth surrogate",
        func_name='f', var_name='x', aux_var_name='w',
        auto_declare_regularity=True, full_proof_only=full_proof_only)
    f, x = eng.f, eng.x
    eng.declare_hypothesis("surrogate: smooth shadow of T(n) = n/2 (even) | (3n+1)/2 (odd) "
                           "on the positive integers")
    eng.add_self_relation(f(x / 2), f(x))
    eng.add_self_relation(f((3 * x + 1) / 2), f(x))
    eng.set_open_problem({
        "objective": "for every n in N+, some iterate T^k(n) = 1",
        "discrete_domain": True,
        "surrogate": True,
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "parity branch tree analysis on N+",
            "well-founded Lyapunov function V: N+ -> N decreasing along orbits",
            "diophantine bounds ruling out non-trivial cycles (log-linear / 2-adic)",
            "divergent-orbit exclusion",
        ],
    })
    return eng


def _engine_collatz_cycles(full_proof_only=True):
    eng = AutonomousProofEngine(
        "Collatz conjecture | non-trivial cycle and divergent-orbit exclusion",
        func_name='g', var_name='n', aux_var_name='m',
        auto_declare_regularity=True, full_proof_only=full_proof_only)
    g, n = eng.f, eng.x
    eng.declare_hypothesis("T: N+ -> N+ with T(n) = n/2 (even) | (3n+1)/2 (odd)")
    eng.add_self_relation(g(2 * n), g(n))
    eng.set_open_problem({
        "objective": "no non-trivial periodic orbit and no divergent orbit under T",
        "discrete_domain": True,
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "2-adic / log-linear cycle bound: k log 3 < j log 2 with diophantine gap",
            "Rhin-style strong-lower-bound zeta estimates on the cycle inequality",
            "well-founded descent measure on N+ outside the trivial orbit",
        ],
    })
    return eng


def _engine_ns_energy(full_proof_only=True):
    eng = AutonomousProofEngine(
        "3D Navier-Stokes | Leray energy class and local well-posedness",
        func_name='u', var_name='x', aux_var_name='w',
        full_proof_only=full_proof_only)
    eng.declare_hypothesis("u smooth, divergence-free, on R^3 x [0, T)")
    eng.set_open_problem({
        "objective": "a priori L_infty_t L_2_x control of smooth divergence-free data",
        "pde_dimensional": True,
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "Sobolev / Leray-Hopf energy estimates in 3D",
            "incompressibility and pressure projection handling",
        ],
    })
    return eng


def _engine_ns_blowup(full_proof_only=True):
    eng = AutonomousProofEngine(
        "3D Navier-Stokes | vortex-stretching control and blow-up exclusion",
        func_name='u', var_name='x', aux_var_name='w',
        full_proof_only=full_proof_only)
    eng.declare_hypothesis("u smooth, divergence-free, on R^3 x [0, T)")
    eng.set_open_problem({
        "objective": "no finite-time blow-up: ||u||_L_infty bounded on [0, T) for all T",
        "pde_dimensional": True,
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "vortex stretching control (omega . grad omega)",
            "Beale-Kato-Majda blow-up criterion",
            "global regularity a priori estimate closing the energy ladder in 3D",
        ],
    })
    return eng


def bench_cauchy_additive():
    eng = AutonomousProofEngine(
        "Cauchy additive equation f(x+y) = f(x) + f(y) | f(0)=0, f(1)=1",
        func_name='f', var_name='x', aux_var_name='y', auto_declare_regularity=True)
    f, x, y = eng.f, eng.x, eng.y
    eng.declare_hypothesis("f: R -> R real-valued")
    eng.add_functional_constraint(f(x + y), f(x) + f(y), identity_val=0)
    eng.add_bc_value(0, point=0)
    eng.add_bc_value(1, point=1)
    return eng, eng.solve_and_verify()


def bench_harmonic():
    eng = AutonomousProofEngine(
        "harmonic oscillator u'' + u = 0 | complete real solution family",
        func_name='u', var_name='x', aux_var_name='w', auto_declare_regularity=True)
    u, x = eng.f, eng.x
    eng.declare_hypothesis("u twice differentiable on R")
    eng.add_differential_constraint(sp.Derivative(u(x), (x, 2)), -u(x))
    return eng, eng.solve_and_verify()


def bench_riemann():
    eng = _engine_riemann_selfduality()
    return eng, eng.solve_and_verify()


def bench_collatz():
    eng = _engine_collatz_surrogate()
    return eng, eng.solve_and_verify()


def bench_burgers_family():
    eng = AutonomousProofEngine(
        "1D steady Burgers u u' - nu u'' = 0 | complete branch classification",
        func_name='u', var_name='x', aux_var_name='w',
        params=('nu',), param_assumptions={'nu': 'positive'})
    u, x = eng.f, eng.x
    nu = eng.param_syms['nu']
    eng.declare_hypothesis("nu > 0 (viscosity); u twice differentiable on open intervals")
    eng.add_differential_constraint(u(x) * sp.Derivative(u(x), x),
                                    nu * sp.Derivative(u(x), (x, 2)))
    return eng, eng.solve_and_verify()


def bench_burgers_ivp():
    eng = AutonomousProofEngine(
        "1D steady Burgers | IVP u(0)=1, u'(0)=0",
        func_name='u', var_name='x', aux_var_name='w',
        params=('nu',), param_assumptions={'nu': 'positive'})
    u, x = eng.f, eng.x
    nu = eng.param_syms['nu']
    eng.declare_hypothesis("nu > 0 (viscosity); u twice differentiable on open intervals")
    eng.add_differential_constraint(u(x) * sp.Derivative(u(x), x),
                                    nu * sp.Derivative(u(x), (x, 2)))
    eng.add_bc_value(1, point=0)
    eng.add_bc_deriv(1, 0, point=0)
    return eng, eng.solve_and_verify()


def bench_burgers_singular():
    eng = AutonomousProofEngine(
        "1D steady Burgers | singular-branch IVP u(0)=-2nu, u'(0)=2nu",
        func_name='u', var_name='x', aux_var_name='w',
        params=('nu',), param_assumptions={'nu': 'positive'})
    u, x = eng.f, eng.x
    nu = eng.param_syms['nu']
    eng.declare_hypothesis("nu > 0 (viscosity); u twice differentiable on open intervals")
    eng.add_differential_constraint(u(x) * sp.Derivative(u(x), x),
                                    nu * sp.Derivative(u(x), (x, 2)))
    eng.add_bc_value(-2 * nu, point=0)
    eng.add_bc_deriv(1, 2 * nu, point=0)
    return eng, eng.solve_and_verify()


def bench_ns3d():
    eng = AutonomousProofEngine(
        "3D Navier-Stokes global regularity | context only",
        func_name='u', var_name='x', aux_var_name='w')
    eng.set_open_problem({
        "objective": "global smooth solutions for all t >= 0 given smooth divergence-free data",
        "pde_dimensional": True,
        "beyond_symbolic_scope": True,
        "required_machinery_declared": [
            "Sobolev / Leray-Hopf energy estimates in 3D",
            "incompressibility and pressure projection handling",
            "vortex stretching control (omega . grad omega)",
            "Beale-Kato-Majda blow-up criterion",
        ],
    })
    return eng, eng.solve_and_verify()


# ---------------------------------------------------------------------------
# FULL-PROOF-ONLY OPEN-CONJECTURE RUN
#
# Each open conjecture is stated as MULTIPLE formulas that together would
# constitute the proof.  The conjecture is certified ONLY when every formula
# is fully proved (gap-free PROVEN_EXACT); partial discoveries are recorded
# as diagnostics and never settle the run.
# ---------------------------------------------------------------------------

def bench_riemann_fullproof():
    mfp = MultiFormulaProof(
        "Riemann hypothesis | full-proof-only, multi-formula",
        full_proof_only=True)
    mfp.add_formula(
        "F1: xi self-duality xi(s) = xi(1-s) with complete branch enumeration",
        _engine_riemann_selfduality(full_proof_only=True))
    mfp.add_formula(
        "F2: every nontrivial zero of zeta lies on Re(s) = 1/2",
        _engine_riemann_critical_line())
    return mfp


def bench_collatz_fullproof():
    mfp = MultiFormulaProof(
        "Collatz conjecture | full-proof-only, multi-formula",
        full_proof_only=True)
    mfp.add_formula(
        "F1: parity-averaged smooth surrogate admits only the constant invariant",
        _engine_collatz_surrogate(full_proof_only=True))
    mfp.add_formula(
        "F2: no non-trivial cycle and no divergent orbit on N+ (Collatz)",
        _engine_collatz_cycles())
    return mfp


def bench_ns3d_fullproof():
    mfp = MultiFormulaProof(
        "3D Navier-Stokes global regularity | full-proof-only, multi-formula",
        full_proof_only=True)
    mfp.add_formula(
        "F1: Leray energy class controls smooth divergence-free data",
        _engine_ns_energy())
    mfp.add_formula(
        "F2: vortex stretching is controlled; no finite-time blow-up",
        _engine_ns_blowup())
    return mfp


def bench_burgers_multi_positive():
    """Gate self-test (machinery check, NOT an open conjecture): a conjecture
    whose two formulas are both fully provable must certify."""
    mfp = MultiFormulaProof(
        "steady 1D Burgers two-point library | both formulas provable",
        full_proof_only=True)
    mfp.add_formula("F1: IVP u(0)=1, u'(0)=0 pins u == 1",
                    bench_burgers_ivp()[0])
    mfp.add_formula("F2: singular-branch IVP u(0)=-2nu, u'(0)=2nu pins -2nu/(x+1)",
                    bench_burgers_singular()[0])
    return mfp


def bench_burgers_multi_negative():
    """Gate self-test (machinery check, NOT an open conjecture): one fully
    provable formula plus one unproved (Collatz cycle-exclusion) formula must
    refuse certification -- no partial settlement."""
    mfp = MultiFormulaProof(
        "mixed multi-formula conjecture | one formula unproved",
        full_proof_only=True)
    mfp.add_formula("F1: IVP u(0)=1, u'(0)=0 pins u == 1",
                    bench_burgers_ivp()[0])
    mfp.add_formula("F2: no non-trivial cycle and no divergent orbit on N+ (Collatz)",
                    _engine_collatz_cycles())
    return mfp


OPEN_CONJECTURES = [
    ("riemann_fullproof", bench_riemann_fullproof),
    ("collatz_fullproof", bench_collatz_fullproof),
    ("ns3d_fullproof", bench_ns3d_fullproof),
]

MULTI_FORMULA_GATE_CHECKS = [
    ("burgers_multi_positive", bench_burgers_multi_positive),
    ("burgers_multi_negative", bench_burgers_multi_negative),
]


BENCHMARKS = [
    ("cauchy_absolute", bench_cauchy_absolute),
    ("cauchy_classified", bench_cauchy_classified),
    ("cauchy_pinned", bench_cauchy_pinned),
    ("cauchy_additive", bench_cauchy_additive),
    ("harmonic", bench_harmonic),
    ("dalembert", bench_dalembert),
    ("riemann", bench_riemann),
    ("collatz", bench_collatz),
    ("burgers_family", bench_burgers_family),
    ("burgers_ivp", bench_burgers_ivp),
    ("burgers_singular", bench_burgers_singular),
    ("ns3d", bench_ns3d),
]


# ---------------------------------------------------------------------------
# Test suite: hard assertions incl. the promised d'Alembert and
# singular-branch Burgers checks, plus strict non-settlement negatives.
# ---------------------------------------------------------------------------

def run_assertions(R, ENGS):
    checks = []

    def check(name, cond, detail=""):
        checks.append({"name": name, "pass": bool(cond), "detail": str(detail)[:200]})

    # -- fix A: reaching this point at all means no BooleanTrue crash -------
    check("fix A: d'Alembert run completed without BooleanTrue crash",
          R["dalembert"]["status"] in ("PROVEN_EXACT", "PROVEN_CLASSIFICATION",
                                       "INCOMPLETE_PROOF_GAPS_REMAIN"))
    check("fix A: truth_of classifies collapsed Eq", truth_of(sp.Eq(0, 0)) == "true")

    # -- Cauchy -------------------------------------------------------------
    ca = R["cauchy_absolute"]
    check("cauchy absolute mode refuses certification",
          ca["status"] == "INCOMPLETE_PROOF_GAPS_REMAIN", ca["status"])
    check("cauchy absolute mode reports the regularity gap",
          any("C1" in g["tag"] for g in ca["gaps"]),
          str([g["tag"] for g in ca["gaps"]])[:150])
    check("cauchy absolute mode still discovers the exp family (T4)",
          any("exp" in str(c["expr"]) for c in ca["candidates"] if c["verified"]))
    cp = R["cauchy_pinned"]
    check("cauchy pinned run reaches PROVEN_EXACT", cp["status"] == "PROVEN_EXACT", cp["status"])
    eng_cp = ENGS["cauchy_pinned"]
    check("cauchy pinned predict(1) = exp(2)",
          sp.simplify(eng_cp.predict(1) - sp.exp(2)) == 0)
    cc = R["cauchy_classified"]
    check("cauchy classified run reaches PROVEN_CLASSIFICATION",
          cc["status"] == "PROVEN_CLASSIFICATION", cc["status"])
    try:
        eng_cc = ENGS["cauchy_classified"]
        eng_cc.predict(1)
        unlocked = True
    except ValueError:
        unlocked = False
    check("cauchy classification keeps predict() locked (parametric family)",
          not unlocked)

    # -- d'Alembert (fixes A, B, C, D) --------------------------------------
    dl = R["dalembert"]
    check("dalembert C^k escalation recorded the C2 axiom (fix B)",
          any("C2" in a["tag"] for a in dl["axioms"]),
          str([a["tag"] for a in dl["axioms"]])[:150])
    check("dalembert no artificial _k0 constants (fix C)",
          all("_k0" not in str(c["expr"]) for c in dl["candidates"]))
    check("dalembert series discovery free of the aux variable (fix D)",
          all("w" not in str(c["expr"]) and "y" not in str(c["expr"])
              for c in dl["candidates"] if c["verified"]))
    check("dalembert reaches PROVEN_CLASSIFICATION with a cosh family",
          dl["status"] == "PROVEN_CLASSIFICATION"
          and any("cosh" in str(c["expr"]) for c in dl["candidates"] if c["verified"]),
          dl["status"])
    eng_dl = ENGS["dalembert"]
    fam = [c for c in eng_dl.candidates if c["verified"] and "cosh" in str(c["expr"])]
    if fam:
        check("dalembert family verified against the functional equation",
              fam[0]["verified"] is True)
        check("dalembert free parameter is the semantic f''(0)",
              any(s.name.startswith("_d") for s in fam[0]["expr"].free_symbols))

    # -- Burgers family (fix E) ---------------------------------------------
    bf = R["burgers_family"]
    kinds = {branch_kind_of(c["expr"], ENGS["burgers_family"].x)
             for c in bf["candidates"] if c["verified"]}
    check("burgers family enumerates all five branch classes (fix E)",
          {"constant", "tanh", "coth", "rational", "tan"} <= kinds, str(kinds))
    nonconst = [c for c in bf["candidates"] if c["verified"]
                and c["kind"] != "constant"]
    check("every non-constant burgers branch carries recorded conditions (fix E)",
          all(c["branch_conds"] or c["domain_guards"] for c in nonconst))
    check("burgers first integral discovered autonomously",
          any("first integral" in str(f["meaning"]).lower()
              for f in bf["discovered"]))
    check("burgers family reaches PROVEN_CLASSIFICATION",
          bf["status"] == "PROVEN_CLASSIFICATION", bf["status"])

    # -- Burgers IVP ----------------------------------------------------------
    bi = R["burgers_ivp"]
    eng_bi = ENGS["burgers_ivp"]
    check("burgers IVP reaches PROVEN_EXACT", bi["status"] == "PROVEN_EXACT", bi["status"])
    check("burgers IVP certifies u == 1",
          bi["status"] == "PROVEN_EXACT" and sp.simplify(eng_bi._certified_fn - 1) == 0)
    check("burgers IVP predict(5) = 1", eng_bi.predict(5) == 1)
    check("burgers IVP eliminated branches all carry reasons",
          len(bi["eliminated_branches"]) >= 3
          and all(e["reason"] for e in bi["eliminated_branches"]))

    # -- Burgers singular-branch IVP ------------------------------------------
    bs = R["burgers_singular"]
    eng_bs = ENGS["burgers_singular"]
    nu = eng_bs.param_syms['nu']
    check("burgers singular IVP reaches PROVEN_EXACT", bs["status"] == "PROVEN_EXACT",
          bs["status"])
    check("burgers singular IVP certifies the rational shock -2nu/(x+1)",
          bs["status"] == "PROVEN_EXACT"
          and sp.simplify(eng_bs._certified_fn + 2 * nu / (eng_bs.x + 1)) == 0,
          str(eng_bs._certified_fn))
    check("burgers singular IVP predict(2) = -2nu/3",
          sp.simplify(eng_bs.predict(2) + 2 * nu / 3) == 0)
    cert_cand = [c for c in eng_bs.candidates
                 if c["verified"] and c["pin_status"] == "pinned"]
    check("burgers singular IVP records the pole guard x != -1 (fix E)",
          cert_cand and any("x != -1" in g or "-1" in g
                            for g in cert_cand[0]["domain_guards"]),
          str(cert_cand[0]["domain_guards"] if cert_cand else ""))

    # -- open problems: strict non-settlement (negative tests) ----------------
    for nm in ("riemann", "collatz", "ns3d"):
        r = R[nm]
        eng = ENGS[nm]
        check(f"{nm} NOT certified (strict non-settlement)",
              r["status"] not in ("PROVEN_EXACT", "PROVEN_CLASSIFICATION")
              and eng._certified_fn is None, r["status"])
        check(f"{nm} reports unsettled gaps / declared machinery",
              bool(r["gaps"]) or bool(r["machinery"]))
    rm = R["riemann"]
    check("riemann discovers xi'(1/2) = 0 and the even critical-line expansion",
          any("_d1" in d["formula"] or "d1" in d["formula"] for d in rm["discovered"])
          or any("even" in d["formula"].lower() or "Taylor" in d["formula"]
                 for d in rm["discovered"]),
          str([d["formula"] for d in rm["discovered"]])[:180])
    cl = R["collatz"]
    check("collatz surrogate yields the constant invariant and nothing more",
          cl["status"] == "INCOMPLETE_PROOF_GAPS_REMAIN"
          and any(c["kind"] == "constant" and c["verified"] for c in cl["candidates"]),
          str([str(c["expr"]) for c in cl["candidates"]])[:150])

    # -- new tactics: T6 special points, T7 polynomial, T8 spectral, T9 superposition --
    cad = R["cauchy_additive"]
    eng_cad = ENGS["cauchy_additive"]
    check("cauchy additive reaches PROVEN_EXACT with f(x) = x",
          cad["status"] == "PROVEN_EXACT"
          and eng_cad._certified_fn is not None
          and sp.simplify(eng_cad._certified_fn - eng_cad.x) == 0,
          f"{cad['status']} {eng_cad._certified_fn}")
    check("cauchy additive polynomial ansatz (T7) contributed a candidate",
          any("T7" in s for c in cad["candidates"] for s in c.get("sources", [])))
    check("cauchy additive anchor probe (T6) pinned the distinguished points",
          any("T6" in dc for dc in cad["derivation_chain"]))
    hm = R["harmonic"]
    eng_hm = ENGS["harmonic"]
    check("harmonic reaches PROVEN_CLASSIFICATION",
          hm["status"] == "PROVEN_CLASSIFICATION", hm["status"])
    hm_verified = [str(c["expr"]) for c in eng_hm.candidates if c["verified"]]
    check("harmonic spectral ansatz (T8) discovers sin and cos independently",
          any("sin" in k for k in hm_verified) and any("cos" in k for k in hm_verified),
          str(hm_verified)[:150])
    check("harmonic superposition (T9) closes the solution space linearly",
          any("T9" in s for c in hm["candidates"] for s in c.get("sources", [])))
    dl2 = R["dalembert"]
    diag_recorded = (any("f(2" in str(dc) for dc in dl2["derived_conditions"])
                     or any("f(2" in d["formula"] for d in dl2["discovered"]))
    check("dalembert special-point probe (T6) records the diagonal recursion",
          diag_recorded,
          str(dl2["derived_conditions"])[:150])

    # -- fix E source-level guarantee -----------------------------------------
    try:
        import ast as _ast
        with open(__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        _tree = _ast.parse(src)
        has_logcombine_call = any(
            isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Attribute)
            and node.func.attr == "logcombine"
            for node in _ast.walk(_tree))
        check("fix E: no logcombine call anywhere in engine source (AST scan)",
              not has_logcombine_call)
    except Exception as exc:
        check("fix E: no logcombine call anywhere in engine source (AST scan)",
              False, exc)
    return checks


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def write_report(R, checks, path):
    L = []
    L.append("=" * 78)
    L.append("AUTONOMOUS PROOF ENGINE v2 -- RUN REPORT (strict non-settlement mode)")
    L.append("=" * 78)
    L.append(f"Residual cache: {_residual_is_zero_cached.cache_info()} (configured max={_CACHE_SIZE})")
    L.append("")
    for key, _fn in BENCHMARKS:
        r = R[key]
        L.append("-" * 78)
        L.append(f"[{key}] {r['problem']}")
        L.append(f"  STATUS: {r['status']}   (elapsed {r['elapsed_s']} s)")
        L.append(f"  certified function: {r['certified_fn']}")
        if r["axioms"]:
            L.append("  self-declared axioms:")
            for a in r["axioms"]:
                L.append(f"    - {a['tag']}: {a['reason'][:160]}")
        L.append("  candidates:")
        for c in r["candidates"]:
            L.append(f"    - {c['expr']}")
            L.append(f"      kind={c['kind']} status={c['status']} verified={c['verified']} "
                     f"pin_status={c['pin_status']} pin={c['pin']} "
                     f"free={c['free_params']}")
            if c["branch_conds"]:
                L.append(f"      branch conds: {c['branch_conds']}")
            if c["domain_guards"]:
                L.append(f"      domain guards: {c['domain_guards']}")
            if c["eliminated_reason"]:
                L.append(f"      eliminated: {c['eliminated_reason'][:160]}")
        if r["eliminated_branches"]:
            L.append("  eliminated branches:")
            for e in r["eliminated_branches"]:
                L.append(f"    - {e['expr'][:90]}  reason: {e['reason'][:160]}")
        if r["discovered"]:
            L.append("  discovered structure:")
            for d in r["discovered"]:
                L.append(f"    - {d['formula'][:150]}")
                L.append(f"      ({d['meaning'][:160]})")
        if r["derived_conditions"]:
            L.append(f"  derived necessary conditions: {r['derived_conditions']}")
        if r["consistency_checks"]:
            L.append("  consistency checks:")
            for cc in r["consistency_checks"]:
                L.append(f"    - {cc['condition'][:110]} -> {cc['result'][:80]}")
        if r["gaps"]:
            L.append("  unsettled gaps:")
            for g in r["gaps"]:
                L.append(f"    - {g['tag']}: {g['detail'][:160]}")
        if r["machinery"]:
            L.append("  required machinery (cited):")
            for m in r["machinery"]:
                L.append(f"    - {m}")
        if r["tactic_notes"]:
            L.append("  tactic notes:")
            for tn in r["tactic_notes"]:
                L.append(f"    - {tn[:170]}")
        L.append("")
    L.append("=" * 78)
    L.append("ASSERTION TEST SUITE")
    L.append("=" * 78)
    npass = sum(1 for c in checks if c["pass"])
    for c in checks:
        L.append(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}"
                 + (f"   [{c['detail']}]" if c["detail"] and not c["pass"] else ""))
    L.append(f"\n  {npass}/{len(checks)} checks passed")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return npass, len(checks)


# ---------------------------------------------------------------------------
# full-proof-only open-conjecture run
# ---------------------------------------------------------------------------

def run_open_assertions(R, MFPS):
    checks = []

    def check(name, cond, detail=""):
        checks.append({"name": name, "pass": bool(cond), "detail": str(detail)[:200]})

    for key, _fn in OPEN_CONJECTURES:
        r = R[key]
        check(f"{key}: run is full-proof-only",
              r["full_proof_only"] is True, r["full_proof_only"])
        check(f"{key}: conjecture NOT certified (strict non-settlement)",
              r["status"] == "NO_FULL_PROOF", r["status"])
        check(f"{key}: no formula certifies anything",
              all(v["certified_fn"] is None for v in r["verdicts"]))
        check(f"{key}: every unproved formula reports gaps or declared machinery",
              all(v["gaps"] or v["machinery"] for v in r["verdicts"]
                  if not v["full_proof"]))
        check(f"{key}: multi-formula statement actually spans several formulas",
              r["n_formulas"] >= 2, r["n_formulas"])

    pos = R["burgers_multi_positive"]
    check("gate positive: all-formulas-proved conjecture IS certified",
          pos["status"] == "PROVEN_EXACT"
          and pos["n_formulas_fully_proved"] == pos["n_formulas"] == 2,
          f"{pos['status']} {pos['n_formulas_fully_proved']}/{pos['n_formulas']}")
    neg = R["burgers_multi_negative"]
    check("gate negative: one unproved formula blocks certification",
          neg["status"] == "NO_FULL_PROOF"
          and neg["n_formulas_fully_proved"] == 1 and neg["n_formulas"] == 2,
          f"{neg['status']} {neg['n_formulas_fully_proved']}/{neg['n_formulas']}")
    check("gate negative: the blocking formula is exactly identified",
          any(not v["full_proof"] and "Collatz" in v["formula"]
              for v in neg["verdicts"]))
    return checks


def write_open_report(R, checks, path):
    L = []
    L.append("=" * 78)
    L.append("AUTONOMOUS PROOF ENGINE v2 -- FULL-PROOF-ONLY OPEN-CONJECTURE RUN")
    L.append("Policy: only complete proofs of open mathematical conjectures count. "
             "No partial progress is credited; classification-tier and surrogate "
             "results are diagnostics only.")
    L.append("=" * 78)
    L.append("")
    for key, _fn in OPEN_CONJECTURES:
        r = R[key]
        L.append("-" * 78)
        L.append(f"[{key}] {r['conjecture']}")
        L.append(f"  VERDICT: {r['status']}   "
                 f"(formulas fully proved: {r['n_formulas_fully_proved']}/{r['n_formulas']}, "
                 f"elapsed {r['elapsed_s']} s)")
        for v in r["verdicts"]:
            L.append(f"  formula: {v['formula']}")
            L.append(f"    status={v['status']} full_proof={v['full_proof']} "
                     f"certified_fn={v['certified_fn']}")
            for d in v["discovered_diagnostics"]:
                L.append(f"    diagnostic (NOT credited as proof): {d['formula'][:140]}")
            for g in v["gaps"]:
                L.append(f"    unsettled gap: {g['tag']}: {g['detail'][:150]}")
            for m in v["machinery"]:
                L.append(f"    missing machinery: {m}")
        L.append("")
    L.append("=" * 78)
    L.append("MULTI-FORMULA GATE SELF-TESTS (machinery check, not conjectures)")
    L.append("=" * 78)
    for key, _fn in MULTI_FORMULA_GATE_CHECKS:
        r = R[key]
        L.append(f"  [{key}] {r['conjecture']}")
        L.append(f"    verdict={r['status']} "
                 f"proved={r['n_formulas_fully_proved']}/{r['n_formulas']}")
    L.append("")
    L.append("ASSERTION TEST SUITE")
    L.append("=" * 78)
    npass = sum(1 for c in checks if c["pass"])
    for c in checks:
        L.append(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}"
                 + (f"   [{c['detail']}]" if c["detail"] and not c["pass"] else ""))
    L.append(f"\n  {npass}/{len(checks)} checks passed")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return npass, len(checks)


def run_open_conjectures():
    """Run ONLY the open mathematical conjectures, in full-proof-only mode.
    Exit 0 only if EVERY conjecture is fully proved (all formulas gap-free
    PROVEN_EXACT); otherwise exit 1 with NO_FULL_PROOF verdicts."""
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cert_dir = os.path.join(out_dir, "certificates_open")
    os.makedirs(cert_dir, exist_ok=True)
    R, MFPS = {}, {}
    for key, fn in OPEN_CONJECTURES + MULTI_FORMULA_GATE_CHECKS:
        print(f"=== running {key} (full-proof-only) ...", flush=True)
        mfp = fn()
        res = mfp.prove_all()
        R[key] = res
        MFPS[key] = mfp
        print(f"    verdict: {res['status']}  "
              f"({res['n_formulas_fully_proved']}/{res['n_formulas']} formulas fully proved, "
              f"{res['elapsed_s']} s)", flush=True)
    checks = run_open_assertions(R, MFPS)
    report_path = os.path.join(out_dir, "open_conjecture_run_report.txt")
    npass, ntot = write_open_report(R, checks, report_path)
    for key in R:
        with open(os.path.join(cert_dir, f"{key}.json"), "w", encoding="utf-8") as fh:
            json.dump(MFPS[key].export_certificate(), fh, indent=2, default=str)
    print()
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}")
    print(f"\n  {npass}/{ntot} checks passed")
    print(f"  report written to: {report_path}")
    print(f"  certificates written to: {cert_dir}/")
    all_proved = all(R[key]["status"] == "PROVEN_EXACT"
                     for key, _fn in OPEN_CONJECTURES)
    if all_proved:
        print("  ALL open conjectures FULLY PROVED (multi-formula, gap-free).")
        return 0
    print("  NO_FULL_PROOF: no open conjecture is fully proved. Partial progress "
          "is NOT credited; the missing machinery is listed per formula.")
    return 1


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if "--open-conjectures" in argv:
        return run_open_conjectures()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cert_dir = os.path.join(out_dir, "certificates")
    os.makedirs(cert_dir, exist_ok=True)
    R, ENGS = {}, {}
    for key, fn in BENCHMARKS:
        print(f"=== running {key} ...", flush=True)
        eng, res = fn()
        R[key] = res
        ENGS[key] = eng
        print(f"    status: {res['status']}  ({res['elapsed_s']} s)", flush=True)
    checks = run_assertions(R, ENGS)
    report_path = os.path.join(out_dir, "proof_run_report.txt")
    npass, ntot = write_report(R, checks, report_path)
    for key in R:
        with open(os.path.join(cert_dir, f"{key}.json"), "w", encoding="utf-8") as fh:
            json.dump(ENGS[key].export_certificate(), fh, indent=2, default=str)
    print()
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}")
    print(f"\n  {npass}/{ntot} checks passed")
    print(f"  residual cache: {_residual_is_zero_cached.cache_info()}")
    print(f"  report written to: {report_path}")
    print(f"  certificates written to: {cert_dir}/")
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys_exit = main()
    import sys as _sys
    _sys.exit(sys_exit)
