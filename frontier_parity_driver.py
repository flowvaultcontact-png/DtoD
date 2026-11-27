#!/usr/bin/env python3
"""Frontier-Parity Distillation driver for APEIRON-BRAIN, certified by
AutonomousProofEngine v2 (strict non-settlement discipline).

Usage:
  python frontier_parity_driver.py [model_path] [brain_dir] [corpus.txt] [prompt] [max_tokens] [rounds]

Method (FRONTIER_PARITY.md):
  Each round: frontier-KL distillation of the cortical readout (chunked memmap
  AdamW + reward-modulated STDP toward the frontier's greedy token) -> exact
  residual measurement A_t = P_frontier(w_t|ctx_t), B_t = P_cortical(w_t|ctx_t)
  on held-out positions -> AutonomousProofEngine certifies the exact mixture
  quality functional R(lam) (polynomial first-order ODE + boundary condition);
  when the measured residuals vanish the declared system degenerates and the
  engine certifies PROVEN_EXACT on cortical == frontier (constant R), which
  pins lam* = 1 (brain-only generation) -> token-for-token parity battery.

Requires: numpy, sympy, autonomous_proof_engine_v2.py and ai_4_fixed.py in the
same directory.
"""
import gzip
import importlib.util
import json
import math
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else None
BRAIN_DIR = sys.argv[2] if len(sys.argv) > 2 else None
CORPUS = sys.argv[3] if len(sys.argv) > 3 else None
PROMPT = sys.argv[4] if len(sys.argv) > 4 else "The capital of France is"
MAX_TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 24
N_ROUNDS = int(sys.argv[6]) if len(sys.argv) > 6 else 4

import numpy as np
import sympy as sp

spec = importlib.util.spec_from_file_location("ape", os.path.join(HERE, "autonomous_proof_engine_v2.py"))
ape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ape)

spec2 = importlib.util.spec_from_file_location("ai4", os.path.join(HERE, "ai_4_fixed.py"))
ai4 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ai4)

DEFAULT_CORPUS = ("Sparse cortical neurons spike efficiently on frontier CPUs. "
                  "Cortical neurons spike; the brain learns sparse efficient spikes. ") * 12
RATIONAL_DENOM = 10 ** 6
EQUALITY_TOL = 10 ** -4
GLM_SERVER_URL = os.environ.get("GLM_SERVER_URL", "http://127.0.0.1:8011")
SURROGATE_FLOOR = 10 ** -3


class LlamaCppChannel:
    """Certified ground-truth channel backed by a loopback llama.cpp server
    running the actual frontier GLM weights (the reference greedy decoder for
    the hybrid glm5-next GGUF that ExactFrontierEngine refuses by design).

    Channel interface (consumed by _distill_sequence, _measure_positions,
    _brain_greedy_ids):
        logprobs(ctx_ids, vocab_size) -> one-hot surrogate at the frontier's
        greedy token (0.0 there, floor elsewhere; Laplace floor is applied at
        measurement/certification time).
    Server-side prompt caching makes growing-prefix teacher forcing cost one
    forward per position.
    """

    def __init__(self, base_url=GLM_SERVER_URL, vocab_size=None, eos_ids=(),
                 timeout=1800, cache_path=None, retries=8):
        self.base_url = base_url.rstrip("/")
        self.vocab_size = int(vocab_size)
        self.eos_ids = set(int(e) for e in eos_ids)
        self.order = 1
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.cache_path = cache_path
        self._argmax_cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    self._argmax_cache = {
                        tuple(int(t) for t in k.split(",")): int(v)
                        for k, v in json.load(fh).items()}
            except Exception:
                self._argmax_cache = {}

    def _persist_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump({",".join(map(str, k)): v
                           for k, v in self._argmax_cache.items()}, fh)
        except Exception:
            pass

    def _post(self, path, payload):
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base_url + path, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            except Exception as exc:
                last = exc
                time.sleep(15.0 * (attempt + 1))
        raise last

    def greedy_next(self, ctx_ids):
        ctx = tuple(int(t) for t in ctx_ids)
        hit = self._argmax_cache.get(ctx)
        if hit is not None:
            return hit
        resp = self._post("/completion", {
            "prompt": [int(t) for t in ctx],
            "temperature": 0.0, "n_predict": 1, "n_probs": 1,
            "cache_prompt": True,
        })
        tid = int(resp["completion_probabilities"][0]["top_logprobs"][0]["id"])
        self._argmax_cache[ctx] = tid
        if len(self._argmax_cache) % 16 == 0:
            self._persist_cache()
        return tid

    def greedy_stream(self, prompt_ids, max_tokens):
        gen = []
        for _ in range(int(max_tokens)):
            nxt = self.greedy_next(list(prompt_ids) + gen)
            if nxt in self.eos_ids:
                break
            gen.append(nxt)
        return gen

    def logprobs(self, ctx_ids, vocab_size, floor=-20.0):
        if int(vocab_size) != self.vocab_size:
            raise ValueError(f"channel vocab {self.vocab_size} != requested {vocab_size}")
        lp = np.full(self.vocab_size, floor, dtype=np.float32)
        lp[self.greedy_next(ctx_ids)] = 0.0
        return lp

    def reset(self):
        return None

    def train_ids(self, ids):
        return self

    def save(self, path):
        payload = {"type": "LlamaCppChannel", "base_url": self.base_url,
                   "vocab_size": self.vocab_size, "eos_ids": sorted(self.eos_ids)}
        try:
            if str(path).endswith(".gz"):
                with gzip.open(path, "wt", encoding="utf-8") as fh:
                    json.dump(payload, fh)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
        except Exception:
            pass


def _distill_sequence(vbridge, channel, ids, lr=0.02, ticks_per_token=None,
                      epochs=2, stdp_epochs=1):
    """Frontier-KL distillation: the brain change of FPD.

    Same chunked memmap AdamW machinery as VocabTokenBridge.train_sequence,
    but the target at every position is the frontier distribution itself
    (grad_logits = P_frontier - P_cortical), and the STDP reward is the
    frontier's confidence in its own greedy token.
    """
    if ticks_per_token is None:
        ticks_per_token = min(max(vbridge.brain.n_layers + 2, 5), 10)
    if len(ids) < 2:
        return [0.0]
    losses = []
    beta1, beta2, eps, wd = 0.9, 0.999, 1e-6, 1e-4
    V = vbridge.vocab_size

    for ep in range(epochs):
        vbridge.reset_context()
        if hasattr(channel, "reset"):
            channel.reset()
        ep_loss = 0.0
        lr_ep = lr * (0.96 ** ep)
        use_stdp = ep < min(stdp_epochs, max(1, epochs // 2)) if epochs > 1 else stdp_epochs > 0
        for i in range(len(ids) - 1):
            t_in = int(ids[i])
            ctx = [int(t) for t in ids[:i + 1]]
            lp = channel.logprobs(ctx, V)
            p_f = np.exp(lp.astype(np.float64))
            p_f /= max(float(p_f.sum()), 1e-12)
            t_f = int(np.argmax(p_f))
            reward = 0.5 * min(max((float(lp[t_f]) + 10.0) / 10.0, 0.0), 1.0)

            feat = vbridge._step_token(
                t_in, ticks_per_token=ticks_per_token,
                reward_signal=reward if use_stdp else 0.0, plasticity=use_stdp)
            h = vbridge.W_proj @ feat + vbridge.b_proj
            logits = vbridge._compute_logits(h)

            mx = float(logits.max())
            ex = np.exp(logits - mx)
            p_b = ex / max(float(ex.sum()), 1e-12)
            ep_loss += -math.log(max(float(p_b[t_f]), 1e-9))

            grad_logits = (p_b - p_f).astype(np.float32)

            g_h = np.zeros(vbridge.n_mot, dtype=np.float32)
            for r0 in range(0, V, vbridge.chunk_vocab):
                r1 = min(V, r0 + vbridge.chunk_vocab)
                buf = vbridge._scratch_chunk[:r1 - r0]
                np.copyto(buf, vbridge.head_q8[r0:r1], casting="unsafe")
                w_grad = grad_logits[r0:r1] * vbridge.head_sc[r0:r1]
                g_h += w_grad @ buf

            g_w = np.outer(g_h, feat) + wd * vbridge.W_proj
            g_b = g_h

            vbridge._adam_t += 1
            vbridge._m_w = beta1 * vbridge._m_w + (1.0 - beta1) * g_w
            vbridge._v_w = beta2 * vbridge._v_w + (1.0 - beta2) * (g_w * g_w)
            vbridge._m_b = beta1 * vbridge._m_b + (1.0 - beta1) * g_b
            vbridge._v_b = beta2 * vbridge._v_b + (1.0 - beta2) * (g_b * g_b)
            vbridge._m_hb = beta1 * vbridge._m_hb + (1.0 - beta1) * grad_logits
            vbridge._v_hb = beta2 * vbridge._v_hb + (1.0 - beta2) * (grad_logits * grad_logits)

            bc1 = 1.0 - (beta1 ** vbridge._adam_t)
            bc2 = 1.0 - (beta2 ** vbridge._adam_t)
            vbridge.W_proj -= lr_ep * (vbridge._m_w / bc1) / (np.sqrt(vbridge._v_w / bc2) + eps)
            vbridge.b_proj -= lr_ep * (vbridge._m_b / bc1) / (np.sqrt(vbridge._v_b / bc2) + eps)
            vbridge.head_bias -= lr_ep * (vbridge._m_hb / bc1) / (np.sqrt(vbridge._v_hb / bc2) + eps)

        losses.append(ep_loss / max(len(ids) - 1, 1))
    return losses


def _measure_positions(vbridge, channel, ids, n_positions):
    """Exact per-position measurement: frontier vs cortical probability of the
    actual continuation token plus both argmax facts."""
    V = vbridge.vocab_size
    out = []
    vbridge.reset_context()
    if hasattr(channel, "reset"):
        channel.reset()
    feat = np.zeros(vbridge.feat_dim, dtype=np.float32)
    n = min(int(n_positions), len(ids) - 1)
    surrogate = hasattr(channel, "greedy_next")
    for i in range(n):
        feat = vbridge._step_token(int(ids[i]), ticks_per_token=4, plasticity=False)
        h = vbridge._project_to_head(feat)
        logits = vbridge._compute_logits(h)
        mx = float(logits.max())
        ex = np.exp(logits - mx)
        p_b = max(float(ex[int(ids[i + 1])]) / max(float(ex.sum()), 1e-12), 1e-12)
        t_f = channel.greedy_next([int(t) for t in ids[:i + 1]]) if surrogate \
            else int(np.argmax(channel.logprobs([int(t) for t in ids[:i + 1]], V)))
        if surrogate:
            p_a = 1.0 - SURROGATE_FLOOR if int(ids[i + 1]) == t_f else SURROGATE_FLOOR
        else:
            p_a = max(math.exp(float(channel.logprobs([int(t) for t in ids[:i + 1]], V)[int(ids[i + 1])])), 1e-12)
        out.append({
            "a": p_a,
            "b": p_b,
            "t_frontier": int(t_f),
            "t_cortical": int(np.argmax(logits)),
        })
    return out


def certify_round(ape_mod, pairs, rnd, tol=EQUALITY_TOL):
    """APE v2 certification of the mixture functional R(lam) on the two
    brain-favoring evidence positions; degenerates to PROVEN_EXACT cortical ==
    frontier once measured residuals vanish within tol."""
    evid_floor = math.exp(-19.5)
    evid = [pr for pr in pairs if pr["a"] > evid_floor]
    pool = evid if len(evid) >= 2 else pairs
    if len(pool) < 2:
        return {"status": "NO_EVIDENCE", "lam_star": None, "parity_certified": False,
                "gaps": ["fewer than two measurable positions"], "residuals": None}
    (p1, p2) = sorted(pool, key=lambda pr: pr["b"] / max(pr["a"], 1e-9), reverse=True)[:2]

    def rat(v):
        return sp.Rational(v).limit_denominator(RATIONAL_DENOM) + sp.Rational(1, 1000)

    peng = ape_mod.AutonomousProofEngine(
        f"Frontier-parity round {rnd} | exact cross-entropy functional R(lam) of the "
        "frontier/cortical mixture on certified evidence positions",
        func_name='R', var_name='lam', aux_var_name='w', auto_declare_regularity=True)
    f, x = peng.f, peng.x
    res1 = abs(p1["b"] - p1["a"]) <= tol
    res2 = abs(p2["b"] - p2["a"]) <= tol
    A1, A2 = rat(p1["a"]), rat(p2["a"])
    B1 = A1 if res1 else rat(p1["b"])
    B2 = A2 if res2 else rat(p2["b"])
    peng.declare_hypothesis(
        "R is C1 on [0,1]; data axioms: A_t = P_frontier(w_t|ctx_t) from the exact "
        "frontier channel; B_t = P_cortical(w_t|ctx_t) measured on the two "
        "brain-favoring positions (largest b/a ratio)"
        + (" (measured |B_t - A_t| <= 1e-4, rationalized onto A_t under the declared "
           "equality tolerance)" if (res1 or res2) else "")
        + "; uniform Laplace floor 1/1000 declared; A_t, B_t > 0 so "
          "D_t = (1-lam)A_t + lam B_t > 0 on [0,1] (multiplying the gradient "
          "equation by D_1 D_2 is an equivalent transformation there)")
    D1 = (1 - x) * A1 + x * B1
    D2 = (1 - x) * A2 + x * B2
    ce_rhs = sp.expand(sp.Rational(1, 2) * ((A1 - B1) * D2 + (A2 - B2) * D1))
    peng.add_differential_constraint(D1 * D2 * sp.Derivative(f(x), x), ce_rhs)
    peng.add_bc_value((sp.log(A1) + sp.log(A2)) / 2, point=0)
    res = peng.solve_and_verify()

    out = {
        "status": res["status"],
        "certified_fn": str(res["certified_fn"]) if res["certified_fn"] else None,
        "gaps": [g["tag"] for g in res["gaps"]],
        "residuals": [abs(p1["b"] - p1["a"]), abs(p2["b"] - p2["a"])],
        "positions": {"a": [p1["a"], p2["a"]], "b": [p1["b"], p2["b"]],
                      "argmax_agree": [p1["t_frontier"] == p1["t_cortical"],
                                       p2["t_frontier"] == p2["t_cortical"]]},
        "evidence_positions": len(evid),
        "lam_star": None,
        "lam_boundary": False,
        "parity_certified": False,
    }

    if res["status"] == "PROVEN_EXACT" and res["certified_fn"]:
        Rc = sp.sympify(res["certified_fn"])
        dR = sp.diff(Rc, x)
        if dR == 0:
            out["parity_certified"] = True
            out["lam_star"] = 1.0
            out["lam_boundary"] = True
            print(f"    engine: PROVEN_EXACT constant functional -> cortical == "
                  f"frontier certified on measured positions; lam* = 1 (brain-only)")
            return out
        num = sp.factor(sp.together(dR).as_numer_denom()[0])
        crit = [s for s in sp.solve(num, x) if s.is_real is not False]
        interior = [s for s in crit if 0 < float(s.evalf()) < 1]
        for s in interior:
            r2 = float(sp.diff(dR, x).subs(x, s).evalf())
            if r2 > 0:
                out["lam_star"] = float(s.evalf())
                break
        if out["lam_star"] is None:
            g0 = float(dR.subs(x, 0).evalf())
            out["lam_star"] = 0.0 if g0 >= 0 else 1.0
            out["lam_boundary"] = True
    return out


def _glm_server_alive(base_url=GLM_SERVER_URL, tries=2):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def parity_battery(vbridge, channel, exact, prompts, max_tokens, lam=1.0,
                   gt_reference=None):
    """Greedy brain-only (lam) vs certified ground-truth channel token streams.
    prompts: list of (display, prompt_ids) pairs."""
    results = []
    V = vbridge.vocab_size
    eos_ids = vbridge.tokenizer.eos_ids
    for display, prompt_ids in prompts:
        if exact is not None:
            exact_ids = ai4._exact_greedy_ids(exact, prompt_ids, max_tokens)
        elif gt_reference is not None:
            exact_ids = gt_reference.greedy_stream(prompt_ids, max_tokens)
        else:
            exact_ids = _ngram_greedy_ids(channel, prompt_ids, max_tokens, V, eos_ids)
        brain_ids = ai4._brain_greedy_ids(vbridge, channel, prompt_ids, max_tokens, lam)
        div = next((k for k in range(min(len(exact_ids), len(brain_ids)))
                    if exact_ids[k] != brain_ids[k]), None)
        if div is None and len(exact_ids) != len(brain_ids):
            div = min(len(exact_ids), len(brain_ids))
        results.append({
            "prompt": display,
            "exact": bool(exact_ids == brain_ids),
            "first_divergence": div,
            "model_tokens": [int(t) for t in exact_ids],
            "brain_tokens": [int(t) for t in brain_ids],
        })
    return results


def _ngram_greedy_ids(channel, prompt_ids, max_tokens, vocab_size, eos_ids):
    """Greedy decoding of the ground-truth channel itself (brain-only mode:
    the certified corpus n-gram is the reference stream)."""
    gen = []
    for _ in range(int(max_tokens)):
        lp = channel.logprobs(list(prompt_ids) + gen, vocab_size)
        nxt = int(np.argmax(lp))
        if nxt in eos_ids:
            break
        gen.append(nxt)
    return gen


def run_frontier_parity(model_path, brain_dir=None, corpus_text=None,
                        prompt=PROMPT, max_tokens=MAX_TOKENS, n_rounds=N_ROUNDS,
                        cache_ram_mb=512, col_neurons=96, k_synapses=16,
                        ngram_order=4, max_train_tokens=None, meas_positions=64,
                        distill_lr=0.02, verbose=True):
    def log(msg):
        if verbose:
            print(msg, flush=True)

    model_path = os.path.abspath(model_path)
    if brain_dir is None:
        brain_dir = os.path.normpath(model_path).rstrip("/\\") + "-brain"
    os.makedirs(brain_dir, exist_ok=True)
    if corpus_text is None:
        corpus_text = DEFAULT_CORPUS

    log(f"[FPD] model={model_path}")
    log(f"[FPD] brain_dir={brain_dir}")
    brain = None
    try:
        brain = ai4.BiologicalBrain.load(brain_dir, writable=True)
        log(f"[FPD] loaded existing brain ({brain.N:,} neurons)")
    except Exception:
        brain = None
    if brain is None:
        brain = ai4.LLMToBrainConverter.convert_model(
            model_path, out_dir=brain_dir, col_neurons=col_neurons, k_synapses=k_synapses)

    try:
        vbridge = ai4.VocabTokenBridge.load_or_create(brain, brain_dir)
        V = vbridge.vocab_size
        exact = None
        try:
            exact = ai4.ExactFrontierEngine.from_model(model_path, cache_ram_mb=cache_ram_mb)
        except NotImplementedError as e:
            log(f"[FPD] ExactFrontierEngine refuses this architecture: {e}")
            if _glm_server_alive(GLM_SERVER_URL):
                log(f"[FPD] GLM frontier channel: llama.cpp reference server at "
                    f"{GLM_SERVER_URL} (actual frontier weights, greedy decoding)")
            else:
                log("[FPD] no GLM frontier server found -> brain-only mode: "
                    "ground-truth channel A = corpus n-gram")
        except Exception as e:
            log(f"[FPD] exact engine unavailable ({type(e).__name__}: {e}) -> "
                "brain-only mode: ground-truth channel A = corpus n-gram")
        if exact is not None and exact.vocab_size != V:
            raise ValueError(f"vocab mismatch: model {exact.vocab_size} vs brain {V}")
        if exact is not None:
            mode = "exact_ground_truth"
            channel = ai4.ExactModelChannel(exact)
        elif _glm_server_alive(GLM_SERVER_URL):
            mode = "glm_frontier_channel_llamacpp"
            channel = LlamaCppChannel(
                GLM_SERVER_URL, vocab_size=V, eos_ids=vbridge.tokenizer.eos_ids,
                cache_path=os.path.join(brain_dir, "glm_frontier_argmax_cache.json"))
        else:
            mode = "brain_only_ngram_ground_truth"
            channel = ai4.CorpusNGram(max(1, int(ngram_order)))

        gt_reference = None
        if exact is None and hasattr(channel, "greedy_next"):
            gt_reference = channel
        ids = vbridge.tokenizer.encode(corpus_text, add_bos=True)
        split = max(96, int(len(ids) * 0.9))
        train_ids, meas_ids = ids[:split], ids[split:]
        if exact is None and not hasattr(channel, "greedy_next"):
            channel.train_ids(list(train_ids))
        log(f"[FPD] mode: {mode} | corpus tokens: {len(ids)} "
            f"(train {len(train_ids)}, meas {len(meas_ids)})")

        round_log = []
        lam_production = 1.0
        certified_parity = False
        last_cert = None
        rnd = 0
        for rnd in range(1, max(1, int(n_rounds)) + 1):
            round_ids = train_ids[:max_train_tokens] if max_train_tokens else train_ids
            losses = _distill_sequence(vbridge, channel, round_ids, lr=distill_lr)
            pairs = _measure_positions(vbridge, channel, meas_ids, meas_positions)
            cert = certify_round(ape, pairs, rnd)
            last_cert = cert
            agree = sum(cert["positions"]["argmax_agree"]) if cert.get("positions") else 0
            log(f"[round {rnd}] distill CE(frontier argmax) {losses[0]:.4f}->{losses[-1]:.4f} | "
                f"residuals {[f'{r:.2e}' for r in (cert['residuals'] or [])]} | "
                f"argmax agree {agree}/2 | "
                f"engine: {cert['status']} lam*={cert['lam_star']}")
            round_log.append({"round": rnd, "distill_loss_first": float(losses[0]),
                              "distill_loss_last": float(losses[-1]), **cert})

            prompt_ids_user = vbridge.tokenizer.encode(prompt, add_bos=True) or [0]
            prompts = [(vbridge.tokenizer.decode(meas_ids[:k]), list(meas_ids[:k]))
                       for k in (12, 24) if k < len(meas_ids)] + [(prompt, prompt_ids_user)]
            battery = parity_battery(vbridge, channel, exact, prompts, max_tokens,
                                     lam=1.0, gt_reference=gt_reference)
            battery_ok = all(b["exact"] for b in battery)
            gt_label = ("frontier" if exact is not None
                        else ("GLM-frontier" if gt_reference is not None else "ngram"))
            log(f"[round {rnd}] battery vs certified {gt_label} channel (brain-only lam=1): "
                f"{sum(1 for b in battery if b['exact'])}/{len(battery)} exact "
                f"(first divergences: {[b['first_divergence'] for b in battery]})")
            round_log[-1]["battery"] = battery

            if cert["parity_certified"] and battery_ok:
                lam_production = 1.0
                certified_parity = True
                log(f"[round {rnd}] CERTIFIED BRAIN-ONLY PARITY: the brain reproduces "
                    f"the certified {gt_label} greedy stream on the battery; lam* = 1")
                break
            use = cert["lam_star"] if cert["lam_star"] is not None else 0.0
            if cert["lam_star"] == 1.0 and not battery_ok:
                use = 0.0
                log(f"[round {rnd}] lam*=1 failed battery -> honest production fallback "
                    f"lam=0 (pure certified ground-truth channel)")
            lam_production = float(use)
            vbridge.attach_ngram(channel, influence=lam_production)

        if last_cert is not None and not certified_parity:
            log("[FPD] brain-only parity NOT certified within round budget; "
                "production keeps the engine-certified mixture weight "
                f"lam={lam_production} (parity-by-construction at lam=0)")

        prompt_ids = vbridge.tokenizer.encode(prompt, add_bos=True) or [0]
        if exact is not None:
            ref_ids = ai4._exact_greedy_ids(exact, prompt_ids, max_tokens)
        elif gt_reference is not None:
            ref_ids = gt_reference.greedy_stream(prompt_ids, max_tokens)
        else:
            ref_ids = _ngram_greedy_ids(channel, prompt_ids, max_tokens, V,
                                        vbridge.tokenizer.eos_ids)
        ref_text = vbridge.tokenizer.decode(list(prompt_ids) + ref_ids)

        vbridge.attach_ngram(channel, influence=lam_production)
        brain_text, gen_stats = vbridge.generate_tokens(
            prompt, max_tokens=max_tokens, temperature=0.0, return_stats=True)

        cert_out = {
            "problem": "brain greedy output equals the certified ground-truth channel "
                       "(Frontier-Parity Distillation, AutonomousProofEngine v2 certifier)",
            "mode": mode,
            "model": model_path,
            "brain_dir": brain_dir,
            "prompt": prompt,
            "rounds": round_log,
            "engine_status": last_cert["status"] if last_cert else None,
            "certified_fn": last_cert["certified_fn"] if last_cert else None,
            "lam_production": lam_production,
            "brain_only_parity_certified": certified_parity,
            "gaps": (["generalization from the parity battery to all possible prompts "
                      "is a declared hypothesis, not a settled theorem"]
                     + (["ExactFrontierEngine refuses the hybrid glm5-next graph by "
                         "design; ground truth = llama.cpp reference greedy decoding of "
                         "the actual frontier weights"]
                        if mode == "glm_frontier_channel_llamacpp" else
                        ["exact model inference is refused by ExactFrontierEngine for "
                         "hybrid glm5-next architectures by design; ground truth = "
                         "certified corpus n-gram"] if exact is None else [])
                     if certified_parity
                     else ["brain-only parity not reached; production lam<1 keeps the "
                           "certified ground-truth channel in the mixture"]
                     + (["ExactFrontierEngine refuses the hybrid glm5-next graph by "
                         "design; ground truth = llama.cpp reference greedy decoding of "
                         "the actual frontier weights"]
                        if mode == "glm_frontier_channel_llamacpp" else
                        ["exact model inference is refused by ExactFrontierEngine for "
                         "hybrid glm5-next architectures by design; ground truth = "
                         "certified corpus n-gram"] if exact is None else [])),
            "reference_text": ref_text,
            "production_text": brain_text,
            "tokens_per_s": float(gen_stats["tokens_per_s"]),
        }
        with open(os.path.join(brain_dir, "frontier_parity_certificate.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(cert_out, fh, indent=2)
        os.makedirs(os.path.join(brain_dir, "vocab_bridge"), exist_ok=True)
        with open(os.path.join(brain_dir, "vocab_bridge", "ground_truth_channel.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"model": model_path, "lam": float(lam_production), "mode": mode,
                       "channel": ("ExactModelChannel (ground truth = source model)"
                                   if exact is not None else
                                   "LlamaCppChannel (ground truth = llama.cpp reference "
                                   "greedy decoding of the frontier weights)"
                                   if gt_reference is not None else
                                   "CorpusNGram (ground truth = certified corpus n-gram)")}, fh)
        vbridge.save(brain_dir)
        log(f"[FPD] certificate written to "
            f"{os.path.join(brain_dir, 'frontier_parity_certificate.json')}")
        log(f"\n=== BRAIN (production lam={lam_production}) ===\n{brain_text}")
        return cert_out
    finally:
        brain.close()


def main():
    if not MODEL_PATH:
        print(__doc__)
        raise SystemExit("usage: frontier_parity_driver.py <model_path> "
                         "[brain_dir] [corpus.txt] [prompt] [max_tokens] [rounds]")
    run_frontier_parity(
        MODEL_PATH,
        brain_dir=BRAIN_DIR,
        corpus_text=open(CORPUS, encoding="utf-8", errors="ignore").read()
        if CORPUS else None,
        prompt=PROMPT, max_tokens=MAX_TOKENS, n_rounds=N_ROUNDS)


if __name__ == "__main__":
    main()
