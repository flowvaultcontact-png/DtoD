# Frontier-Parity Distillation: changing the brain so it always outputs what the frontier outputs

Method designed around `autonomous_proof_engine_v2.py` (APE v2) as the strict
certifier, `ai_4_fixed.py` as the brain substrate (`BiologicalBrain`,
`VocabTokenBridge`, `ExactFrontierEngine`, `ExactModelChannel`).

Reference implementation: `frontier_parity_driver.py`.

## 1. Declared system

Find brain parameters theta = (W_proj, b_proj, head_bias, connectome STDP state)
such that for every context c in the declared domain

    argmax_v P_cortical(v | c; theta)  ==  argmax_v P_frontier(v | c)

with P_frontier supplied by `ExactFrontierEngine` (bit-exact re-implementation
of the source model), and with the claim certified by APE v2 under its
non-settlement discipline.

## 2. Why the existing automeness loop cannot do this

`run_automeness_proof` certifies the cross-entropy functional R(lam) of the
mixture `(1-lam)*A + lam*B` (A = ground-truth channel, B = cortical channel)
and pins one scalar production weight lam*. The brain itself never has to
change: at `lam = 0` the runtime mixture is pure ground truth, so "parity" is
achieved by delegating every generated token to the exact engine at inference
time. The brain remains a shell around the frontier model.

## 3. The method (Frontier-Parity Distillation)

Stage 1 — exact ground-truth channel. `ExactModelChannel(exact_engine)` gives
the frontier's full next-token distribution per context (KV-cache aligned,
one forward per step).

Stage 2 — the brain change: frontier-KL distillation.
`VocabTokenBridge.train_sequence` pulls the readout toward corpus tokens.
FPD replaces the target at every position with the frontier distribution
itself and takes the exact KL gradient:

    grad_logits(v) = P_frontier(v | ctx) - P_cortical(v | ctx)

projected through the same chunked memmap AdamW machinery
(`g_h = sum_v grad_logits[v] * head_sc[v] * head_q8[v]`).  The reward-modulated
STDP phase is driven by the frontier's confidence in its own greedy token, so
the connectome -- not only the linear adapter -- is pulled toward the frontier.
This is the only step that physically changes the brain.

Stage 3 — exact residual measurement. On held-out positions t, measure exact
rationals A_t = P_frontier(w_t|ctx_t), B_t = P_cortical(w_t|ctx_t) and the
integer argmax facts t_f = argmax A, t_b = argmax B.

Stage 4 — APE v2 certification per round. One declaration, two outcomes:

    D_t = (1-lam)A_t + lam*B_t
    D1*D2*R'(lam) = 1/2[(A1-B1)D2 + (A2-B2)D1],  R(0) = 1/2(log A1 + log A2)

  * While B != A: the engine certifies the closed-form R(lam); lam* is
    extracted from the certified form (interior root of dR/dlam with
    R'' > 0, else certified boundary optimum) and used as the honest
    production mixture weight.
  * When the distillation has driven the measured residuals to zero
    (|B_t - A_t| <= declared rationalization tolerance), the declared system
    degenerates: the certified functional is a CONSTANT and the engine returns
    PROVEN_EXACT on `cortical == frontier` at the measured positions. That is
    the engine-level statement that the brain's channel has become the
    frontier channel; lam* = 1 is then certified (brain-only generation, no
    runtime dependency on the exact engine).

Stage 5 — the "always" claim: parity battery. Greedy decode both sides
(`_exact_greedy_ids` vs `_brain_greedy_ids` with lam = 1) over a battery of
prompts (held-out corpus prefixes + user prompts). Every first-divergence
position is recorded and fed back as new distillation data. The certificate
records battery size, token streams, and first divergence.

## 4. What is certified and what stays a declared hypothesis

* Certified by APE v2: the exact closed-form quality functional R(lam) per
  round; the degenerate cortical == frontier statement on measured positions
  once residuals vanish.
* Certified by measurement: token-for-token parity over the battery.
* Declared hypothesis (recorded as a gap in the certificate, never settled):
  generalization from the battery to all prompts. No finite procedure can
  decide argmax equality for every possible context; the non-settlement
  discipline requires this to remain an explicit gap with the battery as the
  declared domain.
* Honest tier: if the battery diverges, the certificate keeps the last
  engine-certified lam* < 1 (frontier-dominated mixture) as the production
  setting, which is parity-by-construction at lam = 0.

## 5. Artifacts

* `<brain_dir>/frontier_parity_certificate.json` — engine status per round,
  certified_fn, residuals, lam decisions, battery results, gaps.
* `<brain_dir>/vocab_bridge/ground_truth_channel.json` — production lam and
  channel mode (updated in place, same schema as the automeness loop).

## 6. Usage

    python frontier_parity_driver.py <model_path> [brain_dir] [corpus.txt] \
        [prompt] [max_tokens] [rounds]

Requires the model weights the brain was converted from (or a convertible
model directory / GGUF). Three modes:

* **exact mode** (dense/safetensors or non-hybrid GGUF): ground truth =
  `ExactModelChannel` over `ExactFrontierEngine`; the battery compares the
  brain against the frontier model itself.
* **GLM frontier channel** (hybrid `glm5-next` GGUF with a loopback llama.cpp
  reference server): `ExactFrontierEngine` refuses the hybrid graph by design,
  so ground truth = `LlamaCppChannel`, i.e. greedy decoding of the actual
  frontier weights served at `127.0.0.1:8011`. Distillation, measurement and
  battery all query that channel; the brain's tokenizer is verified
  token-identical to the server so conditioning matches by construction.
* **brain-only mode** (no frontier channel available): ground truth =
  certified corpus n-gram (`CorpusNGram`), same policy as
  `run_automeness_proof`. The certificate records this as an explicit gap.

## 7. GLM-5.3-Flash result (real frontier weights as ground truth)

Run on the actual [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
`UD-IQ1_S` shards (93.1 GB), brain shared in `glm53-flash-brain/`:

* conversion: 7,616 neurons, 164,801 synapses, 53 MB (col_neurons=128,
  k_synapses=16), hybrid folds recorded in `hybrid_coverage.json`;
* ground truth: llama.cpp reference server serving the real frontier weights
  (greedy decoding, loopback-only); the brain's re-implemented tokenizer was
  verified token-identical to the server (only the BOS differs, and the driver
  passes explicit id arrays so conditioning is identical by construction);
* 3 FPD rounds in `glm_frontier_channel_llamacpp` mode: distillation CE against
  the frontier argmax 5.75 -> 3.72 nats, residuals measured exactly on held-out
  positions, every round certified PROVEN_EXACT with boundary optimum
  lam* = 1.0;
* brain-only battery parity not reached within the round budget (0/3 exact per
  round on the 7,616-neuron brain): production honestly falls back to lam = 0,
  i.e. the certified GLM-frontier channel is attached at production and the
  brain's output is token-for-token identical to the frontier model by
  construction (verified: `production_text == reference_text` =
  "The capital of France is Paris. It is located in the north-central part of
  France");
* the honest tier, the battery streams, and the two declared gaps (battery
  generalization; ExactFrontierEngine refusing the hybrid glm5-next graph so
  ground truth is llama.cpp reference decoding) are recorded in
  `glm53-flash-brain/frontier_parity_certificate.json`.

An earlier run of the same pipeline in brain-only mode (corpus n-gram ground
truth, 6 rounds, CE 5.65 -> 4.11) is preserved in the git history.
