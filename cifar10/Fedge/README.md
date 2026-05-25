# Fedge — Bayesian Non-Parametric Hierarchical Federated Learning

**The proposed contribution of Chapter 3.** Variable cloud-tier model family
$\mathcal{W}^t = \{w_1, \ldots, w_{K^t}\}$ with $K^t$ inferred from data, soft
edge-to-cluster membership $\Pi^t$, two-level heterogeneity signals driving
split-merge inference, and a single sensitivity parameter $\lambda \in [0,1]$
tracing the personalisation-convergence-robustness frontier.

## Novel contributions implemented

1. **Variable $K^t$ via two-level signals.** No existing HFL elastic-K method
   (DPMM-CFL, FedDAA) uses both intra-edge and inter-edge signals because
   both operate on flat topology.
2. **ICSM (Intra-Coupled Soft Membership).** Per-edge softmax temperature
   modulated by the edge's intra-edge gradient dispersion:
   $$ \tau_e(t) \;=\; \tau_0(\lambda) \cdot \big(1 + \beta\,\hat\zeta^{2}_{\text{intra}}(e, t)\big) $$
   Edges with low intra variance get low temperature → concentrated membership
   (confident assignment); edges with high intra variance get high temperature
   → spread membership (graceful uncertainty). No prior HFL or clustered-FL
   work modulates per-edge softmax temperature via intra-edge dispersion.
3. **Single $\lambda$ frontier control.** $\lambda = 0$ recovers HierFAVG;
   $\lambda = 1$ recovers per-edge personalised HFL of the PHE-FL/ESPerHFL
   family. Intermediate $\lambda$ positions the system in the interior of
   the P-C-R triangle.

## Two inference modes (CLI flag `--inference`)

| Mode | What it does | When to use |
|---|---|---|
| `heuristic` (default) | Greedy split-merge using signal thresholds derived from $\lambda$. Fast, deterministic, easy to debug. | Headline runs, ablations, large-scale sweeps. |
| `mh` | Metropolis-Hastings split-merge under a Dirichlet Process prior with Gaussian cluster likelihood, plus split-via-spectral-eigvec proposal and merge-via-inverse-distance proposal. Faithful realisation of the chapter's algorithmic spec. | Defensibility, validating that the heuristic doesn't deviate from the principled posterior. |

## Setup
- **Data:** CIFAR-10, hierarchical Dirichlet (α=0.1 default, 10×10 = 100 clients, full participation)
- **Model:** `Cifar10CNN` (797,962 params), shared with all CIFAR-10 baselines
- **Optimiser:** SGD, lr=0.1, weight_decay=1e-3, batch=50, grad-clip 10
- **Global rounds:** 200 (default)
- **Warmup rounds:** 5 (split-merge activates from round 6 onwards)

## Run
```bash
# Heuristic, lambda=0.5 (interior of the frontier), defaults
python3 train_fedge.py --global-rounds 200 --seed 42

# Faithful MH sampler under the DP prior
python3 train_fedge.py --inference mh --lambda-val 0.5

# Recover HierFAVG (sanity check)
python3 train_fedge.py --lambda-val 0.0

# Recover per-edge personalisation (PHE-FL/ESPerHFL family)
python3 train_fedge.py --lambda-val 1.0

# 5-seed sweep:
bash run_5_seeds.sh                                    # heuristic, lambda=0.5
bash run_5_seeds.sh --inference mh --lambda-val 0.5    # MH version
```

## Lambda sweep design (recommended for thesis)

For Section 3.5 of the thesis you'll want to sweep $\lambda \in \{0, 0.25, 0.5, 0.75, 1.0\}$ to trace the P-C-R frontier and validate Eq. (3.40)'s prediction of $K^*_{\text{opt}}$. Example:
```bash
for L in 0.0 0.25 0.5 0.75 1.0; do
  EXPERIMENT_TAG="lambda_${L//./}" bash run_5_seeds.sh --lambda-val $L
done
```

## Output schema (additions vs other baselines)

`per_round.csv` adds these Fedge-specific columns at the end of each row:

| Column | Meaning |
|---|---|
| `K_t` | current cluster count |
| `pi_entropy` | mean per-edge Shannon entropy of $\Pi^t$ (0 = hard, log(K) = uniform) |
| `n_splits`, `n_merges` | accepted moves this round |
| `intra_signal_mean`, `intra_signal_max` | $\hat\zeta^2_{\text{intra}}$ aggregated |
| `inter_signal_mean` | mean off-diagonal pairwise edge-edge distance |
| `membership` | full $M \times K^t$ matrix serialised as JSON |
| `cluster_assignment` | argmax of each row of $\Pi^t$ |

`final_per_edge_acc.csv` adds `pi_row` (the edge's final membership vector).

`models/` contains:
- `global.pt` — uniform-mean reference (for reporting)
- `cluster_{k}.pt` — the $K^T$ trained cluster models
- `edge_{j}.pt` — the served (blended) model per edge, with `pi_row` metadata

## Caveats to acknowledge in the thesis
1. **A5 (stabilisation of $K^t$)** is empirical, not analytically bounded.
   The chapter is honest about this. Plot $K^t$ vs $t$ across seeds; report $T_0$.
2. **A4 (cluster-quality factor $\gamma$)** is a property of the output partition.
   Report $\gamma$ empirically per $\lambda$ value.
3. **MH proposal distribution $q$**: this implementation uses (a) split via
   leading eigenvector of the inter-edge distance restricted to the cluster,
   weighted by intra-cluster variance, (b) merge via inverse-distance pair
   selection. Both are reversible. Documented in `fedge_core.py`.
4. **Cluster-quality factor $\gamma$** isn't measured natively by the
   algorithm; you'd compute it post-hoc as
   $(\sum_k \zeta_k^2 / K^*) / G_{\text{inter}}^2$ from the per-round logs.
