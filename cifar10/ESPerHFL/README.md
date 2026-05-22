# ESPerHFL — Per-Edge Personalized Hierarchical FL with Learnable Mixing

**Reference:** Ma, Liu, Wang, et al. *Personalized client-edge-cloud
hierarchical federated learning in mobile edge computing.* Journal of Cloud
Computing 2024. DOI: [10.1186/s13677-024-00721-w](https://doi.org/10.1186/s13677-024-00721-w)
Official code: [github.com/xiangqianL/ESPerHFL](https://github.com/xiangqianL/ESPerHFL)

## What it is
Per-edge personalization via APFL-style learnable mixing extended to a
three-tier hierarchy. Two-level personalization: (i) client-side learnable
mixing scalar, (ii) cloud-side similarity-weighted per-edge aggregation.

Per client `c` in edge `j` we keep two models and a scalar:
- `w_c` — shared/global path
- `v_c` — local/personal path
- `m_c ∈ [0, 1]` — mixing scalar

Per local SGD step on `(x, y)`:
1. **Update `w_c`** with `loss = CE(w_c(x), y)`.
2. **Update `v_c`** with `loss = CE(m_c · v_c(x) + (1 − m_c) · w_c(x), y)`.
3. After the local update finishes, **update `m_c`** via the APFL
   alpha-update rule (gradient of the mixed loss w.r.t. `m`, with a small
   regularizer; clipped to `[0, 1]`).

Aggregation:
- **Edge** `j`: weighted mean of `w_c` → `W_j`; weighted mean of `v_c` → `V_j`;
  uniform mean of `m_c` → `m_j`. Weights are by sample count.
- **Cloud (paper-faithful `p_average`)**: similarity-weighted per-edge
  aggregation. For each edge `e`,
  `W_cloud[e] = (1 − h) · W_e + h · Σ_{k≠e} softmax(g · jaccard(W_e, W_k))[k] · W_k`
  with `g = 5`, `h = 0.7`. Produces **M distinct cloud models**, one per
  edge. `V_j` and `m_j` STAY at the edge.

**Served model at edge j** = `m_j · V_j + (1 − m_j) · W_cloud[j]` (parameter-space mix).

## Setup in this folder
Identical to MTGC and HierFAVG for direct comparability:
- **Data:** CIFAR-10 with **severe non-IID** (α_server = α_client = **0.1**)
- **Hierarchy:** 10 edges × 10 clients = 100 clients, full participation
- **Model:** `Cifar10CNN` (797,962 params) — byte-identical copy of MTGC's
- **Optimizer:** SGD, lr=0.1, weight_decay=1e-3, batch=50, grad-clip 10
- **Local work:** E=1 edge round per global round; ~1 local epoch per client
- **APFL alpha-update:** `ALPHA_LR = 0.1`, `ALPHA_REG = 0.02` (matches paper)

## What changes in the output schema
ESPerHFL is the first personalized baseline, so `per_edge_pers_accs` now
**differs from `per_edge_accs`**:

| Column | Meaning here |
|---|---|
| `per_edge_accs[j]` | Reference **uniform-weighted global** evaluated on edge j's local test (kept for cross-baseline comparison) |
| `per_edge_pers_accs[j]` | **Served (mixed) model** at edge j: `m_j · V_j + (1−m_j) · W_cloud[j]` |
| `mix_mean / min / max` | Diagnostics on the per-edge `m_j` distribution (ESPerHFL-only columns) |
| `final_per_edge_acc.csv → local_test_acc` | Served model on edge's local test |
| `final_per_edge_acc.csv → global_test_acc` | Served model on full 10k test (differs per row) |
| `final_per_edge_acc.csv → mix_scalar` | Final `m_j` per edge |
| `models/edge_{j}.pt` | The **served** state dict (parameter-space mix), plus `mix_scalar` metadata |
| `models/cloud_{j}.pt` | The per-edge `W_cloud[j]` from `p_average` (one per edge) |
| `models/global.pt` | Reference uniform-weighted global (used for `global_acc` reporting only) |

## Run
```bash
cd Chapter3-Baselines/cifar10/ESPerHFL
python3 train_esperhfl.py --global-rounds 100 --seed 42
# or all 5 seeds:
bash run_5_seeds.sh
```

## Deviations from the reference repo (documented)
1. **Optimizer momentum.** Reference SGD uses `momentum = args.momentum`
   (default 0.9). For cross-baseline consistency with MTGC / HierFAVG / CHPFL
   in this chapter (which all use no momentum), this implementation uses
   `momentum = 0`. The `--lr` and `--weight-decay` defaults match the paper.
2. **Partition.** Reference uses hard-coded label-range partitions
   (`initialize_edges_niid`); this chapter uses two-level hierarchical
   Dirichlet (`α_server = α_client = 0.1`) shared across all baselines so
   partition determinism is verifiable across methods.

## NaN guards
- `alpha_update` clips the new mixing scalar to `[0, 1]` and falls back to the
  previous value if the update is NaN/Inf or if either model has no `.grad`.
- `evaluate()` returns `NaN` only for empty loaders (defensive).
- `p_average` clamps the Jaccard denominator at 1e-12 and falls back to the
  uniform mean for any edge row whose cloud aggregation came out non-finite.
- Gradient norm is clipped at `clip_norm=10` on both `w` and `v` updates.
