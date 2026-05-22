# PHE-FL — Personalizing FL for Hierarchical Edge Networks (Lee et al., 2025)

**Reference:** Lee et al. *Personalizing Federated Learning for Hierarchical
Edge Networks with Non-IID Data.* 2025. [arXiv:2504.08872](https://arxiv.org/abs/2504.08872)

## What it is
Per-edge personalization via two paper-specific mechanisms:

1. **Leave-one-out complement aggregation at the cloud.** Instead of a single
   global model, the cloud builds a *different* complement model for each
   edge:
   `CAM_e = Σ_{k ≠ e} (|D_k| / |D ∖ D_e|) · EAM_k`
   (M distinct cloud models, one per edge, each computed by *excluding* the
   target edge's own contribution.)

2. **Accuracy-ratio mixing.** Each edge evaluates its own model `EAM_e` and
   the complement `CAM_e` on its personalization test data `PTD_e`, then sets
   `α_e = Acc(EAM_e, PTD_e) / [Acc(EAM_e, PTD_e) + Acc(CAM_e, PTD_e)]`
   and forms `PEAM_e = α_e · EAM_e + (1 − α_e) · CAM_e`. The mixing weight
   is **not** gradient-learnable — it's a per-round performance ratio.

The clients in edge `e` start the next round from `PEAM_e`. **No client-side
persistent state, no second model**, no APFL-style learnable scalar — that's
the structural difference vs ESPerHFL.

## Setup in this folder
Identical to MTGC / HierFAVG / ESPerHFL / CHPFL for direct comparability:
- **Data:** CIFAR-10 with **severe non-IID** (α_server = α_client = **0.1**)
- **Hierarchy:** 10 edges × 10 clients = 100 clients, full participation
- **Model:** `Cifar10CNN` (797,962 params) — byte-identical copy of MTGC's
- **Optimizer:** SGD, lr=0.1, weight_decay=1e-3, batch=50, grad-clip 10
- **Local work:** E=1 edge round per global round; ~1 local epoch per client
- **PTD_e** = `edge_test_loaders[j]` (the per-edge local test slice we already
  build for all baselines). See the *Implementation notes* below.

## How it differs from ESPerHFL (NOT a one-line variant)
| Aspect | PHE-FL | ESPerHFL |
|---|---|---|
| Cloud→edge transmission | M **complement** models (one per edge) | M **similarity-weighted** models from `p_average` |
| Personalization state at clients | none — clients hold one model | two models `w_c`, `v_c` plus scalar `m_c` |
| Mixing weight source | **accuracy ratio** recomputed each round | **APFL gradient** on the mixed loss |
| Persistent edge-local model `V_j` | no | yes — `V_j` accumulates over rounds, never sees the cloud |

## What changes in the output schema
| Column | Meaning here |
|---|---|
| `per_edge_accs[j]` | Reference uniform-weighted global on edge j's local test |
| `per_edge_pers_accs[j]` | `PEAM_j` on edge j's local test |
| `alpha_mean / min / max`, `per_edge_alphas` | Per-round α_e diagnostics (PHE-FL-only) |
| `per_edge_eam_accs`, `per_edge_cam_accs` | Sanity columns: the two ingredient accuracies used to compute α_e |
| `final_per_edge_acc.csv → local_test_acc` | PEAM_e on edge's local test |
| `final_per_edge_acc.csv → global_test_acc` | PEAM_e on full 10k test |
| `final_per_edge_acc.csv → alpha_e` | Final α_e per edge |
| `models/edge_{j}.pt` | PEAM_e (parameter-space mix) with `alpha_e` metadata |
| `models/cam_{j}.pt` | The per-edge complement CAM_e |
| `models/global.pt` | Reference uniform-weighted global (reporting only) |

## Implementation notes / caveats
1. **PTD_e source.** The PHE-FL paper uses a designated personalization test
   partition at each edge to compute α_e. Here we use the same per-edge test
   loaders `edge_test_loaders[j]` for both α computation *and* final
   reporting. This is consistent with the paper's pseudocode but does mean
   the per-round α benefits from a small amount of "looking at the test set"
   information. All baselines see the same partition, so this is not a
   cross-method asymmetry; it is documented for transparency.
2. **NaN guards.** `accuracy_ratio_alpha` falls back to 0.5 (neutral mix)
   if either accuracy is non-finite or if `acc_eam + acc_cam = 0`. α is
   always clipped to [0, 1]. The leave-one-out aggregator falls back to
   `EAM_e` itself if edge e somehow holds all data (defensive only).
3. **Gradient norm** clipped at `clip_norm=10` on local SGD.

## Run
```bash
cd Chapter3-Baselines/cifar10/PHE-FL
python3 train_phefl.py --global-rounds 100 --seed 42
# or all 5 seeds:
bash run_5_seeds.sh
```
