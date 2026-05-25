# CHPFL — Fixed-K Clustered Hierarchical FL with K-Means++

**Reference:** Song et al. *CHPFL: Clustered adaptive hierarchical federated
learning for edge-level personalization.* 2025.
[ScienceDirect — S2667295225000479](https://www.sciencedirect.com/science/article/pii/S2667295225000479)

## What it is
Three-tier HFL that **hard-clusters the M edges into K (< M) groups** via
K-Means++ on edge-model deltas. Edges in the same cluster share a model;
the cloud still computes a global model but only for reporting after the
clustering takes effect.

### Two phases
1. **Warm-up** (rounds `1 … T_warmup`): vanilla hierarchical FedAvg with
   weighted aggregation at both edge and cloud. Every edge is served the
   single global model. `cluster_id = 0` for everyone.
2. **Clustered** (rounds `T_warmup+1 … T`): at the end of warm-up, the cloud
   runs K-Means++ on edge-model deltas `θ_j − W_global` to produce a fixed
   assignment `cluster_id[j] ∈ {0, …, K−1}`. The assignment is held for
   the rest of training (no re-clustering — that's the "Fixed-K" part).
   Per round thereafter:
   - Each edge `j` starts from `cluster_model[cluster_id[j]]`.
   - Clients in edge `j` do local SGD; edge weighted-aggregates → new edge model.
   - Within each cluster `k`, edges weighted-aggregate → `cluster_model[k]`.
   - The cloud also computes a global model (weighted mean over all edges)
     **for reporting only** — it is never served back to edges post-warmup.

**Served model at edge `j`** = `cluster_model[cluster_id[j]]`.

## Setup in this folder
Identical to MTGC / HierFAVG / ESPerHFL for direct comparability:
- **Data:** Fashion-MNIST with **severe non-IID** (α_server = α_client = **0.1**)
- **Hierarchy:** 10 edges × 10 clients = 100 clients, full participation
- **Model:** `Cifar10CNN` (797,962 params) — byte-identical copy of MTGC's
- **Optimizer:** SGD, lr=0.1, weight_decay=1e-3, batch=50, grad-clip 10
- **Local work:** E=1 edge round per global round; ~1 local epoch per client
- **Clustering:** K = **3** by default, warm-up = **5** rounds, K-Means++ on
  edge-delta features, run **once** at warm-up end. Override with `--K`
  and `--warmup-rounds`.

## What changes in the output schema
| Column | Meaning here |
|---|---|
| `per_edge_accs[j]` | `W_global` evaluated on edge `j`'s local test partition (same semantics as other baselines) |
| `per_edge_pers_accs[j]` | **Cluster (served) model** on edge `j`'s local test |
| `K`, `phase` | CHPFL-only diagnostics. `phase ∈ {warmup, clustered}` |
| `cluster_assignment` | JSON list of length M with each edge's cluster id (constant after warmup) |
| `final_per_edge_acc.csv → local_test_acc` | Served (cluster) model on edge's local test |
| `final_per_edge_acc.csv → global_test_acc` | Served (cluster) model on full 10k test |
| `final_per_edge_acc.csv → cluster_id` | K-Means++ hard-cluster id ∈ {0, …, K-1} |
| `models/global.pt` | Cloud's global model (reporting only post-warmup) |
| `models/cluster_{k}.pt` | The K cluster models |
| `models/edge_{j}.pt` | Per-edge served state dict (= its cluster's model), with metadata `{edge_id, cluster_id, note}` |

## Run
```bash
cd Chapter3-Baselines/fmnist/CHPFL
python3 train_chpfl.py --global-rounds 100 --seed 42
# or all 5 seeds:
bash run_5_seeds.sh
# different K or warm-up:
bash run_5_seeds.sh --K 5 --warmup-rounds 10
```

## NaN guards
- `evaluate()` returns `NaN` only for empty loaders (defensive; not reached
  under full participation at α=0.1).
- K-Means++ has an empty-cluster fallback: if a cluster ends with no
  members during Lloyd's iterations, its center is reseeded from the
  farthest point in the largest cluster and that point is reassigned.
- If the cloud aggregator is invoked on an empty cluster (shouldn't happen
  after the fallback), it falls back to the previous cluster model.
- All gradient norms clipped at `clip_norm=10`.
