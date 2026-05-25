# HierFAVG — Foundational Hierarchical FedAvg

**Reference:** Liu, Zhang, Song, Letaief. *Client-Edge-Cloud Hierarchical
Federated Learning.* IEEE ICC 2020.

## What it is
Nested FedAvg over a three-tier hierarchy (clients → edge servers → cloud).
No control variates, no proximal term — just weighted averaging at each tier:

- **Edge aggregation:** `θ_edge_j = Σ_c (n_c / N_j) · θ_c`
- **Global aggregation:** `θ_global = Σ_j (N_j / N) · θ_edge_j`

Weights are by sample count, matching the original paper.

## Setup in this folder
Identical to MTGC for direct comparability:
- **Data:** HHAR
- **Hierarchy:** 10 edges × 10 clients = 100 clients, full participation
- **Non-IID:** two-level Dirichlet, α_server = α_client = 0.5, **same seed → same split as MTGC**
- **Model:** `Cifar10CNN` (797,962 params) — byte-identical copy of `MTGC/model.py`
- **Optimizer:** SGD, lr=0.1, weight_decay=1e-3, batch=50, grad-clip 10
- **Local work:** E=1 edge round per global round; ~1 local epoch per client

## Run
```bash
cd Chapter3-Baselines/hhar/HierFAVG
python3 train_hierfavg.py --global-rounds 100 --seed 42
# or all 5 seeds:
bash run_5_seeds.sh
```

## Output layout
Same schema as MTGC. A run with `--seed 42` writes to `../gc_results/hierfavg_seed42/`:

| File | What's in it |
|---|---|
| `per_round.csv` | Same columns as MTGC's. `per_edge_pers_accs` equals `per_edge_accs` (no personalization). |
| `final_per_edge_acc.csv` | One row per edge. `cluster_id` = `edge_id` (no clustering). |
| `metadata.json` | Adds `"aggregation": "weighted_by_sample_count"`. |
| `models/global.pt`, `models/edge_{j}.pt` | Final state dicts. Every `edge_{j}.pt` equals `global.pt`. |

## Reproducibility check
With the same `--seed`, the `edge_label_distribution` matrix in `metadata.json`
is **identical** to MTGC's — proving both baselines see the exact same data
partition. Diff to verify:
```bash
diff <(jq .edge_label_distribution ../gc_results/mtgc_seed42/metadata.json) \
     <(jq .edge_label_distribution ../gc_results/hierfavg_seed42/metadata.json)
```

## HHAR-specific notes
- **Model**: `HARNet` (1D-CNN + GRU + FC, ~152k params), defined in `model.py`
- **Input**: 6-channel acc+gyro windows of length 100 (2 s @ 50 Hz)
- **Classes**: 6 activities — walking, sitting, standing, biking, stairsup, stairsdown
- **Data**: downloaded from UCI on first run; cached as `.npz` under `<data-root>/hhar_cache/`
- **Note**: HHAR has 9 natural users; we apply hierarchical Dirichlet over the *windowed* label list (ignoring user identity) to match the protocol of the image baselines.
