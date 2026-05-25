# MTGC — Hierarchical FL with Multi-Timescale Gradient Correction

**Reference:** Fang, Han, Chen, Wang, Brinton. *Hierarchical Federated Learning with
Multi-Timescale Gradient Correction.* NeurIPS 2024. https://arxiv.org/abs/2409.18448
Official code: https://github.com/wenzhifang/MTGC

## What it is
Three-level FL (clients → edge servers → cloud) with two control variates:
- **Z[c]** — corrects client gradient toward the edge gradient (reset every global round).
- **Y[j]** — corrects edge gradient toward the global gradient (persists across rounds).

The client local objective adds a linear penalty `<w, Z[c] + Y[j]>` (same shape as the
SCAFFOLD term), which kills multi-timescale drift under hierarchical non-IID.

## Setup in this folder
- **Data:** Fashion-MNIST
- **Hierarchy:** 10 edges × 10 clients = 100 clients, full participation
- **Non-IID:** two-level Dirichlet, α_server = α_client = 0.5
- **Model:** Fashion-MNIST CNN from the MTGC paper (2× conv-5 64ch + FC 384/192/10, ~890K params)
- **Optimizer:** SGD, lr=0.1, weight_decay=1e-3, batch=50, grad-clip 10
- **Local work:** E=1 edge round per global round; local steps ≈ 1 epoch (10 minibatches/client)

These defaults match the MTGC paper's Fashion-MNIST setup. Override with CLI flags.

## Run
```bash
cd Chapter3-Baselines/fmnist/MTGC
python3 train_mtgc.py --global-rounds 100 --seed 42
```

For a fast smoke test on CPU:
```bash
python3 train_mtgc.py --global-rounds 2 --eval-every 1 --num-workers 0
```

## Output layout
A run with `--seed 42` writes to `../gc_results/mtgc_seed42/`:

| File | What's in it |
|---|---|
| `per_round.csv` | One row per evaluated round. Columns: `round`, `per_edge_acc_{mean,std,min,max}`, `per_edge_loss_mean`, `per_edge_accs` (JSON list of M floats), `per_edge_losses`, `per_edge_pers_accs`, `global_acc`, `global_loss`, `round_time_s`, `bits_transmitted`. |
| `final_per_edge_acc.csv` | One row per edge. Columns: `edge_id`, `local_test_acc`, `local_test_loss`, `global_test_acc`, `n_train_samples`, `n_test_samples`, `dominant_class`, `cluster_id`. |
| `metadata.json` | Seed, M/C/ρ/α values, edge label distribution (M×10 matrix), client→edge map, all hyperparameters, hardware, version hash, start/end time, final global accuracy. |
| `models/global.pt`, `models/edge_{j}.pt` | Final state dicts. For MTGC every `edge_{j}.pt` equals `global.pt` (no personalization). |

## Notes on bookkeeping
- **Per-edge local test partition** is drawn by re-running the hierarchical
  Dirichlet partitioner on the test labels with the same `alpha_server` and
  `seed` (`clients_per_server=1`), so each edge sees a test slice that mirrors
  its training class mix.
- **`bits_transmitted`** uses a default formula:
  `2 · N · E · n_par · 32 + 2 · M · n_par · 32` (uplink+downlink, 32-bit
  floats, full participation). Swap in the Chapter 3 Table 3.1 formulas
  later if they differ.
- **`per_edge_pers_accs`** equals `per_edge_accs` for MTGC because MTGC has
  no per-edge served model — every edge gets the same global model. The
  column exists so personalized baselines (PHE-FL, ESPerHFL, Fedge, CHPFL)
  can fill it with their distinct per-edge models without changing the
  schema.
