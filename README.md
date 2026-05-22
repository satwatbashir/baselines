# Chapter 3 Baselines — Hierarchical Federated Learning

Six baselines compared against the proposed Fedge method on CIFAR-10 and
Fashion-MNIST, all using the same hierarchical setup:
**10 edges × 10 clients = 100 clients, full participation,
α_server = α_client = 0.5 (two-level Dirichlet).**

| Baseline | Paper |
|---|---|
| `HierFAVG`  | Liu, Zhang, Song, Letaief — ICC 2020 |
| `MTGC`      | Fang et al. — NeurIPS 2024 |
| `PHE-FL`    | Lee et al. — 2025 |
| `ESPerHFL`  | Ma et al. — J. Cloud Comp. 2024 |
| `CHPFL`     | Song et al. — 2025 |
| `Fedge`     | proposed |

## Layout

```
Chapter3-Baselines/
├── cifar10/
│   ├── HierFAVG/   MTGC/   PHE-FL/   ESPerHFL/   CHPFL/   Fedge/
│   └── gc_results/        ← per-seed run folders land here
└── fmnist/
    └── (same structure)
```

Per baseline a run produces `{method}_seed{N}/` containing:

- `per_round.csv` — round-by-round metrics (mean/std/min/max + per-edge JSON arrays + global + bits)
- `final_per_edge_acc.csv` — one row per edge (local + global acc, sample counts, dominant class, cluster id)
- `metadata.json` — seed, M/C/ρ/α, edge label distribution, hparams, hardware, version hash, timings
- `models/global.pt`, `models/edge_{j}.pt` — final state dicts

## Quick start

```bash
# Local
cd cifar10/MTGC
pip install -r ../../requirements.txt
python train_mtgc.py --global-rounds 2     # smoke test
bash run_5_seeds.sh --global-rounds 100    # full 5-seed sweep
```

After the sweep, results are under `cifar10/gc_results/mtgc_seed{42..46}/`.
Zip the directory and download for offline analysis:

```bash
(cd cifar10 && zip -r mtgc_cifar10_5seeds.zip gc_results)
```
