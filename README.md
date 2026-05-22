# Chapter 3 Baselines — Hierarchical Federated Learning

Six baselines compared against the proposed Fedge method on CIFAR-10 and
Fashion-MNIST, all using the same hierarchical setup:
**10 edges × 10 clients = 100 clients, full participation,
α_server = α_client = 0.5 (two-level Dirichlet).**

| Baseline | Paper | Type | Distinct trained models |
|---|---|---|---|
| `HierFAVG`  | Liu, Zhang, Song, Letaief — ICC 2020 | Global only        | **1** (one global model shared by all edges) |
| `MTGC`      | Fang et al. — NeurIPS 2024            | Global only        | **1** (control variates `Y`, `Z` correct drift, do not personalize) |
| `PHE-FL`    | Lee et al. — 2025                     | Per-edge personalized | **M = 10** (one model per edge) |
| `ESPerHFL`  | Ma et al. — J. Cloud Comp. 2024       | Per-edge personalized w/ learnable mixing | **M = 10** (each edge mixes global + local) |
| `CHPFL`     | Song et al. — 2025                    | Hard-clustered     | **K** distinct cluster models (K < M); each edge served by its cluster's model |
| `Fedge`     | proposed                              | Soft per-edge (similarity-weighted) | **M = 10** distinct per-edge models |

## How methods map onto the shared output schema

Every baseline writes the **same files with the same columns**, but the
semantics shift depending on whether the method is global or personalized.

| Output | Global-only methods (HierFAVG, MTGC) | Personalized methods (PHE-FL, ESPerHFL, CHPFL, Fedge) |
|---|---|---|
| `per_edge_accs` (per_round.csv) | The single global model evaluated on each edge's local test partition — M numbers from the **same** model | The **served model for that edge** evaluated on its local test partition — M numbers from **M different** models |
| `per_edge_pers_accs` (per_round.csv) | Equal to `per_edge_accs` (no personalization to report separately) | Same as `per_edge_accs` *unless* the method also fine-tunes locally, in which case this is the post-personalization accuracy |
| `global_acc` (per_round.csv) | Global model on full 10k test set | Global model on full 10k test set (still well-defined; for CHPFL it's the cluster-1 model, for Fedge it's the pre-personalization global) |
| `final_per_edge_acc.csv` → `local_test_acc` | Global model on edge's local test slice | Edge's served model on edge's local test slice |
| `final_per_edge_acc.csv` → `global_test_acc` | Same number on every row (global model on 10k) | Each edge's served model on 10k test |
| `final_per_edge_acc.csv` → `cluster_id` | `= edge_id` (no clustering) | CHPFL: hard cluster index. Fedge: argmax of the SWPA mixing matrix |
| `models/global.pt` | The trained global model | The pre-personalization global model (for methods that maintain one) |
| `models/edge_{j}.pt` | Identical copies of `global.pt` (kept for schema uniformity) | The **distinct** per-edge served model |

**Rule of thumb when reading results:** for HierFAVG and MTGC, the 10
`per_edge_acc` numbers in any row should differ only because the
**test slices** differ, not because the models do. For the four
personalized methods they differ because the **models** differ too —
that's the whole point of personalization.

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

### Local
```bash
cd cifar10/MTGC
pip install -r ../../requirements.txt
python train_mtgc.py --global-rounds 2     # smoke test
bash run_5_seeds.sh --global-rounds 100    # full 5-seed sweep
```

### Fresh GCP VM (g2-standard-8 + 1× L4, Ubuntu 22.04)
```bash
# On the VM:
git clone https://github.com/satwatbashir/baselines.git
cd baselines
bash setup_vm.sh                  # installs driver, reboots
# log back in, then:
bash setup_vm.sh --skip-driver    # installs Python deps in .venv
source ~/baselines/.venv/bin/activate
cd cifar10/MTGC
tmux new -s mtgc
bash run_5_seeds.sh               # ~30-50 min on L4
```

After the sweep, results are under `cifar10/gc_results/mtgc_seed{42..46}/`.
Zip the directory and download for offline analysis:

```bash
(cd cifar10 && zip -r mtgc_cifar10_5seeds.zip gc_results)
```
