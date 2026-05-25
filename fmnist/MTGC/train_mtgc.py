"""MTGC training driver for CIFAR-10 (10 edges x 10 clients, full participation).

Implements Hierarchical FL with Multi-Timescale Gradient Correction
(Fang, Han, Chen, Wang, Brinton — NeurIPS 2024, https://arxiv.org/abs/2409.18448),
faithful to the official reference code at github.com/wenzhifang/MTGC.

Output layout (per run, written under --out-dir):
    per_round.csv             — round-by-round metrics (incl. per-edge JSON arrays)
    final_per_edge_acc.csv    — one row per edge at end of training
    metadata.json             — seed, partition, hparams, hardware, timings
    models/global.pt          — final global model (state_dict)
    models/edge_{j}.pt        — per-edge served model (= global, for MTGC)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import Cifar10CNN, get_flat_params, set_flat_params, num_params
from partitioning import hier_dirichlet_indices


METHOD_NAME = "MTGC"
FMNIST_MEAN = (0.2860, 0.2860, 0.2860)  # F-MNIST single-channel stats, replicated to 3 channels
FMNIST_STD = (0.3530, 0.3530, 0.3530)   # F-MNIST single-channel stats, replicated to 3 channels
BITS_PER_PARAM = 32


# ---------------------------------------------------------------------------
# Data / partition
# ---------------------------------------------------------------------------
def build_train_loaders(trainset, mapping, batch_size, num_workers):
    edge_ids = sorted(mapping.keys(), key=lambda s: int(s))
    client_loaders: list[DataLoader] = []
    client_to_edge: list[int] = []
    edge_train_indices: list[list[int]] = [[] for _ in range(len(edge_ids))]
    for j, eid in enumerate(edge_ids):
        client_ids = sorted(mapping[eid].keys(), key=lambda s: int(s))
        for cid in client_ids:
            idxs = mapping[eid][cid]
            edge_train_indices[j].extend(idxs)
            loader = DataLoader(
                Subset(trainset, idxs), batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False, drop_last=False,
            )
            client_loaders.append(loader)
            client_to_edge.append(j)
    return client_loaders, client_to_edge, edge_train_indices


def build_edge_test_loaders(testset, num_edges, alpha_server, seed, num_workers):
    """One test loader per edge. Use the same hierarchical Dirichlet (outer level)
    as training, with `clients_per_server=1` so each edge gets one test slice."""
    test_mapping = hier_dirichlet_indices(
        labels=testset.targets,
        num_servers=num_edges,
        clients_per_server=1,
        alpha_server=alpha_server,
        alpha_client=alpha_server,  # ignored at clients_per_server=1
        seed=seed,
    )
    edge_test_loaders: list[DataLoader] = []
    edge_test_indices: list[list[int]] = []
    for j in range(num_edges):
        idxs = test_mapping[str(j)]["0"]
        edge_test_indices.append(idxs)
        edge_test_loaders.append(DataLoader(
            Subset(testset, idxs), batch_size=256, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        ))
    return edge_test_loaders, edge_test_indices


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, int]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item()
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += yb.size(0)
    if total_n == 0:
        return float("nan"), float("nan"), 0
    return total_loss / total_n, total_correct / total_n, total_n


# ---------------------------------------------------------------------------
# Local MTGC update
# ---------------------------------------------------------------------------
def local_mtgc_update(
    model: nn.Module,
    loader: DataLoader,
    state_diff: torch.Tensor,
    n_minibatch: int,
    lr: float,
    weight_decay: float,
    clip_norm: float,
    device: torch.device,
) -> None:
    """Run `n_minibatch` SGD steps on loss = CE + <w, state_diff>."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    state_diff = state_diff.to(device)
    steps = 0
    while steps < n_minibatch:
        for xb, yb in loader:
            if steps >= n_minibatch:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            ce = loss_fn(logits, yb)
            flat_w = torch.cat([p.reshape(-1) for p in model.parameters()])
            algo = torch.dot(flat_w, state_diff)
            loss = ce + algo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            steps += 1


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------
def edge_label_matrix(labels: np.ndarray, edge_train_indices: list[list[int]], n_classes: int = 10) -> np.ndarray:
    M = len(edge_train_indices)
    mat = np.zeros((M, n_classes), dtype=np.int64)
    for j, idxs in enumerate(edge_train_indices):
        if idxs:
            counts = np.bincount(labels[idxs], minlength=n_classes)
            mat[j] = counts
    return mat


def code_version_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def round_bits_default(n_par: int, n_clients: int, n_edges: int, edge_rounds: int) -> int:
    """Default per-round comms cost (full-participation hierarchical FL).

    Each of the E edge rounds within a global round: every client uploads its
    model to its edge and receives the new edge model back. Then once per
    global round each edge uploads to the cloud and the cloud broadcasts back.
    Counts both uplink and downlink, 32-bit floats by default.
    """
    client_edge = 2 * n_clients * edge_rounds * n_par * BITS_PER_PARAM
    edge_cloud = 2 * n_edges * n_par * BITS_PER_PARAM
    return int(client_edge + edge_cloud)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data",
                   help="Where CIFAR-10 lives (or will be downloaded on first run).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: ../gc_results/mtgc_seed{seed}/  (under fmnist/gc_results/)")
    p.add_argument("--num-edges", type=int, default=10)
    p.add_argument("--clients-per-edge", type=int, default=10)
    p.add_argument("--alpha-server", type=float, default=0.1,
                   help="Outer Dirichlet concentration (severe non-IID = 0.1).")
    p.add_argument("--alpha-client", type=float, default=0.1,
                   help="Inner Dirichlet concentration (severe non-IID = 0.1).")
    p.add_argument("--participation", type=float, default=1.0, help="rho — full participation = 1.0")
    p.add_argument("--global-rounds", type=int, default=100)
    p.add_argument("--edge-rounds", type=int, default=1, help="E in the paper")
    p.add_argument("--local-epochs", type=float, default=1.0,
                   help="Used to derive n_minibatch = ceil(epochs * samples/batch).")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-decay", type=float, default=1.0, help="Per-round multiplier on lr.")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--clip-norm", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers per client. 2 is a good default on "
                        "g2-standard-8 (8 vCPU). Use 0 for deterministic "
                        "augmentation orderings on CPU smoke tests.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eval-every", type=int, default=1)
    args = p.parse_args()

    if abs(args.participation - 1.0) > 1e-9:
        raise NotImplementedError("Full participation only (rho=1.0).")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / ".." / "gc_results" / f"mtgc_seed{args.seed}")
    out_dir = out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # ---- Data + partition --------------------------------------------------
    print(f"[mtgc] device={device}  edges={args.num_edges}  clients/edge={args.clients_per_edge}  out={out_dir}")
    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
    ])
    download = not (Path(args.data_root) / "FashionMNIST").exists()
    trainset = datasets.FashionMNIST(root=args.data_root, train=True, download=download, transform=train_tf)
    testset = datasets.FashionMNIST(root=args.data_root, train=False, download=download, transform=test_tf)
    train_labels = np.asarray(trainset.targets, dtype=np.int64)

    mapping = hier_dirichlet_indices(
        labels=train_labels,
        num_servers=args.num_edges,
        clients_per_server=args.clients_per_edge,
        alpha_server=args.alpha_server,
        alpha_client=args.alpha_client,
        seed=args.seed,
    )
    client_loaders, client_to_edge, edge_train_indices = build_train_loaders(
        trainset, mapping, args.batch_size, args.num_workers
    )
    edge_test_loaders, edge_test_indices = build_edge_test_loaders(
        testset, args.num_edges, args.alpha_server, args.seed, args.num_workers
    )
    n_clients = len(client_loaders)
    print(f"[mtgc] total clients={n_clients}  "
          f"per-client sizes: min={min(len(l.dataset) for l in client_loaders)} "
          f"max={max(len(l.dataset) for l in client_loaders)}")

    # Edge label matrix (M x 10) from training partition
    label_matrix = edge_label_matrix(train_labels, edge_train_indices, n_classes=10)
    dominant_class = label_matrix.argmax(axis=1).tolist()

    # ---- Model + state-vector machinery ------------------------------------
    avg_model = Cifar10CNN().to(device)
    n_par = num_params(avg_model)
    print(f"[mtgc] model params={n_par:,}")

    avg_flat = get_flat_params(avg_model)
    client_flats = avg_flat.unsqueeze(0).expand(n_clients, -1).clone()
    Y = torch.zeros(args.num_edges, n_par, dtype=torch.float32)
    samples_per_client = float(np.mean([len(l.dataset) for l in client_loaders]))
    n_iter_per_epoch = int(np.ceil(samples_per_client / args.batch_size))
    n_minibatch = max(1, int(np.ceil(args.local_epochs * n_iter_per_epoch)))
    print(f"[mtgc] n_minibatch per local update = {n_minibatch}")

    # ---- Global test loader (for sanity / final reporting) -----------------
    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # ---- Metadata.json (initial) ------------------------------------------
    start_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "method_name": METHOD_NAME,
        "version_hash": code_version_hash([
            script_dir / "train_mtgc.py",
            script_dir / "model.py",
            script_dir / "partitioning.py",
        ]),
        "random_seed": args.seed,
        "dataset": "Fashion-MNIST",
        "M": args.num_edges,
        "C": args.clients_per_edge,
        "rho": args.participation,
        "alpha_server": args.alpha_server,
        "alpha_client": args.alpha_client,
        "edge_label_distribution": label_matrix.tolist(),
        "client_to_edge": client_to_edge,
        "edge_train_sizes": [len(idxs) for idxs in edge_train_indices],
        "edge_test_sizes": [len(idxs) for idxs in edge_test_indices],
        "hyperparameters": vars(args),
        "n_par": n_par,
        "n_minibatch": n_minibatch,
        "hardware": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cpu_count": os.cpu_count(),
        },
        "start_time": start_iso,
        "end_time": None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ---- CSV headers -------------------------------------------------------
    per_round_csv = out_dir / "per_round.csv"
    with per_round_csv.open("w", newline="") as fp:
        csv.writer(fp).writerow([
            "round",
            "per_edge_acc_mean", "per_edge_acc_std", "per_edge_acc_min", "per_edge_acc_max",
            "per_edge_loss_mean",
            "per_edge_accs",         # JSON list[M]
            "per_edge_losses",       # JSON list[M]
            "per_edge_pers_accs",    # JSON list[M] (== per_edge_accs for MTGC)
            "global_acc", "global_loss",
            "round_time_s", "bits_transmitted",
        ])

    # ---- Main training loop ------------------------------------------------
    work_model = Cifar10CNN().to(device)
    edge_model = Cifar10CNN().to(device)

    for t in range(args.global_rounds):
        round_start = time.time()
        Z = torch.zeros(n_clients, n_par, dtype=torch.float32)
        lr_t = args.lr * (args.lr_decay ** t)

        for j in range(args.num_edges):
            set_flat_params(edge_model, avg_flat)
            edge_flat = avg_flat.clone()
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                continue

            for _e in range(args.edge_rounds):
                for c in members:
                    set_flat_params(work_model, edge_flat)
                    state_diff = Z[c] + Y[j]
                    local_mtgc_update(
                        work_model, client_loaders[c], state_diff,
                        n_minibatch=n_minibatch, lr=lr_t,
                        weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                        device=device,
                    )
                    client_flats[c] = get_flat_params(work_model)

                edge_flat = client_flats[members].mean(dim=0)
                for c in members:
                    Z[c] = Z[c] + (1.0 / (n_minibatch * lr_t)) * (client_flats[c] - edge_flat)

        avg_flat_new = client_flats.mean(dim=0)
        for j in range(args.num_edges):
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                continue
            edge_flat_j = client_flats[members].mean(dim=0)
            Y[j] = Y[j] + (1.0 / (n_minibatch * lr_t * args.edge_rounds)) * (edge_flat_j - avg_flat_new)
        avg_flat = avg_flat_new
        set_flat_params(avg_model, avg_flat)

        wall = time.time() - round_start
        if (t + 1) % args.eval_every == 0 or t == args.global_rounds - 1:
            per_edge_accs = []
            per_edge_losses = []
            for j in range(args.num_edges):
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                per_edge_accs.append(acc_j)
                per_edge_losses.append(loss_j)
            g_loss, g_acc, _ = evaluate(avg_model, global_test_loader, device)
            accs = np.asarray(per_edge_accs)
            losses = np.asarray(per_edge_losses)
            bits = round_bits_default(n_par, n_clients, args.num_edges, args.edge_rounds)
            print(f"[mtgc] round {t + 1:3d}/{args.global_rounds}  "
                  f"global_acc={g_acc:.4f}  per_edge mean={accs.mean():.4f}±{accs.std():.4f}  "
                  f"({wall:.1f}s)")
            with per_round_csv.open("a", newline="") as fp:
                csv.writer(fp).writerow([
                    t + 1,
                    f"{accs.mean():.6f}", f"{accs.std():.6f}",
                    f"{accs.min():.6f}", f"{accs.max():.6f}",
                    f"{losses.mean():.6f}",
                    json.dumps([round(x, 6) for x in per_edge_accs]),
                    json.dumps([round(x, 6) for x in per_edge_losses]),
                    json.dumps([round(x, 6) for x in per_edge_accs]),  # pers == global for MTGC
                    f"{g_acc:.6f}", f"{g_loss:.6f}",
                    f"{wall:.2f}", bits,
                ])

    # ---- Final per-edge CSV + model snapshots ------------------------------
    final_csv = out_dir / "final_per_edge_acc.csv"
    with final_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "edge_id", "local_test_acc", "local_test_loss", "global_test_acc",
            "n_train_samples", "n_test_samples", "dominant_class", "cluster_id",
        ])
        g_loss_final, g_acc_final, _ = evaluate(avg_model, global_test_loader, device)
        for j in range(args.num_edges):
            loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
            w.writerow([
                j,
                f"{acc_j:.6f}", f"{loss_j:.6f}", f"{g_acc_final:.6f}",
                len(edge_train_indices[j]), len(edge_test_indices[j]),
                int(dominant_class[j]), j,  # cluster_id = edge_id for MTGC
            ])

    # Save the global model + per-edge served snapshots (= global for MTGC)
    torch.save(avg_model.state_dict(), out_dir / "models" / "global.pt")
    for j in range(args.num_edges):
        torch.save(avg_model.state_dict(), out_dir / "models" / f"edge_{j}.pt")

    # Finalize metadata
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["final_global_acc"] = g_acc_final
    metadata["final_global_loss"] = g_loss_final
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[mtgc] done. Final global_acc={g_acc_final:.4f}. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
