"""CHPFL training driver for CIFAR-10 (10 edges x 10 clients, full participation).

Implements CHPFL (Song et al., 2025) — Fixed-K clustered HFL with K-Means++:
    Clustered Hierarchical Personalized Federated Learning. The cloud groups
    the M edge servers into K (< M) hard clusters via K-Means++ on edge-model
    deltas; thereafter each cluster maintains its own model and edges in the
    same cluster share it.

Phases:
    Phase 1 (rounds 1..T_warmup):
        Vanilla hierarchical FedAvg (weighted at edge and cloud). Every edge
        is served the single global model. cluster_id = 0 for everyone.
    Phase 2 (rounds T_warmup+1..T):
        At t = T_warmup+1, run K-Means++ on edge-model deltas (current
        edge model minus global model) to produce a fixed assignment
        cluster_id[j] in {0,...,K-1}. The cluster assignment is then held
        for the remainder of training.
        Per round:
            (a) Each edge j starts from cluster_model[cluster_id[j]].
            (b) Clients in edge j do local SGD; edge weighted-aggregates.
            (c) Within each cluster, edges weighted-aggregate -> new
                cluster_model[k]. Edges in the same cluster share this model.
            (d) The cloud also computes a global model (weighted mean over
                ALL edges) for reporting only — it is not served post-warmup.

Served model at edge j after warmup = cluster_model[cluster_id[j]].
Output schema matches MTGC/HierFAVG/ESPerHFL. `per_edge_pers_accs[j]` is the
served (cluster) model on edge j's local test; `per_edge_accs[j]` is the
global model on the same slice (for cross-baseline comparison).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import Cifar10CNN, get_flat_params, set_flat_params, num_params
from partitioning import hier_dirichlet_indices


METHOD_NAME = "CHPFL"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
BITS_PER_PARAM = 32


# ---------------------------------------------------------------------------
# K-Means++ (pure numpy, deterministic given seed)
# ---------------------------------------------------------------------------
def kmeans_plusplus(X: np.ndarray, k: int, seed: int, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """K-Means++ on (n, d) feature matrix X. Returns labels (n,).

    Picks initial centers via D^2-weighted sampling, then runs Lloyd's
    iterations. Empty-cluster guard: if a cluster ends up empty during an
    iteration, reseed its center from the farthest point of the largest
    cluster (standard fallback)."""
    rng = np.random.default_rng(int(seed))
    n, d = X.shape
    if k >= n:
        return np.arange(n, dtype=np.int64) % k

    # ---- K-Means++ initialization ------------------------------------------
    centers_idx = [int(rng.integers(n))]
    for _ in range(k - 1):
        dist_sq = np.min(
            np.stack([np.sum((X - X[c]) ** 2, axis=1) for c in centers_idx], axis=0),
            axis=0,
        )
        total = float(dist_sq.sum())
        if total <= 0:
            remaining = [i for i in range(n) if i not in centers_idx]
            centers_idx.append(int(rng.choice(remaining)))
        else:
            probs = dist_sq / total
            centers_idx.append(int(rng.choice(n, p=probs)))
    centers = X[centers_idx].copy()

    # ---- Lloyd's iterations ------------------------------------------------
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for ki in range(k):
            mask = new_labels == ki
            if mask.any():
                new_centers[ki] = X[mask].mean(axis=0)
            else:
                # Empty-cluster fallback: reseed from farthest point of largest cluster.
                sizes = np.bincount(new_labels, minlength=k)
                largest = int(np.argmax(sizes))
                in_largest = np.where(new_labels == largest)[0]
                farthest = int(in_largest[np.argmax(np.linalg.norm(X[in_largest] - centers[largest], axis=1))])
                new_centers[ki] = X[farthest]
                new_labels[farthest] = ki
        if np.allclose(new_centers, centers, atol=tol):
            centers = new_centers
            labels = new_labels
            break
        centers = new_centers
        labels = new_labels
    return labels


# ---------------------------------------------------------------------------
# Data / partition (shared with MTGC/HierFAVG/ESPerHFL)
# ---------------------------------------------------------------------------
def build_train_loaders(trainset, mapping, batch_size, num_workers):
    edge_ids = sorted(mapping.keys(), key=lambda s: int(s))
    client_loaders, client_to_edge, edge_train_indices = [], [], [[] for _ in range(len(edge_ids))]
    for j, eid in enumerate(edge_ids):
        for cid in sorted(mapping[eid].keys(), key=lambda s: int(s)):
            idxs = mapping[eid][cid]
            edge_train_indices[j].extend(idxs)
            client_loaders.append(DataLoader(
                Subset(trainset, idxs), batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False, drop_last=False,
            ))
            client_to_edge.append(j)
    return client_loaders, client_to_edge, edge_train_indices


def build_edge_test_loaders(testset, num_edges, alpha_server, seed, num_workers):
    test_mapping = hier_dirichlet_indices(
        labels=testset.targets, num_servers=num_edges, clients_per_server=1,
        alpha_server=alpha_server, alpha_client=alpha_server, seed=seed,
    )
    edge_test_loaders, edge_test_indices = [], []
    for j in range(num_edges):
        idxs = test_mapping[str(j)]["0"]
        edge_test_indices.append(idxs)
        edge_test_loaders.append(DataLoader(
            Subset(testset, idxs), batch_size=256, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        ))
    return edge_test_loaders, edge_test_indices


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, int]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item()
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += yb.size(0)
    if total_n == 0:
        return float("nan"), float("nan"), 0
    return total_loss / total_n, total_correct / total_n, total_n


def local_fedavg_update(
    model: nn.Module, loader: DataLoader, n_minibatch: int, lr: float,
    weight_decay: float, clip_norm: float, device: torch.device,
) -> None:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    steps = 0
    while steps < n_minibatch:
        for xb, yb in loader:
            if steps >= n_minibatch:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            steps += 1


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def edge_label_matrix(labels: np.ndarray, edge_train_indices: list[list[int]], n_classes: int = 10) -> np.ndarray:
    M = len(edge_train_indices)
    mat = np.zeros((M, n_classes), dtype=np.int64)
    for j, idxs in enumerate(edge_train_indices):
        if idxs:
            mat[j] = np.bincount(labels[idxs], minlength=n_classes)
    return mat


def code_version_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def round_bits_default(n_par: int, n_clients: int, n_edges: int, edge_rounds: int) -> int:
    """Same default formula as MTGC/HierFAVG: 2*(N*E + M)*n_par*32 bits."""
    return int(2 * n_clients * edge_rounds * n_par * BITS_PER_PARAM
               + 2 * n_edges * n_par * BITS_PER_PARAM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default=None,
                   help="Default: ../gc_results/chpfl_seed{seed}/")
    p.add_argument("--num-edges", type=int, default=10)
    p.add_argument("--clients-per-edge", type=int, default=10)
    p.add_argument("--alpha-server", type=float, default=0.1,
                   help="Outer Dirichlet concentration (severe non-IID = 0.1).")
    p.add_argument("--alpha-client", type=float, default=0.1,
                   help="Inner Dirichlet concentration (severe non-IID = 0.1).")
    p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--global-rounds", type=int, default=200)
    p.add_argument("--edge-rounds", type=int, default=1)
    p.add_argument("--local-epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-decay", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--clip-norm", type=float, default=10.0)
    p.add_argument("--K", type=int, default=3,
                   help="Number of clusters (CHPFL is 'Fixed-K'; K < M).")
    p.add_argument("--warmup-rounds", type=int, default=5,
                   help="Vanilla HierFAVG rounds before K-Means++ clusters edges.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers per client. 2 is a good default on "
                        "g2-standard-8 (8 vCPU). Use 0 for deterministic "
                        "augmentation orderings on CPU smoke tests.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eval-every", type=int, default=1)
    args = p.parse_args()

    if abs(args.participation - 1.0) > 1e-9:
        raise NotImplementedError("Full participation only (rho=1.0).")
    if args.K >= args.num_edges:
        raise ValueError(f"K (={args.K}) must be < num_edges (={args.num_edges}) for CHPFL.")
    if args.warmup_rounds < 1:
        raise ValueError("warmup_rounds must be >= 1 so K-Means++ has signal to cluster on.")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / ".." / "gc_results" / f"chpfl_seed{args.seed}")
    out_dir = out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # ---- Data + partition --------------------------------------------------
    print(f"[chpfl] device={device}  edges={args.num_edges}  clients/edge={args.clients_per_edge}  "
          f"K={args.K}  warmup={args.warmup_rounds}  out={out_dir}")
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    download = not (Path(args.data_root) / "cifar-10-batches-py").exists()
    trainset = datasets.CIFAR10(root=args.data_root, train=True, download=download, transform=train_tf)
    testset = datasets.CIFAR10(root=args.data_root, train=False, download=download, transform=test_tf)
    train_labels = np.asarray(trainset.targets, dtype=np.int64)

    mapping = hier_dirichlet_indices(
        labels=train_labels, num_servers=args.num_edges,
        clients_per_server=args.clients_per_edge,
        alpha_server=args.alpha_server, alpha_client=args.alpha_client, seed=args.seed,
    )
    client_loaders, client_to_edge, edge_train_indices = build_train_loaders(
        trainset, mapping, args.batch_size, args.num_workers
    )
    edge_test_loaders, edge_test_indices = build_edge_test_loaders(
        testset, args.num_edges, args.alpha_server, args.seed, args.num_workers
    )
    n_clients = len(client_loaders)
    client_sizes = np.asarray([len(l.dataset) for l in client_loaders], dtype=np.float64)
    edge_sizes = np.asarray([sum(client_sizes[c] for c in range(n_clients) if client_to_edge[c] == j)
                             for j in range(args.num_edges)], dtype=np.float64)
    total_size = float(edge_sizes.sum())
    print(f"[chpfl] total clients={n_clients}  per-client sizes: "
          f"min={int(client_sizes.min())} max={int(client_sizes.max())}")

    label_matrix = edge_label_matrix(train_labels, edge_train_indices, n_classes=10)
    dominant_class = label_matrix.argmax(axis=1).tolist()

    # ---- Models + state ----------------------------------------------------
    avg_model = Cifar10CNN().to(device)
    n_par = num_params(avg_model)
    print(f"[chpfl] model params={n_par:,}")
    W_global = get_flat_params(avg_model)
    edge_flats = W_global.unsqueeze(0).expand(args.num_edges, -1).clone()  # M x n_par
    cluster_flats = W_global.unsqueeze(0).expand(args.K, -1).clone()        # K x n_par
    cluster_id = np.zeros(args.num_edges, dtype=np.int64)                   # all in cluster 0 pre-warmup

    samples_per_client = float(np.mean(client_sizes))
    n_iter_per_epoch = int(np.ceil(samples_per_client / args.batch_size))
    n_minibatch = max(1, int(np.ceil(args.local_epochs * n_iter_per_epoch)))
    print(f"[chpfl] n_minibatch per local update = {n_minibatch}")

    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # ---- Metadata.json (initial) ------------------------------------------
    start_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "method_name": METHOD_NAME,
        "version_hash": code_version_hash([
            script_dir / "train_chpfl.py",
            script_dir / "model.py",
            script_dir / "partitioning.py",
        ]),
        "random_seed": args.seed,
        "dataset": "CIFAR-10",
        "M": args.num_edges, "C": args.clients_per_edge, "rho": args.participation,
        "alpha_server": args.alpha_server, "alpha_client": args.alpha_client,
        "edge_label_distribution": label_matrix.tolist(),
        "client_to_edge": client_to_edge,
        "edge_train_sizes": [len(idxs) for idxs in edge_train_indices],
        "edge_test_sizes": [len(idxs) for idxs in edge_test_indices],
        "hyperparameters": vars(args),
        "n_par": n_par, "n_minibatch": n_minibatch,
        "aggregation": "weighted_by_sample_count",
        "clustering": {"K": args.K, "warmup_rounds": args.warmup_rounds,
                       "init": "k-means++", "feature": "edge_model_delta_from_global",
                       "schedule": "once_at_warmup_end"},
        "hardware": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cpu_count": os.cpu_count(),
        },
        "start_time": start_iso, "end_time": None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    per_round_csv = out_dir / "per_round.csv"
    with per_round_csv.open("w", newline="") as fp:
        csv.writer(fp).writerow([
            "round",
            "per_edge_acc_mean", "per_edge_acc_std", "per_edge_acc_min", "per_edge_acc_max",
            "per_edge_loss_mean",
            "per_edge_accs", "per_edge_losses",
            "per_edge_pers_accs",      # cluster (served) model on edge local test
            "global_acc", "global_loss",
            "round_time_s", "bits_transmitted",
            "K", "phase",              # CHPFL-only: phase = "warmup" or "clustered"
            "cluster_assignment",      # JSON list[M] of cluster_id (constant after warmup)
        ])

    work_model = Cifar10CNN().to(device)
    cluster_eval_model = Cifar10CNN().to(device)
    clustered = False  # flips True at round T_warmup+1

    # ---- Main training loop ------------------------------------------------
    for t in range(args.global_rounds):
        round_start = time.time()
        lr_t = args.lr * (args.lr_decay ** t)
        phase = "clustered" if clustered else "warmup"

        new_edge_flats = torch.zeros(args.num_edges, n_par, dtype=torch.float32)
        for j in range(args.num_edges):
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                new_edge_flats[j] = edge_flats[j]
                continue
            # Starting point for edge j: cluster model post-warmup, else global.
            if clustered:
                start_flat = cluster_flats[cluster_id[j]].clone()
            else:
                start_flat = W_global.clone()

            members_sizes = client_sizes[members]
            members_total = float(members_sizes.sum())
            edge_flat = start_flat

            for _e in range(args.edge_rounds):
                client_flats = torch.zeros(len(members), n_par, dtype=torch.float32)
                for k, c in enumerate(members):
                    set_flat_params(work_model, edge_flat)
                    local_fedavg_update(
                        work_model, client_loaders[c],
                        n_minibatch=n_minibatch, lr=lr_t,
                        weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                        device=device,
                    )
                    client_flats[k] = get_flat_params(work_model)
                weights = torch.tensor(members_sizes / members_total, dtype=torch.float32).unsqueeze(1)
                edge_flat = (client_flats * weights).sum(dim=0)
            new_edge_flats[j] = edge_flat

        edge_flats = new_edge_flats

        # Always compute the global (for reporting; pre-warmup also feeds it back)
        edge_weights = torch.tensor(edge_sizes / total_size, dtype=torch.float32).unsqueeze(1)
        W_global = (edge_flats * edge_weights).sum(dim=0)
        set_flat_params(avg_model, W_global)

        # ---- Clustering trigger (once, at end of warmup) -------------------
        if not clustered and (t + 1) == args.warmup_rounds:
            # Feature = edge model delta from current global (size M x n_par).
            deltas = (edge_flats - W_global.unsqueeze(0)).numpy().astype(np.float64)
            cluster_id = kmeans_plusplus(deltas, args.K, seed=args.seed)
            # Initialize cluster models = weighted mean of edges in each cluster.
            for k in range(args.K):
                in_k = np.where(cluster_id == k)[0]
                if len(in_k) == 0:
                    cluster_flats[k] = W_global.clone()
                    continue
                w_k = torch.tensor(edge_sizes[in_k] / edge_sizes[in_k].sum(), dtype=torch.float32).unsqueeze(1)
                cluster_flats[k] = (edge_flats[in_k] * w_k).sum(dim=0)
            clustered = True
            print(f"[chpfl] >>> Clustering done at round {t+1}: "
                  f"cluster_id = {cluster_id.tolist()}  "
                  f"cluster_sizes = {np.bincount(cluster_id, minlength=args.K).tolist()}")
        elif clustered:
            # Update cluster models = weighted mean of edges in each cluster.
            new_cluster_flats = torch.zeros(args.K, n_par, dtype=torch.float32)
            for k in range(args.K):
                in_k = np.where(cluster_id == k)[0]
                if len(in_k) == 0:
                    new_cluster_flats[k] = cluster_flats[k]
                    continue
                w_k = torch.tensor(edge_sizes[in_k] / edge_sizes[in_k].sum(), dtype=torch.float32).unsqueeze(1)
                new_cluster_flats[k] = (edge_flats[in_k] * w_k).sum(dim=0)
            cluster_flats = new_cluster_flats

        wall = time.time() - round_start

        # ---- Evaluation + CSV row ------------------------------------------
        if (t + 1) % args.eval_every == 0 or t == args.global_rounds - 1:
            per_edge_accs, per_edge_losses, per_edge_pers_accs = [], [], []
            for j in range(args.num_edges):
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                per_edge_accs.append(acc_j); per_edge_losses.append(loss_j)
                # Served model = global pre-warmup, cluster_model[k] post-warmup
                if clustered:
                    set_flat_params(cluster_eval_model, cluster_flats[cluster_id[j]])
                    _, acc_p, _ = evaluate(cluster_eval_model, edge_test_loaders[j], device)
                else:
                    acc_p = acc_j  # warmup: served = global
                per_edge_pers_accs.append(acc_p)
            g_loss, g_acc, _ = evaluate(avg_model, global_test_loader, device)
            accs = np.asarray(per_edge_accs); losses = np.asarray(per_edge_losses)
            pers = np.asarray(per_edge_pers_accs)
            bits = round_bits_default(n_par, n_clients, args.num_edges, args.edge_rounds)
            print(f"[chpfl] round {t + 1:3d}/{args.global_rounds}  "
                  f"global_acc={g_acc:.4f}  per_edge mean={accs.mean():.4f}  "
                  f"pers mean={pers.mean():.4f}  phase={phase}  ({wall:.1f}s)")
            with per_round_csv.open("a", newline="") as fp:
                csv.writer(fp).writerow([
                    t + 1,
                    f"{accs.mean():.6f}", f"{accs.std():.6f}",
                    f"{accs.min():.6f}", f"{accs.max():.6f}",
                    f"{losses.mean():.6f}",
                    json.dumps([round(x, 6) for x in per_edge_accs]),
                    json.dumps([round(x, 6) for x in per_edge_losses]),
                    json.dumps([round(x, 6) for x in per_edge_pers_accs]),
                    f"{g_acc:.6f}", f"{g_loss:.6f}",
                    f"{wall:.2f}", bits,
                    args.K, phase,
                    json.dumps(cluster_id.tolist()),
                ])

    # ---- Final per-edge CSV + model snapshots ------------------------------
    final_csv = out_dir / "final_per_edge_acc.csv"
    with final_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "edge_id", "local_test_acc", "local_test_loss", "global_test_acc",
            "n_train_samples", "n_test_samples", "dominant_class", "cluster_id",
        ])
        # local_test_acc/loss = SERVED (cluster) model on edge's local test slice
        # global_test_acc    = SERVED (cluster) model on full 10k test
        for j in range(args.num_edges):
            if clustered:
                set_flat_params(cluster_eval_model, cluster_flats[cluster_id[j]])
                loss_j, acc_j, _ = evaluate(cluster_eval_model, edge_test_loaders[j], device)
                _, acc_glob_j, _ = evaluate(cluster_eval_model, global_test_loader, device)
            else:
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                _, acc_glob_j, _ = evaluate(avg_model, global_test_loader, device)
            w.writerow([
                j,
                f"{acc_j:.6f}", f"{loss_j:.6f}", f"{acc_glob_j:.6f}",
                len(edge_train_indices[j]), len(edge_test_indices[j]),
                int(dominant_class[j]), int(cluster_id[j]),
            ])

    # Save global, K cluster models, and a per-edge snapshot (= cluster model)
    torch.save(avg_model.state_dict(), out_dir / "models" / "global.pt")
    for k in range(args.K):
        set_flat_params(cluster_eval_model, cluster_flats[k])
        torch.save(cluster_eval_model.state_dict(), out_dir / "models" / f"cluster_{k}.pt")
    for j in range(args.num_edges):
        if clustered:
            set_flat_params(cluster_eval_model, cluster_flats[cluster_id[j]])
        else:
            cluster_eval_model.load_state_dict(avg_model.state_dict())
        torch.save({
            "state_dict": cluster_eval_model.state_dict(),
            "edge_id": j, "cluster_id": int(cluster_id[j]),
            "note": "Served model for this edge = its cluster's model.",
        }, out_dir / "models" / f"edge_{j}.pt")

    g_loss_final, g_acc_final, _ = evaluate(avg_model, global_test_loader, device)
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["final_global_acc"] = g_acc_final
    metadata["final_global_loss"] = g_loss_final
    metadata["final_cluster_assignment"] = cluster_id.tolist()
    metadata["clustered"] = bool(clustered)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[chpfl] done. Final global_acc={g_acc_final:.4f}. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
