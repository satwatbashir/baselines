"""HierFAVG training driver for HHAR (10 edges x 10 clients, full participation).

Same algorithm as cifar10/HierFAVG/train_hierfavg.py (nested weighted FedAvg)
but for the HHAR sensor-time-series dataset:
  - model:      HARNet (1D-CNN + GRU + FC, ~152k params)
  - input:      6-channel acc+gyro windows of length 100 (2 s at 50 Hz)
  - n_classes:  6 (walking, sitting, standing, biking, stairsup, stairsdown)
Output schema identical to the image-dataset baselines.
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import load_hhar_data, HHARTensorDataset, NUM_CLASSES
from model import HARNet, get_flat_params, set_flat_params, num_params
from partitioning import hier_dirichlet_indices


METHOD_NAME = "HierFAVG"
BITS_PER_PARAM = 32
DATASET_NAME = "HHAR"


# ---------------------------------------------------------------------------
# Train/test split + loader build (HHAR-specific)
# ---------------------------------------------------------------------------
def load_and_split(args) -> tuple[HHARTensorDataset, HHARTensorDataset, np.ndarray, np.ndarray]:
    """Load HHAR, deterministic 80/20 split, return (trainset, testset, train_y, test_y)."""
    X, y = load_hhar_data(
        data_root=args.data_root,
        use_watches=True,
        sample_rate_hz=50,
        window_seconds=2,
        window_stride_seconds=1,
        cache_dir=str(Path(args.data_root) / "hhar_cache"),
    )
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(len(X))
    n_test = int(0.2 * len(X))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]
    trainset = HHARTensorDataset(train_X, train_y, compute_stats=True)
    testset = HHARTensorDataset(test_X, test_y, means=trainset.means, stds=trainset.stds)
    return trainset, testset, np.asarray(train_y, dtype=np.int64), np.asarray(test_y, dtype=np.int64)


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


def build_edge_test_loaders(testset, test_y, num_edges, alpha_server, seed, num_workers):
    test_mapping = hier_dirichlet_indices(
        labels=test_y, num_servers=num_edges, clients_per_server=1,
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


# ---------------------------------------------------------------------------
# Evaluation + local update  (same shape as cifar10 version)
# ---------------------------------------------------------------------------
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


def local_fedavg_update(model, loader, n_minibatch, lr, weight_decay, clip_norm, device):
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
def edge_label_matrix(labels: np.ndarray, edge_train_indices, n_classes: int) -> np.ndarray:
    M = len(edge_train_indices)
    mat = np.zeros((M, n_classes), dtype=np.int64)
    for j, idxs in enumerate(edge_train_indices):
        if idxs:
            mat[j] = np.bincount(labels[idxs], minlength=n_classes)
    return mat


def code_version_hash(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def round_bits_default(n_par, n_clients, n_edges, edge_rounds):
    return int(2 * n_clients * edge_rounds * n_par * BITS_PER_PARAM
               + 2 * n_edges * n_par * BITS_PER_PARAM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../data",
                   help="Where HHAR lives (downloads on first run). Shared across baselines.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: ../gc_results/hierfavg_seed{seed}/")
    p.add_argument("--num-edges", type=int, default=10)
    p.add_argument("--clients-per-edge", type=int, default=10)
    p.add_argument("--alpha-server", type=float, default=0.1,
                   help="Outer Dirichlet concentration (severe non-IID = 0.1).")
    p.add_argument("--alpha-client", type=float, default=0.1,
                   help="Inner Dirichlet concentration (severe non-IID = 0.1).")
    p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--global-rounds", type=int, default=100)
    p.add_argument("--edge-rounds", type=int, default=1)
    p.add_argument("--local-epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-decay", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--clip-norm", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
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
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / ".." / "gc_results" / f"hierfavg_seed{args.seed}")
    out_dir = out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    print(f"[hierfavg] device={device}  edges={args.num_edges}  clients/edge={args.clients_per_edge}  out={out_dir}")

    # ---- Data + partition --------------------------------------------------
    trainset, testset, train_labels, test_labels = load_and_split(args)
    mapping = hier_dirichlet_indices(
        labels=train_labels,
        num_servers=args.num_edges, clients_per_server=args.clients_per_edge,
        alpha_server=args.alpha_server, alpha_client=args.alpha_client,
        seed=args.seed,
    )
    client_loaders, client_to_edge, edge_train_indices = build_train_loaders(
        trainset, mapping, args.batch_size, args.num_workers
    )
    edge_test_loaders, edge_test_indices = build_edge_test_loaders(
        testset, test_labels, args.num_edges, args.alpha_server, args.seed, args.num_workers
    )
    n_clients = len(client_loaders)
    client_sizes = np.asarray([len(l.dataset) for l in client_loaders], dtype=np.float64)
    edge_sizes = np.asarray([sum(client_sizes[c] for c in range(n_clients) if client_to_edge[c] == j)
                             for j in range(args.num_edges)], dtype=np.float64)
    total_size = float(edge_sizes.sum())
    print(f"[hierfavg] total clients={n_clients}  per-client sizes: "
          f"min={int(client_sizes.min())} max={int(client_sizes.max())}  total_train={int(total_size)}")

    label_matrix = edge_label_matrix(train_labels, edge_train_indices, n_classes=NUM_CLASSES)
    dominant_class = label_matrix.argmax(axis=1).tolist()

    # ---- Model -------------------------------------------------------------
    in_ch = trainset.X.shape[1]
    avg_model = HARNet(in_ch=in_ch, num_classes=NUM_CLASSES).to(device)
    n_par = num_params(avg_model)
    print(f"[hierfavg] model params={n_par:,}  (in_ch={in_ch}, n_classes={NUM_CLASSES})")

    avg_flat = get_flat_params(avg_model)
    client_flats = avg_flat.unsqueeze(0).expand(n_clients, -1).clone()
    edge_flats = avg_flat.unsqueeze(0).expand(args.num_edges, -1).clone()

    samples_per_client = float(np.mean(client_sizes))
    n_iter_per_epoch = int(np.ceil(samples_per_client / args.batch_size))
    n_minibatch = max(1, int(np.ceil(args.local_epochs * n_iter_per_epoch)))
    print(f"[hierfavg] n_minibatch per local update = {n_minibatch}")

    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # ---- Metadata.json -----------------------------------------------------
    start_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "method_name": METHOD_NAME,
        "version_hash": code_version_hash([
            script_dir / "train_hierfavg.py", script_dir / "model.py",
            script_dir / "partitioning.py", script_dir / "dataset.py",
        ]),
        "random_seed": args.seed,
        "dataset": DATASET_NAME,
        "M": args.num_edges, "C": args.clients_per_edge, "rho": args.participation,
        "alpha_server": args.alpha_server, "alpha_client": args.alpha_client,
        "edge_label_distribution": label_matrix.tolist(),
        "client_to_edge": client_to_edge,
        "edge_train_sizes": [len(idxs) for idxs in edge_train_indices],
        "edge_test_sizes": [len(idxs) for idxs in edge_test_indices],
        "hyperparameters": vars(args),
        "n_par": n_par, "n_minibatch": n_minibatch, "in_ch": in_ch, "n_classes": NUM_CLASSES,
        "aggregation": "weighted_by_sample_count",
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
            "per_edge_accs", "per_edge_losses", "per_edge_pers_accs",
            "global_acc", "global_loss",
            "round_time_s", "bits_transmitted",
        ])

    # ---- Main training loop ------------------------------------------------
    work_model = HARNet(in_ch=in_ch, num_classes=NUM_CLASSES).to(device)
    edge_model = HARNet(in_ch=in_ch, num_classes=NUM_CLASSES).to(device)

    for t in range(args.global_rounds):
        round_start = time.time()
        lr_t = args.lr * (args.lr_decay ** t)

        for j in range(args.num_edges):
            edge_flat = avg_flat.clone()
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                continue
            members_sizes = client_sizes[members]
            members_total = float(members_sizes.sum())

            for _e in range(args.edge_rounds):
                set_flat_params(edge_model, edge_flat)
                for c in members:
                    set_flat_params(work_model, edge_flat)
                    local_fedavg_update(
                        work_model, client_loaders[c],
                        n_minibatch=n_minibatch, lr=lr_t,
                        weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                        device=device,
                    )
                    client_flats[c] = get_flat_params(work_model)
                weights = torch.tensor(members_sizes / members_total, dtype=torch.float32).unsqueeze(1)
                edge_flat = (client_flats[members] * weights).sum(dim=0)
            edge_flats[j] = edge_flat

        edge_weights = torch.tensor(edge_sizes / total_size, dtype=torch.float32).unsqueeze(1)
        avg_flat = (edge_flats * edge_weights).sum(dim=0)
        set_flat_params(avg_model, avg_flat)

        wall = time.time() - round_start
        if (t + 1) % args.eval_every == 0 or t == args.global_rounds - 1:
            per_edge_accs, per_edge_losses = [], []
            for j in range(args.num_edges):
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                per_edge_accs.append(acc_j); per_edge_losses.append(loss_j)
            g_loss, g_acc, _ = evaluate(avg_model, global_test_loader, device)
            accs = np.asarray(per_edge_accs); losses = np.asarray(per_edge_losses)
            bits = round_bits_default(n_par, n_clients, args.num_edges, args.edge_rounds)
            print(f"[hierfavg] round {t + 1:3d}/{args.global_rounds}  "
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
                    json.dumps([round(x, 6) for x in per_edge_accs]),
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
                int(dominant_class[j]), j,
            ])

    torch.save(avg_model.state_dict(), out_dir / "models" / "global.pt")
    for j in range(args.num_edges):
        torch.save(avg_model.state_dict(), out_dir / "models" / f"edge_{j}.pt")

    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["final_global_acc"] = g_acc_final
    metadata["final_global_loss"] = g_loss_final
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[hierfavg] done. Final global_acc={g_acc_final:.4f}. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
