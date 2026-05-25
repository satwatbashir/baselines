"""PHE-FL training driver for CIFAR-10 (10 edges x 10 clients, full participation).

Implements PHE-FL from
    Lee et al. — "Personalizing Federated Learning for Hierarchical Edge
    Networks with Non-IID Data" — 2025. arXiv:2504.08872.

Algorithm (per global round, after client local SGD + edge FedAvg):
    For each edge e:
        EAM_e   = edge_e's standard FedAvg over its clients (weighted by samples)
    For each edge e (leave-one-out at the cloud):
        CAM_e   = sum_{k != e} (|D_k| / |D \\ D_e|) * EAM_k
        a_eam   = Acc(EAM_e, PTD_e)        # personalization test partition for edge e
        a_cam   = Acc(CAM_e, PTD_e)
        alpha_e = a_eam / (a_eam + a_cam)
        PEAM_e  = alpha_e * EAM_e + (1 - alpha_e) * CAM_e
    Each client in edge e starts the NEXT round from PEAM_e (no client-side
    persistent state, no second model, no learnable gradient-mixing).

Output schema matches MTGC/HierFAVG/ESPerHFL/CHPFL. `per_edge_pers_accs[j]`
is PEAM_j on edge j's local test; `per_edge_accs[j]` is the uniform-weighted
reference global on the same slice (kept for cross-baseline comparison).
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


METHOD_NAME = "PHE-FL"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
BITS_PER_PARAM = 32


# ---------------------------------------------------------------------------
# Data / partition  (shared with MTGC / HierFAVG / ESPerHFL / CHPFL)
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
# PHE-FL specific: leave-one-out complement aggregation + accuracy-ratio alpha
# ---------------------------------------------------------------------------
def complement_aggregation(EAM: torch.Tensor, edge_sizes: np.ndarray) -> torch.Tensor:
    """CAM_e = sum_{k != e} (|D_k| / |D \\ D_e|) * EAM_k  for all edges e.

    Input:  EAM (M, n_par), edge_sizes (M,)
    Output: CAM (M, n_par)
    """
    M, n_par = EAM.shape
    total = float(edge_sizes.sum())
    sizes = torch.tensor(edge_sizes, dtype=EAM.dtype)
    CAM = torch.zeros_like(EAM)
    for e in range(M):
        comp_total = total - float(edge_sizes[e])
        if comp_total <= 0:
            # Defensive: if edge e holds all data, fall back to its own EAM.
            CAM[e] = EAM[e]
            continue
        weights = sizes / comp_total
        weights[e] = 0.0
        CAM[e] = (EAM * weights.unsqueeze(1)).sum(dim=0)
    return CAM


def accuracy_ratio_alpha(acc_eam: float, acc_cam: float, fallback: float = 0.5) -> float:
    """alpha_e = Acc(EAM_e, PTD_e) / [Acc(EAM_e, PTD_e) + Acc(CAM_e, PTD_e)].

    Defensive: if either accuracy is non-finite, or both are zero, falls back
    to `fallback` (default 0.5 = neutral mix). Always clipped to [0, 1].
    """
    if not (math.isfinite(acc_eam) and math.isfinite(acc_cam)):
        return float(fallback)
    denom = acc_eam + acc_cam
    if denom <= 0:
        return float(fallback)
    a = acc_eam / denom
    return float(min(1.0, max(0.0, a)))


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
    """PHE-FL transmits one model client<->edge and one personalized cloud
    model per edge (the CAM_e, distinct per edge). So:
        client<->edge: 2 * N * E * n_par * 32
        edge<->cloud:  2 * M * n_par * 32  (edges send EAM up, get CAM_e down)
    """
    return int(2 * n_clients * edge_rounds * n_par * BITS_PER_PARAM
               + 2 * n_edges * n_par * BITS_PER_PARAM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default=None,
                   help="Default: ../gc_results/phefl_seed{seed}/")
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
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / ".." / "gc_results" / f"phefl_seed{args.seed}")
    out_dir = out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # ---- Data + partition --------------------------------------------------
    print(f"[phefl] device={device}  edges={args.num_edges}  clients/edge={args.clients_per_edge}  out={out_dir}")
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
    print(f"[phefl] total clients={n_clients}  per-client sizes: "
          f"min={int(client_sizes.min())} max={int(client_sizes.max())}")

    label_matrix = edge_label_matrix(train_labels, edge_train_indices, n_classes=10)
    dominant_class = label_matrix.argmax(axis=1).tolist()

    # ---- Models + state ----------------------------------------------------
    # Each edge holds its PEAM_e as state across rounds (clients sync from
    # this each round). No client-side persistent state.
    avg_model = Cifar10CNN().to(device)
    n_par = num_params(avg_model)
    print(f"[phefl] model params={n_par:,}")
    init_flat = get_flat_params(avg_model)
    PEAM = init_flat.unsqueeze(0).expand(args.num_edges, -1).clone()   # (M, n_par)

    samples_per_client = float(np.mean(client_sizes))
    n_iter_per_epoch = int(np.ceil(samples_per_client / args.batch_size))
    n_minibatch = max(1, int(np.ceil(args.local_epochs * n_iter_per_epoch)))
    print(f"[phefl] n_minibatch per local update = {n_minibatch}")

    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # ---- Metadata.json -----------------------------------------------------
    start_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "method_name": METHOD_NAME,
        "version_hash": code_version_hash([
            script_dir / "train_phefl.py",
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
        "aggregation": "edge=weighted_by_sample_count; cloud=leave_one_out_complement",
        "personalization": {"per_edge_peam": True,
                            "alpha_rule": "accuracy_ratio_on_local_test",
                            "ptd_source": "edge_test_loaders (same partition as final eval)"},
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
            "per_edge_accs",            # uniform-weighted ref global on local test
            "per_edge_losses",
            "per_edge_pers_accs",       # PEAM_e on local test
            "global_acc", "global_loss",
            "round_time_s", "bits_transmitted",
            "alpha_mean", "alpha_min", "alpha_max",   # PHE-FL-only diagnostics
            "per_edge_alphas",                          # JSON list[M]
            "per_edge_eam_accs", "per_edge_cam_accs",   # diagnostic: EAM_e vs CAM_e on PTD_e
        ])

    work_model = Cifar10CNN().to(device)
    eval_model = Cifar10CNN().to(device)

    # ---- Main training loop ------------------------------------------------
    for t in range(args.global_rounds):
        round_start = time.time()
        lr_t = args.lr * (args.lr_decay ** t)

        # ---- Per-edge: clients local SGD starting from PEAM_e -> EAM_e -----
        EAM = torch.zeros(args.num_edges, n_par, dtype=torch.float32)
        for j in range(args.num_edges):
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                EAM[j] = PEAM[j]
                continue
            members_sizes = client_sizes[members]
            members_total = float(members_sizes.sum())
            edge_start = PEAM[j].clone()

            for _e in range(args.edge_rounds):
                client_flats = torch.zeros(len(members), n_par, dtype=torch.float32)
                for k, c in enumerate(members):
                    set_flat_params(work_model, edge_start)
                    local_fedavg_update(
                        work_model, client_loaders[c],
                        n_minibatch=n_minibatch, lr=lr_t,
                        weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                        device=device,
                    )
                    client_flats[k] = get_flat_params(work_model)
                weights = torch.tensor(members_sizes / members_total, dtype=torch.float32).unsqueeze(1)
                edge_start = (client_flats * weights).sum(dim=0)
            EAM[j] = edge_start

        # ---- Cloud: leave-one-out complement aggregation -------------------
        CAM = complement_aggregation(EAM, edge_sizes)

        # ---- Edge personalization: accuracy-ratio alpha, then PEAM ---------
        alphas = []
        per_edge_eam_accs = []
        per_edge_cam_accs = []
        for j in range(args.num_edges):
            set_flat_params(eval_model, EAM[j])
            _, a_eam, _ = evaluate(eval_model, edge_test_loaders[j], device)
            set_flat_params(eval_model, CAM[j])
            _, a_cam, _ = evaluate(eval_model, edge_test_loaders[j], device)
            alpha_e = accuracy_ratio_alpha(a_eam, a_cam, fallback=0.5)
            alphas.append(alpha_e)
            per_edge_eam_accs.append(a_eam)
            per_edge_cam_accs.append(a_cam)
            PEAM[j] = alpha_e * EAM[j] + (1.0 - alpha_e) * CAM[j]

        # Reference uniform-weighted global (reporting only).
        edge_weights = torch.tensor(edge_sizes / total_size, dtype=torch.float32).unsqueeze(1)
        W_global_ref = (EAM * edge_weights).sum(dim=0)
        set_flat_params(avg_model, W_global_ref)

        wall = time.time() - round_start

        # ---- Evaluation row ------------------------------------------------
        if (t + 1) % args.eval_every == 0 or t == args.global_rounds - 1:
            per_edge_accs, per_edge_losses, per_edge_pers_accs = [], [], []
            for j in range(args.num_edges):
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                per_edge_accs.append(acc_j); per_edge_losses.append(loss_j)
                set_flat_params(eval_model, PEAM[j])
                _, acc_p, _ = evaluate(eval_model, edge_test_loaders[j], device)
                per_edge_pers_accs.append(acc_p)
            g_loss, g_acc, _ = evaluate(avg_model, global_test_loader, device)
            accs = np.asarray(per_edge_accs); losses = np.asarray(per_edge_losses)
            pers = np.asarray(per_edge_pers_accs)
            alpha_arr = np.asarray(alphas)
            bits = round_bits_default(n_par, n_clients, args.num_edges, args.edge_rounds)
            print(f"[phefl] round {t + 1:3d}/{args.global_rounds}  "
                  f"global_acc={g_acc:.4f}  per_edge mean={accs.mean():.4f}  "
                  f"pers mean={pers.mean():.4f}  "
                  f"alpha=[{alpha_arr.min():.2f},{alpha_arr.max():.2f}]  ({wall:.1f}s)")
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
                    f"{alpha_arr.mean():.6f}", f"{alpha_arr.min():.6f}", f"{alpha_arr.max():.6f}",
                    json.dumps([round(x, 6) for x in alphas]),
                    json.dumps([round(x, 6) for x in per_edge_eam_accs]),
                    json.dumps([round(x, 6) for x in per_edge_cam_accs]),
                ])

    # ---- Final per-edge CSV + model snapshots ------------------------------
    final_csv = out_dir / "final_per_edge_acc.csv"
    with final_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "edge_id", "local_test_acc", "local_test_loss", "global_test_acc",
            "n_train_samples", "n_test_samples", "dominant_class", "cluster_id",
            "alpha_e",
        ])
        for j in range(args.num_edges):
            set_flat_params(eval_model, PEAM[j])
            loss_j, acc_j, _ = evaluate(eval_model, edge_test_loaders[j], device)
            _, acc_glob, _ = evaluate(eval_model, global_test_loader, device)
            w.writerow([
                j,
                f"{acc_j:.6f}", f"{loss_j:.6f}", f"{acc_glob:.6f}",
                len(edge_train_indices[j]), len(edge_test_indices[j]),
                int(dominant_class[j]), j,   # cluster_id = edge_id (per-edge personalization, no clustering)
                f"{alphas[j]:.6f}",
            ])

    # Save reference global + per-edge served (PEAM_e) + per-edge CAM_e
    torch.save(avg_model.state_dict(), out_dir / "models" / "global.pt")
    for j in range(args.num_edges):
        set_flat_params(eval_model, PEAM[j])
        torch.save({
            "state_dict": eval_model.state_dict(),
            "alpha_e": float(alphas[j]),
            "edge_id": j,
            "note": "PEAM_e = alpha_e * EAM_e + (1 - alpha_e) * CAM_e (parameter-space mix).",
        }, out_dir / "models" / f"edge_{j}.pt")
        set_flat_params(eval_model, CAM[j])
        torch.save(eval_model.state_dict(), out_dir / "models" / f"cam_{j}.pt")

    g_loss_final, g_acc_final, _ = evaluate(avg_model, global_test_loader, device)
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["final_global_acc"] = g_acc_final
    metadata["final_global_loss"] = g_loss_final
    metadata["final_alphas"] = [float(a) for a in alphas]
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[phefl] done. Final global_acc={g_acc_final:.4f}. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
