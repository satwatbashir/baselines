"""Fedge training driver for CIFAR-10 (10 edges x 10 clients, full participation).

Implements the framework of Chapter 3 with:
    - Variable cloud-tier family W^t via split-merge inference
    - Intra-Coupled Soft Membership (ICSM, novel) for Pi^t
    - Sensitivity parameter lambda controlling P-C-R trade-off position
    - Two selectable inference modes: heuristic vs Metropolis-Hastings

Output schema matches the other baselines, plus Fedge-specific columns:
    K_t              -- current cluster count
    pi_entropy       -- mean per-edge Shannon entropy of Pi^t
    n_splits         -- splits accepted this round
    n_merges         -- merges accepted this round
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
from torchvision import datasets, transforms

from model import Cifar10CNN, get_flat_params, set_flat_params, num_params
from partitioning import hier_dirichlet_indices
from fedge_core import (
    FedgeHParams, lambda_to_hparams,
    compute_intra_signals, compute_inter_signals,
    compute_membership_icsm,
    heuristic_split_merge, mh_split_merge,
    aggregate_by_membership, blend_to_edge, membership_entropy,
)


METHOD_NAME = "Fedge"
DATASET_NAME = "CIFAR-10"
BITS_PER_PARAM = 32
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ---------------------------------------------------------------------------
# Data loaders (same as cifar10 baselines)
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
def edge_label_matrix(labels, edge_train_indices, n_classes=10):
    M = len(edge_train_indices)
    mat = np.zeros((M, n_classes), dtype=np.int64)
    for j, idxs in enumerate(edge_train_indices):
        if idxs:
            mat[j] = np.bincount(np.asarray(labels)[idxs], minlength=n_classes)
    return mat


def code_version_hash(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def round_bits_default(n_par, n_clients, n_edges, edge_rounds):
    """Per-round comms. Fedge adds M scalars per round (intra-edge dispersion);
    negligible vs the model transmission."""
    base = 2 * n_clients * edge_rounds * n_par * BITS_PER_PARAM + 2 * n_edges * n_par * BITS_PER_PARAM
    return int(base + n_edges * BITS_PER_PARAM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--num-edges", type=int, default=10)
    p.add_argument("--clients-per-edge", type=int, default=10)
    p.add_argument("--alpha-server", type=float, default=0.1)
    p.add_argument("--alpha-client", type=float, default=0.1)
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
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eval-every", type=int, default=1)
    # ---- Fedge-specific args ----
    p.add_argument("--lambda-val", type=float, default=0.5,
                   help="Sensitivity parameter in [0, 1]. 0 = K=1 (HierFAVG); 1 = K=M (per-edge).")
    p.add_argument("--K-max", type=int, default=10, help="Cap on cluster count (default M).")
    p.add_argument("--inference", choices=["heuristic", "mh"], default="heuristic",
                   help="Split-merge inference mode: 'heuristic' (signal-thresholded) or 'mh' (Metropolis-Hastings).")
    p.add_argument("--warmup-rounds", type=int, default=5,
                   help="Number of initial rounds before split-merge is activated.")
    args = p.parse_args()

    if abs(args.participation - 1.0) > 1e-9:
        raise NotImplementedError("Full participation only (rho=1.0).")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / ".." / "gc_results" / f"fedge_seed{args.seed}")
    out_dir = out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    print(f"[fedge] device={device}  edges={args.num_edges}  clients/edge={args.clients_per_edge}  "
          f"lambda={args.lambda_val}  inference={args.inference}  out={out_dir}")

    # ---- Hparams (lambda-derived) -----------------------------------------
    hp = lambda_to_hparams(args.lambda_val, K_max=min(args.K_max, args.num_edges))
    print(f"[fedge] hp: tau_0={hp.tau_0:.3f}  theta_split={hp.theta_split:.3f}  "
          f"theta_merge={hp.theta_merge:.3f}  alpha_0={hp.alpha_0:.3e}  K_max={hp.K_max}")

    # ---- Data + partition --------------------------------------------------
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
    print(f"[fedge] total clients={n_clients}  per-client sizes: "
          f"min={int(client_sizes.min())} max={int(client_sizes.max())}")

    label_matrix = edge_label_matrix(train_labels, edge_train_indices, n_classes=10)
    dominant_class = label_matrix.argmax(axis=1).tolist()

    # ---- Initial state -----------------------------------------------------
    avg_model = Cifar10CNN().to(device)
    n_par = num_params(avg_model)
    print(f"[fedge] model params={n_par:,}")

    init_flat = get_flat_params(avg_model)
    W = init_flat.unsqueeze(0).clone()                        # (K=1, n_par)
    Pi = torch.ones(args.num_edges, 1, dtype=torch.float32)   # (M, 1) all edges in cluster 0
    centroids = init_flat.unsqueeze(0).clone()                # (K=1, n_par)
    member_indices = [list(range(args.num_edges))]            # cluster 0 has all edges

    samples_per_client = float(np.mean(client_sizes))
    n_iter_per_epoch = int(np.ceil(samples_per_client / args.batch_size))
    n_minibatch = max(1, int(np.ceil(args.local_epochs * n_iter_per_epoch)))
    print(f"[fedge] n_minibatch per local update = {n_minibatch}")

    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # ---- Metadata.json -----------------------------------------------------
    start_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "method_name": METHOD_NAME,
        "version_hash": code_version_hash([
            script_dir / "train_fedge.py", script_dir / "fedge_core.py",
            script_dir / "model.py", script_dir / "partitioning.py",
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
        "n_par": n_par, "n_minibatch": n_minibatch,
        "fedge_hparams": {
            "lambda_val": hp.lambda_val, "tau_0": hp.tau_0,
            "theta_split": hp.theta_split, "theta_merge": hp.theta_merge,
            "alpha_0": hp.alpha_0, "K_max": hp.K_max,
            "beta_intra": hp.beta_intra, "rho": hp.rho,
            "inference": args.inference, "warmup_rounds": args.warmup_rounds,
        },
        "aggregation": "Fedge: membership-weighted (Eq. 3.4) + ICSM Pi update",
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
            # Fedge-specific:
            "K_t", "pi_entropy", "n_splits", "n_merges",
            "intra_signal_mean", "intra_signal_max",
            "inter_signal_mean", "membership", "cluster_assignment",
        ])

    # ---- Reusable work models ----------------------------------------------
    work_model = Cifar10CNN().to(device)
    edge_model = Cifar10CNN().to(device)
    eval_model = Cifar10CNN().to(device)

    # State across rounds
    client_flats = init_flat.unsqueeze(0).expand(n_clients, -1).clone()
    edge_flats = init_flat.unsqueeze(0).expand(args.num_edges, -1).clone()

    T0_stabilized = None  # round at which K stops changing

    for t in range(args.global_rounds):
        round_start = time.time()
        lr_t = args.lr * (args.lr_decay ** t)

        # ---- 1) Blend cloud family -> per-edge starting model (Eq. 3.2) ----
        edge_start_models = blend_to_edge(W, Pi)   # (M, n_par)

        # ---- 2) Edge training: each edge runs FedAvg on its blended model --
        for j in range(args.num_edges):
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                edge_flats[j] = edge_start_models[j]
                continue
            members_sizes = client_sizes[members]
            members_total = float(members_sizes.sum())
            edge_w = edge_start_models[j].clone()

            for _e in range(args.edge_rounds):
                set_flat_params(edge_model, edge_w)
                for c in members:
                    set_flat_params(work_model, edge_w)
                    local_fedavg_update(
                        work_model, client_loaders[c],
                        n_minibatch=n_minibatch, lr=lr_t,
                        weight_decay=args.weight_decay, clip_norm=args.clip_norm, device=device,
                    )
                    client_flats[c] = get_flat_params(work_model)
                weights = torch.tensor(members_sizes / members_total, dtype=torch.float32).unsqueeze(1)
                edge_w = (client_flats[members] * weights).sum(dim=0)
            edge_flats[j] = edge_w

        # ---- 3) Compute two-level signals ----------------------------------
        zeta_intra = compute_intra_signals(client_flats, client_to_edge, args.num_edges)
        inter = compute_inter_signals(edge_flats)

        # ---- 4) Inference: split-merge after warm-up -----------------------
        n_splits, n_merges = 0, 0
        K_old = len(centroids)
        if t + 1 > args.warmup_rounds:
            # Update centroids to current edge means within each cluster
            centroids = torch.stack(
                [edge_flats[m].mean(dim=0) if len(m) > 0 else centroids[i]
                 for i, m in enumerate(member_indices)], dim=0,
            )
            if args.inference == "heuristic":
                K_new, centroids, member_indices, n_splits, n_merges = heuristic_split_merge(
                    K_old, centroids, member_indices, edge_flats, zeta_intra, inter, hp, rng,
                )
            else:  # mh
                K_new, centroids, member_indices, n_splits, n_merges = mh_split_merge(
                    K_old, centroids, member_indices, edge_flats, zeta_intra, inter, hp, rng,
                )
        else:
            K_new = K_old

        if T0_stabilized is None and t > args.warmup_rounds and n_splits == 0 and n_merges == 0 and K_new == K_old:
            # Two consecutive rounds with no change -> mark stabilization
            pass  # we let it drift; T0 logged when no change for N consecutive (simple: first such round)

        # ---- 5) ICSM membership update (novel) -----------------------------
        # Reshape Pi if K changed; fall back to None to skip smoothing
        prev_pi_for_smoothing = Pi if Pi.shape[1] == K_new else None
        Pi = compute_membership_icsm(edge_flats, centroids, zeta_intra, hp, prev_pi=prev_pi_for_smoothing)

        # ---- 6) Aggregate -> new cloud family W (Eq. 3.4) ------------------
        edge_n = torch.tensor(edge_sizes, dtype=torch.float32)
        W = aggregate_by_membership(edge_flats, Pi, edge_n)
        # Also track a "global reference" model = uniform-weighted mean over edges, for reporting
        global_ref = (edge_flats * torch.tensor(edge_sizes / total_size, dtype=torch.float32).unsqueeze(1)).sum(dim=0)
        set_flat_params(avg_model, global_ref)

        wall = time.time() - round_start

        if (t + 1) % args.eval_every == 0 or t == args.global_rounds - 1:
            # ---- Eval ------------------------------------------------------
            per_edge_accs, per_edge_losses, per_edge_pers_accs = [], [], []
            # Pre-compute blended models for served accuracy
            served = blend_to_edge(W, Pi)  # (M, n_par)
            for j in range(args.num_edges):
                # per_edge_acc: GLOBAL reference (uniform mean over edges)
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                per_edge_accs.append(acc_j); per_edge_losses.append(loss_j)
                # per_edge_pers_acc: edge j's BLENDED served model
                set_flat_params(eval_model, served[j])
                _, acc_p, _ = evaluate(eval_model, edge_test_loaders[j], device)
                per_edge_pers_accs.append(acc_p)
            g_loss, g_acc, _ = evaluate(avg_model, global_test_loader, device)
            accs = np.asarray(per_edge_accs); losses = np.asarray(per_edge_losses)
            pers = np.asarray(per_edge_pers_accs)
            ent = membership_entropy(Pi)
            cluster_assign = Pi.argmax(dim=1).tolist()
            bits = round_bits_default(n_par, n_clients, args.num_edges, args.edge_rounds)
            print(f"[fedge] r {t + 1:3d}/{args.global_rounds}  "
                  f"K={K_new}  ent={ent:.3f}  splits={n_splits} merges={n_merges}  "
                  f"global_acc={g_acc:.4f}  per_edge mean={accs.mean():.4f}  "
                  f"pers mean={pers.mean():.4f}  ({wall:.1f}s)")
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
                    K_new, f"{ent:.6f}", n_splits, n_merges,
                    f"{zeta_intra.mean().item():.6f}", f"{zeta_intra.max().item():.6f}",
                    f"{inter[inter > 0].mean().item():.6f}" if (inter > 0).any() else "0.0",
                    json.dumps([[round(float(x), 4) for x in row] for row in Pi.tolist()]),
                    json.dumps(cluster_assign),
                ])

    # ---- Final per-edge CSV + model snapshots ------------------------------
    final_csv = out_dir / "final_per_edge_acc.csv"
    served_final = blend_to_edge(W, Pi)
    g_loss_final, g_acc_final, _ = evaluate(avg_model, global_test_loader, device)
    with final_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "edge_id", "local_test_acc", "local_test_loss", "global_test_acc",
            "n_train_samples", "n_test_samples", "dominant_class", "cluster_id",
            "pi_row",
        ])
        for j in range(args.num_edges):
            set_flat_params(eval_model, served_final[j])
            loss_j, acc_j, _ = evaluate(eval_model, edge_test_loaders[j], device)
            _, acc_glob_j, _ = evaluate(eval_model, global_test_loader, device)
            w.writerow([
                j,
                f"{acc_j:.6f}", f"{loss_j:.6f}", f"{acc_glob_j:.6f}",
                len(edge_train_indices[j]), len(edge_test_indices[j]),
                int(dominant_class[j]), int(Pi[j].argmax().item()),
                json.dumps([round(float(x), 4) for x in Pi[j].tolist()]),
            ])

    torch.save(avg_model.state_dict(), out_dir / "models" / "global.pt")
    # Save K cluster models
    for k in range(W.shape[0]):
        set_flat_params(eval_model, W[k])
        torch.save(eval_model.state_dict(), out_dir / "models" / f"cluster_{k}.pt")
    # Save each edge's served (blended) model
    for j in range(args.num_edges):
        set_flat_params(eval_model, served_final[j])
        torch.save({
            "state_dict": eval_model.state_dict(),
            "edge_id": j,
            "pi_row": [round(float(x), 6) for x in Pi[j].tolist()],
            "cluster_id": int(Pi[j].argmax().item()),
            "note": "Served model = sum_k Pi_{e,k} * w_k (parameter-space blend; Eq. 3.2).",
        }, out_dir / "models" / f"edge_{j}.pt")

    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["final_global_acc"] = g_acc_final
    metadata["final_global_loss"] = g_loss_final
    metadata["final_K"] = int(W.shape[0])
    metadata["final_membership"] = [[round(float(x), 6) for x in row] for row in Pi.tolist()]
    metadata["final_cluster_assignment"] = Pi.argmax(dim=1).tolist()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[fedge] done. Final K={W.shape[0]}  global_acc={g_acc_final:.4f}. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
