"""ESPerHFL training driver for CIFAR-10 (10 edges x 10 clients, full participation).

Implements ESPerHFL from
    Ma et al. — "Personalized client-edge-cloud hierarchical federated learning
    in mobile edge computing" — Journal of Cloud Computing 2024.
    Official code: github.com/xiangqianL/ESPerHFL

Algorithm (APFL-style learnable mixing extended to 3 tiers):
    Per client c: two models w_c (shared), v_c (local), scalar m_c in [0,1].
    Per local SGD step on (x, y):
        Update w_c via plain CE(w_c(x), y).
        Update v_c via CE(m_c * v_c(x) + (1 - m_c) * w_c(x), y).
        After the local update finishes, update m_c via APFL alpha-update.
    Edge j aggregates: W_j, V_j = weighted mean of w_c, v_c (by sample count).
                       m_j = uniform mean of m_c.
    Cloud aggregates ONLY W_j -> W_global (weighted by per-edge sample count).
    V_j and m_j STAY AT THE EDGE -> personalization persists across rounds.
    Served model at edge j = m_j * V_j + (1 - m_j) * W_global.

Output schema is identical to MTGC / HierFAVG. ESPerHFL fills
`per_edge_pers_accs` with the served (mixed) model accuracy, which now
differs from `per_edge_accs` (global model on local partition).
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


METHOD_NAME = "ESPerHFL"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
BITS_PER_PARAM = 32

# APFL alpha-update constants (from the paper code, alpha_update in ESPerHFL.py)
ALPHA_REG = 0.02
ALPHA_LR = 0.1

# p_average constants (from average.p_average in reference code)
P_AVERAGE_G = 5.0       # softmax sharpness on Jaccard similarity
P_AVERAGE_H = 0.7       # fraction of weight on other edges (1 - H goes to self)


def p_average(edge_flats: torch.Tensor, g: float = P_AVERAGE_G, h: float = P_AVERAGE_H) -> torch.Tensor:
    """Per-edge similarity-weighted cloud aggregation (paper's `p_average`).

    For each edge e, produce a personalized cloud model:
        cloud_e = (1 - h) * edge_e + h * sum_{k!=e} softmax(g * jaccard(edge_e, edge_k))[k] * edge_k
    where jaccard(a, b) = <a, b> / (||a||^2 + ||b||^2 - <a, b>).
    Defaults g=5, h=0.7 match `average.py::p_average` in the official repo.

    Input  : edge_flats (M, n_par)
    Output : cloud_flats (M, n_par) — one personalized cloud model per edge.
    """
    M = edge_flats.shape[0]
    inner = edge_flats @ edge_flats.T                  # (M, M)
    norm_sq = (edge_flats * edge_flats).sum(dim=1)     # (M,)
    denom = norm_sq.unsqueeze(0) + norm_sq.unsqueeze(1) - inner
    denom = torch.clamp(denom, min=1e-12)              # avoid /0 (defensive)
    sim = inner / denom                                 # Jaccard (Tanimoto) similarity
    # Mask self in softmax: set diag to -inf so softmax row puts 0 mass there.
    sim_g = g * sim
    sim_g = sim_g.masked_fill(torch.eye(M, dtype=torch.bool), float("-inf"))
    weights_off = torch.softmax(sim_g, dim=1) * h
    weights = weights_off + torch.eye(M, dtype=edge_flats.dtype) * (1.0 - h)
    cloud_flats = weights @ edge_flats                  # (M, n_par)
    # Defensive NaN scrub: if any row is non-finite (e.g. all-zero edges),
    # fall back to the uniform mean of all edges.
    bad = ~torch.isfinite(cloud_flats).all(dim=1)
    if bad.any():
        fallback = edge_flats.mean(dim=0)
        cloud_flats[bad] = fallback
    return cloud_flats


# ---------------------------------------------------------------------------
# Data / partition  (shared with MTGC and HierFAVG)
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


# ---------------------------------------------------------------------------
# Evaluation
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


@torch.no_grad()
def evaluate_mixed(model_w: nn.Module, model_v: nn.Module, mix: float,
                   loader: DataLoader, device: torch.device) -> tuple[float, float, int]:
    """Evaluate `mix * v(x) + (1 - mix) * w(x)` on the loader."""
    model_w.eval(); model_v.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = mix * model_v(xb) + (1.0 - mix) * model_w(xb)
        total_loss += loss_fn(logits, yb).item()
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += yb.size(0)
    if total_n == 0:
        return float("nan"), float("nan"), 0
    return total_loss / total_n, total_correct / total_n, total_n


# ---------------------------------------------------------------------------
# Local update (joint w + v + m, APFL-style)
# ---------------------------------------------------------------------------
def alpha_update(model_w: nn.Module, model_v: nn.Module, m: float) -> float:
    """APFL alpha-update from Ma et al. (mirrors `alpha_update` in ESPerHFL.py).

    Uses .grad from the latest backward pass on both models.
    Returns the new mixing scalar, clipped to [0, 1]. Falls back to the
    previous value if grads are missing or NaN (defensive guard)."""
    grad_alpha = torch.zeros((), dtype=torch.float32)
    have_any_grad = False
    for w_p, v_p in zip(model_w.parameters(), model_v.parameters()):
        if w_p.grad is None or v_p.grad is None:
            continue
        have_any_grad = True
        dif = v_p.data - w_p.data
        grad = m * v_p.grad.data + (1.0 - m) * w_p.grad.data
        grad_alpha = grad_alpha + dif.view(-1).cpu().dot(grad.view(-1).cpu())
    if not have_any_grad:
        return float(m)
    grad_alpha = grad_alpha + ALPHA_REG * m
    new_m = float(m) - ALPHA_LR * float(grad_alpha.item())
    if not math.isfinite(new_m):
        return float(m)
    return float(min(1.0, max(0.0, new_m)))


def local_esperhfl_update(
    model_w: nn.Module, model_v: nn.Module, m: float, loader: DataLoader,
    n_minibatch: int, lr: float, weight_decay: float, clip_norm: float,
    device: torch.device,
) -> float:
    """One client's local update: alternating SGD on w then v per batch.
    After the n_minibatch steps, run the APFL alpha-update once."""
    model_w.train(); model_v.train()
    opt_w = torch.optim.SGD(model_w.parameters(), lr=lr, weight_decay=weight_decay)
    opt_v = torch.optim.SGD(model_v.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    steps = 0
    while steps < n_minibatch:
        for xb, yb in loader:
            if steps >= n_minibatch:
                break
            xb, yb = xb.to(device), yb.to(device)

            # 1) w step on plain CE
            opt_w.zero_grad()
            loss_w = loss_fn(model_w(xb), yb)
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(model_w.parameters(), clip_norm)
            opt_w.step()

            # 2) v step on mixed-output CE
            # We need fresh grads on BOTH models for alpha_update later,
            # so don't zero w's grads here — they get repopulated below.
            opt_w.zero_grad(); opt_v.zero_grad()
            # Re-forward w (without grad accumulation issues) and v
            out_w = model_w(xb)
            out_v = model_v(xb)
            mixed = m * out_v + (1.0 - m) * out_w
            loss_v = loss_fn(mixed, yb)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(model_v.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(model_w.parameters(), clip_norm)
            opt_v.step()  # only updates v; w.grad is left intact for alpha_update

            steps += 1
    # APFL alpha-update once per local_update, using last-batch grads.
    m = alpha_update(model_w, model_v, m)
    return m


# ---------------------------------------------------------------------------
# Bookkeeping helpers
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
    """ESPerHFL transmits TWO models client<->edge (w and v), but cloud<->edge
    still moves only W. Per global round:
        client<->edge: 2 (uplink+downlink) * (w + v) per client per edge round
                       = 4 * N * E * n_par * 32 bits
        edge<->cloud:  2 * M * n_par * 32 bits (only W moves)
    """
    client_edge = 4 * n_clients * edge_rounds * n_par * BITS_PER_PARAM
    edge_cloud = 2 * n_edges * n_par * BITS_PER_PARAM
    return int(client_edge + edge_cloud)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default=None,
                   help="Default: ../gc_results/esperhfl_seed{seed}/")
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
    p.add_argument("--init-mix", type=float, default=0.5,
                   help="Initial value of the per-edge mixing scalar m_j (paper uses 0.5).")
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

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / ".." / "gc_results" / f"esperhfl_seed{args.seed}")
    out_dir = out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # ---- Data + partition --------------------------------------------------
    print(f"[esperhfl] device={device}  edges={args.num_edges}  clients/edge={args.clients_per_edge}  out={out_dir}")
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
    print(f"[esperhfl] total clients={n_clients}  per-client sizes: "
          f"min={int(client_sizes.min())} max={int(client_sizes.max())}")

    label_matrix = edge_label_matrix(train_labels, edge_train_indices, n_classes=10)
    dominant_class = label_matrix.argmax(axis=1).tolist()

    # ---- Models + state ----------------------------------------------------
    # Cloud aggregation = p_average (per-edge similarity-weighted) -> M cloud
    # models, one per edge. NOT a single global. We still track a reference
    # global = weighted mean of edges, used only for the `global_acc` column.
    # Edge state persists: V_j, m_j never touched by the cloud.
    avg_model = Cifar10CNN().to(device)
    n_par = num_params(avg_model)
    print(f"[esperhfl] model params={n_par:,}  (per client: 2x w+v = {2*n_par:,})")
    init_flat = get_flat_params(avg_model)
    W_cloud = init_flat.unsqueeze(0).expand(args.num_edges, -1).clone()   # per-edge personalized cloud
    V_edges = init_flat.unsqueeze(0).expand(args.num_edges, -1).clone()   # same init for all edges
    m_edges = torch.full((args.num_edges,), float(args.init_mix), dtype=torch.float32)
    W_global_ref = init_flat.clone()                                       # for reporting only

    samples_per_client = float(np.mean(client_sizes))
    n_iter_per_epoch = int(np.ceil(samples_per_client / args.batch_size))
    n_minibatch = max(1, int(np.ceil(args.local_epochs * n_iter_per_epoch)))
    print(f"[esperhfl] n_minibatch per local update = {n_minibatch}")

    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # ---- Metadata.json (initial) ------------------------------------------
    start_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "method_name": METHOD_NAME,
        "version_hash": code_version_hash([
            script_dir / "train_esperhfl.py",
            script_dir / "model.py",
            script_dir / "partitioning.py",
        ]),
        "random_seed": args.seed,
        "dataset": "CIFAR-10",
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
        "aggregation": "edge=weighted_by_sample_count; cloud=p_average (paper-faithful)",
        "personalization": {"per_edge_v_model": True, "per_edge_mix_scalar": True,
                            "alpha_update_lr": ALPHA_LR, "alpha_update_reg": ALPHA_REG,
                            "init_mix": args.init_mix,
                            "p_average_g": P_AVERAGE_G, "p_average_h": P_AVERAGE_H,
                            "cloud_model_count": args.num_edges},
        "hardware": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
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
            "per_edge_accs",         # global model on edge's local test
            "per_edge_losses",
            "per_edge_pers_accs",    # served (mixed) model on edge's local test
            "global_acc", "global_loss",
            "round_time_s", "bits_transmitted",
            "mix_mean", "mix_min", "mix_max",     # ESPerHFL-only diagnostics
        ])

    # ---- Reusable work models (one pair, reused across clients) ------------
    work_w = Cifar10CNN().to(device)
    work_v = Cifar10CNN().to(device)
    served_model = Cifar10CNN().to(device)  # for evaluation only

    # ---- Main training loop ------------------------------------------------
    for t in range(args.global_rounds):
        round_start = time.time()
        lr_t = args.lr * (args.lr_decay ** t)

        new_W_edges = torch.zeros(args.num_edges, n_par, dtype=torch.float32)

        for j in range(args.num_edges):
            members = [c for c in range(n_clients) if client_to_edge[c] == j]
            if not members:
                new_W_edges[j] = W_cloud[j]
                continue
            members_sizes = client_sizes[members]
            members_total = float(members_sizes.sum())

            # Each edge starts from its OWN personalized cloud model (p_average output).
            # V_j and m_j persist across rounds at the edge.
            W_j = W_cloud[j].clone()
            V_j = V_edges[j].clone()
            m_j = float(m_edges[j].item())

            for _e in range(args.edge_rounds):
                w_flats = torch.zeros(len(members), n_par, dtype=torch.float32)
                v_flats = torch.zeros(len(members), n_par, dtype=torch.float32)
                m_vals = []
                for k, c in enumerate(members):
                    set_flat_params(work_w, W_j)
                    set_flat_params(work_v, V_j)
                    m_c = local_esperhfl_update(
                        work_w, work_v, m_j, client_loaders[c],
                        n_minibatch=n_minibatch, lr=lr_t,
                        weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                        device=device,
                    )
                    w_flats[k] = get_flat_params(work_w)
                    v_flats[k] = get_flat_params(work_v)
                    m_vals.append(m_c)

                # Weighted aggregation of w and v by sample count; uniform mean of m.
                weights = torch.tensor(members_sizes / members_total, dtype=torch.float32).unsqueeze(1)
                W_j = (w_flats * weights).sum(dim=0)
                V_j = (v_flats * weights).sum(dim=0)
                m_j = float(np.mean(m_vals))

            new_W_edges[j] = W_j
            V_edges[j] = V_j
            m_edges[j] = m_j

        # Cloud aggregation: PAPER-FAITHFUL p_average -> M personalized clouds.
        W_cloud = p_average(new_W_edges)                       # (M, n_par)
        # Reference uniform-weighted global, ONLY for the `global_acc` report column.
        edge_weights = torch.tensor(edge_sizes / total_size, dtype=torch.float32).unsqueeze(1)
        W_global_ref = (new_W_edges * edge_weights).sum(dim=0)
        set_flat_params(avg_model, W_global_ref)

        wall = time.time() - round_start
        if (t + 1) % args.eval_every == 0 or t == args.global_rounds - 1:
            per_edge_accs, per_edge_losses, per_edge_pers_accs = [], [], []
            for j in range(args.num_edges):
                # Reference global model on edge's local test (for cross-baseline comparability)
                loss_j, acc_j, _ = evaluate(avg_model, edge_test_loaders[j], device)
                per_edge_accs.append(acc_j); per_edge_losses.append(loss_j)
                # Served (mixed) model on edge's local test: m_j * V_j + (1-m_j) * cloud_j
                set_flat_params(work_w, W_cloud[j])
                set_flat_params(work_v, V_edges[j])
                _, acc_p, _ = evaluate_mixed(work_w, work_v, float(m_edges[j].item()),
                                             edge_test_loaders[j], device)
                per_edge_pers_accs.append(acc_p)
            g_loss, g_acc, _ = evaluate(avg_model, global_test_loader, device)
            accs = np.asarray(per_edge_accs); losses = np.asarray(per_edge_losses)
            pers = np.asarray(per_edge_pers_accs)
            mix_arr = m_edges.cpu().numpy()
            bits = round_bits_default(n_par, n_clients, args.num_edges, args.edge_rounds)
            print(f"[esperhfl] round {t + 1:3d}/{args.global_rounds}  "
                  f"global_acc={g_acc:.4f}  per_edge mean={accs.mean():.4f}  "
                  f"pers mean={pers.mean():.4f}  "
                  f"m=[{mix_arr.min():.2f},{mix_arr.max():.2f}]  ({wall:.1f}s)")
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
                    f"{mix_arr.mean():.6f}", f"{mix_arr.min():.6f}", f"{mix_arr.max():.6f}",
                ])

    # ---- Final per-edge CSV + model snapshots ------------------------------
    final_csv = out_dir / "final_per_edge_acc.csv"
    with final_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "edge_id", "local_test_acc", "local_test_loss", "global_test_acc",
            "n_train_samples", "n_test_samples", "dominant_class", "cluster_id", "mix_scalar",
        ])
        # local_test_acc = served (personalized) model on edge's local test slice
        # global_test_acc = served (personalized) model on full 10k test set
        for j in range(args.num_edges):
            set_flat_params(work_w, W_cloud[j])
            set_flat_params(work_v, V_edges[j])
            m_j = float(m_edges[j].item())
            loss_j, acc_j, _ = evaluate_mixed(work_w, work_v, m_j, edge_test_loaders[j], device)
            _, acc_glob, _ = evaluate_mixed(work_w, work_v, m_j, global_test_loader, device)
            w.writerow([
                j,
                f"{acc_j:.6f}", f"{loss_j:.6f}", f"{acc_glob:.6f}",
                len(edge_train_indices[j]), len(edge_test_indices[j]),
                int(dominant_class[j]), j,  # cluster_id = edge_id (per-edge personalization, no clustering)
                f"{m_j:.6f}",
            ])

    # Save: reference global (uniform-weighted across edges), M personalized
    # cloud models from p_average, and each edge's served state (the actual
    # personalized model it serves to its clients).
    torch.save(avg_model.state_dict(), out_dir / "models" / "global.pt")
    for j in range(args.num_edges):
        set_flat_params(served_model, W_cloud[j])
        torch.save(served_model.state_dict(), out_dir / "models" / f"cloud_{j}.pt")
        m_j = float(m_edges[j].item())
        # Served model = m_j * V_j + (1 - m_j) * W_cloud[j]  (param-space mix)
        served_flat = m_j * V_edges[j] + (1.0 - m_j) * W_cloud[j]
        set_flat_params(served_model, served_flat)
        torch.save({
            "state_dict": served_model.state_dict(),
            "mix_scalar": m_j,
            "edge_id": j,
            "note": "Served model = m_j * V_j + (1 - m_j) * W_cloud[j] "
                    "(p_average's per-edge personalized cloud).",
        }, out_dir / "models" / f"edge_{j}.pt")

    g_loss_final, g_acc_final, _ = evaluate(avg_model, global_test_loader, device)
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["final_global_acc"] = g_acc_final
    metadata["final_global_loss"] = g_loss_final
    metadata["final_mix_scalars"] = m_edges.tolist()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[esperhfl] done. Final global_acc={g_acc_final:.4f}. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
