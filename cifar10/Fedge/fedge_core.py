"""Fedge core: variable K via split-merge + Intra-Coupled Soft Membership (ICSM).

Implements the framework described in Chapter 3 of the thesis:
    - Variable cloud-tier model family W^t = {w_1, ..., w_{K^t}}, K^t inferred.
    - Soft membership matrix Pi^t in R^{M x K^t}, row-stochastic.
    - Two-level signals: hat_zeta^2_intra(e, t) and hat_d^2(e, e', t).
    - Sensitivity parameter lambda in [0, 1] mapping to algorithm hparams.
    - Two inference modes selectable at runtime:
        * 'heuristic': greedy split-merge driven by signal thresholds (fast).
        * 'mh'       : Metropolis-Hastings split-merge under a DP prior (faithful).
    - ICSM (Intra-Coupled Soft Membership) -- novel: per-edge softmax temperature
      modulated by the edge's intra-edge gradient dispersion.

Aggregation Eq. (3.4) and blending Eq. (3.2) from the chapter are exact (no
approximation). Only the inference of K and Pi differs between modes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lambda -> hyperparameter mapping (chapter Eq. 3.5 generalized for both modes)
# ---------------------------------------------------------------------------
@dataclass
class FedgeHParams:
    """All lambda-derived hparams in one place. Pulled at init time and reused.

    lambda = 0 -> single cluster regime (HierFAVG)
    lambda = 1 -> max-K regime (PHE-FL/ESPerHFL family)
    """
    lambda_val: float
    tau_0: float          # base softmax temperature for ICSM
    theta_split: float    # heuristic: split when max intra > theta_split
    theta_merge: float    # heuristic: merge when min inter < theta_merge
    alpha_0: float        # DP concentration (MH mode)
    K_max: int            # cap on cluster count
    beta_intra: float = 1.0   # ICSM intra-coupling strength
    rho: float = 0.5          # exponential smoothing on Pi across rounds


def lambda_to_hparams(lambda_val: float, K_max: int) -> FedgeHParams:
    """Map lambda in [0, 1] -> all algorithm constants.

    Calibrated so that lambda=0 forces K=1 (high merge threshold, low split,
    low alpha_0) and lambda=1 favors K=K_max (low merge threshold, high split,
    high alpha_0). Intermediate values trace the trade-off frontier.
    """
    lam = float(np.clip(lambda_val, 0.0, 1.0))

    # Geometric interpolation: alpha_0 spans [1e-3, 1e2] across lambda
    alpha_min, alpha_max = 1e-3, 1e2
    alpha_0 = alpha_min * (alpha_max / alpha_min) ** lam

    # Softmax base temperature: high tau at lambda=0 (uniform Pi),
    # low tau at lambda=1 (peaked Pi).
    tau_0 = 10.0 ** (1.0 - 2.0 * lam)   # lambda=0 -> tau_0=10; lambda=1 -> tau_0=0.1

    # Heuristic thresholds (normalized; the actual values are scaled by current
    # signal magnitudes at runtime to remain dimensionless).
    theta_split = 1.0 + 4.0 * (1.0 - lam)    # lambda=0 -> 5.0 (hard); lambda=1 -> 1.0 (easy)
    theta_merge = 0.05 + 0.95 * (1.0 - lam)  # lambda=0 -> 1.0 (easy merge); lambda=1 -> 0.05 (hard)

    return FedgeHParams(
        lambda_val=lam, tau_0=tau_0,
        theta_split=theta_split, theta_merge=theta_merge,
        alpha_0=alpha_0, K_max=K_max,
    )


# ---------------------------------------------------------------------------
# Two-level signals (chapter Eq. 3.3, 3.4 / 3.13, 3.14 in algorithm section)
# ---------------------------------------------------------------------------
def compute_intra_signals(
    client_flats: torch.Tensor,   # (N, D)
    client_to_edge: list[int],    # length N
    num_edges: int,
) -> torch.Tensor:
    """hat_zeta^2_intra(e, t) per Eq. (3.3): mean squared client-vs-edge distance."""
    intra = torch.zeros(num_edges, dtype=torch.float32)
    for j in range(num_edges):
        members = [c for c, e in enumerate(client_to_edge) if e == j]
        if len(members) < 2:
            continue
        sub = client_flats[members]            # (n_e, D)
        mean = sub.mean(dim=0, keepdim=True)
        d2 = ((sub - mean) ** 2).sum(dim=1)    # (n_e,)
        intra[j] = d2.mean()
    return intra


def compute_inter_signals(edge_flats: torch.Tensor) -> torch.Tensor:
    """hat_d^2(e, e', t) per Eq. (3.4): pairwise squared edge-update distances.
    Returns (M, M) matrix with diagonal = 0."""
    diffs = edge_flats.unsqueeze(0) - edge_flats.unsqueeze(1)   # (M, M, D)
    return (diffs ** 2).sum(dim=2)


# ---------------------------------------------------------------------------
# ICSM -- Intra-Coupled Soft Membership (novel contribution)
# ---------------------------------------------------------------------------
def compute_membership_icsm(
    edge_flats: torch.Tensor,        # (M, D)
    centroids: torch.Tensor,          # (K, D)
    zeta_intra: torch.Tensor,         # (M,)
    hp: FedgeHParams,
    prev_pi: torch.Tensor | None = None,   # (M, K_prev) for exponential smoothing
) -> torch.Tensor:
    """ICSM (novel): per-edge softmax temperature modulated by intra-edge dispersion.

    tau_e(t) = tau_0(lambda) * (1 + beta * hat_zeta^2_intra(e, t))

    Edges with consistent client data (low intra) -> low tau -> concentrated Pi.
    Edges with inconsistent client data (high intra) -> high tau -> spread Pi.

    No existing HFL or clustered-FL method modulates per-edge softmax temperature
    by intra-edge dispersion. This coupling between the two heterogeneity levels
    is the novelty.
    """
    M, D = edge_flats.shape
    K = centroids.shape[0]

    # Pairwise squared distance edge -> centroid (M, K)
    d2 = ((edge_flats.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2)

    # Normalize zeta_intra to a unit-free scale so beta is interpretable
    zeta_mean = zeta_intra.mean().clamp_min(1e-8)
    zeta_norm = zeta_intra / zeta_mean              # (M,)
    tau_e = hp.tau_0 * (1.0 + hp.beta_intra * zeta_norm)   # (M,)

    # Normalize d2 to unit scale per row for numerical stability
    d2_norm = d2 / d2.mean(dim=1, keepdim=True).clamp_min(1e-8)
    logits = -d2_norm / tau_e.unsqueeze(1)
    new_pi = torch.softmax(logits, dim=1)

    # Exponential smoothing across rounds when K is unchanged
    if prev_pi is not None and prev_pi.shape == new_pi.shape:
        new_pi = (1.0 - hp.rho) * prev_pi + hp.rho * new_pi
        new_pi = new_pi / new_pi.sum(dim=1, keepdim=True).clamp_min(1e-12)

    return new_pi


# ---------------------------------------------------------------------------
# Heuristic split-merge (signal-threshold driven)
# ---------------------------------------------------------------------------
def heuristic_split_merge(
    K: int,
    centroids: torch.Tensor,        # (K, D)
    member_indices: list[list[int]],  # length K, lists of edge ids
    edge_flats: torch.Tensor,        # (M, D)
    zeta_intra: torch.Tensor,        # (M,)
    inter_signals: torch.Tensor,     # (M, M)
    hp: FedgeHParams,
    rng: np.random.Generator,
) -> tuple[int, torch.Tensor, list[list[int]], int, int]:
    """Greedy split-merge driven by the two-level signals. Returns
    (new_K, new_centroids, new_member_indices, n_splits, n_merges)."""
    M = edge_flats.shape[0]
    n_splits, n_merges = 0, 0

    # ---- SPLIT pass: split clusters with high internal intra-edge variance ---
    new_centroids = list(centroids.unbind(0))
    new_members = [list(m) for m in member_indices]
    changed = True
    while changed and len(new_centroids) < min(hp.K_max, M):
        changed = False
        for k in range(len(new_centroids)):
            members = new_members[k]
            if len(members) < 2:
                continue
            # Trigger: max member intra-zeta exceeds normalized split threshold
            max_intra = zeta_intra[members].max().item()
            mean_intra = zeta_intra[members].mean().clamp_min(1e-8).item()
            if max_intra / mean_intra > hp.theta_split:
                # Split via leading eigenvector of inter-edge dissim within cluster
                sub_idx = members
                sub_D = inter_signals[sub_idx][:, sub_idx].numpy()
                # Spectral partition: sign of leading non-trivial eigenvector
                try:
                    eigvals, eigvecs = np.linalg.eigh(sub_D)
                    v = eigvecs[:, -1]
                    mask_a = v > np.median(v)
                except np.linalg.LinAlgError:
                    rng.shuffle(sub_idx)
                    half = len(sub_idx) // 2
                    mask_a = np.array([i < half for i in range(len(sub_idx))])
                a_members = [sub_idx[i] for i in range(len(sub_idx)) if mask_a[i]]
                b_members = [sub_idx[i] for i in range(len(sub_idx)) if not mask_a[i]]
                if len(a_members) == 0 or len(b_members) == 0:
                    continue
                cent_a = edge_flats[a_members].mean(dim=0)
                cent_b = edge_flats[b_members].mean(dim=0)
                new_centroids[k] = cent_a
                new_members[k] = a_members
                new_centroids.append(cent_b)
                new_members.append(b_members)
                n_splits += 1
                changed = True
                if len(new_centroids) >= min(hp.K_max, M):
                    break

    # ---- MERGE pass: merge close cluster pairs --------------------------------
    changed = True
    while changed and len(new_centroids) > 1:
        changed = False
        K_cur = len(new_centroids)
        cent_stack = torch.stack(new_centroids, dim=0)
        cent_d2 = ((cent_stack.unsqueeze(0) - cent_stack.unsqueeze(1)) ** 2).sum(dim=2)
        cent_d2 += torch.eye(K_cur) * 1e18    # ignore diagonal
        # Normalize by mean off-diagonal so threshold is unit-free
        mean_d = cent_d2[cent_d2 < 1e17].mean().clamp_min(1e-8)
        min_d_ratio = (cent_d2 / mean_d).min().item()
        if min_d_ratio < hp.theta_merge:
            flat_min = int(torch.argmin(cent_d2).item())
            k1, k2 = flat_min // K_cur, flat_min % K_cur
            if k1 > k2:
                k1, k2 = k2, k1
            # Merge: combined members, sample-size-weighted centroid
            combined = new_members[k1] + new_members[k2]
            new_cent = edge_flats[combined].mean(dim=0)
            new_centroids[k1] = new_cent
            new_members[k1] = combined
            new_centroids.pop(k2)
            new_members.pop(k2)
            n_merges += 1
            changed = True

    new_K = len(new_centroids)
    new_centroids_t = torch.stack(new_centroids, dim=0) if new_K > 0 else centroids
    return new_K, new_centroids_t, new_members, n_splits, n_merges


# ---------------------------------------------------------------------------
# Metropolis-Hastings split-merge (faithful DP version)
# ---------------------------------------------------------------------------
def _log_dp_prior(member_sizes: Sequence[int], alpha_0: float) -> float:
    """Log of the DP/CRP prior over partitions: log[alpha^K * prod_k (n_k - 1)!]."""
    K = len(member_sizes)
    val = K * np.log(alpha_0)
    for n_k in member_sizes:
        # log of (n_k - 1)!  (use lgamma since n_k - 1 may be 0)
        val += float(math.lgamma(n_k))
    return val


def _log_data_likelihood(
    member_indices: list[list[int]],
    edge_flats: torch.Tensor,
    sigma2: float = 1.0,
) -> float:
    """Diagonal-Gaussian likelihood approximation: each cluster's edges are
    assumed to be IID Gaussian around the cluster centroid with variance sigma2.
    Numerically: -0.5 * sum_e ||w_e - centroid_{c(e)}||^2 / sigma2."""
    ll = 0.0
    for members in member_indices:
        if len(members) == 0:
            continue
        sub = edge_flats[members]
        cent = sub.mean(dim=0, keepdim=True)
        ll += -0.5 * float(((sub - cent) ** 2).sum().item()) / sigma2
    return ll


def mh_split_merge(
    K: int,
    centroids: torch.Tensor,
    member_indices: list[list[int]],
    edge_flats: torch.Tensor,
    zeta_intra: torch.Tensor,
    inter_signals: torch.Tensor,
    hp: FedgeHParams,
    rng: np.random.Generator,
    n_sweeps: int = 3,
) -> tuple[int, torch.Tensor, list[list[int]], int, int]:
    """One sweep of MH split-merge updates under the DP prior with Gaussian
    cluster likelihood. Proposal distribution: split via spectral partition,
    merge via inverse-distance pairwise selection.

    This is a *practical* MH realization of the chapter's framework. It is
    less rigorous than a full conjugate Gibbs sampler but matches the
    chapter's spec: proposes splits when intra-cluster signal is high,
    merges when inter-cluster signal is low, and accepts via Eq. (3.6).
    """
    M = edge_flats.shape[0]
    new_centroids = list(centroids.unbind(0))
    new_members = [list(m) for m in member_indices]
    n_splits, n_merges = 0, 0

    for _ in range(n_sweeps):
        K_cur = len(new_centroids)

        # Decide move type: split with prob proportional to (K_max - K)/(K_max),
        # merge with prob proportional to K/K_max
        p_split = max(0.0, (hp.K_max - K_cur) / max(1, hp.K_max))
        do_split = rng.random() < p_split and K_cur < hp.K_max

        # Current state metrics
        sizes_cur = [len(m) for m in new_members]
        lp_cur = _log_dp_prior(sizes_cur, hp.alpha_0) + _log_data_likelihood(new_members, edge_flats)

        if do_split:
            # Pick cluster to split, weighted by intra-cluster heterogeneity
            cand_weights = []
            for k in range(K_cur):
                if len(new_members[k]) < 2:
                    cand_weights.append(0.0)
                else:
                    sub = edge_flats[new_members[k]]
                    cand_weights.append(float(((sub - sub.mean(0, keepdim=True)) ** 2).sum().item()))
            tot = sum(cand_weights)
            if tot <= 0:
                continue
            probs = np.array(cand_weights) / tot
            k = int(rng.choice(K_cur, p=probs))
            members_k = new_members[k]
            sub_D = inter_signals[members_k][:, members_k].numpy()
            try:
                eigvals, eigvecs = np.linalg.eigh(sub_D)
                v = eigvecs[:, -1]
                mask_a = v > np.median(v)
            except np.linalg.LinAlgError:
                mask_a = rng.choice([True, False], size=len(members_k))
            a_members = [members_k[i] for i in range(len(members_k)) if mask_a[i]]
            b_members = [members_k[i] for i in range(len(members_k)) if not mask_a[i]]
            if len(a_members) == 0 or len(b_members) == 0:
                continue
            cand_centroids = list(new_centroids)
            cand_centroids[k] = edge_flats[a_members].mean(dim=0)
            cand_centroids.append(edge_flats[b_members].mean(dim=0))
            cand_members = list(new_members)
            cand_members[k] = a_members
            cand_members.append(b_members)
            sizes_new = [len(m) for m in cand_members]
            lp_new = _log_dp_prior(sizes_new, hp.alpha_0) + _log_data_likelihood(cand_members, edge_flats)
            log_accept = lp_new - lp_cur
            if np.log(rng.random() + 1e-30) < min(0.0, log_accept):
                new_centroids = cand_centroids
                new_members = cand_members
                n_splits += 1
        else:
            if K_cur < 2:
                continue
            cent_stack = torch.stack(new_centroids, dim=0)
            cent_d2 = ((cent_stack.unsqueeze(0) - cent_stack.unsqueeze(1)) ** 2).sum(dim=2)
            cent_d2 += torch.eye(K_cur) * 1e18
            # Pair selection: weight ∝ 1/(d^2 + eps)
            inv = 1.0 / (cent_d2.numpy() + 1.0)
            np.fill_diagonal(inv, 0.0)
            inv_flat = inv.flatten()
            inv_flat = inv_flat / inv_flat.sum()
            pair_idx = int(rng.choice(K_cur * K_cur, p=inv_flat))
            k1, k2 = pair_idx // K_cur, pair_idx % K_cur
            if k1 == k2:
                continue
            if k1 > k2:
                k1, k2 = k2, k1
            cand_centroids = list(new_centroids)
            cand_members = list(new_members)
            combined = cand_members[k1] + cand_members[k2]
            cand_centroids[k1] = edge_flats[combined].mean(dim=0)
            cand_members[k1] = combined
            cand_centroids.pop(k2)
            cand_members.pop(k2)
            sizes_new = [len(m) for m in cand_members]
            lp_new = _log_dp_prior(sizes_new, hp.alpha_0) + _log_data_likelihood(cand_members, edge_flats)
            log_accept = lp_new - lp_cur
            if np.log(rng.random() + 1e-30) < min(0.0, log_accept):
                new_centroids = cand_centroids
                new_members = cand_members
                n_merges += 1

    new_K = len(new_centroids)
    new_centroids_t = torch.stack(new_centroids, dim=0) if new_K > 0 else centroids
    return new_K, new_centroids_t, new_members, n_splits, n_merges


# ---------------------------------------------------------------------------
# Aggregation Eq. (3.4) — exact
# ---------------------------------------------------------------------------
def aggregate_by_membership(
    edge_flats: torch.Tensor,         # (M, D) updated edge models
    Pi: torch.Tensor,                  # (M, K) soft membership
    edge_sample_counts: torch.Tensor, # (M,) n_e^t (participating clients)
) -> torch.Tensor:
    """Eq. (3.4) of the chapter:  w_k^{t+1} = sum_e Pi_{e,k} n_e^t w_e / sum_e Pi_{e,k} n_e^t"""
    weights = Pi * edge_sample_counts.unsqueeze(1)     # (M, K)
    denom = weights.sum(dim=0).clamp_min(1e-12)        # (K,)
    numer = weights.transpose(0, 1) @ edge_flats        # (K, D)
    return numer / denom.unsqueeze(1)


def blend_to_edge(W: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Eq. (3.2) of the chapter: tilde_w_e^{t+1} = sum_k Pi_{e,k}^t w_k^t.

    Cloud computes the blended model for each edge before transmission so
    that the cloud-to-edge download size is one model (not K), matching the
    Fedge communication-cost claim in Section 3.3.6."""
    return Pi @ W   # (M, K) @ (K, D) -> (M, D)


# ---------------------------------------------------------------------------
# Pi entropy diagnostic (per-row Shannon entropy, averaged over edges)
# ---------------------------------------------------------------------------
def membership_entropy(Pi: torch.Tensor) -> float:
    """Returns mean per-edge entropy of the soft membership matrix."""
    p = Pi.clamp_min(1e-12)
    H = -(p * p.log()).sum(dim=1).mean().item()
    return float(H)
