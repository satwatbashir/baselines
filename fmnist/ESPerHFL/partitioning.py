"""Two-level Dirichlet partition for hierarchical FL.

Outer Dirichlet across edge servers (alpha_server), inner Dirichlet across
clients inside each edge server (alpha_client). Same scheme used by the
Fedge HierFAVG/HierFL baselines so MTGC sees identical non-IID structure.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np


def hier_dirichlet_indices(
    labels: Sequence[int] | np.ndarray,
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    alpha_server: float,
    alpha_client: float,
    seed: int,
) -> Dict[str, Dict[str, List[int]]]:
    """Hierarchical Dirichlet mapping: {server_id: {client_id: [indices]}}."""
    if isinstance(clients_per_server, int):
        cps_list = [clients_per_server] * num_servers
    else:
        cps_list = list(clients_per_server)
        if len(cps_list) != num_servers:
            raise ValueError(
                f"len(clients_per_server)={len(cps_list)} != num_servers={num_servers}"
            )

    labels_np = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels_np)
    rng_outer = np.random.default_rng(int(seed))

    server_to_indices: list[list[int]] = [[] for _ in range(num_servers)]
    for cls in unique_labels:
        cls_idx = np.where(labels_np == cls)[0]
        rng_outer.shuffle(cls_idx)
        probs = rng_outer.dirichlet([alpha_server] * num_servers)
        counts = rng_outer.multinomial(cls_idx.size, probs)
        start = 0
        for sid, cnt in enumerate(counts):
            if cnt > 0:
                server_to_indices[sid].extend(cls_idx[start:start + cnt].tolist())
            start += cnt
    for sid in range(num_servers):
        rng_outer.shuffle(server_to_indices[sid])

    mapping: Dict[str, Dict[str, List[int]]] = {}
    for sid in range(num_servers):
        s_indices = np.array(server_to_indices[sid], dtype=np.int64)
        s_labels = labels_np[s_indices]
        n_clients = int(cps_list[sid])
        rng_inner = np.random.default_rng(int(seed) + sid)

        client_lists: list[list[int]] = [[] for _ in range(n_clients)]
        if s_indices.size == 0:
            mapping[str(sid)] = {str(cid): [] for cid in range(n_clients)}
            continue

        for cls in np.unique(s_labels):
            cls_abs_idx = s_indices[s_labels == cls]
            rng_inner.shuffle(cls_abs_idx)
            probs = rng_inner.dirichlet([alpha_client] * n_clients)
            counts = rng_inner.multinomial(cls_abs_idx.size, probs)
            start = 0
            for cid, cnt in enumerate(counts):
                if cnt > 0:
                    client_lists[cid].extend(cls_abs_idx[start:start + cnt].tolist())
                start += cnt

        # Guarantee no empty client: donate one sample from the largest
        sizes = [len(lst) for lst in client_lists]
        while any(sz == 0 for sz in sizes):
            src = int(np.argmax(sizes))
            if sizes[src] <= 1:
                break
            for cid, sz in enumerate(sizes):
                if sz == 0:
                    client_lists[cid].append(client_lists[src].pop())
                    sizes = [len(lst) for lst in client_lists]
                    break
        for lst in client_lists:
            rng_inner.shuffle(lst)

        mapping[str(sid)] = {str(cid): client_lists[cid] for cid in range(n_clients)}

    return mapping
