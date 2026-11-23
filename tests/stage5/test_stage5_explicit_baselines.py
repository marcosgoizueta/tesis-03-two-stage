import numpy as np
import torch
from types import SimpleNamespace

from recsys.stage5.explicit_baselines import FunkSVD, NeuMF, train_neumf, rmse_neumf, score_all_items_neumf
from scripts.run_stage5_explicit_baselines import topk_from_scores


def _toy(n_u=30, n_i=40, seed=0):
    rng = np.random.RandomState(seed)
    pu, qi = rng.normal(0, 1, (n_u, 3)), rng.normal(0, 1, (n_i, 3))
    u, i = np.meshgrid(np.arange(n_u), np.arange(n_i - 5), indexing="ij")   # los últimos 5 ítems quedan sin datos
    u, i = u.ravel(), i.ravel()
    r = np.clip(3 + 0.7 * (pu[u] * qi[i]).sum(1), 0.5, 5)
    return u, i, r, n_u, n_i


def test_funksvd_learns_and_falls_back_for_unknown_items():
    u, i, r, n_u, n_i = _toy()
    m = FunkSVD(n_factors=5, n_epochs=30, seed=1).fit(u, i, r, n_u, n_i)
    rmse = np.sqrt(np.mean((m.predict_pairs(u, i) - r) ** 2))
    assert rmse < np.std(r)                       # mejor que predecir la media
    unk = np.array([n_i - 1] * 3); uu = np.array([0, 1, 2])
    assert np.allclose(m.predict_pairs(uu, unk), m.mu + m.bu[uu])   # ítem sin datos: mu + bu, como Surprise
    full = m.predict_matrix([0, 1])
    assert full.shape == (2, n_i) and np.allclose(full[:, n_i - 1], m.mu + m.bu[[0, 1]])


def test_funksvd_is_deterministic_per_seed():
    u, i, r, n_u, n_i = _toy()
    a = FunkSVD(n_factors=4, n_epochs=3, seed=7).fit(u, i, r, n_u, n_i).predict_matrix([0, 1, 2])
    b = FunkSVD(n_factors=4, n_epochs=3, seed=7).fit(u, i, r, n_u, n_i).predict_matrix([0, 1, 2])
    assert np.allclose(a, b)


def test_neumf_output_range_and_training_reduces_error():
    u, i, r, n_u, n_i = _toy()
    dev = torch.device("cpu")
    model, hist = train_neumf(u, i, r, n_u, n_i, dev, seed=3, epochs=3, val=(u, i, r))
    assert hist[-1]["train_mse"] < hist[0]["train_mse"] and "val_rmse" in hist[0]
    s = score_all_items_neumf(model, [0, 1], n_i, dev)
    assert s.shape == (2, n_i) and s.min() >= 0.5 and s.max() <= 5.0
    assert abs(rmse_neumf(model, u, i, r, dev) - hist[-1]["val_rmse"]) < 1e-6


def test_topk_masks_seen_and_cold_items():
    n_i = 20
    seen = np.zeros((2, n_i), dtype=bool); seen[0, :3] = True
    bundle = SimpleNamespace(seen_mask_np=seen)
    cold = np.zeros(n_i, dtype=bool); cold[3:5] = True
    scores = np.tile(np.arange(n_i, dtype=float)[::-1], (2, 1))     # ítem 0 es el mejor
    top = topk_from_scores(scores, [0, 1], bundle, cold, k=5)
    assert list(top[0]) == [5, 6, 7, 8, 9]                           # 0-2 vistos, 3-4 fríos
    assert list(top[1]) == [0, 1, 2, 5, 6]                           # usuario 1: solo se excluyen los fríos
