import numpy as np
import pandas as pd
import pytest

from recsys.data import compute_feature_matrices
from recsys.evaluation import compute_user_metrics, make_rec_popularity
from recsys.ranker import sample_ranker_negatives
from recsys.retrieval import sample_negatives, sample_negatives_fast


def test_ndcg_perfect_list_is_one():
    """El NDCG de una lista con todos los ítems relevantes en las primeras posiciones debe ser 1.0."""
    relevant = {10, 20, 30, 40, 50}
    recommended = np.array([10, 20, 30, 40, 50, 1, 2, 3, 4, 5])

    precision, recall, ndcg = compute_user_metrics(recommended, relevant, top_k=10)
    assert pytest.approx(ndcg, rel=1e-6) == 1.0
    assert pytest.approx(precision, rel=1e-6) == 5 / 10
    assert pytest.approx(recall, rel=1e-6) == 1.0


def test_ndcg_empty_or_zero_relevance():
    """Si no hay ítems relevantes o ninguno está en la lista recomendada, las métricas son 0.0."""
    relevant = {99, 100}
    recommended = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    precision, recall, ndcg = compute_user_metrics(recommended, relevant, top_k=10)
    assert precision == 0.0
    assert recall == 0.0
    assert ndcg == 0.0


def test_deterministic_tie_breaking():
    """Verifica que ante empates de score, el desempate sea estrictamente por item_idx ascendente."""
    n_items = 8
    # Scores con empates intencionales:
    # item 0: 50, item 1: 100, item 2: 100, item 3: 100, item 4: 200, item 5: 200, item 6: 10, item 7: 100
    scores = np.array([50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 10.0, 100.0])
    seen_mask = np.zeros((1, n_items), dtype=bool)

    # 1. Popularidad
    rec_fn = make_rec_popularity(scores, seen_mask, top_k=6)
    rec = rec_fn(0)
    # Esperado:
    # 1ro y 2do: items con score 200 -> [4, 5] (orden ascendente de item_idx)
    # 3ro, 4to, 5to, 6to: items con score 100 -> [1, 2, 3, 7] (orden ascendente de item_idx)
    expected = np.array([4, 5, 1, 2, 3, 7])
    assert np.array_equal(rec, expected), f"Esperado {expected}, obtenido {rec}"

    # 2. Two-stage tie breaking sobre candidatos arbitrarios
    cands = np.array([20, 10, 50, 5, 30])
    # scores con empate entre cands 10 y 30 (score 0.8), y entre 20 y 5 (score 0.5)
    cands_scores = np.array([0.5, 0.8, 0.1, 0.5, 0.8])
    order = np.lexsort((cands, -cands_scores))
    rec_cands = cands[order]
    # Esperado:
    # score 0.8: cands 10 y 30 -> 10 antes que 30
    # score 0.5: cands 5 y 20 -> 5 antes que 20
    # score 0.1: cand 50
    expected_cands = np.array([10, 30, 5, 20, 50])
    assert np.array_equal(rec_cands, expected_cands), f"Esperado {expected_cands}, obtenido {rec_cands}"


def test_negative_sampling_never_returns_seen_train():
    """El muestreo de negativos nunca debe retornar ítems vistos por el usuario en train."""
    n_items = 50
    user_seen_train = {
        0: {0, 1, 2, 3, 4, 5},
        1: {10, 20, 30, 40},
        2: set(range(20)),
    }
    users = np.array([0, 0, 1, 2, 1])

    seen_mask = np.zeros((3, n_items), dtype=bool)
    for u, items in user_seen_train.items():
        for i in items:
            seen_mask[u, i] = True

    np.random.seed(42)
    neg_ref = sample_negatives(users, user_seen_train, n_items, n_neg_per_pos=5)
    assert neg_ref.shape == (len(users), 5)
    for u, row in zip(users, neg_ref):
        seen = user_seen_train[u]
        for item in row:
            assert item not in seen

    np.random.seed(42)
    neg_fast = sample_negatives_fast(users, seen_mask, n_items, n_neg_per_pos=5)
    assert neg_fast.shape == (len(users), 5)
    for u, row in zip(users, neg_fast):
        seen = user_seen_train[u]
        for item in row:
            assert item not in seen

    dummy_topk = np.zeros((3, 20), dtype=np.int64)
    for u in range(3):
        unseen = [i for i in range(n_items) if i not in user_seen_train[u]]
        dummy_topk[u, : len(unseen)] = unseen[:20]

    ranker_neg = sample_ranker_negatives(
        pos_users=users,
        n_uniform=3,
        n_hard=2,
        n_items=n_items,
        seen_mask_np=seen_mask,
        retrieval_topk=dummy_topk,
        hard_pool=10,
    )
    assert ranker_neg.shape == (len(users), 5)
    for u, row in zip(users, ranker_neg):
        seen = user_seen_train[u]
        for item in row:
            assert item not in seen


def test_feature_matrices_shapes_and_user_genre_sums_to_one():
    """Las matrices deben tener las dimensiones esperadas y los perfiles de usuario sumar 1."""
    movies_data = {
        "movieId": [101, 102, 103, 104],
        "genres": ["Action|Comedy", "Drama", "Comedy", "(no genres listed)"],
    }
    movies_df = pd.DataFrame(movies_data)
    item2idx = {101: 0, 102: 1, 103: 2, 104: 3}
    n_users = 3
    n_items = 4

    train_positive = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [0, 2, 1],
            "rating": [4.0, 5.0, 4.5],
        }
    )

    (
        item_genre_np,
        user_genre_np,
        item_popularity,
        item_logpop_np,
        all_genres,
    ) = compute_feature_matrices(
        movies_df, item2idx, train_positive, n_users, n_items
    )

    n_genres = len(all_genres)
    assert n_genres == 3
    assert item_genre_np.shape == (n_items, n_genres)
    assert user_genre_np.shape == (n_users, n_genres)
    assert item_popularity.shape == (n_items,)
    assert item_logpop_np.shape == (n_items,)
    assert item_logpop_np.dtype == np.float32

    assert pytest.approx(user_genre_np[0].sum(), rel=1e-6) == 1.0
    assert pytest.approx(user_genre_np[1].sum(), rel=1e-6) == 1.0
    assert user_genre_np[2].sum() == 0.0
