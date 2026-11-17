"""Tests para las variantes A, B, C, C2, D y E de la Etapa 4."""

import numpy as np
import pytest
import torch
import random

from recsys.config import Stage3Config
from recsys.data import load_and_prepare_data
from recsys.evaluation import evaluate_model, make_rec_retrieval_only
from recsys.retrieval import RetrievalDataset
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4, train_retrieval_stage4


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def test_parameter_counts_and_shapes_all_variants():
    """Verifica formas y conteos de parámetros para A, B, C, C2, D y E."""
    n_users = 610
    n_items = 9724
    embed_dim = 32
    text_dim = 384
    genre_dim = 19

    dummy_text_embs = np.zeros((n_items, text_dim), dtype=np.float32)
    dummy_genre_embs = np.zeros((n_items, genre_dim), dtype=np.float32)

    # Variante A: Solo IDs
    mA = TwoTowerStage4("A", n_users, n_items, embed_dim)
    paramsA = mA.count_parameters()
    assert paramsA["trainable_params"] == 330688
    assert paramsA["buffer_params"] == 0
    assert paramsA["total_params"] == 330688

    # Variante B: Solo contenido texto
    mB = TwoTowerStage4("B", n_users, n_items, embed_dim, text_embeddings_np=dummy_text_embs)
    paramsB = mB.count_parameters()
    assert paramsB["trainable_params"] == 46240
    assert paramsB["buffer_params"] == 3734016
    assert paramsB["total_params"] == 46240 + 3734016

    # Variante C: Híbrida
    mC = TwoTowerStage4("C", n_users, n_items, embed_dim, text_embeddings_np=dummy_text_embs)
    paramsC = mC.count_parameters()
    assert paramsC["trainable_params"] == 357408
    assert paramsC["buffer_params"] == 3734016
    assert paramsC["total_params"] == 357408 + 3734016

    # Variante C2: Híbrida con ID Dropout
    mC2 = TwoTowerStage4("C2", n_users, n_items, embed_dim, text_embeddings_np=dummy_text_embs)
    paramsC2 = mC2.count_parameters()
    assert paramsC2["trainable_params"] == 357408
    assert paramsC2["buffer_params"] == 3734016 + n_items  # text + is_cold_mask
    assert paramsC2["total_params"] == 357408 + 3734016 + n_items

    # Variante D: Control géneros con ID
    mD = TwoTowerStage4("D", n_users, n_items, embed_dim, item_genres_np=dummy_genre_embs)
    paramsD = mD.count_parameters()
    assert paramsD["trainable_params"] == 334048
    assert paramsD["buffer_params"] == 184756

    # Variante D2: Control géneros con ID Dropout
    mD2 = TwoTowerStage4("D2", n_users, n_items, embed_dim, item_genres_np=dummy_genre_embs)
    paramsD2 = mD2.count_parameters()
    assert paramsD2["trainable_params"] == 334048
    assert paramsD2["buffer_params"] == 184756 + n_items
    assert paramsD2["total_params"] == 334048 + 184756 + n_items

    # Variante E: Control de B sin ID
    mE = TwoTowerStage4("E", n_users, n_items, embed_dim, item_genres_np=dummy_genre_embs)
    paramsE = mE.count_parameters()
    assert paramsE["trainable_params"] == 22880
    assert paramsE["buffer_params"] == 184756

    # Variante R: Control del texto (embeddings aleatorios normalizados)
    mR = TwoTowerStage4("R", n_users, n_items, embed_dim)
    paramsR = mR.count_parameters()
    assert paramsR["trainable_params"] == 46240
    assert paramsR["buffer_params"] == 3734016
    assert paramsR["total_params"] == 46240 + 3734016


def test_rng_isolation_across_all_variants():
    """Verifica que instanciar A, B, C, C2, D, D2, E y R deja el RNG global de PyTorch en el EXACTO mismo estado."""
    n_users = 610
    n_items = 9724
    dummy_text_embs = np.zeros((n_items, 384), dtype=np.float32)
    dummy_genre_embs = np.zeros((n_items, 19), dtype=np.float32)

    next_randoms = {}
    for var, kwargs in [
        ("A", {}),
        ("B", {"text_embeddings_np": dummy_text_embs}),
        ("C", {"text_embeddings_np": dummy_text_embs}),
        ("C2", {"text_embeddings_np": dummy_text_embs}),
        ("D", {"item_genres_np": dummy_genre_embs}),
        ("D2", {"item_genres_np": dummy_genre_embs}),
        ("E", {"item_genres_np": dummy_genre_embs}),
        ("R", {}),
    ]:
        torch.manual_seed(42)
        _ = TwoTowerStage4(var, n_users, n_items, seed=42, **kwargs)
        next_randoms[var] = torch.randn(10).numpy()

    for var in ["B", "C", "C2", "D", "D2", "E", "R"]:
        np.testing.assert_array_equal(next_randoms["A"], next_randoms[var], err_msg=f"RNG mismatch for {var}")


def test_c_zero_inference_control():
    """Verifica que el modo C-cero sobre el modelo C anula el ID para ítems fríos y conserva ID para warm."""
    n_users = 10
    n_items = 20
    text_dim = 384
    dummy_text = np.ones((n_items, text_dim), dtype=np.float32)

    is_cold = np.zeros(n_items, dtype=bool)
    is_cold[:5] = True  # 0..4 son cold

    model_c = TwoTowerStage4("C", n_users, n_items, text_embeddings_np=dummy_text, is_cold_mask_np=is_cold, seed=42)
    model_c.eval()

    with torch.no_grad():
        # Evaluación C normal: todos tienen id_emb + content
        normal_embs = model_c.forward_all_items(zero_cold_id=False)
        expected_normal = model_c.item_id_embedding.weight + model_c.content_net(model_c.fixed_text_embeddings)
        assert torch.allclose(normal_embs, expected_normal, atol=1e-6)

        # Evaluación C-cero: ítems cold tienen solo content, warm tienen id_emb + content
        c_zero_embs = model_c.forward_all_items(zero_cold_id=True)

        # Cold items (0..4)
        expected_cold = model_c.content_net(model_c.fixed_text_embeddings[:5])
        assert torch.allclose(c_zero_embs[:5], expected_cold, atol=1e-6)

        # Warm items (5..19)
        expected_warm = model_c.item_id_embedding.weight[5:] + model_c.content_net(model_c.fixed_text_embeddings[5:])
        assert torch.allclose(c_zero_embs[5:], expected_warm, atol=1e-6)


def test_c2_id_dropout_behavior():
    """Verifica el comportamiento de C2 en train (dropout p=0.5) y en eval (filtro de cold items)."""
    n_users = 10
    n_items = 20
    text_dim = 384
    dummy_text = np.ones((n_items, text_dim), dtype=np.float32)

    # Definir que los ítems 0..4 son fríos (True) y 5..19 son warm (False)
    is_cold = np.zeros(n_items, dtype=bool)
    is_cold[:5] = True

    model = TwoTowerStage4("C2", n_users, n_items, text_embeddings_np=dummy_text, is_cold_mask_np=is_cold, seed=42)

    # 1. En eval mode:
    model.eval()
    with torch.no_grad():
        cold_idx = torch.tensor([0, 1, 2, 3, 4])
        warm_idx = torch.tensor([5, 6, 7])

        cold_rep = model.forward_item(cold_idx)
        # Para ítems fríos, el ID embedding debe estar anulado, por lo que cold_rep == content_net(text)
        expected_content = model.content_net(model.fixed_text_embeddings[cold_idx])
        assert torch.allclose(cold_rep, expected_content, atol=1e-6)

        warm_rep = model.forward_item(warm_idx)
        # Para ítems warm, debe ser la suma completa id_emb + content
        expected_warm = model.item_id_embedding(warm_idx) + model.content_net(model.fixed_text_embeddings[warm_idx])
        assert torch.allclose(warm_rep, expected_warm, atol=1e-6)

    # 2. En train mode:
    model.train()
    with torch.no_grad():
        reps = [model.forward_item(torch.tensor([5])) for _ in range(20)]
        # Con p=0.5 debe alternar entre suma y solo contenido
        content_only = model.content_net(model.fixed_text_embeddings[torch.tensor([5])])
        has_content_only = any(torch.allclose(r, content_only, atol=1e-5) for r in reps)
        has_full_sum = any(not torch.allclose(r, content_only, atol=1e-5) for r in reps)
        assert has_content_only and has_full_sum


def test_d2_id_dropout_behavior():
    """Verifica el comportamiento de D2 en train (dropout p=0.5) y en eval (filtro de cold items sobre géneros)."""
    n_users = 10
    n_items = 20
    genre_dim = 19
    dummy_genres = np.ones((n_items, genre_dim), dtype=np.float32)

    is_cold = np.zeros(n_items, dtype=bool)
    is_cold[:5] = True

    model = TwoTowerStage4("D2", n_users, n_items, item_genres_np=dummy_genres, is_cold_mask_np=is_cold, seed=42)

    # 1. En eval mode:
    model.eval()
    with torch.no_grad():
        cold_idx = torch.tensor([0, 1, 2, 3, 4])
        warm_idx = torch.tensor([5, 6, 7])

        cold_rep = model.forward_item(cold_idx)
        expected_genre = model.genre_net(model.fixed_genre_features[cold_idx])
        assert torch.allclose(cold_rep, expected_genre, atol=1e-6)

        warm_rep = model.forward_item(warm_idx)
        expected_warm = model.item_id_embedding(warm_idx) + model.genre_net(model.fixed_genre_features[warm_idx])
        assert torch.allclose(warm_rep, expected_warm, atol=1e-6)

    # 2. En train mode:
    model.train()
    with torch.no_grad():
        reps = [model.forward_item(torch.tensor([5])) for _ in range(20)]
        genre_only = model.genre_net(model.fixed_genre_features[torch.tensor([5])])
        has_genre_only = any(torch.allclose(r, genre_only, atol=1e-5) for r in reps)
        has_full_sum = any(not torch.allclose(r, genre_only, atol=1e-5) for r in reps)
        assert has_genre_only and has_full_sum


@pytest.mark.slow
@pytest.mark.regression
def test_variant_a_exact_reproducibility():
    """Verifica que la Variante A en TwoTowerStage4 reproduce EXACTAMENTE la Etapa 3 con semilla 42:
    P@10: 0.1478, R@10: 0.1275, NDCG@10: 0.1907.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    set_seed(seed, device)
    cfg = Stage3Config(seed=seed, device="cuda" if device.type == "cuda" else "cpu")
    data = load_and_prepare_data(cfg)

    retrieval_dataset = RetrievalDataset(
        positive_df=data.train_positive,
        user_seen_train=data.user_seen_train,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
        fast_sampling=False,
    )

    model_A = TwoTowerStage4(
        variant="A",
        n_users=data.n_users,
        n_items=data.n_items,
        embed_dim=cfg.embed_dim_retrieval,
        seed=seed,
    ).to(device)

    train_retrieval_stage4(
        model=model_A,
        dataset=retrieval_dataset,
        cfg=cfg,
        device=device,
    )

    retrieval_topk = compute_retrieval_candidates_stage4(
        model=model_A,
        n_users=data.n_users,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        max_candidates=cfg.max_candidates,
        device=device,
    )

    metrics = evaluate_model(
        name="Variante A (Solo IDs)",
        rec_fn=make_rec_retrieval_only(retrieval_topk, cfg.top_k),
        eval_users=data.eval_users,
        test_relevant_by_user=data.test_relevant_by_user,
        item_popularity=data.item_popularity,
        n_items=data.n_items,
        device=device,
        top_k=cfg.top_k,
    )

    p10 = metrics["precision_at_10"]
    r10 = metrics["recall_at_10"]
    n10 = metrics["ndcg_at_10"]

    print(f"\n[Test Variante A Etapa 4] P@10: {p10:.6f}, R@10: {r10:.6f}, NDCG@10: {n10:.6f}")

    assert abs(p10 - 0.147754) < 1e-4, f"P@10 esperado 0.1478, obtenido {p10:.4f}"
    assert abs(r10 - 0.127539) < 1e-4, f"R@10 esperado 0.1275, obtenido {r10:.4f}"
    assert abs(n10 - 0.190736) < 1e-4, f"NDCG@10 esperado 0.1907, obtenido {n10:.4f}"
