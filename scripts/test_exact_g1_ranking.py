import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import torch

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer
from recsys.stage5.models import T5Recommender

def build_prefix_factorized_scorer(tok: GenerativeTokenizer, n_items: int):
    item_tokens = []
    for i in range(n_items):
        item_tokens.append(tok.encode_target_item(i)) # [c1, c2, c3, d, EOS]
    return np.array(item_tokens, dtype=np.int64)

def score_unseen_items_factorized(
    model: T5Recommender,
    tok: GenerativeTokenizer,
    history_indices: list,
    seen_items: set,
    n_items: int,
    item_tokens: np.ndarray,
    device: torch.device,
    chunk_size: int = 256,
) -> list:
    """Calcula el log-prob exacto de todos los ítems válidos no vistos factorizado por el árbol de prefijos en chunks seguros."""
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            inp = tok.encode_input_sequence(history_indices, max_tokens=150)
            inp_t = torch.tensor([inp], dtype=torch.long, device=device)
            enc_out = model.model.encoder(input_ids=inp_t)
            H_x = enc_out.last_hidden_state # (1, L_in, D)

            unseen_items = [i for i in range(n_items) if i not in seen_items]
            unseen_tokens = item_tokens[unseen_items] # (M, 5)

            # Paso 1: Logits sobre BOS -> P(c1 | BOS)
            bos_t = torch.tensor([[tok.BOS_TOKEN_ID]], dtype=torch.long, device=device)
            dec1 = model.model.decoder(input_ids=bos_t, encoder_hidden_states=H_x)
            logits1 = model.model.lm_head(dec1.last_hidden_state)[0, 0] # (V,)
            logp1 = torch.log_softmax(logits1, dim=-1)

            # Paso 2: dec_in = [BOS, c1]
            unique_c1, inv_c1 = np.unique(unseen_tokens[:, 0], return_inverse=True)
            b2_dec_in = torch.zeros((len(unique_c1), 2), dtype=torch.long, device=device)
            b2_dec_in[:, 0] = tok.BOS_TOKEN_ID
            b2_dec_in[:, 1] = torch.tensor(unique_c1, dtype=torch.long, device=device)
            H2 = H_x.expand(len(unique_c1), -1, -1)
            dec2 = model.model.decoder(input_ids=b2_dec_in, encoder_hidden_states=H2)
            logits2 = model.model.lm_head(dec2.last_hidden_state)[:, 1] # (B2, V)
            logp2 = torch.log_softmax(logits2, dim=-1)

            # Paso 3: dec_in = [BOS, c1, c2] en chunks de chunk_size
            unique_c12, inv_c12 = np.unique(unseen_tokens[:, :2], axis=0, return_inverse=True)
            N_c12 = len(unique_c12)
            logp3_list = []
            for s in range(0, N_c12, chunk_size):
                e = min(s + chunk_size, N_c12)
                bsz = e - s
                b3_in = torch.zeros((bsz, 3), dtype=torch.long, device=device)
                b3_in[:, 0] = tok.BOS_TOKEN_ID
                b3_in[:, 1:] = torch.tensor(unique_c12[s:e], dtype=torch.long, device=device)
                Hb = H_x.expand(bsz, -1, -1)
                dec3 = model.model.decoder(input_ids=b3_in, encoder_hidden_states=Hb)
                lg3 = model.model.lm_head(dec3.last_hidden_state)[:, 2]
                logp3_list.append(torch.log_softmax(lg3, dim=-1))
            logp3 = torch.cat(logp3_list, dim=0)

            # Paso 4: dec_in = [BOS, c1, c2, c3] en chunks de chunk_size
            unique_c123, inv_c123 = np.unique(unseen_tokens[:, :3], axis=0, return_inverse=True)
            N_c123 = len(unique_c123)
            logp4_list = []
            for s in range(0, N_c123, chunk_size):
                e = min(s + chunk_size, N_c123)
                bsz = e - s
                b4_in = torch.zeros((bsz, 4), dtype=torch.long, device=device)
                b4_in[:, 0] = tok.BOS_TOKEN_ID
                b4_in[:, 1:] = torch.tensor(unique_c123[s:e], dtype=torch.long, device=device)
                Hb = H_x.expand(bsz, -1, -1)
                dec4 = model.model.decoder(input_ids=b4_in, encoder_hidden_states=Hb)
                lg4 = model.model.lm_head(dec4.last_hidden_state)[:, 3]
                logp4_list.append(torch.log_softmax(lg4, dim=-1))
            logp4 = torch.cat(logp4_list, dim=0)

            # Paso 5: Acumular log-probs
            score_c1 = logp1[torch.tensor(unseen_tokens[:, 0], device=device)]
            idx_c1 = torch.tensor(inv_c1, dtype=torch.long, device=device)
            target_c2 = torch.tensor(unseen_tokens[:, 1], dtype=torch.long, device=device)
            score_c2 = logp2[idx_c1, target_c2]

            idx_c12 = torch.tensor(inv_c12, dtype=torch.long, device=device)
            target_c3 = torch.tensor(unseen_tokens[:, 2], dtype=torch.long, device=device)
            score_c3 = logp3[idx_c12, target_c3]

            idx_c123 = torch.tensor(inv_c123, dtype=torch.long, device=device)
            target_d = torch.tensor(unseen_tokens[:, 3], dtype=torch.long, device=device)
            score_d = logp4[idx_c123, target_d]

            total_scores = (score_c1 + score_c2 + score_c3 + score_d).cpu().float().numpy()
            order = np.argsort(total_scores)[::-1][:50]
            return [unseen_items[idx] for idx in order]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=42)
    n_items = bundle.n_items

    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")
    train_indices = sorted(list(bundle.train_item_indices))
    cold_indices = sorted(list(bundle.cold_item_indices))
    tr_embs = all_embeddings[train_indices]
    mu = tr_embs.mean(axis=0)
    sig = tr_embs.std(axis=0) + 1e-8
    X_std = (all_embeddings - mu) / sig

    torch.manual_seed(42)
    rq = RQVAE().to("cpu")
    rq.rvq.init_from_kmeans(torch.tensor(X_std[train_indices], dtype=torch.float32, device="cpu"), seed=42)
    rq.eval()
    codes = rq.quantize(torch.tensor(X_std, dtype=torch.float32, device="cpu")).numpy()
    tr_counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
    mgr = SemanticIdManager(max_disambiguation=16)
    mgr.fit_train_items(train_indices, codes, tr_counts)
    mgr.assign_cold_items(cold_indices, codes)

    tok_g1 = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=mgr.item_to_semantic_id)
    model_g1 = T5Recommender(tok_g1).to(device)
    model_g1.load_state_dict(torch.load("scratch/stage5_g1_fase2c.pt", map_location=device))

    item_tokens = build_prefix_factorized_scorer(tok_g1, n_items)

    test_users = bundle.eval_users_test[:50] # Medir sobre 50 usuarios
    print(f"Midiendo tiempo de ranking exacto factorizado sobre 50 usuarios...")
    t0 = time.perf_counter()
    for u in test_users:
        recs = score_unseen_items_factorized(
            model_g1, tok_g1, bundle.user_train_seq[u],
            bundle.user_seen_train.get(u, set()), n_items, item_tokens, device, chunk_size=256
        )
    t1 = time.perf_counter()
    ms_per_user = (t1 - t0) / len(test_users) * 1000.0
    est_total_s = (t1 - t0) / len(test_users) * 592
    print(f"Tiempo por usuario: {ms_per_user:.2f} ms")
    print(f"Tiempo total estimado para los 592 usuarios de test: {est_total_s:.2f} s ({est_total_s/60:.2f} min)")

if __name__ == "__main__":
    main()
