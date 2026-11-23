"""
run_stage5_save_recs.py — Reentrena los modelos de la Etapa 5 con las recetas ya publicadas y
GUARDA las listas top-50 por usuario (para popularidad media, Gini, diversidad intra-lista y
bootstrap por usuarios).  No modifica results/stage5/per_seed*/.

Salida:
  results/stage5/recs/{modelo}_seed_{s}.npz   (users: int32[n], recs: int32[n,50], -1 = relleno)
  results/stage5/recs/repro_check.json        (NDCG@10 nuevo vs. el JSON publicado, por modelo y semilla)
Modelos: ES_Two-Tower_{A,B,C2,D2}, ES_Two-Stage, FP_Two-Tower_{A,B,C2,D2}, FP_Two-Stage (30 épocas),
         G1, G2, G3_beam, G3_exact, G1b (semillas 42-46).
Reanuda por archivo.  Orden por semilla: discriminativos primero (verificación temprana), luego generativos.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from recsys.stage4.models import compute_retrieval_candidates_stage4
from recsys.ranker import train_ranker
from recsys.evaluation import make_two_stage
from recsys.config import Stage3Config
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50
from scripts.run_baselines_with_val_es import train_two_tower_with_early_stopping
from scripts.run_stage5_sensitivity_30ep import train_two_tower_fixed_epochs
from scripts.run_stage5_g1_exact import build_g1

OUT = ROOT / "results" / "stage5" / "recs"
SEL = None   # None = todos los modelos; si no, conjunto de nombres a (re)generar (ver --models)
ALL_MODELS = ["FP_Two-Tower_A", "FP_Two-Tower_B", "FP_Two-Tower_C2", "FP_Two-Tower_D2", "FP_Two-Stage",
              "ES_Two-Tower_A", "ES_Two-Tower_B", "ES_Two-Tower_C2", "ES_Two-Tower_D2", "ES_Two-Stage",
              "G1", "G2", "G3_beam", "G3_exact", "G1b"]
ES_DIR, FP_DIR = ROOT / "results/stage5/per_seed", ROOT / "results/stage5/per_seed_30ep"
GEN_CFG = {"G1": (100, 1e-3, 0.1, 18), "G2": (100, 1e-3, 0.1, 18),
           "G3": (20, 5e-4, 0.2, 16), "G1b": (100, 1e-3, 0.1, 18)}  # (max_len, lr, dropout, epochs)


def pad(recs, users, k=50):
    arr = np.full((len(users), k), -1, dtype=np.int32)
    for i, u in enumerate(users):
        r = list(recs[u])[:k]
        arr[i, :len(r)] = r
    return arr


def save(name, seed, users, recs, bundle, saved_ndcg, log):
    f = OUT / f"{name}_seed_{seed}.npz"
    arr = pad(recs, users)
    np.savez_compressed(f, users=np.array(users, dtype=np.int32), recs=arr)
    m = compute_metrics_at_10_and_50({u: list(recs[u]) for u in users}, bundle.test_relevant_by_user, users,
                                     bundle.cold_item_indices, bundle.item_buckets, bundle.n_items, beam_size=50)
    log[f"{name}|{seed}"] = {"ndcg_new": m["ndcg@10"], "ndcg_published": saved_ndcg,
                             "diff": None if saved_ndcg is None else m["ndcg@10"] - saved_ndcg,
                             "n_short_lists": int((arr < 0).any(axis=1).sum())}
    print(f"  {name} seed {seed}: NDCG@10={m['ndcg@10']:.4f} (publicado {saved_ndcg}) "
          f"diff={log[f'{name}|{seed}']['diff']}", flush=True)


def pub(d, pre, seed, sub=None):
    p = d / f"{pre}_seed_{seed}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    return (j[sub] if sub else j)["ndcg@10"]


def train_gen(tok, bundle, seed, device, cfg):
    max_len, lr, drop, epochs = cfg
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    win = get_windows_for_policy(bundle.user_train_seq, policy="balanceado", max_len=max_len, seed=seed)
    loader = DataLoader(SequenceDataset(win, tok, max_tokens=150), batch_size=128, shuffle=True, generator=gen,
                        pin_memory=True, collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID))
    model = T5Recommender(tok, dropout_rate=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for bx, ba, by in loader:
            bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(bx, attention_mask=ba, labels=by).loss
            loss.backward(); opt.step()
    return model


def beam_recs(model, tok, bundle, device):
    trie = ItemTrie(tok)
    return {u: model.recommend_for_user(bundle.user_train_seq[u], trie, bundle.user_seen_train.get(u, set()),
                                        device, num_beams=50, top_k=50, max_context_tokens=150)
            for u in bundle.eval_users_test}


def g3_exact_recs(model, tok, bundle, device):
    model.eval(); out = {}
    with torch.no_grad():
        for u in bundle.eval_users_test:
            seen = bundle.user_seen_train.get(u, set())
            inp = torch.tensor([tok.encode_input_sequence(bundle.user_train_seq[u], max_tokens=150)], dtype=torch.long, device=device)
            enc = model.model.encoder(input_ids=inp)
            bos = torch.tensor([[tok.BOS_TOKEN_ID]], dtype=torch.long, device=device)
            dec = model.model.decoder(input_ids=bos, encoder_hidden_states=enc.last_hidden_state)
            logits = model.model.lm_head(dec.last_hidden_state)[0, 0].cpu().numpy().copy()
            sc = np.full(bundle.n_items, -1e9, dtype=np.float32)
            for i in range(bundle.n_items):
                if i not in seen:
                    sc[i] = logits[tok.encode_target_item(i)[0]]
            out[u] = list(np.argsort(sc)[::-1][:50])
    return out


def two_stage_recs(model_a, bundle, item_genres_np, cfg, device, seed):
    n_items = bundle.n_items
    top100 = compute_retrieval_candidates_stage4(model_a, bundle.n_users, n_items, bundle.seen_mask_np,
                                                 max_candidates=100, device=device)
    pos = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]
    pu, pi = pos["user_idx"].values, pos["item_idx"].values
    up = np.zeros((bundle.n_users, n_items), dtype=np.float32); up[pu, pi] = 1.0
    ug = up @ item_genres_np
    ug = ug / np.maximum(ug.sum(axis=1, keepdims=True), 1.0)
    lp = np.log1p(bundle.item_popularity).astype(np.float32)
    ranker, _ = train_ranker(pos_users=pu, pos_items=pi, n_users=bundle.n_users, n_items=n_items,
                             user_genre_np=ug, item_genre_np=item_genres_np, item_logpop_np=lp,
                             seen_mask_np=bundle.seen_mask_np, retrieval_topk=top100,
                             n_uniform=2, n_hard=2, cfg=cfg, device=device, seed=seed)
    fn = make_two_stage(ranker, top100, n_candidates=50, device=device, top_k=50)
    return {u: list(fn(u)) for u in bundle.eval_users_test}, top100


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS,
                    help="Solo estos modelos (default: todos). Útil para la reproducción liviana en Colab.")
    ap.add_argument("--out-dir", default=None,
                    help="Directorio de salida (default: results/stage5/recs). El chequeo contra los NDCG publicados no cambia.")
    args = ap.parse_args()
    global OUT, SEL
    if args.out_dir:
        OUT = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    SEL = set(args.models) if args.models else None
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "repro_check.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    movies = pd.read_csv(ROOT / "data/ml-latest-small/movies.csv")
    emb = np.load(ROOT / "data/cache/movie_embeddings_minilm.npy")
    gd = dict(zip(movies["movieId"], movies["genres"].str.split("|")))
    genres = sorted({g for gl in gd.values() for g in gl if g != "(no genres listed)"})
    g2c = {g: c for c, g in enumerate(genres)}

    for seed in args.seeds:
        t_seed = time.perf_counter()
        print(f"\n=== SEMILLA {seed} ===", flush=True)
        b = create_temporal_split(ratings, movies, seed=seed)
        n_items, users = b.n_items, b.eval_users_test
        ig = np.zeros((n_items, len(genres)), dtype=np.float32)
        for mid, idx in b.item2idx.items():
            for g in gd.get(mid, []):
                if g in g2c:
                    ig[idx, g2c[g]] = 1.0
        cold = np.zeros(n_items, dtype=bool); cold[list(b.cold_item_indices)] = True
        cfg = Stage3Config(data_dir=ROOT / "data", seed=seed, device="cuda" if device.type == "cuda" else "cpu")

        def todo(name):
            return (SEL is None or name in SEL) and not (OUT / f"{name}_seed_{seed}.npz").exists()

        def flush_log():
            log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

        # --- 30 épocas fijas ---
        for v, pre in [("A", "Two-Tower_A"), ("B", "Two-Tower_B"), ("C2", "Two-Tower_C2"), ("D2", "Two-Tower_D2")]:
            need_ts = (v == "A" and todo("FP_Two-Stage"))
            if todo(f"FP_{pre}") or need_ts:
                m = train_two_tower_fixed_epochs(v, b, emb, ig, cold, device, seed=seed, epochs=30)
                if todo(f"FP_{pre}"):
                    top = compute_retrieval_candidates_stage4(m, b.n_users, n_items, b.seen_mask_np, max_candidates=50, device=device)
                    save(f"FP_{pre}", seed, users, {u: list(top[u]) for u in users}, b, pub(FP_DIR, pre, seed), log)
                if need_ts:
                    r, _ = two_stage_recs(m, b, ig, cfg, device, seed)
                    save("FP_Two-Stage", seed, users, r, b, pub(FP_DIR, "Two-Stage", seed), log)
                flush_log()
        # --- parada por validación ---
        for v, pre in [("A", "Two-Tower_A"), ("B", "Two-Tower_B"), ("C2", "Two-Tower_C2"), ("D2", "Two-Tower_D2")]:
            need_ts = (v == "A" and todo("ES_Two-Stage"))
            if todo(f"ES_{pre}") or need_ts:
                m, *_ = train_two_tower_with_early_stopping(v, b, emb, ig, cold, cfg, device, seed=seed)
                if todo(f"ES_{pre}"):
                    top = compute_retrieval_candidates_stage4(m, b.n_users, n_items, b.seen_mask_np, max_candidates=50, device=device)
                    save(f"ES_{pre}", seed, users, {u: list(top[u]) for u in users}, b, pub(ES_DIR, pre, seed), log)
                if need_ts:
                    r, _ = two_stage_recs(m, b, ig, cfg, device, seed)
                    save("ES_Two-Stage", seed, users, r, b, pub(ES_DIR, "Two-Stage", seed), log)
                flush_log()
        # --- generativos ---
        tok1, _ = build_g1(b, emb, seed)
        if todo("G1"):
            m = train_gen(tok1, b, seed, device, GEN_CFG["G1"])
            save("G1", seed, users, beam_recs(m, tok1, b, device), b, pub(ES_DIR, "G1", seed), log); flush_log()
        if todo("G2"):
            tok2 = GenerativeTokenizer("G2", n_items=n_items, random_id_map=build_random_id_map(tok1.id_map, seed=seed))
            m = train_gen(tok2, b, seed, device, GEN_CFG["G2"])
            save("G2", seed, users, beam_recs(m, tok2, b, device), b, pub(ES_DIR, "G2", seed), log); flush_log()
        if todo("G3_beam") or todo("G3_exact"):
            tok3 = GenerativeTokenizer("G3", n_items=n_items)
            m = train_gen(tok3, b, seed, device, GEN_CFG["G3"])
            if todo("G3_beam"):
                save("G3_beam", seed, users, beam_recs(m, tok3, b, device), b, pub(ES_DIR, "G3", seed, "beam_50"), log)
            if todo("G3_exact"):
                save("G3_exact", seed, users, g3_exact_recs(m, tok3, b, device), b, pub(ES_DIR, "G3", seed, "exact"), log)
            flush_log()
        if seed <= 46 and todo("G1b"):
            emb_ny = np.load(ROOT / "data/cache/movie_embeddings_minilm_noyear.npy")   # solo G1b
            tokb, _ = build_g1(b, emb_ny, seed)
            m = train_gen(tokb, b, seed, device, GEN_CFG["G1b"])
            save("G1b", seed, users, beam_recs(m, tokb, b, device), b, pub(ES_DIR, "G1b", seed), log); flush_log()
        print(f"Semilla {seed} completa en {(time.perf_counter()-t_seed)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
