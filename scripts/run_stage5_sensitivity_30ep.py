"""
run_stage5_sensitivity_30ep.py — Tarea T2: Sensibilidad de protocolo.

Pregunta central: ¿supera Two-Stage al retrieval solo (Two-Tower A) tanto con
parada temprana (protocolo actual, E*≈2 para A) como con 30 épocas fijas
(protocolo de las Etapas 3 y 4)?

Este script corre Two-Tower A, B, C2, D2 y Two-Stage con 30 épocas fijas en
las 10 semillas (42-51), entrenando en train_full (mismo split temporal que el
multi-seed principal).  Los resultados van a results/stage5/per_seed_30ep/ y
no tocan los resultados de parada temprana en results/stage5/per_seed/.

Al terminar genera:
  - results/stage5/sensitivity_30ep_summary.json   (cifras por semilla y media)
  - results/stage5/sensitivity_30ep_analysis.md    (tabla comparativa y conclusiones)

Uso:
    python scripts/run_stage5_sensitivity_30ep.py
    python scripts/run_stage5_sensitivity_30ep.py --epochs 30 --seeds 42 43
    python scripts/run_stage5_sensitivity_30ep.py --resume   (reanuda corrida parcial)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.stage5.data_split import create_temporal_split
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4
from recsys.retrieval import RetrievalDataset
from recsys.ranker import train_ranker
from recsys.evaluation import make_two_stage
from recsys.config import Stage3Config
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50

ROOT = Path(__file__).resolve().parent.parent
PER_SEED_ES_DIR  = ROOT / "results" / "stage5" / "per_seed"       # resultados con early stopping (ya existen)
PER_SEED_30_DIR  = ROOT / "results" / "stage5" / "per_seed_30ep"  # resultados con 30 épocas fijas (nuevos)
RESULTS_DIR      = ROOT / "results" / "stage5"

FIXED_EPOCHS = 30


# ---------------------------------------------------------------------------
# Entrenamiento con épocas fijas sobre train_full
# ---------------------------------------------------------------------------

def train_two_tower_fixed_epochs(
    variant: str,
    bundle,
    all_text_embs: np.ndarray,
    item_genres_np: np.ndarray,
    is_cold_mask_np: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = FIXED_EPOCHS,
) -> torch.nn.Module:
    """
    Entrena un TwoTowerStage4 directamente en train_full durante `epochs` épocas
    fijas (sin parada temprana).  Mismo protocolo que las Etapas 3 y 4.
    """
    n_items = bundle.n_items
    train_full_pos = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ds = RetrievalDataset(
        train_full_pos, bundle.user_seen_train, n_items,
        bundle.seen_mask_np, n_neg_per_pos=4,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = TwoTowerStage4(
        variant=variant,
        n_users=bundle.n_users,
        n_items=n_items,
        embed_dim=32,
        text_embeddings_np=all_text_embs if variant in {"B", "C2"} else None,
        item_genres_np=item_genres_np  if variant in {"D2"}        else None,
        is_cold_mask_np=is_cold_mask_np,
        seed=seed,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
    crit = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        ds.resample_negatives()
        model.train()
        for u, p, negs in loader:
            u, p, negs = u.to(device), p.to(device), negs.to(device)
            optimizer.zero_grad()
            pos_scores = model.score(u, p)
            u_exp = u.unsqueeze(1).expand(-1, negs.size(1)).reshape(-1)
            negs_flat = negs.reshape(-1)
            neg_scores = model.score(u_exp, negs_flat)
            loss = crit(
                torch.cat([pos_scores, neg_scores]),
                torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]),
            )
            loss.backward()
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def run():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Saltar semillas que ya tienen JSON guardado")
    args = parser.parse_args()

    PER_SEED_30_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Épocas fijas: {args.epochs} | Semillas: {args.seeds}")

    ratings     = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    movies      = pd.read_csv(ROOT / "data/ml-latest-small/movies.csv")
    all_embeddings = np.load(ROOT / "data/cache/movie_embeddings_minilm.npy")

    movie_genres_dict = dict(zip(movies["movieId"], movies["genres"].str.split("|")))
    all_genres_list   = sorted({g for gl in movie_genres_dict.values() for g in gl if g != "(no genres listed)"})
    genre2col         = {g: c for c, g in enumerate(all_genres_list)}

    variants = [
        ("A",  "Two-Tower_A"),
        ("B",  "Two-Tower_B"),
        ("C2", "Two-Tower_C2"),
        ("D2", "Two-Tower_D2"),
    ]

    for seed in args.seeds:
        t_seed = time.perf_counter()
        print(f"\n{'='*70}")
        print(f"SEMILLA {seed}")
        print("="*70)

        bundle = create_temporal_split(ratings, movies, seed=seed)
        n_items    = bundle.n_items
        test_users = bundle.eval_users_test

        item_genres_np = np.zeros((n_items, len(all_genres_list)), dtype=np.float32)
        for mid, idx in bundle.item2idx.items():
            for g in movie_genres_dict.get(mid, []):
                if g in genre2col:
                    item_genres_np[idx, genre2col[g]] = 1.0

        is_cold_mask_np = np.zeros(n_items, dtype=bool)
        is_cold_mask_np[list(bundle.cold_item_indices)] = True

        retrieval_topk_a_100 = None

        for v_code, v_name in variants:
            out_file = PER_SEED_30_DIR / f"{v_name}_seed_{seed}.json"
            ts_file  = PER_SEED_30_DIR / f"Two-Stage_seed_{seed}.json"
            need_a   = (v_code == "A" and not (args.resume and ts_file.exists()))

            if args.resume and out_file.exists() and not need_a:
                d = json.loads(out_file.read_text())
                if v_code == "A":
                    # necesitamos el modelo A para Two-Stage si ts_file no existe
                    pass
                print(f"  [Skip] {v_name} seed {seed}")
                continue

            print(f"  Entrenando {v_name} ({args.epochs} épocas, seed={seed})...")
            t0 = time.perf_counter()
            model = train_two_tower_fixed_epochs(
                v_code, bundle, all_embeddings, item_genres_np, is_cold_mask_np,
                device, seed=seed, epochs=args.epochs,
            )
            train_time = time.perf_counter() - t0

            if v_code == "A":
                retrieval_topk_a_100 = compute_retrieval_candidates_stage4(
                    model, bundle.n_users, n_items, bundle.seen_mask_np,
                    max_candidates=100, device=device,
                )
                topk = retrieval_topk_a_100[:, :50]
            else:
                topk = compute_retrieval_candidates_stage4(
                    model, bundle.n_users, n_items, bundle.seen_mask_np,
                    max_candidates=50, device=device,
                )

            if not (args.resume and out_file.exists()):
                recs = {u: list(topk[u]) for u in test_users}
                m = compute_metrics_at_10_and_50(
                    recs, bundle.test_relevant_by_user, test_users,
                    bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50,
                )
                m["protocol"] = "fixed_30ep"
                m["epochs"]   = args.epochs
                m["train_time_s"] = train_time
                out_file.write_text(json.dumps(m, indent=2), encoding="utf-8")
                print(f"    -> NDCG@10={m['ndcg@10']:.4f}  Recall@10={m['recall@10']:.4f}  "
                      f"Cov@10={m['cov@10']*100:.2f}%  ({train_time:.1f}s)")

        # Two-Stage con el modelo A de 30 épocas
        ts_file = PER_SEED_30_DIR / f"Two-Stage_seed_{seed}.json"
        if args.resume and ts_file.exists():
            print(f"  [Skip] Two-Stage seed {seed}")
        else:
            if retrieval_topk_a_100 is None:
                # cargar modelo A si fue saltado por resume pero Two-Stage no existe
                print(f"  Re-entrenando Two-Tower A para Two-Stage (seed={seed})...")
                model_a = train_two_tower_fixed_epochs(
                    "A", bundle, all_embeddings, item_genres_np, is_cold_mask_np,
                    device, seed=seed, epochs=args.epochs,
                )
                retrieval_topk_a_100 = compute_retrieval_candidates_stage4(
                    model_a, bundle.n_users, n_items, bundle.seen_mask_np,
                    max_candidates=100, device=device,
                )

            print(f"  Entrenando Ranker (Two-Stage, seed={seed})...")
            train_full_pos = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]
            pos_users = train_full_pos["user_idx"].values
            pos_items = train_full_pos["item_idx"].values
            user_pos_np = np.zeros((bundle.n_users, n_items), dtype=np.float32)
            user_pos_np[pos_users, pos_items] = 1.0
            user_genre_np = user_pos_np @ item_genres_np
            user_genre_np /= np.maximum(user_genre_np.sum(axis=1, keepdims=True), 1.0)
            item_logpop_np = np.log1p(bundle.item_popularity).astype(np.float32)

            cfg = Stage3Config(data_dir=ROOT / "data", seed=seed,
                               device="cuda" if device.type == "cuda" else "cpu")
            t0 = time.perf_counter()
            ranker, _ = train_ranker(
                pos_users=pos_users, pos_items=pos_items,
                n_users=bundle.n_users, n_items=n_items,
                user_genre_np=user_genre_np, item_genre_np=item_genres_np,
                item_logpop_np=item_logpop_np, seen_mask_np=bundle.seen_mask_np,
                retrieval_topk=retrieval_topk_a_100,
                n_uniform=2, n_hard=2, cfg=cfg, device=device, seed=seed,
            )
            ts_recs_fn = make_two_stage(ranker, retrieval_topk_a_100,
                                        n_candidates=50, device=device, top_k=50)
            ts_recs = {u: list(ts_recs_fn(u)) for u in test_users}
            train_time = time.perf_counter() - t0

            m_ts = compute_metrics_at_10_and_50(
                ts_recs, bundle.test_relevant_by_user, test_users,
                bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50,
            )
            m_ts["protocol"]     = "fixed_30ep"
            m_ts["epochs"]       = args.epochs
            m_ts["train_time_s"] = train_time
            ts_file.write_text(json.dumps(m_ts, indent=2), encoding="utf-8")
            print(f"    -> Two-Stage NDCG@10={m_ts['ndcg@10']:.4f}  "
                  f"Recall@10={m_ts['recall@10']:.4f}  "
                  f"Cov@10={m_ts['cov@10']*100:.2f}%  ({train_time:.1f}s)")

        print(f"  Semilla {seed} completa en {time.perf_counter()-t_seed:.1f}s")

    # -------------------------------------------------------------------
    # Generar summary + análisis
    # -------------------------------------------------------------------
    build_summary_and_analysis(args.seeds, args.epochs)


# ---------------------------------------------------------------------------
# Summary y análisis
# ---------------------------------------------------------------------------

MODELS_30EP = [
    ("Two-Tower_A",  "A"),
    ("Two-Tower_B",  "B"),
    ("Two-Tower_C2", "C2"),
    ("Two-Tower_D2", "D2"),
    ("Two-Stage",    "Two-Stage"),
]
METRICS_TO_REPORT = ["ndcg@10", "recall@10", "recall@50", "cold_recall@50", "cov@10", "tail@10"]


def load_per_seed_vals(model_key: str, seeds: list, source_dir: Path) -> dict[str, list]:
    """Carga valores por semilla desde source_dir."""
    vals = {m: [] for m in METRICS_TO_REPORT}
    for seed in seeds:
        path = source_dir / f"{model_key}_seed_{seed}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        for m in METRICS_TO_REPORT:
            if m in d:
                vals[m].append(float(d[m]))
    return vals


def build_summary_and_analysis(seeds: list, epochs: int):
    summary = {
        "protocol_fixed_30ep": {},
        "protocol_early_stopping": {},
    }

    for model_key, _ in MODELS_30EP:
        # 30 épocas fijas
        vals_30 = load_per_seed_vals(model_key, seeds, PER_SEED_30_DIR)
        summary["protocol_fixed_30ep"][model_key] = {
            "n_seeds": len(vals_30["ndcg@10"]),
            "seeds":   seeds,
            **{f"{m}_mean": float(np.mean(v)) if v else None for m, v in vals_30.items()},
            **{f"{m}_std":  float(np.std(v, ddof=1)) if len(v) > 1 else None for m, v in vals_30.items()},
            **{f"{m}_vals": v for m, v in vals_30.items()},
        }

        # Early stopping (desde per_seed existente)
        vals_es = load_per_seed_vals(model_key, seeds, PER_SEED_ES_DIR)
        summary["protocol_early_stopping"][model_key] = {
            "n_seeds": len(vals_es["ndcg@10"]),
            "seeds":   seeds,
            **{f"{m}_mean": float(np.mean(v)) if v else None for m, v in vals_es.items()},
            **{f"{m}_std":  float(np.std(v, ddof=1)) if len(v) > 1 else None for m, v in vals_es.items()},
            **{f"{m}_vals": v for m, v in vals_es.items()},
        }

    out_json = RESULTS_DIR / "sensitivity_30ep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary guardado en: {out_json}")

    # Análisis Markdown
    _write_analysis_md(summary, seeds, epochs)


def _fmt(val, pct=False):
    if val is None:
        return "N/A"
    return f"{val*100:.2f}%" if pct else f"{val:.4f}"


def _write_analysis_md(summary: dict, seeds: list, epochs: int):
    es  = summary["protocol_early_stopping"]
    fp  = summary["protocol_fixed_30ep"]

    lines = []
    lines.append("# Análisis de Sensibilidad de Protocolo — Etapa 5")
    lines.append("")
    lines.append(f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Semillas:** {seeds}  |  **Épocas fijas:** {epochs}")
    lines.append("")
    lines.append(
        "Pregunta central: ¿supera Two-Stage al retrieval solo (Two-Tower A) tanto con "
        "parada temprana como con 30 épocas fijas?"
    )
    lines.append("")

    # Tabla NDCG@10
    lines.append("## NDCG@10 — comparación de protocolos")
    lines.append("")
    lines.append("| Modelo | ES media ± std | 30ep media ± std | Δ (30ep − ES) |")
    lines.append("|--------|---------------|-----------------|--------------|")
    for model_key, label in MODELS_30EP:
        e_mean = es[model_key].get("ndcg@10_mean")
        e_std  = es[model_key].get("ndcg@10_std")
        f_mean = fp[model_key].get("ndcg@10_mean")
        f_std  = fp[model_key].get("ndcg@10_std")
        delta  = (f_mean - e_mean) if (f_mean is not None and e_mean is not None) else None
        d_str  = (f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}") if delta is not None else "N/A"
        e_str  = (f"{e_mean:.4f} ± {e_std:.4f}" if e_std is not None
                  else f"{e_mean:.4f}") if e_mean is not None else "N/A"
        f_str  = (f"{f_mean:.4f} ± {f_std:.4f}" if f_std is not None
                  else f"{f_mean:.4f}") if f_mean is not None else "N/A"
        lines.append(f"| {label} | {e_str} | {f_str} | {d_str} |")
    lines.append("")

    # Tabla Cobertura y Cola
    lines.append("## Cobertura @10 y Fracción de Cola @10")
    lines.append("")
    lines.append("| Modelo | ES cov@10 | 30ep cov@10 | ES tail@10 | 30ep tail@10 |")
    lines.append("|--------|-----------|------------|-----------|-------------|")
    for model_key, label in MODELS_30EP:
        ec = _fmt(es[model_key].get("cov@10_mean"), pct=True)
        fc = _fmt(fp[model_key].get("cov@10_mean"), pct=True)
        et = _fmt(es[model_key].get("tail@10_mean"), pct=True)
        ft = _fmt(fp[model_key].get("tail@10_mean"), pct=True)
        lines.append(f"| {label} | {ec} | {fc} | {et} | {ft} |")
    lines.append("")

    # Diferencias pareadas Two-Stage vs A
    lines.append("## Diferencias pareadas: Two-Stage − Two-Tower A")
    lines.append("")
    for protocol_key, label in [("protocol_early_stopping", "Early Stopping"),
                                  ("protocol_fixed_30ep", f"{epochs} épocas fijas")]:
        d = summary[protocol_key]
        a_vals  = d["Two-Tower_A"]["ndcg@10_vals"]
        ts_vals = d["Two-Stage"]["ndcg@10_vals"]
        if a_vals and ts_vals and len(a_vals) == len(ts_vals):
            deltas = [t - a for t, a in zip(ts_vals, a_vals)]
            n_pos  = sum(1 for x in deltas if x > 0)
            mean_d = np.mean(deltas)
            lines.append(f"**{label}**: Two-Stage − A media = {mean_d:+.4f}  "
                          f"(n+ = {n_pos}/{len(deltas)})")
        else:
            lines.append(f"**{label}**: datos incompletos (n_A={len(a_vals)}, n_TS={len(ts_vals)})")
    lines.append("")

    # Notas de interpretación
    lines.append("## Notas de interpretación")
    lines.append("")
    lines.append(
        "- **Early stopping agresivo en A**: con parada temprana por val_loss "
        "(paciencia 3, mejora mínima 0.005), el modelo A converge en E*≈2 épocas "
        "en la mayoría de las semillas, lo que lo deja sistemáticamente por debajo "
        "de la popularidad. Esto convierte la diferencia Two-Stage − A con early "
        "stopping en un artefacto del protocolo (ver error #6 en CLAUDE.md)."
    )
    lines.append("")
    lines.append(
        "- **Protocolo de 30 épocas fijas**: entrena directamente en train_full "
        "durante 30 épocas, que es el mismo protocolo usado en las Etapas 3 y 4. "
        "Con este protocolo, el modelo A converge a un nivel más representativo de "
        "su capacidad real."
    )
    lines.append("")
    lines.append(
        "- **Hipótesis a verificar con estos datos**: si Two-Stage supera a A con "
        "30 épocas fijas, la ventaja del reranker es robusta. Si no, la mejora "
        "observada con early stopping era un artefacto del E* agresivo."
    )
    lines.append("")
    lines.append(
        "- **Comparación con otros modelos**: los valores de Two-Stage y los "
        "Two-Tower de estas tablas deben compararse con los generativos G1, G2, G3 "
        "del multi-seed principal (results/stage5/multi_seed_summary.json), teniendo "
        "en cuenta la asimetría de protocolo documentada en el README."
    )
    lines.append("")
    lines.append("## Archivos fuente")
    lines.append("")
    lines.append(f"- Per-seed 30 épocas: `results/stage5/per_seed_30ep/`")
    lines.append(f"- Per-seed early stopping: `results/stage5/per_seed/`")
    lines.append(f"- Summary: `results/stage5/sensitivity_30ep_summary.json`")
    lines.append(f"- Stats generales: `results/stage5/stats_report.json`")

    out_md = RESULTS_DIR / "sensitivity_30ep_analysis.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Análisis guardado en: {out_md}")


if __name__ == "__main__":
    run()
