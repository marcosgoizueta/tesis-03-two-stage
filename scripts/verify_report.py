"""
verify_report.py — Recalcula todas las cifras del multi_seed_summary.json
desde los per_seed/*.json y falla si hay discrepancias.

Verifica también:
  - Que los valores de semilla 42 coincidan con las referencias de Fase 2c.
  - Que generate_stage5_report_tables.py pueda ejecutarse (el archivo del que depende
    está en scratch/ y no está versionado; este script lo detecta y avisa).
  - Guarda checksums SHA256 de todos los per_seed JSONs.

Uso:
    python scripts/verify_report.py
    python scripts/verify_report.py --strict   # sale con código 1 si hay cualquier diferencia
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PER_SEED_DIR = ROOT / "results" / "stage5" / "per_seed"
SUMMARY_PATH = ROOT / "results" / "stage5" / "multi_seed_summary.json"
OUTPUT_PATH = ROOT / "results" / "stage5" / "verify_report.json"

METRICS = ["ndcg@10", "recall@10", "recall@50", "cold_recall@50", "cov@10", "tail@10"]

# Referencia Fase 2c (semilla 42, haz 50).  Fuente: CLAUDE.md sección 6.
FASE2C_REF = {
    "G1":      {"ndcg@10": 0.0658},
    "G2":      {"ndcg@10": 0.0399},
    "G3_beam": {"ndcg@10": 0.0832},
}
FASE2C_TOL = 5e-4   # tolerancia para comparar contra los valores redondeados del CLAUDE.md

# Tolerancia para comparar recálculo vs summary guardado (error de punto flotante puro)
RECOMPUTE_TOL = 1e-9

# --- helpers ----------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_flat_metrics(d: dict) -> dict:
    """Devuelve {metric: value} para un JSON con estructura plana."""
    return {m: d[m] for m in METRICS if m in d}


def get_model_config() -> list[dict]:
    """
    Describe cómo extraer métricas de cada entrada del summary a partir de
    los per_seed JSONs.  Cada entrada tiene:
      - summary_key: clave en models_summary
      - file_prefix: prefijo del archivo per_seed (ej. "G1" → G1_seed_42.json)
      - extractor: función(dict) → {metric: float}
      - seeds: lista de semillas esperadas
    """
    flat = get_flat_metrics

    def g3_beam(d):
        return get_flat_metrics(d["beam_50"])

    def g3_exact(d):
        return get_flat_metrics(d["exact"])

    return [
        dict(summary_key="Two-Tower_A",  file_prefix="Two-Tower_A",  extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="Two-Tower_B",  file_prefix="Two-Tower_B",  extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="Two-Tower_C2", file_prefix="Two-Tower_C2", extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="Two-Tower_D2", file_prefix="Two-Tower_D2", extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="Two-Stage",    file_prefix="Two-Stage",    extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="G1",           file_prefix="G1",           extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="G2",           file_prefix="G2",           extractor=flat,     seeds=list(range(42, 52))),
        dict(summary_key="G3_beam",      file_prefix="G3",           extractor=g3_beam,  seeds=list(range(42, 52))),
        dict(summary_key="G3_exact",     file_prefix="G3",           extractor=g3_exact, seeds=list(range(42, 52))),
        dict(summary_key="G1b",          file_prefix="G1b",          extractor=flat,     seeds=list(range(42, 47))),
    ]


def load_per_seed(file_prefix: str, seed: int) -> dict:
    path = PER_SEED_DIR / f"{file_prefix}_seed_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Falta: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --- verificación principal -------------------------------------------------

def verify_summary(summary: dict, strict: bool) -> tuple[list, list]:
    """
    Recalcula media y std de cada métrica para cada modelo y compara contra
    el summary guardado.
    Devuelve (errors, warnings).
    """
    errors = []
    warnings = []
    models_sum = summary["models_summary"]

    for cfg in get_model_config():
        sk = cfg["summary_key"]
        if sk not in models_sum:
            errors.append(f"[FALTA EN SUMMARY] {sk}")
            continue

        entry = models_sum[sk]
        vals_by_metric: dict[str, list] = {m: [] for m in METRICS}

        for seed in cfg["seeds"]:
            try:
                raw = load_per_seed(cfg["file_prefix"], seed)
            except FileNotFoundError as e:
                errors.append(str(e))
                continue
            metrics = cfg["extractor"](raw)
            for m in METRICS:
                if m in metrics:
                    vals_by_metric[m].append(metrics[m])

        for m in METRICS:
            vals = vals_by_metric[m]
            if not vals:
                warnings.append(f"[SIN DATOS] {sk} / {m}")
                continue

            recalc_mean = float(np.mean(vals))
            recalc_std  = float(np.std(vals, ddof=1))

            mean_key = f"{m}_mean"
            std_key  = f"{m}_std"

            if mean_key not in entry:
                warnings.append(f"[CLAVE AUSENTE EN SUMMARY] {sk} / {mean_key}")
                continue

            saved_mean = entry[mean_key]
            saved_std  = entry.get(std_key)

            diff_mean = abs(recalc_mean - saved_mean)
            if diff_mean > RECOMPUTE_TOL:
                errors.append(
                    f"[DISCREPANCIA MEDIA] {sk} / {m}: "
                    f"recalculado={recalc_mean:.10f} guardado={saved_mean:.10f} "
                    f"diff={diff_mean:.2e}"
                )

            if saved_std is not None:
                diff_std = abs(recalc_std - saved_std)
                if diff_std > RECOMPUTE_TOL:
                    errors.append(
                        f"[DISCREPANCIA STD] {sk} / {m}: "
                        f"recalculado={recalc_std:.10f} guardado={saved_std:.10f} "
                        f"diff={diff_std:.2e}"
                    )

    return errors, warnings


def verify_fase2c(strict: bool) -> list:
    """Compara los valores de semilla 42 contra las referencias de Fase 2c."""
    issues = []
    for model_key, refs in FASE2C_REF.items():
        if model_key == "G3_beam":
            file_prefix, extractor = "G3", lambda d: get_flat_metrics(d["beam_50"])
        else:
            file_prefix = model_key
            extractor = get_flat_metrics

        try:
            raw = load_per_seed(file_prefix, 42)
        except FileNotFoundError as e:
            issues.append(f"[FALTA ARCHIVO FASE2C] {e}")
            continue

        metrics = extractor(raw)
        for m, ref_val in refs.items():
            actual = metrics.get(m)
            if actual is None:
                issues.append(f"[FALTA MÉTRICA FASE2C] {model_key} / {m}")
                continue
            diff = abs(actual - ref_val)
            if diff > FASE2C_TOL:
                issues.append(
                    f"[NO COINCIDE FASE2C] {model_key} / {m}: "
                    f"per_seed={actual:.6f} referencia={ref_val:.4f} diff={diff:.4f}"
                )
            else:
                print(f"  OK  {model_key} seed42 {m}: {actual:.6f} ~= {ref_val:.4f}")
    return issues


def compute_checksums() -> dict[str, str]:
    checksums = {}
    for path in sorted(PER_SEED_DIR.glob("*.json")):
        checksums[path.name] = sha256_file(path)
    return checksums


def check_generate_tables_dep() -> list:
    """Avisa si el archivo del que depende generate_stage5_report_tables.py no existe."""
    dep = ROOT / "results" / "stage5" / "fase2c_reference.json"
    if not dep.exists():
        return [
            f"[DEP FALTANTE] results/stage5/fase2c_reference.json no encontrado "
            f"(lo usan generate_stage5_report_tables.py y build_final_table_stage5.py)."
        ]
    return []


# --- main -------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Verifica multi_seed_summary.json contra per_seed/*.json")
    parser.add_argument("--strict", action="store_true",
                        help="Sale con código 1 si hay cualquier diferencia")
    args = parser.parse_args()

    print("=" * 70)
    print("VERIFY REPORT — Etapa 5")
    print("=" * 70)

    if not SUMMARY_PATH.exists():
        print(f"ERROR: No se encontró {SUMMARY_PATH}")
        sys.exit(1)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    # 1. Verificar coincidencia de recálculo vs summary
    print("\n[1] Recalculando medias y desviaciones desde per_seed/*.json …")
    errors, warnings = verify_summary(summary, args.strict)

    if not errors and not warnings:
        print("  Todas las métricas coinciden.")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")

    # 2. Verificar semilla 42 contra Fase 2c
    print("\n[2] Verificando semilla 42 contra referencias de Fase 2c …")
    fase2c_issues = verify_fase2c(args.strict)
    for issue in fase2c_issues:
        print(f"  ERROR {issue}")
    if not fase2c_issues:
        print("  Todos los valores de referencia coinciden dentro de la tolerancia.")

    # 3. Dependencia de generate_stage5_report_tables.py
    print("\n[3] Verificando dependencias de scripts de reporte …")
    dep_issues = check_generate_tables_dep()
    for d in dep_issues:
        print(f"  WARN  {d}")
    if not dep_issues:
        print("  results/stage5/fase2c_reference.json presente.")

    # 4. Checksums
    print("\n[4] Calculando checksums SHA256 de per_seed/*.json …")
    checksums = compute_checksums()
    print(f"  {len(checksums)} archivos procesados.")

    # 5. Guardar reporte
    all_errors = errors + fase2c_issues
    report = {
        "summary_source": str(SUMMARY_PATH),
        "per_seed_dir": str(PER_SEED_DIR),
        "n_per_seed_files": len(checksums),
        "recompute_tolerance": RECOMPUTE_TOL,
        "fase2c_tolerance": FASE2C_TOL,
        "errors": all_errors,
        "warnings": warnings + dep_issues,
        "checksums_sha256": checksums,
        "status": "PASS" if not all_errors else "FAIL",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado en: {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    if all_errors:
        print(f"RESULTADO: FAIL — {len(all_errors)} error(es), {len(warnings + dep_issues)} advertencia(s)")
        if args.strict:
            sys.exit(1)
    else:
        print(f"RESULTADO: PASS — 0 errores, {len(warnings + dep_issues)} advertencia(s)")


if __name__ == "__main__":
    main()
