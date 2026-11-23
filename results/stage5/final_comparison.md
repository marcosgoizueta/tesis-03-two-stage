# Tabla comparativa unificada — Etapa 5 (split temporal, 592 usuarios de test)

Fuente: `results/stage5/final_comparison.json`, generado por `scripts/build_final_table_stage5.py` desde `per_seed/` y `per_seed_30ep/`. 10 semillas (G1b: 5). Media ± desvío muestral; entre corchetes, IC95% bootstrap sobre semillas. Cov@10 y Tail@10 con K=10 para todos los modelos.

| Modelo | NDCG@10 | Recall@10 | Recall@50 | Cold Rec@50 | Cov@10 | Tail@10 |
|---|---|---|---|---|---|---|
| Azar (esperado) | - | - | - | 0.53% | 45.61% | 73.01% |
| Popularidad | 0.0821 | 0.0477 | 0.1454 | 0.0000 | 1.01% | 0.00% |
| *Discriminativo, 30 épocas fijas* | | | | | | |
| Two-Tower A (30 ép.) | 0.0944 ± 0.0045 [0.0920, 0.0972] | 0.0679 | 0.2189 | 0.0000 | 8.60% | 0.12% |
| Two-Tower B (30 ép.) | 0.0988 ± 0.0035 [0.0968, 0.1009] | 0.0648 | 0.2040 | 0.0023 | 8.53% | 2.49% |
| Two-Tower C2 (30 ép.) | 0.1049 ± 0.0049 [0.1021, 0.1078] | 0.0717 | 0.2319 | 0.0033 | 8.94% | 0.91% |
| Two-Tower D2 (30 ép.) | 0.1053 ± 0.0026 [0.1037, 0.1068] | 0.0744 | 0.2246 | 0.0030 | 7.97% | 0.56% |
| Two-Stage (30 ép.) | 0.0957 ± 0.0024 [0.0942, 0.0970] | 0.0695 | 0.2189 | 0.0000 | 8.00% | 0.03% |
| *Discriminativo, parada por validación* | | | | | | |
| Two-Tower A (val.) | 0.0796 ± 0.0024 [0.0782, 0.0810] | 0.0444 | 0.1480 | 0.0000 | 1.44% | 0.23% |
| Two-Tower B (val.) | 0.0916 ± 0.0057 [0.0881, 0.0948] | 0.0578 | 0.1893 | 0.0012 | 6.00% | 3.13% |
| Two-Tower C2 (val.) | 0.1036 ± 0.0026 [0.1022, 0.1052] | 0.0655 | 0.2128 | 0.0010 | 4.49% | 0.34% |
| Two-Tower D2 (val.) | 0.1007 ± 0.0035 [0.0987, 0.1028] | 0.0634 | 0.1926 | 0.0023 | 4.38% | 0.44% |
| Two-Stage (val.) | 0.1050 ± 0.0030 [0.1034, 0.1068] | 0.0681 | 0.1480 | 0.0000 | 2.28% | 0.19% |
| *Generativo, E* fijo* | | | | | | |
| G1 (haz 50) | 0.0571 ± 0.0078 [0.0525, 0.0615] | 0.0456 | 0.1816 | 0.0000 | 2.03% | 0.43% |
| G2 (haz 50) | 0.0466 ± 0.0085 [0.0415, 0.0516] | 0.0357 | 0.1485 | 0.0002 | 1.44% | 0.36% |
| G3 (haz 50) | 0.0868 ± 0.0026 [0.0854, 0.0884] | 0.0630 | 0.1873 | 0.0000 | 2.00% | 0.01% |
| G3 (exacto) | 0.0874 ± 0.0025 [0.0860, 0.0890] | 0.0633 | 0.1873 | 0.0000 | 1.99% | 0.01% |
| G1b (haz 50, 5 sem.) | 0.0535 ± 0.0063 [0.0484, 0.0582] | 0.0422 | 0.1597 | 0.0000 | 1.48% | 0.03% |
| *Regresión de rating, Etapas 1 y 2* | | | | | | |
| SVD (Etapa 1) | 0.0412 ± 0.0026 [0.0397, 0.0428] | 0.0213 | 0.0742 | 0.0000 | 3.31% | 0.18% |
| NCF (Etapa 2, 15 ép.) | 0.0336 ± 0.0027 [0.0320, 0.0352] | 0.0188 | 0.0615 | 0.0000 | 4.62% | 1.72% |
| NCF (épocas por val.) | 0.0421 ± 0.0027 [0.0405, 0.0436] | 0.0236 | 0.0771 | 0.0000 | 2.04% | 0.46% |

## Contrastes primarios (NDCG@10, Holm α=0.05). Mejor discriminativo con 30 épocas: **Two-Tower D2 (30 ép.)** (elegido por NDCG@10 medio sobre test: selección post hoc; la brecha con G1 es grande, pero queda declarado).

| Comparación | Δ medio | IC95% | n+ | p signo | p Wilcoxon | Umbral Holm | ¿Rechaza? |
|---|---|---|---|---|---|---|---|
| G1 (haz 50) - G3 (haz 50) | -0.0296 | [-0.0351, -0.0242] | 0/10 | 0.0020 | 0.0020 | 0.0167 | sí |
| G1 (haz 50) - Two-Tower D2 (30 ép.) | -0.0481 | [-0.0532, -0.0432] | 0/10 | 0.0020 | 0.0020 | 0.0250 | sí |
| G1 (haz 50) - G2 (haz 50) | +0.0106 | [+0.0038, +0.0179] | 8/10 | 0.1094 | 0.0273 | 0.0500 | no |

## Contrastes secundarios (NDCG@10, sin corrección; exploratorios)

| Comparación | Δ medio | IC95% | n+ | p signo | p Wilcoxon |
|---|---|---|---|---|---|
| G3 (haz 50) - Two-Tower D2 (30 ép.) | -0.0185 | [-0.0203, -0.0167] | 0/10 | 0.0020 | 0.0020 |
| G3 (haz 50) - Two-Stage (30 ép.) | -0.0089 | [-0.0112, -0.0063] | 0/10 | 0.0020 | 0.0020 |
| Two-Stage (30 ép.) - Two-Tower A (30 ép.) | +0.0013 | [-0.0018, +0.0040] | 7/10 | 0.3438 | 0.4922 |
| Two-Stage (val.) - Two-Tower A (val.) | +0.0255 | [+0.0227, +0.0283] | 10/10 | 0.0020 | 0.0020 |
| Two-Tower C2 (30 ép.) - Two-Tower D2 (30 ép.) | -0.0003 | [-0.0034, +0.0027] | 5/10 | 1.0000 | 0.9219 |
| Two-Tower C2 (30 ép.) - Two-Tower A (30 ép.) | +0.0105 | [+0.0079, +0.0136] | 10/10 | 0.0020 | 0.0020 |
| G3 (exacto) - G3 (haz 50) | +0.0007 | [+0.0002, +0.0011] | 7/10 | 0.3438 | 0.0273 |
| NCF (Etapa 2, 15 ép.) - SVD (Etapa 1) | -0.0076 | [-0.0095, -0.0058] | 0/10 | 0.0020 | 0.0020 |
| NCF (épocas por val.) - NCF (Etapa 2, 15 ép.) | +0.0085 | [+0.0067, +0.0102] | 10/10 | 0.0020 | 0.0020 |
| Two-Tower A (30 ép.) - SVD (Etapa 1) | +0.0532 | [+0.0499, +0.0566] | 10/10 | 0.0020 | 0.0020 |
| G3 (haz 50) - SVD (Etapa 1) | +0.0456 | [+0.0436, +0.0475] | 10/10 | 0.0020 | 0.0020 |
| G1 (haz 50) - SVD (Etapa 1) | +0.0159 | [+0.0109, +0.0203] | 9/10 | 0.0215 | 0.0039 |

## Contrastes de cobertura @10 (exploratorios)

| Comparación | Δ medio (puntos de cobertura) | IC95% | n+ | p Wilcoxon |
|---|---|---|---|---|
| G1 (haz 50) - G2 (haz 50) | +0.59 | [+0.46, +0.74] | 10/10 | 0.0020 |
| G3 (haz 50) - Two-Tower D2 (30 ép.) | -5.96 | [-6.04, -5.88] | 0/10 | 0.0020 |
| G1 (haz 50) - Two-Tower D2 (30 ép.) | -5.94 | [-6.05, -5.82] | 0/10 | 0.0020 |

## Límites y pendientes

- Con 10 semillas el p mínimo del test de signo es 0.002; ese valor aparece siempre que las 10 diferencias tienen el mismo signo.
- Los IC95% son bootstrap sobre semillas; no hay bootstrap por usuarios porque los JSON no guardan métricas por usuario.
- Faltan popularidad media, Gini y diversidad intra-lista: requieren guardar las listas recomendadas (re-evaluar modelos).
- Popularidad y azar son referencias de la semilla 42 (popularidad idéntica en las 10 semillas; azar es esperado analítico/simulado).
- Asimetría de protocolo: Two-Tower con épocas elegidas por validación o fijas a 30; generativos con E* fijo elegido en validación (G1/G2: 18, G3: 16).