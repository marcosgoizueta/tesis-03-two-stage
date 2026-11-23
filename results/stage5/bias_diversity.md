# Sesgo y diversidad — Etapa 5 (top-10, 592 usuarios, media ± desvío entre semillas)

Fuente: `results/stage5/recs/*.npz`, procesado por `scripts/analyze_stage5_recs.py`.
Popularidad media = interacciones en train_full del ítem recomendado. Gini: desigualdad de la exposición sobre los 9724 ítems (0 = pareja, 1 = todo en un ítem). Diversidad intra-lista = 1 − coseno medio entre pares (MiniLM).

| Modelo | n sem. | NDCG@10 | Cov@10 | Pop. media | Gini | Div. intra-lista |
|---|---|---|---|---|---|---|
| FP_Two-Tower_A | 10 | 0.0944 ± 0.0045 | 8.60% ± 0.13 | 119.6 | 0.971 ± 0.001 | 0.501 ± 0.001 |
| FP_Two-Tower_B | 10 | 0.0988 ± 0.0035 | 8.53% ± 0.37 | 117.0 | 0.974 ± 0.002 | 0.480 ± 0.002 |
| FP_Two-Tower_C2 | 10 | 0.1049 ± 0.0049 | 8.94% ± 0.14 | 112.8 | 0.970 ± 0.001 | 0.489 ± 0.003 |
| FP_Two-Tower_D2 | 10 | 0.1053 ± 0.0026 | 7.97% ± 0.10 | 120.1 | 0.974 ± 0.001 | 0.494 ± 0.003 |
| FP_Two-Stage | 10 | 0.0957 ± 0.0024 | 8.00% ± 0.16 | 110.8 | 0.970 ± 0.001 | 0.503 ± 0.001 |
| ES_Two-Tower_A | 10 | 0.0796 ± 0.0024 | 1.44% ± 0.52 | 197.0 | 0.998 ± 0.001 | 0.488 ± 0.007 |
| ES_Two-Tower_B | 10 | 0.0916 ± 0.0057 | 6.00% ± 0.41 | 128.3 | 0.984 ± 0.001 | 0.470 ± 0.006 |
| ES_Two-Tower_C2 | 10 | 0.1036 ± 0.0026 | 4.49% ± 0.73 | 148.3 | 0.988 ± 0.002 | 0.480 ± 0.004 |
| ES_Two-Tower_D2 | 10 | 0.1007 ± 0.0035 | 4.38% ± 0.55 | 153.3 | 0.989 ± 0.002 | 0.484 ± 0.003 |
| ES_Two-Stage | 10 | 0.1050 ± 0.0030 | 2.28% ± 0.45 | 159.6 | 0.993 ± 0.001 | 0.503 ± 0.004 |
| G1 | 10 | 0.0571 ± 0.0078 | 2.03% ± 0.20 | 103.3 | 0.994 ± 0.001 | 0.480 ± 0.016 |
| G2 | 10 | 0.0466 ± 0.0085 | 1.44% ± 0.14 | 110.0 | 0.996 ± 0.000 | 0.518 ± 0.006 |
| G3_beam | 10 | 0.0868 ± 0.0026 | 2.00% ± 0.13 | 170.4 | 0.994 ± 0.000 | 0.489 ± 0.005 |
| G3_exact | 10 | 0.0874 ± 0.0025 | 1.99% ± 0.13 | 171.7 | 0.994 ± 0.000 | 0.489 ± 0.005 |
| G1b | 5 | 0.0535 ± 0.0063 | 1.48% ± 0.07 | 123.1 | 0.997 ± 0.000 | 0.484 ± 0.014 |
| EXPL_SVD | 10 | 0.0412 ± 0.0026 | 3.31% ± 0.13 | 88.6 | 0.991 ± 0.000 | 0.538 ± 0.002 |
| EXPL_NCF | 10 | 0.0336 ± 0.0027 | 4.62% ± 0.59 | 84.5 | 0.986 ± 0.002 | 0.539 ± 0.004 |
| EXPL_NCF_val | 10 | 0.0421 ± 0.0027 | 2.04% ± 0.32 | 101.1 | 0.994 ± 0.001 | 0.528 ± 0.007 |
| Popularidad | 10 | 0.0821 ± 0.0000 | 1.01% ± 0.00 | 216.1 | 0.998 ± 0.000 | 0.475 ± 0.000 |

## Contrastes primarios con bootstrap por usuarios (Holm). Mejor con 30 épocas: FP_Two-Tower_D2 (post hoc)

| Comparación | Δ NDCG@10 | IC95% (usuarios) | semillas a favor | p signo (semillas) | Umbral Holm | ¿Rechaza? |
|---|---|---|---|---|---|---|
| G1 - G3_beam | -0.0296 | [-0.0370, -0.0227] | 0/10 | 0.0020 | 0.0167 | sí |
| G1 - FP_Two-Tower_D2 | -0.0481 | [-0.0572, -0.0393] | 0/10 | 0.0020 | 0.0250 | sí |
| G1 - G2 | +0.0106 | [+0.0070, +0.0144] | 8/10 | 0.1094 | 0.0500 | no |

## Contrastes secundarios (exploratorios)

| Comparación | Δ NDCG@10 | IC95% (usuarios) | semillas a favor |
|---|---|---|---|
| G3_beam - FP_Two-Tower_D2 | -0.0185 | [-0.0284, -0.0087] | 0/10 |
| G1 - Popularidad | -0.0249 | [-0.0353, -0.0150] | 0/10 |
| G2 - Popularidad | -0.0355 | [-0.0454, -0.0262] | 0/10 |
| G3_beam - Popularidad | +0.0047 | [-0.0037, +0.0133] | 10/10 |
| FP_Two-Stage - FP_Two-Tower_A | +0.0013 | [-0.0050, +0.0074] | 7/10 |
| EXPL_SVD - Popularidad | -0.0409 | [-0.0509, -0.0316] | 0/10 |
| EXPL_NCF - Popularidad | -0.0485 | [-0.0585, -0.0387] | 0/10 |
| EXPL_NCF_val - Popularidad | -0.0400 | [-0.0496, -0.0307] | 0/10 |
| EXPL_NCF - EXPL_SVD | -0.0076 | [-0.0109, -0.0045] | 0/10 |
| FP_Two-Tower_A - EXPL_SVD | +0.0532 | [+0.0450, +0.0614] | 10/10 |
| FP_Two-Tower_A - EXPL_NCF | +0.0609 | [+0.0524, +0.0693] | 10/10 |
| G3_beam - EXPL_SVD | +0.0456 | [+0.0365, +0.0548] | 10/10 |
| G1 - EXPL_SVD | +0.0159 | [+0.0096, +0.0224] | 9/10 |

Nota: el IC por usuarios mide la incertidumbre por la muestra de 592 usuarios; no incluye la variabilidad entre semillas (promedia las semillas dentro de cada usuario). Ambas se reportan por separado.