# T4 — Error del haz en G1 (haz 50 vs ranking exacto)

Script: `scripts/run_stage5_g1_exact.py`. Semillas 42, 43, 44. 592 usuarios de test (pueden variar por semilla; ver `n_eval_users` en cada JSON). Dispositivo: RTX 4060 Laptop, bf16.
Fuentes: `G1_exact_seed_{42,43,44}.json`, `g1_exact_summary.json`.

Cada semilla reentrena G1 con la receta del orquestador (L=100, balanceado, lr 1e-3, dropout 0.1, E*=18). El NDCG@10 del haz reentrenado coincide con el JSON guardado en `per_seed/` (diferencia 0.00000 en las 3 semillas), así que el reentrenamiento reproduce la corrida original.

| Semilla | NDCG@10 haz 50 | NDCG@10 exacto | Δ (exacto − haz) | Cov@10 haz / exacto | Jaccard@10 | Jaccard@50 | Acuerdo top-1 |
|---|---|---|---|---|---|---|---|
| 42 | 0.0658 | 0.0621 | −0.0037 | 1.98% / 2.34% | 0.177 | 0.487 | 4.4% |
| 43 | 0.0687 | 0.0734 | +0.0047 | 2.24% / 2.53% | 0.248 | 0.522 | 7.8% |
| 44 | 0.0610 | 0.0620 | +0.0010 | 1.83% / 2.09% | 0.169 | 0.537 | 4.6% |

Latencia por usuario (hardware local, implementación concreta): haz ~80-93 ms, exacto ~660 ms. No es una propiedad arquitectónica.

## Comprobado
- Con 3 semillas, el ranking exacto no mejora el NDCG@10 de G1 de forma consistente: el signo de la diferencia cambia entre semillas y su tamaño (±0.005) es comparable al desvío entre semillas de G1 (~0.008).
- Los top-10 del haz y del ranking exacto casi no coinciden (Jaccard@10 entre 0.17 y 0.25), y el top-1 coincide en menos del 8% de los usuarios.
- La cobertura @10 es algo mayor con ranking exacto (+0.2 a +0.4 puntos), lo opuesto a que el haz «infle» la diversidad.
- El recall@10 prácticamente no cambia.

## Hipótesis (no comprobadas)
- Parte de la discrepancia puede deberse a que el haz y el scorer exacto no puntúan igual (el exacto suma la log-prob de 4 tokens sin EOS; no verifiqué cómo puntúa el haz final). Entonces el Jaccard bajo mezclaría error de búsqueda con diferencia de criterio de puntaje. Habría que puntuar los top-50 del haz con el scorer exacto para separarlos.
- Con haz 50 sobre ~8.200 ítems no vistos, el haz explora solo una fracción pequeña del árbol; como el modelo es poco discriminativo (NDCG cercano a popularidad), muchos ítems tienen puntajes casi iguales y el orden entre ellos es inestable, lo que también bajaría el Jaccard sin afectar el NDCG.

## Limitaciones
- n = 3 semillas; no se calcularon intervalos ni tests (no alcanzan para inferir).
- El exacto se calculó solo para G1 (G3 ya tenía exacto en las 10 semillas).
- No se evaluó G2.

## Conclusión práctica
Usar el haz 50 en vez del ranking exacto no cambia las conclusiones sobre G1 (sigue bajo popularidad 0.0821 en estas semillas, con ambos métodos), aunque los top-10 individuales sí difieren. El resultado no sugiere que el bajo desempeño de G1 sea un artefacto del haz.
