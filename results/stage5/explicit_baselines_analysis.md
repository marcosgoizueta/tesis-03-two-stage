# Etapas 1 y 2 (SVD y NCF) bajo el protocolo temporal full-ranking

Script: `scripts/run_stage5_explicit_baselines.py`; módulo `src/recsys/stage5/explicit_baselines.py`; tests `tests/stage5/test_stage5_explicit_baselines.py`.
Fuentes: `results/stage5/per_seed_explicit/*.json` (30 archivos), `results/stage5/recs/EXPL_*.npz`, `results/stage5/explicit_gate.json`, `results/stage5/final_comparison.md`, `results/stage5/bias_diversity.md`.
Split temporal de la Etapa 5, 592 usuarios de test, 10 semillas (42-51), K=10 salvo indicación.

## Qué se hizo
Los notebooks de Colab (`notebooks/01_svd_baseline.ipynb`, `02_ncf.ipynb`) miden precision@10 y recall@10 pointwise: solo sobre los ítems que cada usuario ya calificó en test (unos 33 por usuario), con 61% de ratings ≥ 3.5. No son comparables con el NDCG@10 de full-ranking del resto de la tesis. Se reimplementaron ambos modelos con la configuración de los notebooks y se evaluaron rankeando todo el catálogo no visto.

- **SVD:** reimplementación en numpy del algoritmo de `surprise.SVD` (100 factores, 20 épocas, lr 0.005, reg 0.02, SGD sin mezclar). No se instaló Surprise para no arriesgar el entorno (NumPy 1.23.5 fijado).
- **NCF:** la arquitectura exacta del notebook (NeuMF, MSE, Adam lr 1e-3, weight decay 1e-5, 15 épocas, lote 256).
- **NCF_val (sensibilidad):** épocas elegidas por RMSE de validación sobre train_core (E* entre 7 y 9 en las 10 semillas) y reentrenado en train_full.
- Entrenan con todas las valoraciones de train_full (regresión de rating). Se excluyen del ranking los ítems ya vistos y los 1.487 ítems sin interacciones en train (sin señal; Two-Tower A tampoco los recomienda, cold recall 0).

## Compuerta de validación (split aleatorio original, semilla 42)
| Modelo | RMSE Colab | RMSE reimplementado | MAE Colab | MAE reimplementado |
|---|---|---|---|---|
| SVD | 0.8807 | 0.8826 | 0.6766 | 0.6749 |
| NCF | 0.9645 | 0.9629 | 0.7388 | 0.7393 |

Diferencias de ~0.002: las reimplementaciones reproducen los notebooks dentro del ruido esperable (el NCF original corrió en la GPU de Colab).

## Resultados (media ± desvío entre 10 semillas)
| Modelo | NDCG@10 | Recall@10 | Cov@10 | Tail@10 | Pop. media | Gini | RMSE puntual en test |
|---|---|---|---|---|---|---|---|
| Popularidad | 0.0821 | 0.0477 | 1.01% | 0.00% | 216 | 0.998 | - |
| SVD (Etapa 1) | 0.0412 ± 0.0026 | 0.0213 | 3.31% | 0.18% | 88.6 | 0.991 | 0.890 |
| NCF, 15 épocas (Etapa 2) | 0.0336 ± 0.0027 | 0.0188 | 4.62% | 1.72% | 84.5 | 0.986 | 0.963 |
| NCF, épocas por validación | 0.0421 ± 0.0027 | 0.0236 | 2.04% | 0.46% | 101.1 | 0.994 | 0.914 |
| Two-Tower A, 30 épocas (referencia) | 0.0944 ± 0.0045 | 0.0679 | 8.60% | 0.12% | 119.6 | 0.971 | - |
| G1 (referencia) | 0.0571 ± 0.0078 | 0.0456 | 2.03% | 0.43% | 103.3 | 0.994 | - |

Cold recall@50 es 0.0000 en todas las semillas.

## Diferencias pareadas por semilla (NDCG@10; IC95% por usuarios entre corchetes, bootstrap)
- SVD − Popularidad: −0.0409 [−0.051, −0.032]; NCF − Popularidad: −0.0485 [−0.059, −0.039]; NCF_val − Popularidad: −0.0400 [−0.050, −0.031]. Las tres quedan por debajo de popularidad en las 10 semillas.
- NCF − SVD: −0.0076 [−0.011, −0.005] (SVD mejor en 10 de 10).
- NCF con épocas por validación − NCF fijo: +0.0085 (10 de 10).
- Two-Tower A (30 épocas) − SVD: +0.0532 [+0.045, +0.061] (10 de 10). Two-Tower A es esencialmente MF con feedback implícito, así que esta diferencia aísla sobre todo el efecto del objetivo de entrenamiento (rating contra interacción), no de la arquitectura.
- G3 − SVD: +0.0456 (10 de 10). G1 − SVD: +0.0159 [+0.010, +0.022] (9 de 10).

## Comprobado
- Predicen bien el rating pero rankean mal el catálogo: el RMSE puntual bajo el split temporal (SVD 0.890, NCF 0.963) es casi igual al de Colab, y sin embargo el NDCG@10 queda a la mitad del de popularidad.
- Entrenar para predecir rating no es lo mismo que entrenar para ordenar: con el mismo tipo de modelo (factorización), pasar a feedback implícito (Two-Tower A) multiplica el NDCG@10 por más de dos.
- El NCF de 15 épocas sobreajusta (MSE de train 0.44 contra un RMSE de test de 0.96) y, con épocas elegidas en validación, mejora en ranking (+0.0085) pero no supera a popularidad.

## Hipótesis no comprobadas / hipótesis descartada
- Hipótesis previa **no confirmada**: que estos modelos recomienden ítems oscuros con pocos votos. La fracción de cola @10 es baja (SVD 0.18%, NCF 1.72%) y la popularidad media recomendada (85-101) es menor que la de los Two-Tower pero no extrema. Qué ítems prefieren exactamente no se investigó.
- No se probó si una variante implícita de NCF (BCE con negativos) cerraría la brecha; no se corrió y no se afirma.

## Limitaciones
- SVD reimplementado, no la librería Surprise (misma receta, validada solo por RMSE/MAE en la compuerta).
- Sin ajuste de hiperparámetros de SVD ni de NCF más allá de la sensibilidad por épocas: son baselines simples a propósito, con la configuración original de los notebooks.
- El IC por usuarios no incluye variabilidad entre semillas. Dataset único y chico.
