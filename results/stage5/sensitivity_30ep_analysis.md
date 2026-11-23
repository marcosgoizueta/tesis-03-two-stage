# Análisis de Sensibilidad de Protocolo — Etapa 5

**Fecha:** 2026-11-22 03:41
**Semillas:** [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  |  **Épocas fijas:** 30

Pregunta central: ¿supera Two-Stage al retrieval solo (Two-Tower A) tanto con parada temprana como con 30 épocas fijas?

## NDCG@10 — comparación de protocolos

| Modelo | ES media ± std | 30ep media ± std | Δ (30ep − ES) |
|--------|---------------|-----------------|--------------|
| A | 0.0796 ± 0.0024 | 0.0944 ± 0.0045 | +0.0148 |
| B | 0.0916 ± 0.0057 | 0.0988 ± 0.0035 | +0.0071 |
| C2 | 0.1036 ± 0.0026 | 0.1049 ± 0.0049 | +0.0013 |
| D2 | 0.1007 ± 0.0035 | 0.1053 ± 0.0026 | +0.0046 |
| Two-Stage | 0.1050 ± 0.0030 | 0.0957 ± 0.0024 | -0.0094 |

## Cobertura @10 y Fracción de Cola @10

| Modelo | ES cov@10 | 30ep cov@10 | ES tail@10 | 30ep tail@10 |
|--------|-----------|------------|-----------|-------------|
| A | 1.44% | 8.60% | 0.23% | 0.12% |
| B | 6.00% | 8.53% | 3.13% | 2.49% |
| C2 | 4.49% | 8.94% | 0.34% | 0.91% |
| D2 | 4.38% | 7.97% | 0.44% | 0.56% |
| Two-Stage | 2.28% | 8.00% | 0.19% | 0.03% |

## Diferencias pareadas: Two-Stage − Two-Tower A

**Early Stopping**: Two-Stage − A media = +0.0255  (n+ = 10/10)
**30 épocas fijas**: Two-Stage − A media = +0.0013  (n+ = 7/10)

## Notas de interpretación

- **Early stopping agresivo en A**: con parada temprana por val_loss (paciencia 3, mejora mínima 0.005), el modelo A converge en E*≈2 épocas en la mayoría de las semillas, lo que lo deja sistemáticamente por debajo de la popularidad. Esto convierte la diferencia Two-Stage − A con early stopping en un artefacto del protocolo (ver error #6 en CLAUDE.md).

- **Protocolo de 30 épocas fijas**: entrena directamente en train_full durante 30 épocas, que es el mismo protocolo usado en las Etapas 3 y 4. Con este protocolo, el modelo A converge a un nivel más representativo de su capacidad real.

- **Hipótesis a verificar con estos datos**: si Two-Stage supera a A con 30 épocas fijas, la ventaja del reranker es robusta. Si no, la mejora observada con early stopping era un artefacto del E* agresivo.

- **Comparación con otros modelos**: los valores de Two-Stage y los Two-Tower de estas tablas deben compararse con los generativos G1, G2, G3 del multi-seed principal (results/stage5/multi_seed_summary.json), teniendo en cuenta la asimetría de protocolo documentada en el README.

## Archivos fuente

- Per-seed 30 épocas: `results/stage5/per_seed_30ep/`
- Per-seed early stopping: `results/stage5/per_seed/`
- Summary: `results/stage5/sensitivity_30ep_summary.json`
- Stats generales: `results/stage5/stats_report.json`