# Etapa 3 — Arquitectura de Dos Etapas (Candidate Generation + Ranking)

Este módulo implementa y evalúa la arquitectura de dos etapas para sistemas de recomendación (Candidate Generation con Two-Tower + Ranking con MLP) sobre el dataset MovieLens `ml-latest-small`.

Forma parte de la tesis de Ingeniería en Sistemas de Información que compara sistemas de recomendación discriminativos y generativos con análisis de sesgo.

## Entornos de Ejecución y Hardware

### 1. Entorno GPU (Principal)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM, Compute Capability 8.9).
- **Driver NVIDIA**: 555.97 (soporte CUDA hasta 12.5).
- **PyTorch**: `2.6.0+cu124` (instalado desde el índice oficial `https://download.pytorch.org/whl/cu124` con soporte CUDA 12.4).
- **Sistema Operativo**: Windows 11 (build 10.0.26200).
- **Python**: 3.11.9.

### 2. Entorno CPU (Validación y benchmarking)
- **PyTorch**: `2.9.1+cpu`.
- Las corridas de validación histórica de CPU reportadas en la documentación y en `results/cpu/` fueron ejecutadas bajo esta versión antes de la migración a CUDA.

> **Nota metodológica sobre Google Colab**:
> El notebook original `03_two_stage.ipynb` ejecutado en Google Colab no registró el modelo de GPU (ej. T4, P100, V100) ni la versión de CUDA empleada. Por ende, no se asume ninguna configuración específica de hardware para los números de referencia de Colab.

---

## Estructura del Proyecto

```text
03-two-stage/
├── data/                       # Dataset MovieLens (descarga automática si no existe)
├── notebooks/
│   └── 03_two_stage.ipynb      # Notebook original intacto de referencia
├── results/
│   ├── comparativa_modelos.csv # Tabla histórica de corridas CPU
│   ├── figures/                # Curvas de pérdida en CPU
│   ├── stage3_runs/            # JSONs de corridas en CPU
│   └── gpu/                    # Resultados independientes en GPU (sin sobrescritura)
│       ├── comparativa_modelos.csv
│       ├── figures/
│       ├── stage3_runs/
│       ├── stage3_per_seed.csv
│       ├── stage3_summary.csv
│       └── stage3_paired_differences.csv
├── scripts/
│   ├── run_stage3.py           # Pipeline completo ejecutable por CLI
│   └── run_multi_seed.py       # Análisis multi-semilla y contrastes pareados
├── src/
│   └── recsys/                 # Módulos limpios (config, data, retrieval, ranker, eval, tracking)
├── tests/
│   ├── conftest.py             # Configuración de precarga para Windows
│   └── test_pure_functions.py  # Tests unitarios con pytest
├── requirements.txt            # Dependencias fijadas para GPU (CUDA 12.4)
└── README.md                   # Documentación técnica
```

---

## Instalación

```bash
# Instalación con soporte CUDA 12.4 para GPU NVIDIA:
pip install -r requirements.txt
```

---

## Ejecución

### 1. Pruebas unitarias
```bash
pytest tests/ -v
```

### 2. Modo rápido de prueba (`--quick`)
```bash
python scripts/run_stage3.py --quick
```

### 3. Corrida completa (Semilla 42)
```bash
# En GPU (por defecto en auto si hay CUDA disponible):
python scripts/run_stage3.py --device cuda --seed 42

# En CPU explícitamente:
python scripts/run_stage3.py --device cpu --seed 42
```

### 4. Análisis multi-semilla (Estabilidad y Contrastes Pareados)
```bash
# 5 semillas en GPU:
python scripts/run_multi_seed.py --seeds 42 43 44 45 46 --device cuda

# 10 semillas en GPU:
python scripts/run_multi_seed.py --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --suffix _10seeds
```


---

## Etapa 4 — Retrieval Semántico y Análisis de Representaciones de Contenido

> **Nota sobre scripts de ejecución**:
> El script principal `scripts/run_stage4.py` unifica y reproduce de forma integral todas las corridas de la Etapa 4 para las 9 variantes arquitectónicas (A, B, C, C-cero, C2, D, D2, E, R) en ambos escenarios (`standard` y `cold_start`). Reproduce con exactitud binaria los resultados de referencia generados originalmente de forma aislada por `scripts/run_stage4_d2.py` para la variante D2.

### Archivos de Referencia y Reproducibilidad

Para garantizar la reproducibilidad entre entornos y permitir la auditoría de resultados en Google Colab sin necesidad de reejecutar las 10 semillas completas, el proyecto versiona los siguientes artefactos:

1. **Metadatos y Embeddings de Referencia (`reference/stage4/`)**:
   - `reference/stage4/metadata.json`: Metadatos del modelo de lenguaje (`all-MiniLM-L6-v2`), commit exacto de Hugging Face (`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), dimensión (384), plantilla de texto y versiones de librerías utilizadas en la máquina de referencia.
   - `reference/stage4/movie_embeddings_minilm.npy`: Matriz de embeddings precalculada en formato NumPy (~14.9 MB) para verificar similitud coseno contra matrices generadas en otros entornos (ej. Colab).

2. **Tablas de Resultados Consolidados (`results/gpu/stage4/`)**:
   - `stage4_consolidated_summary_10seeds.csv`: Métricas consolidadas (media y desvío estándar) de las 9 variantes en los escenarios *standard* y *cold_start* a lo largo de 10 semillas (leída por `notebooks/colab_stage4.ipynb`).
   - `stage4_paired_contrasts_10seeds.csv` y `stage4_paired_contrasts_d2.csv`: Contrastes pareados primarios con media, desvío, mínimo, máximo y tasa de éxito (positivas / 10).
   - `stage4_bias_summary_10seeds.csv`: Métricas de cobertura de catálogo, popularidad media y fracción recomendada en la cola (cortes 0 y 1-5) para Retrieval Solo y Two-Stage N=50.
   - `stage4_candidate_recall_cuts_standard_10seeds.csv`: Recall de candidatos @200 en cortes de popularidad 0 y 1-5 contrastado con el valor esperado analítico al azar.
   - `stage4_results_10seeds.csv` y `stage4_d2_results_10seeds.csv`: Resultados detallados por semilla y variante.
   - `diagnostic_variant_e.json`: Diagnóstico de redundancia de géneros (tamaño medio de bloque y fracción de empates).

### Limitaciones Metodológicas de la Etapa 4

1. **Split en frío idéntico entre semillas**:
   - En el escenario de evaluación en frío, el 10% de ítems seleccionados para conformar el conjunto frío (972 ítems que se eliminan completamente de train y cuyas interacciones se evalúan en test) se particiona con una semilla fija (`split_seed=42`) a lo largo de las 10 corridas.
   - Esto permite aislar la varianza atribuible a la inicialización y entrenamiento de los modelos frente a un conjunto de evaluación idéntico, pero no captura la varianza que surgiría al muestrear diferentes particiones de ítems en frío del catálogo.

2. **Sesgo temporal por año de estreno en los embeddings de texto**:
   - Al codificar el texto con `all-MiniLM-L6-v2` usando la plantilla estándar `"<title (YYYY)>. Genres: <genres>"`, la presencia explícita del año induce un fuerte agrupamiento temporal en los vecinos más cercanos:
     - En *The Matrix (1999)*, 4 de sus 5 vecinos más cercanos en el catálogo comparten el año 1999 (*eXistenZ*, *Arlington Road*, *The Insider*).
     - En *Pulp Fiction (1994)*, los 5 vecinos más cercanos pertenecen exclusivamente a 1994 (*The Chase*, *Fresh*, *Forrest Gump*, *Amateur*, *Fun*).
   - En el diagnóstico comparativo sin año (`"<title>. Genres: <genres>"` en `data/cache/movie_embeddings_minilm_noyear.npy`), la similitud semántica se desacopla del año de estreno, agrupando secuelas y películas del mismo género a través de distintas décadas (ej. *The Matrix* pasa a tener como vecinos principales a *The Matrix Revolutions (2003)* y *The Matrix Reloaded (2003)* con similitudes $> 0.85$).

3. **Ranker compartido entre variantes de retrieval**:
   - Para una misma semilla y escenario, el modelo de Ranking (MLP de dos capas) se entrena una única vez sobre las interacciones de train y se reutiliza de forma determinista para rerankear los candidatos generados por todas las variantes de retrieval (A, B, C, C-cero, C2, D, D2, E, R).
   - Esta decisión de diseño metodológico elimina la varianza aleatoria del ranker en los contrastes pareados entre arquitecturas de retrieval, pero asume que el ranker es agnóstico a la distribución de candidatos que le suministra cada generador.

4. **Entorno de ejecución único (hardware y software)**:
   - Todas las corridas de la Etapa 4 fueron ejecutadas en un único entorno de cómputo local: Windows 11 (build 10.0.26200), Python 3.11.9, PyTorch `2.6.0+cu124` y GPU NVIDIA GeForce RTX 4060 Laptop (CUDA 12.4).
   - Aunque se fijan rigurosamente las semillas en generadores independientes (`torch.Generator`), los kernels no deterministas de ciertas operaciones en GPU o diferencias de compilación CUDA/cuDNN en otras plataformas (e.g. Linux o Google Colab) podrían introducir pequeñas fluctuaciones de punto flotante.


---

## Etapa 5 — Recuperación Generativa y Análisis Comparativo

Este módulo implementa la arquitectura de recuperación generativa (tipo TIGER a escala reducida: RQ-VAE para cuantización residual de embeddings y Semantic IDs + decodificación restringida sobre modelo encoder-decoder tipo T5 entrenado desde cero) y su análisis comparativo frente a los baselines discriminativos de las Etapas 3 y 4.

### Protocolo de Partición Temporal y Desempates
- **Split temporal por usuario**: 80% entrenamiento (primer 80% de interacciones ordenadas cronológicamente), 20% test (último 20%).
- **Validación**: Último 10% del bloque de entrenamiento de cada usuario (`train_core` 90%, `val` 10%).
- **Criterio de relevancia**: Calificación $r \ge 3.5$.
- **Desempate determinista**: Permutación pseudoaleatoria con semilla fija (`seed=42`) dentro de los grupos con timestamp idéntico, evitando el sesgo de ordenación original por `movieId` presente en `ratings.csv`.
- **Ajuste de bordes (Opción C)**: El corte se desplaza al borde del evento temporal más cercano para preservar la integridad de sesiones y evitar fugas temporales (0 violaciones de monotonía temporal en todos los usuarios).

### Desviaciones Metodológicas Respecto al Plan a Priori

1. **Tamaño de lote y precisión numérica**:
   - **Plan a priori**: Tamaño de lote 64 en precisión simple (`fp32`).
   - **Implementación adoptada**: Tamaño de lote 128 con precisión mixta `bfloat16` (`bf16`).
   - **Justificación**: Aumenta el rendimiento de entrenamiento de ~1.700 a ~2.400 secuencias/segundo (~14,6 s por época con 35.087 ventanas) con un consumo de VRAM de ~1.020 MB en la GPU NVIDIA RTX 4060 Laptop.
   - **Uniformidad**: Esta configuración se aplica de forma idéntica y estricta a todos los modelos generativos (`G1`, `G2`, `G3` y la ablación `G1b`) sin optimizaciones diferenciales por arquitectura.
   - **Nota empírica**: La aceleración observada proviene de la reducción de transferencias en memoria y el procesamiento por lotes; no se atribuye causalmente a la activación de Tensor Cores al no haberse auditado a nivel de instrucciones del silicio mediante perfiladores de hardware.

2. **Haz de búsqueda en validación**:
   - Fijado estrictamente en `beam_size = 50` para todas las arquitecturas evaluadas durante la optimización y parada temprana en validación.

3. **Regla de reporte de métricas según haz de búsqueda**:
   - En evaluación final sobre test con haces 20, 50 y 100, las métricas de ranking se reportan exclusivamente para cortes $K \le \text{haz}$ (e.g. no se computa Recall@50 cuando el haz es 20).

4. **Asimetría en la selección de épocas entre modelos discriminativos y generativos**:
   - Los modelos Two-Tower (A, B, C2, D2) y Two-Stage usan **parada temprana por pérdida de validación** (paciencia 3, mejora mínima 0.005, máximo 30 épocas) y el número de épocas óptimo E* se determina **por semilla** durante la fase de evaluación multi-semilla. Para el modelo A (solo ID), esto produce consistentemente E* ≈ 2, lo que lo deja por debajo del nivel de popularidad en 9 de 10 semillas (ver `results/stage5/stats_report.json`).
   - Los modelos generativos G1, G2, G3 y G1b usan un **E* fijo** para todas las semillas (G1/G2: 18 épocas, G3: 16 épocas), seleccionado una vez en validación a partir de la grilla de hiperparámetros de la Fase 2b y aplicado sin ajuste posterior por semilla.
   - Esta asimetría es una diferencia de protocolo relevante para interpretar comparaciones directas entre los dos grupos de modelos: los Two-Tower con parada temprana agresiva tienen E* determinado en parte por la dinámica estocástica de cada corrida, mientras que los generativos usan E* fijo independiente de la semilla.

5. **Qué varía con la semilla en la corrida multi-semilla (verificado en `scripts/run_stage5_multi_seed.py`)**:
   - `create_temporal_split(seed=seed)` usa la semilla solo para la permutación de desempate de timestamps idénticos. Verificado (`results/stage5/seed_dependence_audit.json`): tamaño de train_full (80.621), usuarios de test (592) y NDCG@10 de popularidad (0.082052) son idénticos en las 10 semillas. Hipótesis no verificada: el ajuste de bordes (Opción C) hace que la pertenencia a train/test no cambie y la semilla solo reordene eventos simultáneos.
   - El RQ-VAE de G1 y G1b se reinicializa por semilla (`init_from_kmeans(seed=seed)`) y G2 regenera sus IDs aleatorios por semilla. Verificado: los hashes SHA256 (16 hex) del mapa difieren en las 10 semillas; el de la semilla 42 (`ed2b07a119f750a4`) coincide con el de la Fase 2c. El mapa de Semantic IDs **no es fijo** entre semillas, de modo que la varianza de G1 incluye la variabilidad del tokenizador además de la del T5, y G1−G2 se compara con la misma lógica por semilla.
   - Los hashes **no están** en `per_seed/*.json` (el orquestador los calcula pero no los guarda); quedan registrados en `results/stage5/seed_dependence_audit.json`, recalculados con el mismo código.

### Modelos Evaluados
1. **G1**: Recuperación generativa con Semantic IDs jerárquicos ($[c_1, c_2, c_3, d]$) generados por RQ-VAE sobre embeddings de texto `all-MiniLM-L6-v2`.
2. **G2**: Control con identificadores aleatorios estructurados con la misma tupla ($[r_1, r_2, r_3, d]$).
3. **G3**: Transformer secuencial con tokens atómicos ($[\text{item\_id}]$).
4. **G1b**: Ablación exploratoria con Semantic IDs derivados de embeddings sin año en el título.
5. **Baselines de contexto**: Popularidad, Two-Tower A (solo ID), Two-Tower B (solo texto), Two-Tower C2 (híbrido ID+texto), Two-Tower D2 (control ID+géneros) y Two-Stage (Two-Tower A + Ranker MLP N=50), reentrenados y evaluados bajo el mismo protocolo temporal.

### Hipótesis Interpretativas (no probadas experimentalmente)

Las siguientes afirmaciones se ofrecen como hipótesis de trabajo, no como conclusiones establecidas. Para elevarlas al estatus de conclusiones serían necesarios experimentos adicionales (ablaciones controladas, datasets más grandes, o análisis de representaciones internas).

**H1 — La jerarquía del RQ-VAE guía la decodificación mejor que IDs aleatorios.**
La comparación G1 vs G2 intenta aislar el efecto de la estructura jerárquica de los Semantic IDs: si G1 supera consistentemente a G2, sugeriría que la organización semántica (ítems similares comparten prefijos de código) facilita al modelo T5 aprender asociaciones útiles entre el historial del usuario y los códigos objetivo. Los resultados de 10 semillas muestran una diferencia media de +0.0106 NDCG@10 favorable a G1 (8 de 10 semillas positivas), pero con test de signo p=0.1094 y Wilcoxon p=0.0273, que no supera el umbral de Holm para las comparaciones primarias (umbral 0.05). La diferencia existe pero no alcanza el nivel de evidencia estadística exigido dado el número de semillas disponible. Una explicación alternativa igualmente plausible es que la ligera ventaja de G1 refleje ruido de entrenamiento o diferencias en la dificultad de los tokens (los IDs aleatorios de G2 no tienen estructura de prefijo, lo que podría hacer más difícil el aprendizaje sin que eso implique nada sobre la utilidad de la jerarquía semántica per se).

**H2 — El token atómico (G3) es una representación más adecuada que los Semantic IDs (G1) para este dataset.**
G3 supera a G1 en las 10 semillas con una diferencia media de +0.0296 NDCG@10 (p=0.002, test de signo y Wilcoxon). Una interpretación posible es que con solo 9.724 ítems y 610 usuarios, el espacio de Semantic IDs (3 niveles × 256 códigos + desambiguación) introduce una dificultad de optimización innecesaria: el modelo debe aprender a generar secuencias de 4 tokens coherentes en lugar de un solo token, con más oportunidades de error de búsqueda. Sin embargo, la diferencia de parámetros (~5.7M para G3 vs ~4.57M para G1) y la posible diferencia en dificultad de la tarea hacen que la comparación no sea perfectamente controlada. La hipótesis de escala —que la ventaja de los Semantic IDs emerja con catálogos de decenas de miles de ítems donde los tokens atómicos se vuelven inmanejables— queda sin verificar en este experimento.

### Resultados de la Etapa 5

Todas las cifras salen de JSON en `results/stage5/`; ninguna fue escrita a mano. Split temporal, 592 usuarios de test, 10 semillas (G1b: 5), K=10 salvo indicación. Intervalos: IC95% bootstrap sobre semillas, y por usuarios cuando se indica.

| Modelo | NDCG@10 (media ± desvío) | Cov@10 | Pop. media recomendada | Gini de exposición |
|---|---|---|---|---|
| Popularidad | 0.0821 | 1,01% | 216 | 0,998 |
| Two-Tower A / B / C2 / D2, 30 épocas fijas | 0.0944 / 0.0988 / 0.1049 / 0.1053 | 8,0 a 8,9% | 113 a 120 | 0,970 a 0,974 |
| Two-Stage (30 épocas) | 0.0957 ± 0.0024 | 8,00% | 111 | 0,970 |
| Two-Tower A / B / C2 / D2, parada por validación | 0.0796 / 0.0916 / 0.1036 / 0.1007 | 1,4 a 6,0% | 128 a 197 | 0,984 a 0,998 |
| Two-Stage (parada por validación) | 0.1050 ± 0.0030 | 2,28% | 160 | 0,993 |
| G1 | 0.0571 ± 0.0078 | 2,03% | 103 | 0,994 |
| G2 | 0.0466 ± 0.0085 | 1,44% | 110 | 0,996 |
| G3 (haz 50 / exacto) | 0.0868 / 0.0874 | 2,00% | 170 | 0,994 |
| G1b (5 semillas) | 0.0535 ± 0.0063 | 1,48% | 123 | 0,997 |

Tablas completas: `results/stage5/final_comparison.md` (con IC95%, recall y frío) y `results/stage5/bias_diversity.md`.

**Comprobado.**
- G1 − G3 = −0.0296 (IC95% por usuarios [−0.037, −0.023]) y G1 − Two-Tower D2 con 30 épocas = −0.0481 ([−0.057, −0.039]); las 10 semillas van en contra de G1 y la corrección de Holm los rechaza.
- G1 − G2 = +0.0106 (8 de 10 semillas; IC por usuarios [+0.007, +0.014]), pero con test de signo p=0,109 y Wilcoxon p=0,027 la corrección de Holm no lo rechaza: es una tendencia, no un efecto establecido.
- G1 y G2 quedan por debajo de popularidad (IC por usuarios excluye 0). **G3 es indistinguible de popularidad**: +0.0047, IC95% por usuarios [−0.004, +0.013]. El test por semillas contra la constante de popularidad (p=0,002 en `stats_report.json`) ignora el error de muestreo de los usuarios y no debe leerse como evidencia de que G3 supere a popularidad.
- Two-Stage − A es +0.0255 con parada por validación (10 de 10 semillas) y +0.0013 con 30 épocas fijas (7 de 10; IC por usuarios [−0.005, +0.007]). La ventaja de Two-Stage depende del protocolo con que se entrena A.
- Todos los modelos concentran la exposición en pocos ítems (Gini ≥ 0,97); ningún modelo alcanza el nivel del azar en recall@50 en frío (0,53%).
- Error del haz en G1 (semillas 42 a 44, `results/stage5/g1_exact/`): el NDCG@10 con ranking exacto no mejora de forma consistente (−0.0037, +0.0047, +0.0010) y los top-10 casi no coinciden (Jaccard@10 de 0,17 a 0,25). Parte de la diferencia puede deberse al criterio de puntaje (el exacto suma 4 tokens y no incluye EOS), lo cual no se separó.

**Hipótesis no comprobadas.** Efecto de escala (610 usuarios es poco para un modelo tipo TIGER); distinto presupuesto de ajuste (grilla de hiperparámetros para los generativos, configuración heredada de las Etapas 3 y 4 para los Two-Tower); que G1 y G2 concentren sus recomendaciones en pocos ítems de popularidad media.

**Limitaciones.** El IC por usuarios promedia las semillas dentro de cada usuario, por lo que no incluye la variabilidad entre semillas (ambos se reportan por separado). D2 con 30 épocas se eligió como «mejor discriminativo» por su NDCG@10 medio sobre test (selección post hoc; la brecha con G1 es grande). La diversidad intra-lista medida con embeddings de MiniLM (0,47 a 0,52 en todos los modelos) no discrimina y es circular para G1, cuyo tokenizador se construye sobre esos mismos embeddings. La latencia (haz ~80 a 93 ms por usuario, exacto ~660 ms en G1) se midió en una RTX 4060 Laptop con esta implementación y no es una propiedad arquitectónica. Dataset único y chico.

### Reproducción de la Etapa 5

```bash
python scripts/run_stage5_multi_seed.py                 # per_seed/ (parada por validación y generativos)
python scripts/run_stage5_sensitivity_30ep.py           # per_seed_30ep/ (30 épocas fijas)
python scripts/run_stage5_g1_exact.py                   # g1_exact/ (error del haz en G1, semillas 42 a 44)
python scripts/run_stage5_save_recs.py                  # recs/ (listas top-50; reproduce los NDCG publicados)
python scripts/verify_report.py                         # recalcula el summary y compara con la Fase 2c
python scripts/compute_stats_stage5.py                  # stats_report.json
python scripts/audit_stage5_seed_dependence.py          # hashes por semilla y popularidad por semilla
python scripts/analyze_stage5_recs.py                   # bias_diversity.md (Gini, popularidad media, bootstrap por usuarios)
python scripts/build_final_table_stage5.py              # final_comparison.md
```

### Etapas 1 y 2 bajo el protocolo temporal full-ranking

Las cifras de los notebooks de Colab (`notebooks/01_svd_baseline.ipynb`: SVD RMSE 0.8807, MAE 0.6766, precision@10 0.7446, recall@10 0.5086; `notebooks/02_ncf.ipynb`: NCF RMSE 0.9645, MAE 0.7388, precision@10 0.7420, recall@10 0.5223) son pointwise: se calculan solo sobre los ítems que cada usuario ya calificó en test (unos 33 por usuario, 61% con rating ≥ 3.5) y **no son comparables** con el NDCG@10 de full-ranking del resto de la tesis. Para poder comparar las Etapas 1 y 2 con las demás se reimplementaron ambos modelos en `src/recsys/stage5/explicit_baselines.py` con la configuración de los notebooks y se evaluaron con el protocolo temporal de la Etapa 5 (`scripts/run_stage5_explicit_baselines.py`; análisis en `results/stage5/explicit_baselines_analysis.md`).

- La reimplementación reproduce los notebooks sobre el split aleatorio original (SVD RMSE 0.8826 contra 0.8807; NCF 0.9629 contra 0.9645; `results/stage5/explicit_gate.json`). El SVD es una reimplementación en numpy del algoritmo de `surprise.SVD`, no la librería.
- Resultados (NDCG@10, 10 semillas): SVD 0.0412 ± 0.0026, NCF con 15 épocas 0.0336 ± 0.0027, NCF con épocas elegidas por validación 0.0421 ± 0.0027, contra popularidad 0.0821 y Two-Tower A con 30 épocas 0.0944. Los tres quedan por debajo de popularidad en las 10 semillas. Cold recall@50 es 0 en todos.
- Predicen el rating tan bien como en Colab (RMSE puntual en test temporal 0.890 y 0.963) pero rankean mal el catálogo: la diferencia con Two-Tower A, que es esencialmente una factorización con feedback implícito, apunta al objetivo de entrenamiento (rating contra interacción) como factor principal. No se probó una variante implícita de NCF.
- Hipótesis previa no confirmada: que recomienden ítems oscuros (fracción de cola @10: SVD 0.18%, NCF 1.72%).
- Son baselines simples a propósito: no se ajustaron hiperparámetros salvo la sensibilidad de épocas del NCF. Los ítems sin interacciones en train se excluyen del ranking (Two-Tower A tampoco los recomienda).

### Reproducción en Google Colab (Etapa 5)

`notebooks/colab_stage5.ipynb` (dependencias en `requirements-colab-stage5.txt`) permite a un tercero verificar la Etapa 5 sin repetir la corrida completa. Clona la etiqueta indicada en `GIT_REF` (por defecto `stage5-final`), instala dependencias, descarga MovieLens, copia las embeddings de referencia de `reference/stage4/`, corre `pytest tests/stage5`, ejecuta `scripts/verify_report.py --strict` y reentrena la(s) semilla(s) elegida(s) (por defecto la 42) comparando cada NDCG@10 con el publicado: se acepta una diferencia de hasta ~2 desvíos entre semillas, porque la GPU y las versiones de Colab no reproducen bit a bit la máquina local. Los resultados de Colab se guardan en `results/colab/` y no pisan los oficiales.

Alcance deliberadamente liviano: no incluye G1b, las 10 semillas, el bootstrap por usuarios ni el análisis de sesgo, que necesitan las listas top-50 de todas las corridas (`results/stage5/recs/*.npz`, ignoradas por git por la regla `*.npz` del `.gitignore`). Esos pasos se reproducen localmente con los comandos de «Reproducción de la Etapa 5». `scripts/run_stage5_save_recs.py` y `scripts/run_stage5_explicit_baselines.py` aceptan `--out-dir` (y el primero `--models`) para no pisar resultados oficiales. Las referencias de la Fase 2c que antes vivían en `scratch/` (popularidad y azar) están ahora en `results/stage5/fase2c_reference.json`.
Limitaciones conocidas del cuaderno: la corrida en Colab no se midió en una GPU T4 (los tiempos del cuaderno son estimaciones a partir de la RTX 4060) y las GPU sin soporte nativo de bf16 pueden entrenar más lento o con otra numérica.
