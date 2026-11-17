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
