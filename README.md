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
