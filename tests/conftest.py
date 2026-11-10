import os
import sys

# Precargar torch para inicializar correctamente las DLLs de C10 en Windows
try:
    import torch
except Exception:
    pass

