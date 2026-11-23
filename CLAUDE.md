# CLAUDE.md — Tesis sobre sistemas de recomendación: contexto, reglas y estado actual

Archivo de traspaso, generado el 22/11/2026 a partir de una conversación larga entre el usuario y un asistente de chat (Claude en claude.ai) que dirigió el proyecto y revisó los reportes de un agente anterior (Antigravity). Todo lo que figura acá proviene de esos reportes y de esa conversación. Lo marcado con [VERIFICAR] no fue comprobado de forma independiente: comprobalo leyendo el repositorio antes de apoyarte en eso.

## 0. Tu rol y el de los demás

- Vos (Claude Code en el IDE) sos desde ahora el **director técnico y el implementador** del proyecto. Tenés una mirada crítica de todo (diseño experimental, validez, coherencia), proponés y corregís los próximos pasos y escribís el código. No sos un ejecutor pasivo: si un pedido del usuario o un resultado previo te parece metodológicamente débil, lo decís antes de implementarlo.
- El usuario (Marcos Goizueta) decide el alcance y aprueba los cambios grandes. Está aprendiendo el campo: explicá el porqué de las decisiones.
- Un segundo asistente (Claude en el chat de claude.ai) pasa a ser **auditor de métricas, consistencia y coherencia de los datos**. El usuario le va a pegar tus reportes (tablas, curvas, comparaciones) para que los revise. Facilitale el trabajo siguiendo el formato de la sección 9.
- El agente anterior (Antigravity) implementó las Etapas 3, 4 y 5. Cometió errores serios (sección 7). No asumas que sus reportes son correctos sin comprobarlos contra los archivos.

## 1. El proyecto en una página

- Tesis de grado de Ingeniería en Sistemas de Información (Plan 2012), Departamento de Ciencias e Ingeniería de la Computación (DCIC), Universidad Nacional del Sur. Título provisorio: «Sistemas de Recomendación Modernos: Diseño, Implementación y Análisis Comparativo de Enfoques Discriminativos y Generativos basados en Modelos de Lenguaje». Directora: Andrea Cohen. Codirector: Federico Schmidt. Alumno: Marcos Goizueta.
- Objetivo: diseñar, implementar y analizar críticamente recomendadores, desde una línea de base clásica hasta recuperación generativa, y evaluar empíricamente los compromisos entre desempeño, sesgo de popularidad, diversidad/cobertura y cold-start.
- El aporte central es la **comparación entre paradigmas**, no optimizar un modelo. Por eso se prioriza ancho antes que profundidad, controles bien elegidos y honestidad sobre lo que los datos permiten concluir.
- Dataset principal: MovieLens ml-latest-small (610 usuarios, 9.724 películas con valoraciones, 100.836 valoraciones). El plan original contemplaba un dataset mayor (ml-latest) en etapas posteriores.
- Etapas: (1) SVD, (2) NCF, (3) arquitectura de dos etapas, (4) embeddings semánticos de texto en la torre de ítems y cold-start, (5) recuperación generativa tipo TIGER más el análisis comparativo final de sesgo y diversidad.

## 2. Reglas de trabajo (no negociables)

1. **Ninguna cifra sin fuente.** Todo número de un reporte debe salir de un script que lee JSON o CSV guardados en disco. Está prohibido interpolar, estimar «a mano» o empalmar columnas de corridas distintas (el agente anterior inventó valores de pérdida de validación y empalmó curvas). Las estimaciones se marcan como estimaciones.
2. **Separar siempre «comprobado» de «no comprobado / hipótesis».** Las explicaciones causales son hipótesis salvo que las pruebes con un experimento. Evitá «comprueba» o «demuestra» para comparaciones entre variantes con una sola semilla.
3. **Una semilla es preliminar.** Reportá media ± desvío sobre semillas y diferencias pareadas por semilla. Regla usada hasta ahora: 10 semillas si el tiempo por semilla es de unos 25 minutos o menos, 5 si es mayor (preferí 10 cuando sea viable: el ruido entre semillas es grande). Diferencias de ~0,01 en NDCG@10 entre semillas son ruido en este dataset.
4. **Nada de selección sobre test.** Hiperparámetros, épocas y tamaño del haz se eligen con validación o se fijan a priori. Los resultados primarios y las hipótesis se anotan **antes** de ver resultados. Declará todas las configuraciones probadas.
5. **Comparaciones justas.** Mismo presupuesto de ajuste para los modelos que se comparan directamente; los controles heredan la configuración de su tratamiento. Cualquier desviación (lote, precisión, criterio de parada) se aplica igual a todos y se documenta en el README.
6. **Controles antes de conclusiones.** Toda afirmación del tipo «el modelo de lenguaje ayuda» necesita su control (géneros simples, vectores aleatorios, IDs aleatorios). Pensá siempre cuál sería la explicación alternativa aburrida.
7. **No tocar lo congelado.** El código y los resultados de las Etapas 3 y 4 no se modifican; los tests de regresión (sección 5) deben seguir pasando. Si necesitás tocar algo compartido, mostrá el diff y pedí OK.
8. **Reproducibilidad.** Semillas explícitas; reseteo de semilla inmediatamente antes de instanciar cada modelo (el orden de las operaciones aleatorias importa); registrar en el JSON de cada corrida versiones, dispositivo, hash del mapa de códigos e hiperparámetros. Las corridas rápidas (`--quick`) van a `results/quick/` y nunca pisan resultados reales. Los registros crudos van en `results/`, no en `scratch/`.
9. **Control de versiones.** No hagas commits, etiquetas ni merges sin que el usuario lo pida o lo autorice. Etiquetas esperadas: `etapa3-final`, `etapa4-final` y luego `etapa5-final`.
10. **Plan antes de código** en cambios grandes, y detenerse al final de cada fase para revisión. Pedir confirmación antes de corridas de más de unas 4 horas.
11. **La latencia** medida en una implementación concreta no es una ventaja arquitectónica: reportala etiquetada con el hardware y sin generalizar.
12. **Bibliografía** solo después de verificarla (sección 10).

## 3. Entorno y repositorio

- Windows; Python 3.11.9; GPU NVIDIA RTX 4060 Laptop (8 GB de VRAM), driver 555.97; PyTorch 2.6.0+cu124; transformers 4.57.6; sentence-transformers 3.0.1; scikit-learn 1.7.2; NumPy 1.23.5. Para evitar un conflicto de DLL (c10.dll) en Windows, PyTorch se importa primero en los scripts y en `tests/conftest.py`.
- Las corridas largas van con la notebook enchufada, plan de energía de alto rendimiento y sin suspensión (una corrida con batería baja tardó varias veces más y sus tiempos no sirven).
- Las corridas en CPU usaron torch 2.9.1+cpu y las de GPU torch 2.6.0+cu124. No son comparables bit a bit entre CPU, GPU local y Colab (el generador aleatorio de la GPU difiere).
- Colab se usa solo para los notebooks livianos de reproducción que se comparten con los directores (`notebooks/colab_stage3.ipynb`, `notebooks/colab_stage4.ipynb`), que clonan el repositorio por etiqueta. No se asume qué GPU asigna Colab.
- Estructura aproximada del repositorio (la carpeta raíz todavía se llama `03-two-stage/`; renombrarla queda pospuesto hasta el final) [VERIFICAR con `git ls-files`]:
  - `src/recsys/` (Etapa 3: datos, retrieval, ranker, evaluación, tracking, config);
  - `src/recsys/stage4/` (embeddings, torres de ítems, cortes de evaluación, split en frío);
  - `src/recsys/stage5/` (split temporal, RQ-VAE, tokenizador, modelos, decodificación restringida, protocolo de evaluación);
  - `scripts/` (run_stage3.py, run_stage4.py, run_stage4_d2.py, scripts de la grilla y de la Fase 2c de la Etapa 5, run_stage5_multi_seed.py, generate_stage5_report_tables.py);
  - `tests/` (suites por etapa; las lentas con marker `slow`);
  - `reference/stage4/` (metadata.json y embeddings de referencia versionados);
  - `results/gpu/` y `results/stage5/` (curvas crudas, per_seed, resúmenes);
  - `scratch/` y `data/cache/` ignorados por git.
- Ramas: `04-semantic-retrieval` (Etapa 4) y `05-generative-retrieval` (actual). Verificá etiquetas y estado de los merges.
- Dependencias: `requirements.txt` (GPU, versiones fijas), `requirements-stage4.txt`, `requirements-colab*.txt` (sin fijar torch).

## 4. Datos y trampas conocidas

- `ratings.csv` está ordenado por (userId, movieId), **no por tiempo**. El 23,04% de las interacciones comparte timestamp con otra del mismo usuario (7.438 grupos), y en el 100% de esos grupos el orden original coincide con movieId ascendente. Desempatar por el orden del archivo introduce una señal falsa: se usa una permutación pseudoaleatoria de semilla fija.
- Texto de las películas: título (que ya incluye el año entre paréntesis en el 99,87% de los casos) y géneros; 34 películas sin género y 13 títulos sin año. Embeddings: all-MiniLM-L6-v2 (384 dimensiones, normalizados, revisión de Hugging Face 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), plantilla `"<título>. Genres: <géneros>"` en inglés. La similitud está dominada en parte por el año del título: la pureza por década de los clusters del primer nivel de Semantic IDs cae de ~71% a nivel de azar al quitar el año.
- Distribución muy sesgada hacia lo popular: con split aleatorio, el 65% de lo relevante en test está en el 10% de ítems más populares, y la mitad del catálogo tiene 2 o menos interacciones en train.
- Cortes de popularidad estándar (interacciones en train): 0, 1-5, 6-20 y más de 20.

## 5. Protocolos de evaluación

**A. Split aleatorio 80/20 (`random_state=42`), Etapas 3 y 4.** Relevante = rating ≥ 3.5 en test. Evaluación **full-ranking**: top-10 por usuario sobre los ítems no vistos en train, con precision, recall y NDCG@10, cobertura, popularidad media y fracción de cola (cortes 0 y 1-5). La Etapa 4 agrega cortes de popularidad, recall de candidatos @50 y @200 y un escenario de **ítems en frío** (10% de ítems al azar, semilla 42, solo en test; 972 ítems).

Guardia de regresión (la variante A debe reproducir la Etapa 3, semilla 42, GPU): retrieval solo con NDCG@10 = 0.190736 (precision 0.147754, recall 0.127539) y Two-Stage N=50 con NDCG@10 = 0.217769. En Colab el retrieval coincide pero Two-Stage puede diferir (opción `--guard warn`).

**B. Split temporal por usuario, Etapa 5.** El primer 80% de las interacciones de cada usuario (ordenadas por timestamp con desempate aleatorio de semilla fija) es train_full, el último 20% es test, y la validación es el último 10% de train_full (train_core es el resto). Los cortes se mueven al borde de evento simultáneo más cercano (Opción C), con fallback; la separación estricta por timestamp está verificada con tests.

Tamaños: train_full 80.621, train_core 72.347, val 8.274 y test 20.215 interacciones; 12.301 relevantes en test; 592 usuarios de test evaluables y 573 de validación. Ítems sin interacciones en train_full: 1.487 (15,29%), que concentran 900 relevantes de test (7,32%). Relevantes de test por corte: 0 → 7,32%, 1-5 → 16,69%, 6-20 → 22,78%, más de 20 → 53,22%. El split temporal es mucho más difícil que el aleatorio (la popularidad pasa de NDCG@10 0.174 a 0.082).

Protocolo de ajuste de la Etapa 5: elegir hiperparámetros y épocas en validación con train_core, **reentrenar con train_full** durante E* épocas y evaluar en test usando como contexto el historial de train_full y excluyendo como vistos todos sus ítems. Parada temprana por pérdida de validación (cada 2 épocas, paciencia 3, mejora mínima 0,005). Haz fijado a priori en 50.

Referencias del azar (split temporal): recall@50 esperado en frío 0,53%, cobertura @10 45,61% y fracción de cola @10 73,01%.

## 6. Qué se hizo y qué se encontró

**Etapa 1, SVD (Surprise, split aleatorio, protocolo pointwise).** RMSE 0.8807, MAE 0.6766, precision@10 0.7446 y recall@10 0.5086 (umbral 3.5). Estas métricas pointwise no son comparables con el full-ranking. Reevaluado el 23/11/2026 bajo el protocolo temporal full-ranking: NDCG@10 0.0412 ± 0.0026 (ver sección 8, tarea A).

**Etapa 2, NCF (NeuMF simplificado, PyTorch).** RMSE 0.9645, MAE 0.7388, precision@10 0.7420 y recall@10 0.5223 (pointwise). Reevaluado el 23/11/2026 bajo el protocolo temporal full-ranking: NDCG@10 0.0336 ± 0.0027 con 15 épocas y 0.0421 ± 0.0027 con épocas por validación.

**Etapa 3, dos etapas (split aleatorio, 10 semillas, GPU).** Retrieval two-tower (embeddings de 32 dimensiones, inicialización con desvío 0,1, BCE con 4 negativos uniformes, lr 5e-3, weight decay 1e-6, lote 256, 30 épocas) más ranker MLP (embeddings, géneros, perfil de géneros del usuario y log-popularidad; BCE; 8 épocas; lote 1024; lr 2e-3). El ranker por regresión de rating inicial empeoraba el retrieval y se rediseñó a feedback implícito.

Resultados: popularidad NDCG@10 0.1735; retrieval solo 0.1933 ± 0.0050; ranker sobre el catálogo completo 0.1689 ± 0.0225 (inestable); **Two-Stage N=50 0.2316 ± 0.0078** (precision 0.1846), mejor que ambos en 10 de 10 semillas (+0.0383 sobre el retrieval). N=50 rinde igual o mejor que N=200. La variante con negativos tomados del retrieval falló (precision 0.0186). La aceleración de Two-Stage depende del hardware (44 veces en CPU, unas 1,7 veces en GPU con este catálogo).

**Etapa 4, embeddings de texto en la torre de ítems del retrieval (split aleatorio, 10 semillas, más escenario en frío).** Variantes: A (solo IDs), B (solo texto: 384→64→32, sin IDs), C (ID + texto), C-cero (C con el ID anulado en ítems fríos al evaluar), C2 (C con ID dropout 0,5 en entrenamiento y ID anulado en ítems fríos), D (ID + géneros), D2 (ID dropout + géneros), E (solo géneros, sin IDs) y R (vectores aleatorios fijos de 384 dimensiones, sin IDs).

Hallazgos en NDCG@10 de retrieval solo (diferencias pareadas medias): B−A +0.0195, C2−A +0.0381 y D2−A +0.0297 (todas positivas en 10 de 10 semillas); C−D +0.0015 (6 de 10, indistinguibles); C2−D2 +0.0085 (9 de 10); B−R +0.0722. Lectura prudente: cualquier contenido más ID dropout explica la mayor parte de la mejora, y el texto del modelo de lenguaje aporta poco sobre los géneros en ítems con historial.

En frío, recall de candidatos @200 (azar 2,09%): B 4,51%, C2 9,18%, D2 10,55%, E 11,20%, C-cero 2,81%, R 0,07% y A 0. A K=10 y K=50, B y C2 están al nivel del azar y solo E lo supera claramente. El ranker compartido re-impone el sesgo de popularidad en Two-Stage (popularidad media 98 a 109, contra 76 a 86 en el retrieval).

**Etapa 5, recuperación generativa tipo TIGER (split temporal).** Variantes:
- G1: Semantic IDs con un RQ-VAE sobre los embeddings de texto (3 niveles de 256 códigos más un token de desambiguación con rango 0 a 15; entrenado solo con ítems de train; los ítems fríos se cuantizan después; KMeans en CPU; el mapa de códigos se regenera por semilla en la corrida multi-semilla [verificado el 22/11/2026: hashes distintos en las 10 semillas, ver `results/stage5/seed_dependence_audit.json`]; el de la semilla 42 tiene hash SHA256 ed2b07a119f750a4…, el de la Fase 2c), más un modelo tipo T5 desde cero (d_model 128, d_ff 1024, 6 cabezas, 4+4 capas, unos 4,57 M de parámetros).
- G2: control con códigos jerárquicos aleatorios y la misma estructura.
- G3: control discriminativo secuencial con un token atómico por película (unos 5,7 M de parámetros).
- G1b: ablación con embeddings sin año (5 semillas).

La decodificación usa búsqueda en haz restringida por un trie a ítems válidos no vistos. Entrenamiento en bf16 con lote 128. Configuración elegida en validación (misma grilla para G1 y G3: L en {20, 50, 100} por política de datos en {tope de 100 ventanas por usuario, muestreo balanceado}; G2 y G1b heredan la de G1): G1 y G2 con L=100, balanceado, lr 1e-3, dropout 0,1 y E*=18; G3 con L=20, balanceado, lr 5e-4, dropout 0,2 y E*=16. Los Two-Tower de comparación eligen sus épocas por pérdida de validación en cada semilla (asimetría documentada).

Resultados con 10 semillas (G1b con 5), NDCG@10 media ± desvío, 592 usuarios: popularidad 0.0821; Two-Tower A 0.0796 ± 0.0024 (E* medio 2,2); B 0.0916 ± 0.0057; C2 0.1036 ± 0.0026; D2 0.1007 ± 0.0035; Two-Stage 0.1050 ± 0.0030; G1 0.0571 ± 0.0078; G2 0.0466 ± 0.0085; G3 0.0868 ± 0.0026 (ranking exacto 0.0874); G1b 0.0535 ± 0.0063.

Cobertura @10: A 1,44%, B 6,00%, C2 4,49%, D2 4,38%, Two-Stage 2,28%, G1 2,03%, G2 1,44%, G3 2,00% (popularidad 1,01%). Recall@50 en frío: todos entre 0,0000 y 0,0023, por debajo del azar (0,0053). Diferencias pareadas (NDCG@10): G3−G1 +0.0296, G1−G2 +0.0106 (8 de 10 semillas; test de signo p=0,109 y Wilcoxon p=0,027; el p=0,0209 citado antes era un t-test pareado; no supera Holm con 3 comparaciones primarias), C2−G3 +0.0169 y G1−G1b +0.0036 (p=0,337).

**Resultados finales de la Etapa 5 (verificados el 22/11/2026; fuentes en `results/stage5/`).** (1) `verify_report.py`: PASS, las cifras de arriba coinciden con `per_seed/*.json` y la semilla 42 con la Fase 2c. (2) Sensibilidad de protocolo (`sensitivity_30ep_analysis.md`): con 30 épocas fijas A 0.0944, B 0.0988, C2 0.1049, D2 0.1053, Two-Stage 0.0957 (cobertura @10 de 8,0% a 8,9% en los cinco); Two-Stage supera a A con parada por validación (+0.0255, 10 de 10) pero no con 30 épocas (+0.0013, indistinguibles); con 30 épocas los mejores discriminativos son C2 y D2 (indistinguibles entre sí). (3) Estadística (`final_comparison.md`, `bias_diversity.md`): con Holm, G1−G3 (−0.0296), G1−D2 con 30 épocas (−0.0481; D2 elegido post hoc como mejor) se rechazan y G1−G2 no; los IC95% por usuarios confirman los dos primeros. G3 contra popularidad: +0.0047 con IC95% por usuarios [−0.004, +0.013], por lo que **G3 es indistinguible de popularidad** (el test por semillas, que trata a popularidad como constante, daba p=0,002 y sobreestimaba). G1 y G2 quedan por debajo de popularidad (IC por usuarios excluye 0). (4) Error del haz en G1, semillas 42-44 (`g1_exact/g1_exact_analysis.md`): NDCG@10 haz/exacto 0.0658/0.0621, 0.0687/0.0734, 0.0610/0.0620 (el signo de la diferencia cambia), Jaccard@10 entre 0,17 y 0,25: el haz no explica el bajo rendimiento de G1, aunque los top-10 individuales difieren mucho (parte puede ser diferencia de criterio de puntaje, no comprobado). (5) Sesgo y diversidad (top-10, 145 listas guardadas en `recs/`, reproducen exactamente los NDCG publicados): popularidad media recomendada y Gini: popularidad 216 y 0,998; discriminativos con 30 épocas 111-120 y 0,970-0,974; A con parada por validación 197 y 0,998 (se comporta casi como popularidad); G1 103 y 0,994; G2 110 y 0,996; G3 170 y 0,994. La diversidad intra-lista con embeddings de MiniLM queda entre 0,47 y 0,52 en todos los modelos y no discrimina (además es circular para G1).

Lectura prudente: los modelos discriminativos con contenido superan a los generativos en este dataset chico de 610 usuarios; G3 es indistinguible de popularidad y G1 y G2 quedan por debajo; todos los modelos concentran sus recomendaciones en pocos ítems (Gini ≥ 0,97), los generativos más que los discriminativos entrenados 30 épocas; ningún modelo llega al azar en recall@50 en frío. G1 y G2 recomiendan ítems menos populares que los discriminativos (103 y 110 contra 111-120) pero con baja precisión: hipótesis no probada, concentran en pocos ítems de popularidad media. Hipótesis no probadas: el efecto de escala (TIGER se evalúa con datasets de decenas de miles de usuarios) y el distinto presupuesto de ajuste entre generativos (grilla en validación) y Two-Tower (configuración fija heredada de las Etapas 3 y 4).

Sensibilidad del haz en G1: los top-10 cambian mucho con el ancho del haz (Jaccard 0,39 entre haz 20 y 50) y la cobertura baja al ensanchar; es probable que parte de la «diversidad» de los haces chicos sea error de búsqueda. El ranking exacto de G1 cuesta unos 655 ms por usuario (6,5 minutos por semilla).

## 7. Errores del agente anterior (para no repetirlos)

1. Desempate de timestamps por el orden del archivo (equivale a movieId): señal falsa en un modelo secuencial. Corregido.
2. Orden de operaciones aleatorias: entrenar el ranker antes que el retrieval desfasaba los generadores y A no reproducía la Etapa 3. Corregido con reseteo de semilla por modelo y una guardia automática.
3. Valores de pérdida de validación **inventados** (G2 y G3) y curvas de G1 **empalmadas** de dos corridas. Corregido con registros crudos. Desconfiá de cualquier reporte que no salga de JSON.
4. Se interpretó mal el sobreajuste de G3 y de G2 con esos datos. Con curvas crudas, G3 sobreajusta de forma moderada (brecha de ~1,8) y G2 no sobreajusta pronto.
5. Afirmaciones fuertes con una sola semilla («comprueba», «significativo»); elección del mejor haz sobre test; cobertura y cola calculadas con K distinto entre modelos; fila del azar mal calculada; filas de sensibilidad copiadas de otro haz.
6. La parada temprana por pérdida de validación con negativos aleatorios selecciona un Two-Tower A peor (E* cercano a 2, por debajo de popularidad). Por eso «Two-Stage mejora a A en +0.0255» es un artefacto del protocolo. Medido el 22/11/2026 con 10 semillas y 30 épocas fijas (`results/stage5/sensitivity_30ep_analysis.md`): A 0.0944 ± 0.0045 y Two-Stage 0.0957 ± 0.0024, diferencia +0.0013 (7 de 10 semillas, Wilcoxon p=0,49; IC95% por usuarios [−0.005, +0.007]), es decir indistinguibles. (La frase anterior de este archivo, «Two-Stage rendía menos que A», no se sostiene con 10 semillas.)
7. Tiempos de corridas con la notebook en ahorro de energía: no representativos.

## 8. Situación actual y próximos pasos

Estado: la corrida multi-semilla de la Etapa 5 terminó según el agente anterior (85 JSON en `results/stage5/per_seed/`, resumen en `results/stage5/multi_seed_summary.json`, 29 tests pasando, rama `05-generative-retrieval`) [VERIFICAR]. Al agente anterior se le pidió lo siguiente; **comprobá qué se completó leyendo el repositorio, no su reporte**:

1. `scripts/verify_report.py`: recalcular todas las cifras de las tablas desde `per_seed/*.json`, fallar si alguna no coincide, comprobar que la semilla 42 del multi-semilla coincide con la de la Fase 2c (G1 0,0658, G2 0,0399 y G3 0,0832 con haz 50) y guardar checksums de los JSON.
2. Sensibilidad de protocolo: Two-Tower A, B, C2, D2 y Two-Stage con 30 épocas fijas (el protocolo de las Etapas 3 y 4) en 10 semillas, junto con la versión de parada por pérdida de validación, con cobertura y cola @10; ver si Two-Stage supera a A en ambos protocolos.
3. Estadística: diferencias pareadas con intervalo de confianza del 95% por bootstrap sobre semillas, bootstrap por usuarios de test, corrección de Holm para las comparaciones primarias (G1 contra G3, G1 contra G2, G1 contra el mejor Two-Stage), y comparación de G1, G2 y G3 contra popularidad; sin etiquetas «muy significativo».
4. Ranking exacto de G1 en las semillas 42 a 44 para medir el error del haz.
5. Redacción: marcar como hipótesis que la jerarquía del RQ-VAE guía mejor la decodificación y que el token atómico es «más eficiente»; documentar las asimetrías (épocas por semilla en los Two-Tower, épocas fijas en los generativos, el mapa de Semantic IDs y los IDs aleatorios de G2 se regeneran por semilla, de modo que la varianza de G1 incluye la del tokenizador; no es una asimetría, pero hay que decirlo).

**Estado verificado el 22/11/2026 (leyendo archivos, no reportes):** las cinco tareas están hechas. (1) `scripts/verify_report.py` → PASS, 85 checksums en `results/stage5/verify_report.json`. (2) `scripts/run_stage5_sensitivity_30ep.py` → `per_seed_30ep/`, `sensitivity_30ep_summary.json` y `.md`. (3) `scripts/compute_stats_stage5.py` → `stats_report.json` (bootstrap sobre semillas, test de signo, Wilcoxon, Holm), más bootstrap por usuarios en `scripts/analyze_stage5_recs.py` → `bias_diversity.json/.md`. (4) `scripts/run_stage5_g1_exact.py` → `g1_exact/`. (5) README actualizado con hipótesis H1 y H2 y las asimetrías. Además: `scripts/audit_stage5_seed_dependence.py` (hashes por semilla, popularidad idéntica en las 10 semillas), `scripts/run_stage5_save_recs.py` (145 listas top-50 en `recs/`, todas reproducen el NDCG publicado con diferencia 0) y `scripts/build_final_table_stage5.py` → `final_comparison.md`. Ramas y etiquetas: rama `05-generative-retrieval`, etiquetas existentes `stage3-final` y `stage4-final` (con ese nombre, no `etapa3-final`); `etapa5-final` aún no existe. Nada de esto está commiteado.

Después (propuesto, a confirmar con el usuario):

- A. Reevaluar SVD y NCF con el protocolo temporal full-ranking (rankear todos los ítems no vistos por puntaje predicho). **Hecho el 23/11/2026** (`scripts/run_stage5_explicit_baselines.py`, `results/stage5/explicit_baselines_analysis.md`): SVD 0.0412 ± 0.0026, NCF (15 épocas) 0.0336 ± 0.0027, NCF con épocas por validación 0.0421 ± 0.0027, todos por debajo de popularidad (0.0821) en las 10 semillas; reproducen los RMSE de Colab en el split aleatorio (0.8826 y 0.9629). SVD es una reimplementación en numpy del algoritmo de Surprise. Los notebooks originales están en `notebooks/01_svd_baseline.ipynb` y `02_ncf.ipynb`.
- B. Tabla final comparativa unificada y análisis de sesgo y diversidad entre todos los modelos bajo el protocolo temporal (cobertura, cola, popularidad media, Gini y diversidad intra-lista), con intervalos de confianza. **Hecho** (`final_comparison.md`, `bias_diversity.md`); queda por decidir si se reporta la diversidad intra-lista con embeddings de MiniLM, que es circular para G1.
- C. Notebook liviano de Colab para la Etapa 5 (mismo esquema que los anteriores), README actualizado con todas las desviaciones declaradas, y luego commit, etiqueta `etapa5-final` y merge, hechos por el usuario o con su autorización. **Notebook escrito el 23/11/2026** (`notebooks/colab_stage5.ipynb`, `requirements-colab-stage5.txt`); clona la etiqueta de `GIT_REF` (por defecto `stage5-final`, igual que `stage4-final`; cambiar el valor si se elige otro nombre), verifica cifras publicadas y reproduce la semilla 42. Simulado en un clon local limpio (sin `scratch/` ni `data/`); **pendiente la prueba real en Colab por el usuario**, que es la que habilita la etiqueta. `scratch/stage5_fase2c_results.json` se copió a `results/stage5/fase2c_reference.json` porque `scratch/` no viaja en git. Las listas `recs/*.npz` están ignoradas por `*.npz` en el `.gitignore` (decisión abierta).
- D. Decisión sobre escala: repetir las comparaciones clave (G1, G2 y G3 contra C2 y Two-Stage) con un dataset mayor (ml-1m o ml-latest) para probar la hipótesis de escala. Conversarlo con los directores antes de ampliar el alcance.
- E. Apoyo a la redacción: tablas, limitaciones y bibliografía verificada.

## 9. Formato de reporte (para facilitar la auditoría)

Cada reporte debe incluir:

1. Qué se ejecutó (comando, rama, hash del commit, semillas, dispositivo).
2. Tablas generadas por script, con la ruta de cada archivo fuente.
3. Número de usuarios evaluados y definición de cada métrica con su K.
4. La línea base de azar y la de popularidad, siempre presentes.
5. **Cambios respecto del reporte anterior**: qué números cambiaron y por qué.
6. Autoauditoría: sumas que cierran, valores repetidos sospechosos, comparación con corridas previas y discrepancias abiertas.
7. Separación entre comprobado, hipótesis y no comprobado.
8. Preguntas abiertas.

Si algo no cierra, decilo vos antes de que lo detecte el auditor.

## 10. Bibliografía (NO verificada: confirmar autores, año y congreso antes de citar)

TIGER (Rajput et al., NeurIPS 2023; verificado en parte: Semantic IDs con RQ-VAE, 3 niveles de 256 códigos más un token de desambiguación, T5 pequeño, leave-one-out y comparación con IDs aleatorios); NCF (He et al., WWW 2017); Rendle et al. 2020 (NCF contra factorización matricial); Covington et al. 2016 (YouTube DNN); Koren, Bell y Volinsky 2009; Funk 2006 (blog); Cremonesi et al. 2010 y Krichene y Rendle 2020 (evaluación top-N); Harper y Konstan 2015 (MovieLens); DropoutNet (Volkovs et al., NeurIPS 2017); Kingma y Ba 2015 (Adam); Ferrari Dacrema et al. 2019.

## 11. Estilo con el usuario

Español rioplatense (voseo), claro y directo. El usuario prefiere prosa fluida a listas largas en los documentos, avanzar de forma concreta una vez fijada la dirección, y que lo critiques con honestidad en lugar de validarlo. Preguntá una sola cosa a la vez cuando necesites una decisión suya. Cerrá cada tarea con un resumen de una o dos frases del resultado y el próximo paso natural. Explicá las decisiones importantes y sus alternativas en lenguaje simple.
