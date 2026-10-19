# Tarea 1 — Noticias peruanas: del titular al clasificador

> **Pregunta del proyecto:** ¿Pueden convertir titulares peruanos en un clasificador de noticias?

**Respuesta corta: sí.** Un BETO ajustado con etiquetas generadas por un LLM local clasifica titulares
de La República en 4 categorías con **F1 macro = 0,88** en un test reservado de 60 noticias. Supera
al LLM que lo entrenó (0,79) y al zero-shot (0,59), y tarda **51 ms por titular en CPU**.

| Enfoque (test, n = 60) | Modelo | Accuracy | F1 macro | Latencia CPU |
|---|---|---|---|---|
| Zero-shot (piso) | `Recognai/bert-base-spanish-wwm-cased-xnli` | 0,617 | 0,592 | 233 ms |
| LLM débil (maestro) | `Qwen/Qwen2.5-1.5B-Instruct` (local) | 0,800 | 0,792 | 4 295 ms |
| **Transformer fine-tuned** | **BETO** (`dccuchile/bert-base-spanish-wwm-cased`) | **0,883** | **0,883** | **51 ms** |
| Extra: respaldo ligero | TF-IDF + regresión logística | 0,800 | 0,789 | 2 ms |

Además: zero-shot sobre 40 titulares (piso del enunciado): accuracy 0,700 · F1 macro 0,697.
Concordancia del LLM con la auditoría humana: **15/20 = 75 %**.

---

## Medio y fechas de extracción

* **Medio:** La República (`https://larepublica.pe`), opción A.
* **Fecha de descarga:** 18 de octubre de 2026, de 21:02 a 21:20 (hora de Lima). El registro está en
  `datos/raw/metadatos_descarga.json`.
* **Fechas de publicación:** del 2019-11-20 al 2026-10-18. El 62 % de las noticias tiene menos de un
  mes de antigüedad.
* **Fuentes:**
  * Los RSS oficiales (`/rss/politica.xml`, `economia`, `deportes`, `ciencia` y `home`). El de
    `tecnologia` no existe: responde `410 Gone`.
  * Como el RSS solo cubre ~48 horas, se completó con el HTML de las páginas de sección y de noticia.
    Ambas están permitidas por `robots.txt`. Se respetaron las rutas prohibidas (`/pag/`,
    `/archivo/`, …).
* **Buenas prácticas:** `User-Agent` descriptivo, pausa aleatoria de 1–2 s entre pedidos, máximo 1
  reintento, y 362 pedidos en 18 min sin ningún bloqueo (403/429). El respaldo crudo quedó en
  `datos/raw/`.
* **Dataset final:** 399 noticias (100 política, 100 economía, 100 deportes, 99 ciencia y tecnología).
  Se descartaron 202 filas crudas, con su motivo reportado en el notebook 1: 53 URLs duplicadas,
  1 titular duplicado y 148 de secciones fuera de las 4 categorías.

## Metodología

1. **`01_recoleccion.ipynb`**: scraping (RSS + HTML) → limpieza (duplicados, fechas a `AAAA-MM-DD`) →
   mapeo explícito `seccion_original` → `etiqueta` → EDA con 3 gráficos de Plotly → `datos/noticias.csv`.
2. **`02_clasificacion.ipynb`**:
   * **Zero-shot** con un modelo NLI.
   * **Supervisión débil**: Qwen2.5-1.5B corre en la laptop con un prompt cerrado. Se puntúa la
     probabilidad de cada palabra-respuesta con la regla de la cadena y se reutiliza la caché KV del
     prompt → `datos/etiquetas_llm.csv`.
   * **Auditoría a ciegas** de 20 casos.
   * **Split estratificado** 70/15/15 (`random_state=42`): train usa las etiquetas del LLM;
     val y test, la etiqueta editorial.
   * **Fine-tuning** con `Trainer` (3 épocas, batch 16, métrica F1 macro) de DistilBERT multilingüe y
     BETO; se elige por F1 en validación.
   * **Evaluación única en test**: accuracy, F1 macro, matriz de confusión con `seaborn` y
     comparativa final.
3. **`app.py`**: dashboard de Streamlit con filtros, métricas, gráfico, tabla descargable y probador
   en vivo.

## Hallazgos analíticos

1. **El estudiante supera a su maestro.** El 17,6 % de las etiquetas del LLM en train contradice a la
   sección del diario. Aun así, BETO logra **F1 0,88 frente a 0,79 del propio LLM** en el mismo test,
   con 14 veces menos parámetros y unas 84 veces menos latencia. La clave fue elegir la época con
   etiquetas editoriales de validación: los dos modelos rinden mejor en la época 1 y después empiezan
   a memorizar el ruido del LLM.
2. **La frontera difícil son las noticias deportivas «sin deporte».** El LLM mandó a política
   **30 de 100** notas deportivas: polémicas entre técnicos y dirigentes, sanciones, el VAR,
   declaraciones. Los verbos de confrontación (*ningunea, arremete*) y los actores institucionales son
   los mismos de la cobertura política. El sesgo se hereda: **4 de los 7 errores del modelo en test
   coinciden con los del LLM**, y Deportes termina siendo la peor categoría (recall 0,73), pese a que
   para el zero-shot era la más fácil (9/10). Política, en cambio, actúa como «sumidero»: es la de
   menor precisión (0,78).

## Limitaciones técnicas

1. **Muestra chica, de un solo medio y con fechas desiguales.**
   * El test tiene 60 noticias (15 por categoría): cada titular mueve ~1,7 puntos de accuracy.
   * Todo viene de La República.
   * 60 de las 99 notas de ciencia y tecnología tienen más de 6 meses (tutoriales y reseñas
     «atemporales»), mientras que política es casi toda de octubre de 2026.

   Por eso el modelo puede no generalizar a otros medios ni a otras coyunturas.
2. **Pesos fuera de GitHub y todo en CPU.**
   * `modelo/model.safetensors` (≈ 440 MB) supera el límite de 100 MB y está en `.gitignore`. Al
     clonar el repositorio, el dashboard usa el respaldo TF-IDF (F1 0,79) hasta que se ejecute
     `02_clasificacion.ipynb` o se publique el modelo en Hugging Face (ver abajo).
   * El etiquetado con el LLM tomó 28,6 min para 399 noticias en CPU, lo que limita escalarlo a
     miles de titulares sin GPU.

---

## Cómo ejecutar en local

Los notebooks y la app se ejecutaron con **Python 3.14 (Anaconda) en Windows 10, solo CPU**.
`requirements.txt` fija las versiones de ese entorno.

```bash
cd tarea_1
python -m venv .venv
.venv\Scripts\activate            # Windows  (en macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # instala PyTorch para CPU

# 1) Datos: ya están en datos/. El notebook no vuelve a descargar si existe datos/raw/noticias_crudas.csv
jupyter notebook 01_recoleccion.ipynb

# 2) Modelo: genera modelo/model.safetensors (~45 min en CPU; baja ~4,5 GB de modelos la primera vez)
jupyter notebook 02_clasificacion.ipynb

# 3) Dashboard
streamlit run app.py
```

Se abre en `http://localhost:8501`. Sin el paso 2, la app funciona igual con el clasificador de
respaldo TF-IDF y lo indica en pantalla.

## Despliegue (opcional) en Streamlit Community Cloud

1. Sube el repositorio a GitHub (público). En `share.streamlit.io` elige el repositorio y el archivo
   `tarea_1/app.py`. En *Advanced settings* elige **Python 3.12, 3.13 o 3.14**: son las versiones
   para las que `requirements.txt` instala PyTorch en versión CPU.
2. Sin más pasos, la app usa el respaldo TF-IDF.
3. Para usar BETO, primero publica la carpeta `modelo/` en Hugging Face (tras iniciar sesión con tu
   token):

   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo("<usuario>/clasificador-noticias-lr", exist_ok=True)
   api.upload_folder(folder_path="modelo", repo_id="<usuario>/clasificador-noticias-lr")
   ```

   Luego, en *Settings → Secrets* de la app, agrega
   `MODELO_CLASIFICADOR = "<usuario>/clasificador-noticias-lr"`. `app.py` lee esa variable.

   No se probó si los recursos gratuitos de Streamlit Cloud alcanzan para PyTorch + BETO.

## Estructura

```
tarea_1/
├── 01_recoleccion.ipynb      # Scraping, limpieza y EDA (ejecutado)
├── 02_clasificacion.ipynb    # Zero-shot, LLM, auditoría, fine-tuning y evaluación (ejecutado)
├── app.py                    # Dashboard Streamlit
├── requirements.txt          # Versiones exactas
├── README.md
├── .gitignore                # Excluye pesos (.safetensors, .bin, checkpoints)
├── datos/
│   ├── noticias.csv          # 399 noticias, 7 columnas
│   ├── etiquetas_llm.csv     # titular, etiqueta_llm, confianza, modelo_llm, fecha
│   ├── auditoria_llm.csv     # 20 casos auditados
│   ├── particion.csv         # url → train / val / test
│   └── raw/                  # robots.txt, XML de cada RSS, articulos_html.jsonl, noticias_crudas.csv
└── modelo/
    ├── config.json, tokenizer.json, tokenizer_config.json   # artefactos ligeros de BETO
    ├── metricas.json         # métricas que lee el dashboard
    ├── tfidf_logreg.joblib   # clasificador de respaldo (< 300 KB)
    └── model.safetensors     # NO se versiona (≈ 440 MB)
```

## Uso de IA asistida

El código de scraping, los notebooks y el dashboard se desarrollaron con asistencia de **Claude Code**
(Anthropic), como permite el enunciado. Los resultados se verificaron así:

* Todas las métricas se calculan en el test reservado.
* Las cifras citadas salen de las salidas ejecutadas de los notebooks.

La tabla de revisión de la auditoría (20 casos) se preparó con asistencia de IA y fue confirmada
manualmente.
