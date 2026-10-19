"""Dashboard: ¿Pueden convertir titulares peruanos en un clasificador de noticias?

Correr en local (desde la carpeta tarea_1/):
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------- configuracion
st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🗞️", layout="wide")

# Los archivos se buscan junto a este app.py, no "donde sea que se lance la app":
# Streamlit Cloud la corre desde la raíz del repositorio.
CARPETA = Path(__file__).parent
CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
LEGIBLE = {"politica": "Política", "economia": "Economía", "deportes": "Deportes",
           "ciencia_tecnologia": "Ciencia y tecnología"}
COLORES = {"Política": "#1f6f8b", "Economía": "#c1663f", "Deportes": "#3a8a4f",
           "Ciencia y tecnología": "#7a5195"}

# El Transformer puede venir de la carpeta modelo/ (local) o de un repositorio de
# Hugging Face (p. ej. en Streamlit Cloud): MODELO_CLASIFICADOR="usuario/nombre-del-modelo".
ORIGEN_MODELO = os.environ.get("MODELO_CLASIFICADOR", str(CARPETA / "modelo"))


# ------------------------------------------------------------------ carga datos
@st.cache_data
def cargar():
    noticias = pd.read_csv(CARPETA / "datos" / "noticias.csv", keep_default_na=False)
    llm = pd.read_csv(CARPETA / "datos" / "etiquetas_llm.csv", keep_default_na=False)
    datos = noticias.merge(llm[["titular", "etiqueta_llm", "confianza"]], on="titular", how="left")
    datos["categoria"] = datos["etiqueta"].map(LEGIBLE)
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    metricas = json.loads((CARPETA / "modelo" / "metricas.json").read_text(encoding="utf-8"))
    return datos, metricas


# @st.cache_resource: el modelo se carga una sola vez y se comparte entre visitas.
@st.cache_resource
def cargar_clasificador():
    """Transformer fine-tuned si sus pesos están disponibles; si no, el respaldo TF-IDF."""
    try:
        local = Path(ORIGEN_MODELO)
        if local.exists() and not (local / "model.safetensors").exists():
            raise FileNotFoundError("Los pesos del Transformer no están en modelo/ (no se suben a GitHub).")
        from transformers import pipeline

        clasificador = pipeline("text-classification", model=ORIGEN_MODELO, tokenizer=ORIGEN_MODELO,
                                device=-1, top_k=None)
        return "transformer", clasificador, ""
    except Exception as e:  # sin pesos, sin internet o sin transformers instalado
        respaldo = joblib.load(CARPETA / "modelo" / "tfidf_logreg.joblib")
        return "tfidf", respaldo, str(e)


def predecir(texto):
    tipo, modelo, _ = cargar_clasificador()
    if tipo == "transformer":
        salida = modelo(texto)
        if salida and isinstance(salida[0], list):
            salida = salida[0]
        probs = {d["label"]: d["score"] for d in salida}
    else:
        probs = dict(zip(modelo.classes_, modelo.predict_proba([texto])[0]))
    return pd.Series(probs).reindex(CATEGORIAS).fillna(0.0)


datos, metricas = cargar()

# ----------------------------------------------------------------------- filtros
st.sidebar.header("Filtros")

medios = sorted(datos["medio"].unique())
medios_elegidos = st.sidebar.multiselect("Medio", medios, default=medios)

categorias = [LEGIBLE[c] for c in CATEGORIAS]
categorias_elegidas = st.sidebar.multiselect("Categoría", categorias, default=categorias)

buscar = st.sidebar.text_input("Buscar en el titular", placeholder="p. ej. Congreso, dólar, Alianza",
                               key="buscar")

filtrado = datos[datos["medio"].isin(medios_elegidos) & datos["categoria"].isin(categorias_elegidas)]
if buscar:
    filtrado = filtrado[filtrado["titular"].str.contains(buscar, case=False, regex=False)]

st.sidebar.caption(f"{len(filtrado)} de {len(datos)} noticias con los filtros actuales.")

# ------------------------------------------------------------------------ titulo
st.title("🗞️ ¿Pueden convertir titulares peruanos en un clasificador de noticias?")
st.caption(f"Titulares de La República · descargados el {metricas['fecha_descarga']} · "
           "Tarea 1 · Python Intermedio para CCSS y Gestión Pública")

with st.expander("Cómo se construyó (resumen metodológico)", expanded=True):
    st.markdown(f"""
1. **Recolección**: {len(datos)} noticias de `larepublica.pe` vía RSS oficial y HTML permitido por
   `robots.txt` (pausas de 1–2 s, máx. 1 reintento). La sección del diario se mapeó a 4 categorías:
   política, economía, deportes, ciencia y tecnología.
2. **Supervisión débil**: un LLM local (`{metricas['llm']['modelo']}`) etiquetó todas las noticias con
   un prompt cerrado; una auditoría a ciegas de {metricas['llm']['n_auditoria']} casos midió su calidad.
3. **Fine-tuning**: `{metricas['modelo_base']}` aprendió de las etiquetas del LLM
   (split estratificado 70/15/15, `random_state=42`, 3 épocas).
4. **Evaluación**: F1 macro en un **test reservado** de {metricas['n_test']} noticias, contra la
   sección editorial del diario.
""")

if filtrado.empty:
    st.warning("No hay noticias que cumplan los filtros. Ajusta la selección.")
    st.stop()

# ---------------------------------------------------------------------- metricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Noticias recolectadas", f"{len(datos):,}",
          delta=f"{len(filtrado):,} con filtros", delta_color="off")
c2.metric("F1 macro en test", f"{metricas['f1_macro_test']:.2f}",
          delta=f"accuracy {metricas['accuracy_test']:.0%}", delta_color="off")
c3.metric("Concordancia LLM vs. auditoría humana", f"{metricas['llm']['concordancia_auditoria']:.0%}",
          delta=f"{metricas['llm']['aciertos_auditoria']}/{metricas['llm']['n_auditoria']} casos",
          delta_color="off")
c4.metric("Piso zero-shot (F1 macro)", f"{metricas['zero_shot_40']['f1_macro']:.2f}",
          delta=f"{metricas['f1_macro_test'] - metricas['zero_shot_40']['f1_macro']:+.2f} del fine-tuned")

st.divider()

# ---------------------------------------------------------------------- graficos
izq, der = st.columns(2)

with izq:
    st.subheader("Distribución por categoría")
    conteo = filtrado["categoria"].value_counts().rename_axis("categoria").reset_index(name="noticias")
    fig = px.bar(conteo.sort_values("noticias"), x="noticias", y="categoria", orientation="h",
                 color="categoria", color_discrete_map=COLORES, text="noticias",
                 labels={"noticias": "Noticias", "categoria": ""})
    fig.update_layout(template="plotly_white", showlegend=False, height=360)
    st.plotly_chart(fig, width="stretch")

with der:
    st.subheader("¿Coincide el LLM con la sección del diario?")
    acuerdo = (filtrado.assign(coincide=lambda d: d["etiqueta_llm"] == d["etiqueta"])
               .groupby("categoria")["coincide"].mean().mul(100).round(1)
               .rename("acuerdo").reset_index().sort_values("acuerdo"))
    fig = px.bar(acuerdo, x="acuerdo", y="categoria", orientation="h", text="acuerdo",
                 color="categoria", color_discrete_map=COLORES,
                 labels={"acuerdo": "% de noticias donde el LLM coincide con la sección", "categoria": ""})
    fig.update_traces(texttemplate="%{text:.0f}%")
    fig.update_layout(template="plotly_white", showlegend=False, height=360, xaxis_range=[0, 105])
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------- tabla
st.subheader("Explorar las noticias")
tabla = filtrado[["fecha", "categoria", "titular", "bajada", "seccion_original", "etiqueta_llm",
                  "confianza", "medio", "url"]].sort_values("fecha", ascending=False)
st.dataframe(
    tabla,
    width="stretch",
    hide_index=True,
    column_config={
        "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
        "url": st.column_config.LinkColumn("Enlace", display_text="abrir"),
        "confianza": st.column_config.ProgressColumn("Confianza LLM", min_value=0, max_value=1, format="%.2f"),
        "etiqueta_llm": "Etiqueta LLM",
        "seccion_original": "Sección original",
    },
)
st.download_button(
    "Descargar datos filtrados (CSV)",
    tabla.assign(fecha=tabla["fecha"].dt.strftime("%Y-%m-%d")).to_csv(index=False).encode("utf-8"),
    "noticias_filtradas.csv",
    "text/csv",
)

# --------------------------------------------------------------- probador en vivo
st.divider()
st.subheader("🔎 Probador en vivo")
st.write("Escribe o pega un titular nuevo y el clasificador te dirá a qué categoría pertenece.")

with st.form("probador"):
    titular = st.text_input("Titular", placeholder="Ej.: BCR mantiene la tasa de referencia en 4,25 %",
                            key="titular_nuevo")
    enviar = st.form_submit_button("Clasificar", type="primary")

if enviar:
    if not titular.strip():
        st.warning("Escribe un titular primero.")
    else:
        with st.spinner("Cargando el modelo y clasificando..."):
            probs = predecir(titular.strip())
            tipo, _, motivo = cargar_clasificador()
        ganadora = probs.idxmax()
        r1, r2 = st.columns([1, 2])
        r1.metric("Categoría predicha", LEGIBLE[ganadora])
        r1.metric("Probabilidad", f"{probs.max():.1%}")
        if probs.max() < 0.5:
            r1.info("El modelo duda: ninguna categoría supera el 50 %.")
        graf = probs.rename(index=LEGIBLE).rename_axis("categoria").reset_index(name="probabilidad")
        fig = px.bar(graf.sort_values("probabilidad"), x="probabilidad", y="categoria", orientation="h",
                     color="categoria", color_discrete_map=COLORES, text="probabilidad",
                     labels={"probabilidad": "Probabilidad", "categoria": ""})
        fig.update_traces(texttemplate="%{text:.1%}")
        fig.update_layout(template="plotly_white", showlegend=False, height=260, xaxis_range=[0, 1.1])
        r2.plotly_chart(fig, width="stretch")
        if tipo == "transformer":
            st.caption(f"Modelo: {metricas['modelo_elegido']} fine-tuned (`{metricas['modelo_base']}`).")
        else:
            st.caption("Modelo de respaldo: TF-IDF + regresión logística (los pesos del Transformer no "
                       f"están disponibles: {motivo}). Ejecuta `02_clasificacion.ipynb` para generarlos.")

with st.expander("Comparativa de modelos en el test reservado"):
    st.dataframe(pd.DataFrame(metricas["comparativa_test"]), width="stretch", hide_index=True)

st.caption("Fuente: La República (larepublica.pe), RSS oficial y HTML. Elaboración propia.")
