"""Dashboard ACA 2 - Aprendizaje Automático (CUN).

Predicción de median_house_value con una red neuronal (MLP) entrenada en el
notebook ACA2_Aprendizaje_Automatico_California_Housing_Redes_Neuronales.ipynb.

Ejecutar con: streamlit run ACA2_Dashboard/app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ACA 2 - California Housing (Red Neuronal)", layout="wide")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def mlp_predict(X, weights):
    """Reproduce el forward-pass del MLP (128-64-32-1, ReLU) entrenado en el
    notebook con Keras/TensorFlow, usando solo NumPy. El Dropout no aparece
    aquí porque es un no-op en inferencia (equivalente a `model.predict`)."""
    h = np.maximum(X @ weights["W1"] + weights["b1"], 0)
    h = np.maximum(h @ weights["W2"] + weights["b2"], 0)
    h = np.maximum(h @ weights["W3"] + weights["b3"], 0)
    out = h @ weights["W4"] + weights["b4"]
    return out.flatten()


@st.cache_resource
def load_artifacts():
    weights = dict(np.load(ARTIFACTS_DIR / "mlp_weights.npz"))
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    ocean_categories = json.loads((ARTIFACTS_DIR / "ocean_categories.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    test_predictions = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    return weights, scaler, feature_columns, ocean_categories, metrics, test_predictions


try:
    weights, scaler, feature_columns, ocean_categories, metrics, test_predictions = load_artifacts()
except FileNotFoundError:
    st.error(
        "No se encontraron los artefactos del modelo. Ejecuta primero el notebook "
        "ACA2_Aprendizaje_Automatico_California_Housing_Redes_Neuronales.ipynb para "
        "generarlos en ACA2_Dashboard/artifacts/."
    )
    st.stop()

TARGET_SCALE = metrics.get("target_scale", 100_000)

st.title("ACA 2 · Predicción del valor de vivienda con Red Neuronal")
st.caption(
    "Perceptrón multicapa (MLP) entrenado sobre el dataset California Housing, "
    "con la misma limpieza y partición de datos usadas en el ACA 1 (Regresión Lineal)."
)

tab_prediccion, tab_desempeno, tab_mapa = st.tabs(
    ["Predicción interactiva", "Desempeño del modelo", "Mapa de predicciones"]
)

with tab_prediccion:
    st.subheader("Características del bloque censal")
    st.caption(
        "Estos valores describen un bloque censal completo, no una sola vivienda "
        "(así está definido el dataset original): `total_rooms`, `total_bedrooms`, "
        "`population` y `households` son sumas de todas las viviendas del bloque."
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = st.slider("Longitud", -124.35, -114.31, -119.57, 0.01)
        latitude = st.slider("Latitud", 32.54, 41.95, 35.63, 0.01)
        housing_median_age = st.slider("Antigüedad mediana de la vivienda (años)", 1, 52, 29)

    with col2:
        total_rooms = st.number_input("Total de habitaciones", min_value=1, max_value=40_000, value=2635)
        total_bedrooms = st.number_input("Total de dormitorios", min_value=1, max_value=6_500, value=538)
        population = st.number_input("Población", min_value=1, max_value=36_000, value=1425)

    with col3:
        households = st.number_input("Hogares", min_value=1, max_value=6_100, value=500)
        median_income = st.slider("Ingreso mediano (decenas de miles de USD)", 0.5, 15.0, 3.87, 0.01)
        ocean_proximity = st.selectbox("Proximidad al océano", ocean_categories)

    # Coherencia entre las variables agregadas: verificamos antes de predecir, no después.
    errores_coherencia = []
    if total_bedrooms > total_rooms:
        errores_coherencia.append(
            f"Los dormitorios ({total_bedrooms:,}) no pueden superar el total de habitaciones ({total_rooms:,})."
        )
    if households > population:
        errores_coherencia.append(
            f"No puede haber más hogares ({households:,}) que personas ({population:,})."
        )

    personas_por_hogar = population / households
    dormitorios_por_habitacion = total_bedrooms / total_rooms
    habitaciones_por_hogar = total_rooms / households

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Personas por hogar", f"{personas_por_hogar:.2f}")
    info_col2.metric("Habitaciones por hogar", f"{habitaciones_por_hogar:.2f}")
    info_col3.metric("Dormitorios / habitaciones", f"{dormitorios_por_habitacion:.0%}")
    st.caption(
        "Como referencia, en el dataset original estos valores suelen rondar "
        "2-4 personas por hogar, 4-7 habitaciones por hogar y 15-20% de dormitorios sobre el total de habitaciones."
    )

    if errores_coherencia:
        for mensaje in errores_coherencia:
            st.warning(mensaje)
        st.button("Predecir valor de la vivienda", type="primary", disabled=True)
        st.caption("Corrige los valores señalados arriba para habilitar la predicción.")
    elif st.button("Predecir valor de la vivienda", type="primary"):
        input_row = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
        }
        for category in ocean_categories:
            input_row[f"ocean_proximity_{category}"] = 1 if category == ocean_proximity else 0

        input_df = pd.DataFrame([input_row]).reindex(columns=feature_columns, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction_scaled = mlp_predict(input_scaled, weights)[0]
        prediction_usd = prediction_scaled * TARGET_SCALE

        st.metric("Valor estimado de la vivienda", f"USD {prediction_usd:,.0f}")
        st.caption(
            "Estimación generada por la red neuronal del ACA 2. No sustituye un avalúo "
            "profesional ni debe usarse como único criterio para decisiones financieras."
        )

with tab_desempeno:
    st.subheader("Comparación Regresión Lineal (ACA 1) vs Red Neuronal (ACA 2)")

    baseline = metrics["regresion_lineal_aca1"]
    nn = metrics["red_neuronal_aca2"]

    comparison_df = pd.DataFrame({
        "Métrica": ["R² prueba", "RMSE prueba (USD)", "MAE prueba (USD)"],
        "Regresión Lineal (ACA 1)": [baseline["r2_test"], baseline["rmse_test_usd"], baseline["mae_test_usd"]],
        "Red Neuronal (ACA 2)": [nn["r2_test"], nn["rmse_test_usd"], nn["mae_test_usd"]],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        "R² de prueba (NN)", f"{nn['r2_test']:.3f}", f"{nn['r2_test'] - baseline['r2_test']:+.3f}"
    )
    metric_col2.metric(
        "RMSE de prueba (NN)",
        f"USD {nn['rmse_test_usd']:,.0f}",
        f"{nn['rmse_test_usd'] - baseline['rmse_test_usd']:+,.0f}",
    )
    metric_col3.metric(
        "MAE de prueba (NN)",
        f"USD {nn['mae_test_usd']:,.0f}",
        f"{nn['mae_test_usd'] - baseline['mae_test_usd']:+,.0f}",
    )

    st.subheader("Valores reales frente a predichos (conjunto de prueba)")
    st.scatter_chart(test_predictions, x="valor_real_usd", y="valor_predicho_usd")

with tab_mapa:
    st.subheader("Distribución geográfica del error de predicción")
    map_data = test_predictions.rename(columns={"latitude": "lat", "longitude": "lon"}).copy()
    map_data["error_absoluto_usd"] = (map_data["valor_real_usd"] - map_data["valor_predicho_usd"]).abs()
    st.map(map_data[["lat", "lon"]], size=20)
    st.caption(
        "Cada punto es un bloque censal del conjunto de prueba. La pestaña de desempeño "
        "muestra el detalle numérico del error asociado."
    )
    st.dataframe(
        map_data[["lat", "lon", "valor_real_usd", "valor_predicho_usd", "error_absoluto_usd"]]
        .sort_values("error_absoluto_usd", ascending=False)
        .head(15),
        use_container_width=True,
        hide_index=True,
    )
