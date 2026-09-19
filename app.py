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


def linear_predict(input_df_raw, linear_model):
    """Predice con la misma Regresión Lineal del ACA 1 (mismo split, mismo
    StandardScaler), reescrita en el espacio de variables sin escalar para no
    necesitar un segundo scaler: coef_crudo = coef_estandarizado / std, con el
    intercepto ajustado para compensar. Ver notebook, sección 8."""
    coef = np.array(linear_model["coefficients"])
    intercept = linear_model["intercept"]
    return input_df_raw.to_numpy() @ coef + intercept


RANGO_FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population", "households", "median_income",
]


@st.cache_resource
def load_artifacts():
    weights = dict(np.load(ARTIFACTS_DIR / "mlp_weights.npz"))
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    ocean_categories = json.loads((ARTIFACTS_DIR / "ocean_categories.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    linear_model = json.loads((ARTIFACTS_DIR / "linear_model.json").read_text(encoding="utf-8"))
    test_predictions = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")

    # Rango típico (percentiles 1-99 del conjunto de prueba) por variable, usado
    # para avisar cuando un input queda fuera de lo que el modelo vio al entrenar:
    # una MLP extrapola mal fuera del rango de sus datos de entrenamiento.
    rangos_tipicos = {
        columna: (
            test_predictions[columna].quantile(0.01),
            test_predictions[columna].quantile(0.99),
        )
        for columna in RANGO_FEATURES
    }

    return (
        weights, scaler, feature_columns, ocean_categories, metrics,
        linear_model, test_predictions, rangos_tipicos,
    )


try:
    (
        weights, scaler, feature_columns, ocean_categories, metrics,
        linear_model, test_predictions, rangos_tipicos,
    ) = load_artifacts()
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

    DEFAULTS = {
        "longitude": -119.57, "latitude": 35.63, "housing_median_age": 29,
        "total_rooms": 2635, "total_bedrooms": 538, "population": 1425,
        "households": 500, "median_income": 3.87, "ocean_proximity": "<1H OCEAN",
    }
    PRESETS = {
        "Costa cara (Bahía de SF)": {
            "longitude": -122.25, "latitude": 37.85, "housing_median_age": 35,
            "total_rooms": 3100, "total_bedrooms": 560, "population": 1000,
            "households": 480, "median_income": 8.5, "ocean_proximity": "NEAR BAY",
        },
        "Interior económico": {
            "longitude": -119.5, "latitude": 36.5, "housing_median_age": 20,
            "total_rooms": 1800, "total_bedrooms": 420, "population": 1200,
            "households": 380, "median_income": 2.0, "ocean_proximity": "INLAND",
        },
    }
    for clave, valor in DEFAULTS.items():
        st.session_state.setdefault(clave, valor)

    st.caption("Escenarios de ejemplo (llenan todos los campos de una vez):")
    preset_cols = st.columns(len(PRESETS) + 1)
    for preset_col, preset_nombre in zip(preset_cols, PRESETS):
        if preset_col.button(preset_nombre):
            for clave, valor in PRESETS[preset_nombre].items():
                st.session_state[clave] = valor
            st.rerun()
    if preset_cols[-1].button("Restablecer valores por defecto"):
        for clave, valor in DEFAULTS.items():
            st.session_state[clave] = valor
        st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = st.slider("Longitud", -124.35, -114.31, step=0.01, key="longitude")
        latitude = st.slider("Latitud", 32.54, 41.95, step=0.01, key="latitude")
        housing_median_age = st.slider("Antigüedad mediana de la vivienda (años)", 1, 52, key="housing_median_age")

    with col2:
        total_rooms = st.number_input("Total de habitaciones", min_value=1, max_value=40_000, key="total_rooms")
        total_bedrooms = st.number_input("Total de dormitorios", min_value=1, max_value=6_500, key="total_bedrooms")
        population = st.number_input("Población", min_value=1, max_value=36_000, key="population")

    with col3:
        households = st.number_input("Hogares", min_value=1, max_value=6_100, key="households")
        median_income = st.slider("Ingreso mediano (decenas de miles de USD)", 0.5, 15.0, step=0.01, key="median_income")
        ocean_proximity = st.selectbox("Proximidad al océano", ocean_categories, key="ocean_proximity")

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

    # Aviso (no bloqueante) cuando un valor queda fuera del rango típico visto en
    # entrenamiento: una red neuronal no "sabe" extrapolar y puede dar una
    # predicción poco confiable ante combinaciones muy atípicas.
    valores_input = {
        "longitude": longitude, "latitude": latitude, "housing_median_age": housing_median_age,
        "total_rooms": total_rooms, "total_bedrooms": total_bedrooms, "population": population,
        "households": households, "median_income": median_income,
    }
    avisos_rango = []
    for columna, valor in valores_input.items():
        p1, p99 = rangos_tipicos[columna]
        if valor < p1 or valor > p99:
            avisos_rango.append(f"`{columna}` = {valor:,.2f} (rango típico: {p1:,.2f} a {p99:,.2f})")

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

    if avisos_rango:
        st.warning(
            "Esta combinación se aleja de lo que el modelo vio al entrenar, la predicción "
            "puede ser poco confiable (extrapolación):\n\n"
            + "\n".join(f"- {aviso}" for aviso in avisos_rango)
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
        prediction_lineal_usd = linear_predict(input_df, linear_model)[0]

        pred_col1, pred_col2 = st.columns(2)
        pred_col1.metric("Regresión Lineal (ACA 1)", f"USD {prediction_lineal_usd:,.0f}")
        pred_col2.metric(
            "Red Neuronal (ACA 2)",
            f"USD {prediction_usd:,.0f}",
            f"{prediction_usd - prediction_lineal_usd:+,.0f} vs. Regresión Lineal",
        )
        st.caption(
            "Ambas estimaciones se calculan con la misma partición de datos usada al entrenar "
            "cada modelo. No sustituyen un avalúo profesional ni deben usarse como único "
            "criterio para decisiones financieras."
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

    # Coloreamos y dimensionamos cada punto según su error: azul (bajo) a rojo (alto).
    # Recortamos en el percentil 95 para que los pocos casos extremos del tope de
    # USD 500.000 no aplasten la escala de color del resto de los puntos.
    error_p95 = map_data["error_absoluto_usd"].quantile(0.95)
    error_min = map_data["error_absoluto_usd"].min()

    def error_a_color(valor: float) -> str:
        rango = max(error_p95 - error_min, 1e-9)
        t = max(0.0, min(1.0, (valor - error_min) / rango))
        r = int(33 + t * (244 - 33))
        g = int(150 + t * (67 - 150))
        b = int(243 + t * (54 - 243))
        return f"#{r:02x}{g:02x}{b:02x}"

    map_data["color"] = map_data["error_absoluto_usd"].apply(error_a_color)
    map_data["tamano_m"] = 600 + (map_data["error_absoluto_usd"].clip(upper=error_p95) / error_p95) * 3000

    st.map(map_data, color="color", size="tamano_m")

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-top:-8px;">
            <span style="font-size:0.85rem;">USD {error_min:,.0f}</span>
            <div style="flex:1; height:10px; border-radius:5px;
                        background:linear-gradient(to right, #2196f3, #f44336);"></div>
            <span style="font-size:0.85rem;">USD {error_p95:,.0f}+</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Cada punto es un bloque censal del conjunto de prueba; el color y el tamaño "
        "representan el error absoluto (escala de azul = bajo a rojo = alto, recortada "
        "en el percentil 95). La pestaña de desempeño muestra el detalle numérico."
    )
    st.dataframe(
        map_data[["lat", "lon", "valor_real_usd", "valor_predicho_usd", "error_absoluto_usd"]]
        .sort_values("error_absoluto_usd", ascending=False)
        .head(15),
        use_container_width=True,
        hide_index=True,
    )
