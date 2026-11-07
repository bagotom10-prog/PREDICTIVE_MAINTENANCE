# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import streamlit as st   # Librería principal para crear la interfaz web interactiva
import pandas as pd      # Para estructurar los datos de entrada en formato tabular
import numpy as np             # Para operaciones y manejo de arrays numéricos
import tensorflow as tf  # Para cargar y ejecutar el modelo de Deep Learning
import joblib            # Para cargar el objeto 'scaler.pkl' guardado previamente

# ==========================================
# 1. CARGA DE ARTEFACTOS (MODELO Y SCALER)
# ==========================================
# Usamos el decorador @st.cache_resource para que Streamlit cargue el modelo y el escalador 
# en la memoria RAM una sola vez, optimizando la velocidad de la aplicación web.
@st.cache_resource
def load_artifacts():
    # Cargamos la red neuronal entrenada en formato .keras
    model = tf.keras.models.load_model("models/nn_model.keras")
    
    # Cargamos el escalador StandardScaler para transformar los datos nuevos exactamente igual que en el entrenamiento
    scaler = joblib.load("models/scaler.pkl")
    
    return model, scaler

# Ejecutamos la función para obtener el modelo y el escalador listos para usar
model, scaler = load_artifacts()

# ==========================================
# 2. CONFIGURACIÓN VISUAL DE LA PÁGINA WEB
# ==========================================
# Configuramos el título de la pestaña del navegador, el icono y el diseño centrado
st.set_page_config(
    page_title="DeepPredict - Mantenimiento con IA", 
    page_icon="🧠", 
    layout="centered"
)

# Título principal y descripción visible en la aplicación web
st.title("🧠 DeepPredict: Mantenimiento Industrial con TensorFlow")
st.markdown("Sistema inteligente impulsado por una **Red Neuronal Profunda (DNN)** construida desde cero con Keras/TensorFlow para predecir fallos mecánicos en tiempo real.")

# ==========================================
# 3. DISEÑO DEL PANEL DE CONTROL (SIDEBAR)
# ==========================================
st.sidebar.header("🎛️ Panel de Sensores")
st.sidebar.markdown("Modifica los valores para simular el estado operativo de la maquinaria:")

# Cargar el dataset para extraer los valores estadísticos de los sensores
df = pd.read_csv("data/ai4i2020.csv")

# Controles deslizantes (Sliders) para simular las lecturas físicas de los sensores en tiempo real
# Los parámetros (mínimo, máximo y valor medio) están basados en los rangos del dataset industrial

# --- Temperatura Ambiente ---
air_temp = st.sidebar.slider(
    "Temperatura Ambiente (K)", 
    min_value=float(df['Air temperature [K]'].min()), 
    max_value=float(df['Air temperature [K]'].max()), 
    value=float(df['Air temperature [K]'].mean())
)
# --- Temperatura de Proceso ---
process_temp = st.sidebar.slider(
    "Temperatura de Proceso (K)", 
    min_value=float(df['Process temperature [K]'].min()), 
    max_value=float(df['Process temperature [K]'].max()), 
    value=float(df['Process temperature [K]'].mean())
)
# --- Velocidad de Rotación ---
rot_speed = st.sidebar.slider(
    "Velocidad de Rotación (rpm)", 
    min_value=int(df['Rotational speed [rpm]'].min()), 
    max_value=int(df['Rotational speed [rpm]'].max()), 
    value=int(df['Rotational speed [rpm]'].mean())
)
# --- Torque / Esfuerzo ---
torque = st.sidebar.slider(
    "Torque / Esfuerzo (Nm)", 
    min_value=float(df['Torque [Nm]'].min()), 
    max_value=float(df['Torque [Nm]'].max()), 
    value=float(df['Torque [Nm]'].mean())
)
# --- Desgaste de Herramienta ---
tool_wear = st.sidebar.slider(
    "Desgaste de Herramienta (min)", 
    min_value=int(df['Tool wear [min]'].min()), 
    max_value=int(df['Tool wear [min]'].max()), 
    value=int(df['Tool wear [min]'].mean())
)

# ==========================================
# 4. ORGANIZACIÓN DE LOS DATOS DE ENTRADA
# ==========================================
# Agrupamos los valores seleccionados en los sliders dentro de un DataFrame de pandas 
# respetando exactamente el mismo orden de columnas con el que se entrenó la red neuronal.
input_data = pd.DataFrame({
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rot_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear]
})

# Mostramos una tabla interactiva en la interfaz web con los datos actuales que el usuario configuró
st.subheader("📊 Lecturas actuales enviadas a la Red Neuronal")
st.dataframe(input_data)

# ==========================================
# 5. LÓGICA DE PREDICCIÓN E INFERENCIA
# ==========================================
# Botón principal para disparar el diagnóstico de la máquina
if st.button("Ejecutar Inferencia con TensorFlow", type="primary"):
    
    # Paso A: Escalamos los datos de entrada usando el scaler previamente entrenado (¡vital!)
    scaled_input = scaler.transform(input_data.values)
    
    # Paso B: Pasamos los datos escalados al modelo de TensorFlow para que calcule la predicción
    # model.predict devuelve una matriz; extraemos el valor escalar de probabilidad entre 0 y 1.
    prediction_prob = model.predict(scaled_input)[0][0] 
    
    st.markdown("---")
    
    # Paso C: Evaluamos la probabilidad resultante y mostramos alertas visuales interactivas
    if prediction_prob > 0.5:
        # Si la probabilidad supera el 50%, la red neuronal advierte riesgo de fallo
        st.error(f"⚠️ **¡ALERTA: FALLO PREDICTIVO DETECTADO!**")
        st.metric(label="Índice de Riesgo (Probabilidad)", value=f"{prediction_prob * 100:.2f}%")
        st.markdown("La red neuronal ha identificado patrones anómalos en los sensores similares a fallos históricos. Se requiere inspección.")
    else:
        # Si está por debajo del 50%, la máquina se encuentra en un estado saludable
        st.success(f"✅ **Estado Óptimo de la Maquinaria**")
        st.metric(label="Índice de Riesgo (Probabilidad)", value=f"{prediction_prob * 100:.2f}%")
        st.markdown("La maquinaria opera dentro de los umbrales seguros estimados por el modelo.")