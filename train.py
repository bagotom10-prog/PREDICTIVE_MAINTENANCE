# ==========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ==========================================
import pandas as pd  # Para manipulación y lectura de datos tabulares (CSV)
import numpy as np   # Para operaciones numéricas y manejo de arrays
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler    # Para normalizar/escalar las variables numéricas
from tensorflow.keras.models import Sequential       # Para crear modelos de red neuronal capa por capa secuencialmente
from tensorflow.keras.layers import Dense, Dropout   # Dense = Capas totalmente conectadas; Dropout = Técnica anti-overfitting
from tensorflow.keras.callbacks import EarlyStopping # Para detener el entrenamiento si el modelo deja de mejorar
import joblib     # Para guardar el objeto 'scaler' en disco y usarlo en la web
import matplotlib.pyplot as plt  # Para graficar la evolución de la pérdida y precisión durante el entrenamiento

# ==========================================
# 1. CARGA Y PREPARACIÓN DE LOS DATOS
# ==========================================
print("Cargando datos...")
# Leemos el archivo CSV de sensores industriales ubicado en la carpeta 'data/'
df = pd.read_csv("data/ai4i2020.csv")

# Definimos cuáles columnas de la tabla son las variables de entrada (características de los sensores)
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Definimos la columna objetivo que queremos predecir: 0 (Funcionamiento normal) o 1 (Fallo inminente)
target = 'Machine failure'

# Extraemos los valores de las características en una matriz X y la variable objetivo en un vector y
X = df[features].values
y = df[target].values

# ==========================================
# 2. DIVISIÓN DEL DATASET (TRAIN / TEST)
# ==========================================
# Dividimos los datos: 80% para entrenar a la red neuronal y 20% para evaluar su rendimiento real.
# stratify=y asegura que la proporción de fallos (1s y 0s) sea igual en ambos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# ==========================================
# 3. NORMALIZACIÓN DE LOS DATOS (OBLIGATORIO EN NN)
# ==========================================
# Las redes neuronales sufren si las escalas de los datos son muy diferentes (ej: RPM en miles vs Desgaste en decenas).
# StandardScaler centra los datos con media 0 y desviación estándar 1.
scaler = StandardScaler()

# Ajustamos el escalador ÚNICAMENTE con los datos de entrenamiento y los transformamos
X_train_scaled = scaler.fit_transform(X_train)

# Transformamos los datos de prueba usando los mismos parámetros calculados en el entrenamiento (evita fuga de datos). Por eso usamos transform() en lugar de fit_transform().
X_test_scaled = scaler.transform(X_test)

# Guardamos el escalador en un archivo .pkl en la carpeta 'models/' para usarlo idéntico en la app web de Streamlit
joblib.dump(scaler, "models/scaler.pkl")

# ==========================================
# 4. DISEÑO DE LA ARQUITECTURA DE LA RED NEURONAL
# ==========================================
print("Construyendo la arquitectura de la Red Neuronal...")

# Sequential indica que las capas se apilarán una encima de otra en orden secuencial
model = Sequential([
    # Primera capa oculta (Dense): 64 neuronas con función de activación ReLU (introduce no linealidad).
    # input_shape recibe el número de columnas de entrada (en este caso, 5 características de sensores), que es el numero de features de entrada. El numero de
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),

    # Dropout(0.3): Apaga aleatoriamente el 30% de las neuronas durante cada época de entrenamiento 
    # para forzar a la red a no memorizar los datos (evita el sobreajuste u 'overfitting').
    Dropout(0.3),
    
    # Segunda capa oculta (Dense): 32 neuronas con activación ReLU para procesar patrones más complejos.
    Dense(32, activation='relu'),
    
    # Dropout(0.2): Apaga el 20% de las neuronas de esta capa por seguridad adicional.
    Dropout(0.2),
    
    # Capa de salida (Dense): 1 sola neurona con función de activación Sigmoide.
    # La sigmoide comprime cualquier valor entre 0 y 1, actuando exactamente como una probabilidad de fallo.
    Dense(1, activation='sigmoid')
])

# ==========================================
# 5. COMPILACIÓN DEL MODELO
# ==========================================
# Configuramos cómo va a aprender la red:
# - optimizer='adam': El optimizador inteligente estándar que ajusta los pesos de forma óptima.
# - loss='binary_crossentropy': La función de pérdida matemática ideal para clasificación binaria (0 o 1). Si usaramos otra función como softmax, tendriamos que usar categorical_crossentropy y cambiar la capa de salida a N neuronas con softmax.
# - metrics=['accuracy']: Para medir el porcentaje de aciertos durante el entrenamiento. Tambien podriamos usar otras como: precision, recall, f1-score, etc.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 6. CONFIGURACIÓN DE EARLY STOPPING
# ==========================================
# EarlyStopping vigila el error de validación (val_loss). 
# Si durante 5 épocas seguidas (patience=5) el modelo deja de mejorar, detiene el entrenamiento automáticamente 
# y recupera los mejores pesos encontrados (restore_best_weights=True).
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# ==========================================
# 7. ENTRENAMIENTO DE LA RED NEURONAL
# ==========================================
print("Entrenando el modelo con TensorFlow...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,                  # Número máximo de pasadas completas por todo el dataset
    batch_size=32,              # Número de muestras procesadas antes de actualizar los pesos internos
    validation_split=0.2,       # Reserva un 20% de los datos de entrenamiento para validar en tiempo real
    callbacks=[early_stop],     # Aplica la regla de parada temprana si deja de aprender
    verbose=1                   # Muestra en la terminal una barra de progreso por cada época
)

# ==========================================
# 8. EVALUACIÓN Y GUARDADO DEL MODELO
# ==========================================
# Evaluamos el rendimiento final de la red neuronal utilizando los datos de test que la red nunca había visto
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\n✅ Precisión del modelo en test: {accuracy * 100:.2f}%")

# Guardamos la estructura, pesos y configuración completa del modelo de TensorFlow en formato moderno (.keras). Es como el cerebro de la red neuronal que luego podremos cargar en la app web para hacer predicciones en tiempo real.
model.save("models/nn_model.keras")
print("¡Modelo guardado con éxito en models/nn_model.keras!")

# ==========================================
# 8. GRAFICOS DE PÉRDIDA Y PRECISIÓN DURANTE EL ENTRENAMIENTO
# ==========================================

# Creamos una figura con dos gráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico de Pérdida (Loss)
ax1.plot(history.history['loss'], label='Pérdida en Entrenamiento (Train Loss)')
ax1.plot(history.history['val_loss'], label='Pérdida en Validación (Val Loss)')
ax1.set_title('Evolución del Error (Loss)')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Gráfico de Precisión (Accuracy)
ax2.plot(history.history['accuracy'], label='Precisión en Entrenamiento (Train Acc)')
ax2.plot(history.history['val_accuracy'], label='Precisión en Validación (Val Acc)')
ax2.set_title('Evolución de la Precisión (Accuracy)')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()