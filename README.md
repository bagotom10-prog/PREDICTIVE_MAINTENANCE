# ⚙️ DeepPredict: Industrial Predictive Maintenance with TensorFlow & Deep Learning

An AI-powered Predictive Maintenance system (Industry 4.0) designed to anticipate unexpected mechanical failures through sensor telemetry analysis and Deep Neural Networks (DNN).

---

## 📊 Data Source and Description

The data used to train and evaluate this model comes from the **AI4I 2020 Predictive Maintenance Dataset**, a synthetic yet highly realistic dataset that closely reflects the operating conditions of industrial machinery in real manufacturing environments.

* **Source Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).
* **Industrial Context:** The dataset simulates telemetry readings extracted from 10,000 machines on a production line, logging a total of 10,000 records with multiple physical and operational variables.
* **Input Features:**
  * `Air temperature [K]`: Ambient temperature recorded around the machine environment.
  * `Process temperature [K]`: Internal operating temperature of the process.
  * `Rotational speed [rpm]`: Rotational speed of the motor spindle.
  * `Torque [Nm]`: Torsion effort applied during operation.
  * `Tool wear [min]`: Cumulative wear time of the cutting tool.
* **Target Variable:**
  * `Machine failure`: Binary label (`0` for normal operation, `1` for impending failure due to various mechanical causes such as tool wear, overheating, or power failures).

---

## 🏗️ Technical Architecture and Data Pipeline

This project avoids simplistic black-box models, implementing a complete Machine Learning and Deep Learning engineering pipeline:

1. **Preprocessing and Standardization:**
   * Uses Scikit-Learn's `StandardScaler` to normalize continuous variables, transforming them into a distribution with zero mean and unit variance.
   * **Data Leakage Prevention:** The scaler is fitted strictly on the training set and independently applied (`transform`) to the test set and real-time inference inputs.
2. **Deep Neural Network Architecture (TensorFlow / Keras):**
   * **Input Layer:** Configured to receive the 5 normalized sensor features.
   * **Hidden Layers (Dense):** Two densely connected layers with 64 and 32 neurons respectively, utilizing **ReLU** activation functions to introduce non-linearity for complex pattern recognition.
   * **Regularization (Dropout):** Dropout layers (0.3 and 0.2) interleaved to randomly drop neurons during training, drastically mitigating *overfitting*.
   * **Output Layer:** A single neuron with a **Sigmoid** activation function, compressing the output into a continuous probability between 0 and 1.
3. **Optimization and Training:**
   * **Optimizer:** Adam (Adaptive Moment Estimation).
   * **Loss Function:** `binary_crossentropy`, optimal for binary classification tasks.
   * **Early Stopping:** Active monitoring of validation loss (`val_loss`) with a patience margin to halt training at the optimal point and automatically restore the best weights.

---

## 🚀 Streamlit Web Interface Features
* **Dynamic Parameter Reading:** Interface sliders automatically fetch minimum, maximum, and mean values directly from the raw dataset, adapting dynamically.
* **Real-Time Inference:** Calculates the machine risk index in milliseconds after processing inputs through the serialized `scaler.pkl` and `.keras` neural network.
* **Operational Visual Alerts:** Interactive success or critical warning panels based on a 50% probability threshold.

---

## 🛠️ Technology Stack
* **Python** (Core programming language)
* **TensorFlow & Keras** (Neural network construction and training)
* **Scikit-Learn** (Data standardization and stratified data splitting)
* **Pandas & NumPy** (Tabular data manipulation and analysis)
* **Streamlit** (Dashboard development and web deployment)
* **Joblib** (Artifact and model persistence)

---


## 📂 Repository Structure
```text
TENSORFLOW_MAINTENANCE/
├── data/
│   └── ai4i2020.csv         # Industrial sensor dataset (UCI/Kaggle)
├── models/
│   ├── nn_model.keras       # Trained Deep Neural Network (TensorFlow)
│   └── scaler.pkl           # Serialized StandardScaler object
├── app.py                   # Interactive web application (Streamlit)
├── train.py                 # Preprocessing, training, and evaluation script
├── requirements.txt         # Production environment dependencies
└── README.md                # Detailed technical documentation
