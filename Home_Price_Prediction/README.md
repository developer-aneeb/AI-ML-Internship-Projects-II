# 🏡 Multimodal Housing Price Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> **"A deep learning architecture combining structured tabular features with image data to accurately predict housing prices."**

---

## 🎯 1. Objective of the task
The objective of this project is to build a robust multimodal deep learning pipeline that predicts house prices. Unlike standard tabular regression models, this approach leverages **both structured tabular features** (e.g., location, numeric specs) and **unstructured visual data** (e.g., images of the house) to achieve richer property representations and more accurate valuations. The project also generates a deployable Streamlit application for interactive testing.

---

## ⚙️ 2. Methodology / Approach
The pipeline processes two distinct modalities using PyTorch and Scikit-Learn:

- **Data Ingestion & Preprocessing**: 
  - **Tabular Data**: Processed using Scikit-Learn's `ColumnTransformer`. Missing values are imputed (median for numeric, most_frequent for categorical). Numeric features are standardized (`StandardScaler`), and categorical features are encoded (`OneHotEncoder`). The target variable (Price) is log-transformed (`log1p`) to handle heavy-tailed price distributions.
  - **Image Data**: Images are dynamically matched to their tabular records. They are resized to 224x224, normalized, and augmented (Random Horizontal Flips and Rotations) during training using `torchvision.transforms`.
- **Model Architecture**:
  - **Tabular Encoder**: A deep feedforward neural network processes the structured tabular features.
  - **Image Encoder**: A pre-trained `ResNet18` acts as the feature extractor for property images. The convolutional layers are frozen to leverage transfer learning and prevent overfitting.
  - **Multimodal Fusion (`MultiNet`)**: The dense embeddings from both the Tabular and Image encoders are concatenated and passed through a shared fusion head to output the final continuous price prediction.
- **Training Strategy**: 
  - Optimized using `AdamW` and `SmoothL1Loss` to remain robust against price outliers.
  - Learning rate drops are handled dynamically via a `ReduceLROnPlateau` scheduler alongside Early Stopping (`PATIENCE=2`).
  - A baseline Tabular-only model (`TabNet`) is trained side-by-side to quantify the value added by the image features.
- **Deployment Generation**: The notebook saves the trained PyTorch weights (`.pt`), the Scikit-Learn preprocessors (`.joblib`), and automatically generates a functional `streamlit_app.py` script for local inference.

---

## 📊 3. Key results or observations
- **Comparative Evaluation**: The pipeline compares the Multimodal Fusion network against a Tabular-only baseline using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics, directly highlighting the impact of visual features on real estate valuation.
- **Visual Analytics**: The notebook tracks model convergence by plotting training and validation loss curves, and generates "Actual vs Predicted" scatter plots to visually verify prediction tightness along the ideal trajectory line.
- **Robust Generalization**: The implementation of Early Stopping and `ReduceLROnPlateau` effectively prevented the fusion network from memorizing the training set, maintaining a minimal gap between train and validation losses.
- **Instant Deployment Artifacts**: The workflow guarantees reproducibility by serializing all models and preprocessing steps. It outputs a complete Streamlit dashboard (`streamlit_app.py`) capable of loading the `ResNet18` multimodal architecture and displaying live predictions alongside the raw house image.