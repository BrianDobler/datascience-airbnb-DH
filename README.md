# 🏡 datascience-airbnb-DH

Proyecto final de Data Science desarrollado en el marco del curso de Digital House.
El objetivo es analizar el mercado de Airbnb y proponer recomendaciones de inversión basadas en análisis descriptivo y modelado predictivo.

Incluye:

📊 Análisis Exploratorio de Datos (EDA)

🧹 Limpieza y Transformación (ETL)

🤖 Modelos de Machine Learning

🧠 Modelos de Deep Learning (MLP y LSTM)

📈 Forecast de precios promedio diarios

## Instalacion del proyecto
🔧 Requisitos

Python 3.10 o superior
pip actualizado

## 📥 Clonar el repositorio

git clone https://github.com/BrianDobler/datascience-airbnb-DH.git

cd datascience-airbnb-DH

## 📦 Instalar dependencias

### pip install -r requirements.txt

## 📂 Descarga de los datos

Los datasets crudos completos (listings.csv, calendar.csv y reviews.csv) no se incluyen en este repositorio por su tamaño. Los mismos Fueron provistos por Digital House.
Para descargar los dataset de manera automáticamente, ejecutar:

### pip install gdown
### python src/download_data.py

O bien descargarlos de:

👉 https://drive.google.com/drive/folders/1xKTfaKZDYLadXDW8H-HKG8FZ6vXaLdul?usp=sharing

Una vez descargados, colóquelos en:

data/
  raw/
    listings.csv
    calendar.csv
    reviews.csv

## 📁 Estructura del proyecto 

<img width="310" height="468" alt="image" src="https://github.com/user-attachments/assets/ff3ccb2a-662d-4488-9bb8-7d7a66182e4a" />


▶️ Orden de Ejecución

Para reproducir el análisis completo:

1️⃣ etapa_01_eda.ipynb
2️⃣ etapa_02_etl.ipynb
3️⃣ etapa_03_ml.ipynb
4️⃣ etapa_04_dl_mlp.ipynb
5️⃣ etapa_04.1_dl_lstm.ipynb

📌 Resultados Destacados

✅ XGBoost optimizado fue el modelo con mejor desempeño para datos tabulares.

📉 Se identificaron patrones de precio según tipo de propiedad y ubicación.

📈 La LSTM permitió modelar la dinámica temporal y generar forecast a 7 días.
