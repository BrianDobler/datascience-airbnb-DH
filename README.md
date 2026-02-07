# datascience-airbnb-DH
Proyecto final de Data Science basado en datos de Airbnb desarrollado para el curso de Digital House. Incluye procesos de EDA y ETL para limpieza y transformación de datos, y la implementación de modelos de ML y Deep Learning orientados a analizar y predecir precios, demanda y patrones de comportamiento en las publicaciones de la plataforma.

## Datos

Los datasets crudos completos (listings.csv, calendar.csv y reviews.csv) no se incluyen en este repositorio por su tamaño.
Fueron provistos por Digital House y pueden descargarse desde:

👉 https://drive.google.com/drive/folders/1xKTfaKZDYLadXDW8H-HKG8FZ6vXaLdul?usp=sharing

Una vez descargados, colóquelos en:

data/
  raw/
    listings.csv
    calendar.csv
    reviews.csv

En caso de querer descargar los datos automáticamente, ejecutar:


pip install gdown
python src/download_data.py