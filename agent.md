
# agent.md — Guía de organización del proyecto

Este documento tiene como objetivo describir **cómo se organiza y estructura el proyecto** de Data Science.
Funciona como una referencia interna para mantener consistencia a lo largo del desarrollo.

La intención de este archivo es ordenar el trabajo y dejar explícitos ciertos criterios adoptados durante el proyecto.

---

## Objetivo general

Desarrollar un proyecto de Data Science aplicado a un caso real (Airbnb), recorriendo las distintas etapas del proceso:
- análisis exploratorio de datos
- limpieza y transformación
- modelado con técnicas de Machine Learning
- experimentación con modelos de Deep Learning
- documentación del proceso

---

## Criterios de trabajo

- Los datos originales se mantienen sin modificaciones.
- El proceso debe ser reproducible.
- Las decisiones técnicas se documentan en los notebooks.
- El código reutilizable se separa del análisis exploratorio.

---

## Estructura del proyecto

```text
airbnb-ml-project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── README.md
└── agent.md
```

---

## data/

### raw/
Contiene los archivos de datos originales descargados.
Estos archivos se utilizan únicamente como entrada del proceso.

### processed/
Incluye los datasets resultantes del proceso de limpieza y transformación.
Estos archivos se usan para el entrenamiento y evaluación de modelos.

---

## notebooks/

Los notebooks se utilizan para desarrollar y documentar el análisis.

- `01_eda.ipynb`: análisis exploratorio y visualizaciones iniciales.
- `02_etl.ipynb`: limpieza de datos y creación de variables.
- `03_ml.ipynb`: entrenamiento y evaluación de modelos de Machine Learning.
- `04_dl.ipynb`: experimentación con modelos de redes neuronales.

El foco de los notebooks está puesto en la explicación del proceso y las decisiones tomadas.

---

## src/

Contiene funciones auxiliares reutilizables utilizadas en el proyecto.
Aquí se centraliza la lógica relacionada con:
- carga de datos
- limpieza
- transformaciones
- creación de variables

El objetivo es evitar duplicación de código en los notebooks.

---

## models/

Se utiliza para almacenar modelos entrenados y objetos necesarios para su utilización posterior.
Permite reutilizar resultados sin necesidad de reentrenar los modelos.

---

## Documentación

- `README.md`: descripción general del proyecto, contexto y forma de ejecución.
- `agent.md`: guía de organización y criterios de trabajo.

---

Este archivo puede actualizarse si la estructura del proyecto cambia o si se agregan nuevas etapas.
