from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    """
    Entrena un modelo de regresión lineal.

    Parámetros:
    X_train (DataFrame): Características de entrenamiento.
    y_train (Series): Etiquetas de entrenamiento.

    Retorna:
    model (LinearRegression): Modelo de regresión lineal entrenado.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model