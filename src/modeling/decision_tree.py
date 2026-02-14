from sklearn.tree import DecisionTreeRegressor

def train_decision_tree(X_train, y_train, params=None):
    if params is None:
        params = {
            "random_state": 42,
            "max_depth": 10, # Limita la profundidad del árbol para evitar el sobreajuste
            "min_samples_leaf": 20 # Minimo de muestras en cada hoja para evitar que el arbol se ajuste demasiado a los datos de entrenamiento - overfitting
        }
    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)
    return model

