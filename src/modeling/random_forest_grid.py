from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_random_forest_grid(X_train, y_train, param_grid, cv=3):
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), # El modelo base que queremos optimizar
        param_grid=param_grid, 
        scoring="neg_root_mean_squared_error", # Utiliza el error cuadratico medio negativo como métrica de evaluación
        cv=cv, # Numero de folds para la validación cruzada
        n_jobs=1,  # Utiliza todos los nucleos disponibles para acelerar la búsqueda de hiperparametros
        verbose=0 # Muestra el progreso de la búsqueda en la consola
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_