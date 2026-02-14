from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, params=None):
   
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        } 
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model