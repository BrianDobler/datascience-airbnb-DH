from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def train_model_xgb_grid(X_train, y_train, param_grid, cv=3):

    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=1
    )

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="r2",  
        cv=cv,
        n_jobs=-1,     
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_
