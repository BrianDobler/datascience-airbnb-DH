from xgboost import XGBRegressor

def train_xgboost(X_train,y_train):
    model = XGBRegressor(
        n_estimators= 500,
        max_depth = 3,
        learning_rate=0.1,
        subsample= 0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train,y_train)
    return model