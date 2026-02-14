from xgboost import XGBRegressor

def train_xgboost(X_train,y_train):
    model = XGBRegressor(
        n_estimators= 400,
        max_depth = 6,
        learning_rate=0.05,
        subsample= 0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train,y_train)
    return model