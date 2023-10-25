import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def split_data(X, Y, percentage):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=percentage, random_state=42)
    return X_train, X_test, y_train, y_test


def get_model(model_name):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge Regression":
        model = make_pipeline(StandardScaler(), Ridge(random_state=42))
    elif model_name == "Lasso Regression":
        model = Lasso(random_state=42)
    elif model_name == "Elastic Net Regression":
        model = ElasticNet(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(random_state=42)
    elif model_name == "LightGBM":
        model = LGBMRegressor(random_state=42)
    elif model_name == "CatBoost":
        model = CatBoostRegressor(verbose=False, random_state=42)
    elif model_name == "KNN":
        model = KNeighborsRegressor(n_neighbors=15, metric="cosine")
    else:
        raise ValueError("Not implemented")
    return model


def get_score(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
