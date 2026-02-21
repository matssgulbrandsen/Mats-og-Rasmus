from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

FEATURES = [
    "is_prebook", "is_weekend", "poi_id", "da_id",
    "sender_lat", "sender_lng", "recipient_lat", "recipient_lng",
    "hour", "day_of_week", "is_peak",
    "merchant_customer_distance_km",
    "estimated_prep_time_minutes",
    "estimated_delivery_duration_minutes", "avg_orders_per_wave"
]
TARGET = "is_late"


def load_data(filtered_filename):
    project_root = Path(__file__).parent.parent
    path = project_root / "data" / filtered_filename
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def time_based_split(df):
    df = df.sort_values("platform_order_time")
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    print(f"Train: {len(train)} rader, Test: {len(test)} rader")
    return train, test


def tune_models(train, test):
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grids = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear"]
            },
            "X_train": X_train_scaled,
            "X_test": X_test_scaled
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_leaf": [1, 5, 10]
            },
            "X_train": X_train,
            "X_test": X_test
        },
        "xgboost": {
            "model": XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=6),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 1.0]
            },
            "X_train": X_train,
            "X_test": X_test
        }
    }

    trained = {}
    for name, config in param_grids.items():
        print(f"Tuner {name}...")
        search = RandomizedSearchCV(
            config["model"],
            config["params"],
            n_iter=min(20, len(config["params"])),  # ikke mer enn antall kombinasjoner
            scoring="recall",
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        search.fit(config["X_train"], y_train)
        print(f"  Beste params: {search.best_params_}")
        print(f"  Beste recall (CV): {search.best_score_:.2%}")

        scaler_out = scaler if name == "logistic_regression" else None
        trained[name] = (search.best_estimator_, scaler_out, config["X_test"], y_test)

    return trained


def save_models(trained):
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "results" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (model, scaler, _, _) in trained.items():
        joblib.dump(model, out_dir / f"{name}.pkl")
        if scaler:
            joblib.dump(scaler, out_dir / f"{name}_scaler.pkl")
    print(f"Modeller lagret til: {out_dir}")


if __name__ == "__main__":
    df = load_data("filtered_all_waybill_info_meituan_0322.csv")
    train, test = time_based_split(df)
    trained = tune_models(train, test)
    save_models(trained)