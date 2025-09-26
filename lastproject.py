#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import joblib


# Preprocessing 
def handle_missing(X):
    imputer = SimpleImputer(strategy="most_frequent")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

def remove_duplicates(X):
    return X.drop_duplicates()

def encode_categorical(X):
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
    return X

def scale_numeric(X):
    scaler = RobustScaler()
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X


# Modeling 
def run_classification(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM": SVC(C=params["svm_c"], gamma=params["svm_gamma"], probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=params["rf_n_estimators"], random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=params["dt_max_depth"], random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=params["knn_k"])
    }

    results, reports, cms = {}, {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = accuracy
        reports[name] = classification_report(y_test, preds, output_dict=True)
        cms[name] = confusion_matrix(y_test, preds)

    best_acc = max(results.values())
    best_models = [n for n, acc in results.items() if acc == best_acc]

    return results, reports, cms, models, best_models, X_test, y_test


def run_regression(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "SVM": SVC(C=params["svm_c"], gamma=params["svm_gamma"], probability=True),
        "Decision Tree": DecisionTreeRegressor(max_depth=params["dt_max_depth"], random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=params["rf_n_estimators"], random_state=42)
    }

    results, metrics_details = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        results[name] = r2
        metrics_details[name] = {"R² Score": r2, "MAE": mae, "MSE": mse}

    best_model = max(results, key=results.get)
    return results, metrics_details, models, best_model


def run_clustering(X, params):
    models = {
        "KMeans": KMeans(n_clusters=params["kmeans_clusters"], random_state=42),
        "DBSCAN": DBSCAN(eps=params["dbscan_eps"], min_samples=params["dbscan_min_samples"])
    }

    results, labels_dict = {}, {}

    for name, model in models.items():
        labels = model.fit_predict(X)
        labels_dict[name] = labels
        if len(set(labels)) > 1 and -1 not in labels:
            score = silhouette_score(X, labels)
            results[name] = score

    best_model = max(results, key=results.get) if results else None
    return results, labels_dict, models, best_model


# Save & Load
def save_model(model, filename="best_model.pkl"):
    joblib.dump(model, filename)

def load_model(filename="best_model.pkl"):
    return joblib.load(filename)