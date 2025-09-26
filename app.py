# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from lastproject import (
    handle_missing, remove_duplicates, encode_categorical, scale_numeric,
    run_classification, run_regression, run_clustering,
    save_model
)

# ================= Functions =================

def classification_task(X, y):
    params = {
        "svm_c": st.sidebar.slider("SVM - C", 0.01, 50.0, 1.0),
        "svm_gamma": st.sidebar.selectbox("SVM gamma", ["scale", "auto"]),
        "rf_n_estimators": st.sidebar.slider("Random Forest - n_estimators", 5, 200, 100),
        "dt_max_depth": st.sidebar.slider("Decision Tree - max_depth", 1, 20, 5),
        "knn_k": st.sidebar.slider("KNN - n_neighbors", 1, 100, 5)
    }
    results, reports, cms, models, best_models, X_test, y_test = run_classification(X, y, params)

    st.write("### Results Table")
    df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    st.write(df_results)

    fig = px.bar(df_results, x="Model", y="Accuracy", color="Model", text="Accuracy")
    st.plotly_chart(fig)

    if len(best_models) == 1:
        st.success(f"🏆 Best Model: {best_models[0]} with Accuracy {results[best_models[0]]:.2f}")
    else:
        st.warning(f"⚖ Multiple models achieved Accuracy {results[best_models[0]]:.2f}: {', '.join(best_models)}")

    for model_name in best_models:
        st.write(f"### 📑 Classification Report for {model_name}")
        st.dataframe(pd.DataFrame(reports[model_name]).transpose())

        st.write(f"### 🔍 Confusion Matrix for {model_name}")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cms[model_name], annot=True, fmt="d", cmap="RdPu",
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax_cm)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    return models, best_models[0]


def regression_task(X, y):
    params = {
        "svm_c": st.sidebar.slider("SVM - C", 0.01, 50.0, 1.0),
        "svm_gamma": st.sidebar.selectbox("SVM gamma", ["scale", "auto"]),
        "rf_n_estimators": st.sidebar.slider("Random Forest - n_estimators", 10, 200, 100),
        "dt_max_depth": st.sidebar.slider("Decision Tree - max_depth", 1, 20, 5),
    }
    results, metrics_details, models, best_model = run_regression(X, y, params)

    st.write("### 📊 Regression Results")
    df_results = pd.DataFrame(metrics_details).T
    st.dataframe(df_results)

    df_melted = df_results.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
    fig = px.bar(df_melted, x="index", y="Value", color="Metric",
                 barmode="group", text=df_melted["Value"].round(2))
    fig.update_layout(xaxis_title="Model", yaxis_title="Score / Error")
    st.plotly_chart(fig)

    st.success(f"🏆 Best Model: {best_model} with R² {results[best_model]:.2f}")
    return models, best_model


def clustering_task(X):
    params = {
        "kmeans_clusters": st.sidebar.slider("KMeans Clusters", 2, 10, 3),
        "dbscan_eps": st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5),
        "dbscan_min_samples": st.sidebar.slider("DBSCAN min_samples", 1, 20, 5)
    }
    results, labels_dict, models, best_model = run_clustering(X, params)
    st.write("### Clustering Results", results)

    if results:
        st.success(f"🏆 Best Clustering Model: {best_model} with Score {results[best_model]:.2f}")

        if best_model in labels_dict:
            labels = labels_dict[best_model]
            X_vis = X.copy()
            X_vis["Cluster"] = labels

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_vis.drop("Cluster", axis=1))

            df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            df_pca["Cluster"] = labels

            fig = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster",
                             title=f"{best_model} Clustering Visualization (PCA 2D)")
            st.plotly_chart(fig)

    return models, best_model


def save_selected_model(task_type, models, best_model, filename):
    save_model(models[best_model], filename)
    st.success(f"✅ Saved {task_type} Model '{best_model}' as {filename}")


# ================= Main App =================

st.title("🤖 Smart AutoML GUI for Machine Learning Problems")

st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload your file", type=["csv", "xlsx"])

if file:
    file_type = file.name.split('.')[-1].lower()
    data = pd.read_csv(file) if file_type == "csv" else pd.read_excel(file)
    st.write("### Preview of Data", data.head())

    st.sidebar.header("⚙ Problem Setup")
    problem_type = st.sidebar.selectbox("Select Problem Type", ["Supervised", "Unsupervised"])

    if problem_type == "Supervised":
        task_type = st.sidebar.radio("Select Task", ["Regression", "Classification"])
        target = st.sidebar.selectbox("Select Target Column", data.columns)
        features = [col for col in data.columns if col != target]
        X, y = data[features], data[target]
    else:
        task_type = "Clustering"
        X, y = data.copy(), None

    # Preprocessing
    st.header("🛠 Preprocessing")
    if st.checkbox("Handle Missing Values"):
        X = handle_missing(X)
        st.success("✅ Missing values handled!")
    if st.checkbox("Remove Duplicates"):
        before, X = X.shape[0], remove_duplicates(X)
        after = X.shape[0]
        st.success(f"✅ Removed {before - after} duplicate rows!")
    if st.checkbox("Encode Categorical Data"):
        X = encode_categorical(X)
        st.success("✅ Encoding Done!")
    if st.checkbox("Scale Numeric Data"):
        X = scale_numeric(X)
        st.success("✅ Scaling Done!")

    # Run Models
    if st.checkbox("🚀 Run Models"):
        if task_type == "Classification":
            models, best_model = classification_task(X, y)
        elif task_type == "Regression":
            models, best_model = regression_task(X, y)
        elif task_type == "Clustering":
            models, best_model = clustering_task(X)

        # Save & Download
        st.subheader("💾 Save Model")
        model_filename = st.text_input("Enter filename to save model", value="best_model.pkl")

        if st.button("Save Selected Model"):
            save_selected_model(task_type, models, best_model, model_filename)

        if os.path.exists(model_filename):
            with open(model_filename, "rb") as f:
                st.download_button(
                    label="⬇ Download Saved Model",
                    data=f,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )