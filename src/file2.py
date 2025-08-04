import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os
import inspect

# 🔗 Initialize DagsHub connection
dagshub.init(repo_owner='SydBurhan', repo_name='Learning-MLFLow', mlflow=True)

# 📍 Set tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/SydBurhan/Learning-MLFLow.mlflow")

# 📊 Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 🧪 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# 🎯 Model params
max_depth = 10
n_estimators = 8

# 🧪 Create or use an experiment
mlflow.set_experiment("YouTube MLOps Exp2")

with mlflow.start_run():
    # 🛠️ Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=43)
    rf.fit(X_train, y_train)

    # ✅ Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 📝 Log metrics and params
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', accuracy)

    # 📊 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    # 🧾 Log current script as artifact
    current_script = inspect.getfile(inspect.currentframe())
    mlflow.log_artifact(current_script)

    # 🏷️ Add tags
    mlflow.set_tags({
        "Author": "Burhan",
        "Project": "Wine classification"
    })

    # 📦 Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(f"Accuracy: {accuracy:.4f}")
