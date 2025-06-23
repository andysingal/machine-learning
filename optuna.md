```py
pip install mlflow datasets scikit-learn xgboost optuna matplotlib

import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import pandas as pd

# Load modern dataset (HuggingFace)
dataset = load_dataset("imdb")
df = pd.DataFrame(dataset["train"])
df = df.sample(5000, random_state=42)  # Keep small for demo
X = df["text"]
y = df["label"]

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer(max_features=1000).fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Track experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IMDB Sentiment Classification")

def objective(trial):
    with mlflow.start_run():
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 3, 20)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

mlflow models serve -m runs:/<run-id>/model -p 5001

```
