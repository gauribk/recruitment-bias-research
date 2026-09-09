
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from preprocessing import load_and_split, encode_categoricals, CATEGORICAL_COLS_FULL

RANDOM_SEED = 42

def train_and_evaluate(X_train, y_train, X_val, y_val):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED)
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)

        results.append({"Model": name, "Accuracy": acc, "ROC-AUC": auc})
        fitted_models[name] = model

        print(f"{name}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")

    return pd.DataFrame(results), fitted_models


if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split("data/raw/job_applicant_dataset.csv")

    X_train_A, X_val_A, X_test_A, encoder_A = encode_categoricals(
        train_df, val_df, test_df, CATEGORICAL_COLS_FULL
    )

    y_train = train_df["Best Match"].reset_index(drop=True)
    y_val = val_df["Best Match"].reset_index(drop=True)

    print("=== Experiment A (with sensitive attributes) ===")
    results_A, models_A = train_and_evaluate(X_train_A, y_train, X_val_A, y_val)

    results_A.to_csv("results/experimentA_baseline_results.csv", index=False)

    for name, model in models_A.items():
        filename = name.lower().replace(" ", "_")
        joblib.dump(model, f"results/model_experimentA_{filename}.joblib")
