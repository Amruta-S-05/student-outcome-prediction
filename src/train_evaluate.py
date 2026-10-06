"""Step 2: preprocessing, training, model comparison, final model selection, feature importance.
Run:  python src/train_evaluate.py

Workflow
  1. Split once into train (80%) and test (20%), stratified by Target.
  2. Build ONE pipeline per model: preprocessing + classifier. All learned preprocessing
     (scaling, one-hot categories) is fitted on training data only.
  3. Compare models with 5-fold cross-validation on the TRAINING set (test set not touched).
  4. Fit both models on the training set and evaluate ONCE on the unseen test set.
  5. Select the final model using CV macro F1 (training data only), save it.
  6. Compute feature importance for the final model.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (BINARY_COLUMNS, CATEGORICAL_COLUMNS, CLASS_ORDER, CV_FOLDS, MODEL_DIR,
                    MODEL_PATH, NUMERIC_COLUMNS, OUTPUT_DIR, RANDOM_STATE)
from data_loader import get_train_test_split


def build_preprocessor():
    """Scale numbers, one-hot encode category codes, keep 0/1 columns unchanged."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        # handle_unknown="ignore": a category never seen in training will not crash prediction
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("binary", "passthrough", BINARY_COLUMNS),
    ])


def build_models():
    """class_weight='balanced' gives the smaller classes more weight DURING TRAINING only."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def make_pipeline(classifier):
    return Pipeline([("preprocess", build_preprocessor()), ("model", classifier)])


def cross_validate_models(pipelines, X_train, y_train):
    """5-fold CV on training data. The pipeline is re-fitted inside every fold (no leakage)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in pipelines.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring=["accuracy", "f1_macro"], n_jobs=1)
        rows.append({
            "Model": name,
            "CV Accuracy": res["test_accuracy"].mean(),
            "CV Macro F1": res["test_f1_macro"].mean(),
            "CV Macro F1 std": res["test_f1_macro"].std(),
        })
    return pd.DataFrame(rows).set_index("Model")


def evaluate_on_test(name, pipe, X_test, y_test):
    """Return overall metrics and per-class metrics, and save the confusion matrix figure."""
    pred = pipe.predict(X_test)
    overall = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Macro Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    report = classification_report(y_test, pred, labels=CLASS_ORDER,
                                   output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[CLASS_ORDER, ["precision", "recall", "f1-score", "support"]]
    per_class.insert(0, "Model", name)

    print(f"\n----- {name}: classification report (TEST SET) -----")
    print(classification_report(y_test, pred, labels=CLASS_ORDER, digits=3, zero_division=0))

    cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASS_ORDER),
                      index=[f"Actual {c}" for c in CLASS_ORDER],
                      columns=[f"Predicted {c}" for c in CLASS_ORDER])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm.to_string())
    cm.to_csv(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASS_ORDER,
                                            ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{name}: counts")
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASS_ORDER, normalize="true",
                                            values_format=".2f", ax=axes[1], cmap="Blues",
                                            colorbar=False)
    axes[1].set_title(f"{name}: row-normalised (= recall per class)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=120)
    plt.close(fig)
    return overall, per_class


def feature_importance(pipe, X_test, y_test):
    """Permutation importance on ORIGINAL columns: shuffle one column at a time on the test
    set and see how much Macro F1 drops. Bigger drop = the model relies on it more."""
    result = permutation_importance(pipe, X_test, y_test, scoring="f1_macro", n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"Feature": X_test.columns,
                        "Macro F1 drop": result.importances_mean,
                        "Std": result.importances_std})
    imp = imp.sort_values("Macro F1 drop", ascending=False).reset_index(drop=True)
    imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["Feature"], top["Macro F1 drop"], xerr=top["Std"], color="#337ab7")
    ax.set_xlabel("Drop in Macro F1 when the feature is shuffled")
    ax.set_title("Top 10 features (permutation importance, test set)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    return imp


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Split
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows | "
          f"Features: {X_train.shape[1]}")
    print("Train class %:\n", (y_train.value_counts(normalize=True) * 100).round(1).to_string())
    print("Test class %:\n", (y_test.value_counts(normalize=True) * 100).round(1).to_string())

    # 2. Pipelines
    pipelines = {name: make_pipeline(clf) for name, clf in build_models().items()}

    # 3. Cross-validation on training data (model selection evidence)
    print("\n=== CROSS-VALIDATION ON TRAINING DATA (5-fold) ===")
    cv_table = cross_validate_models(pipelines, X_train, y_train)
    print(cv_table.round(4).to_string())
    cv_table.round(4).to_csv(OUTPUT_DIR / "cross_validation_results.csv")

    # 4. Fit on the full training set, evaluate once on the test set
    print("\n=== FINAL EVALUATION ON UNSEEN TEST SET ===")
    overall_rows, per_class_tables = [], []
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        overall, per_class = evaluate_on_test(name, pipe, X_test, y_test)
        overall_rows.append(overall)
        per_class_tables.append(per_class)

    comparison = pd.DataFrame(overall_rows).set_index("Model")
    per_class_all = pd.concat(per_class_tables)
    print("\n=== MODEL COMPARISON (TEST SET) ===")
    print(comparison.round(4).to_string())
    comparison.round(4).to_csv(OUTPUT_DIR / "model_comparison.csv")
    per_class_all.round(4).to_csv(OUTPUT_DIR / "per_class_metrics.csv")

    # 5. Select the final model using TRAINING-ONLY evidence (CV Macro F1)
    final_name = cv_table["CV Macro F1"].idxmax()
    final_pipe = pipelines[final_name]
    print(f"\nFinal model (highest CV Macro F1 on training data): {final_name}")
    joblib.dump({"name": final_name, "pipeline": final_pipe}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

    # 6. Feature importance for the final model
    imp = feature_importance(final_pipe, X_test, y_test)
    print("\n=== TOP 10 FEATURES (permutation importance) ===")
    print(imp.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
