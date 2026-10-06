"""Step 3: sample predictions with the saved final model.
Run:  python src/predict.py   (after train_evaluate.py)

The saved object is the WHOLE pipeline (preprocessing + model), so new records go through
exactly the same preprocessing that was learned on the training data.
"""
import joblib
import pandas as pd

from config import MODEL_PATH, OUTPUT_DIR
from data_loader import get_train_test_split


def predict_students(pipeline, students):
    """students: DataFrame with the 24 enrollment-time columns (extra columns are ignored)."""
    needed = list(pipeline.feature_names_in_)
    proba = pipeline.predict_proba(students[needed])
    result = pd.DataFrame(proba, columns=[f"P({c})" for c in pipeline.classes_],
                          index=students.index).round(3)
    result.insert(0, "Predicted", pipeline.predict(students[needed]))
    return result


def main():
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    print(f"Loaded final model: {saved['name']}")

    # Real student records from the held-out test set (random but reproducible choice)
    _, X_test, _, y_test = get_train_test_split()
    sample = X_test.sample(n=8, random_state=42)

    preds = predict_students(pipeline, sample)
    preds.insert(0, "Actual", y_test.loc[sample.index].values)
    preds["Correct"] = preds["Actual"] == preds["Predicted"]
    preds.index.name = "row_in_csv"

    pd.set_option("display.width", 200)
    print("\nSample predictions on 8 unseen test-set students:")
    print(preds.to_string())
    print(f"\nCorrect: {int(preds['Correct'].sum())} / {len(preds)} "
          "(a tiny sample - use the test-set metrics for overall performance)")
    preds.to_csv(OUTPUT_DIR / "sample_predictions.csv")
    print("Saved to outputs/sample_predictions.csv")


if __name__ == "__main__":
    main()
