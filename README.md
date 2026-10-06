# Student Academic Outcome Prediction (Gupio AI/ML Assignment – Option 1)

Predict whether a student will end up **Dropout**, **Enrolled** or **Graduate** using only
information available **at enrollment**.

## 1. Dataset
- UCI Machine Learning Repository – *Predict Students' Dropout and Academic Success* (ID 697),
  DOI 10.24432/C5MC89, licence CC BY 4.0.
- File used: the panel-supplied `data/Option_1_Student_Academic_Outcome.csv` (semicolon-delimited).
  It is **not modified**; the loader only strips whitespace from column names.
- 4,424 rows, 37 columns (36 features + `Target`), 0 missing values, 0 duplicate rows.
- Target: Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%).

## 2. Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib (see `requirements.txt`).
Models: Logistic Regression (baseline) and Random Forest (ensemble).

## 3. Project structure
```
data/        supplied dataset
src/         config.py, data_loader.py, eda.py, train_evaluate.py, predict.py
models/      saved final pipeline (created by train_evaluate.py, not committed – 17 MB)
outputs/     EDA plots, metrics (CSV), confusion matrices, feature importance, sample predictions
requirements.txt, .gitignore, README.md
```

## 4. Setup and run
```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1      |  Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python src/eda.py              # data inspection + EDA plots
python src/train_evaluate.py   # preprocessing, training, evaluation, model selection, feature importance
python src/predict.py          # sample predictions with the saved final model
```
Run the commands from the project root folder. Scripts must be run in this order
(`predict.py` needs the model saved by `train_evaluate.py`).

## 5. Prediction-time rule
The 12 first/second-semester "Curricular units ..." columns are **dropped** (they are not known
at enrollment). `data_loader.py` also asserts that none of them remain in `X`.
24 enrollment-time features remain:
- 9 integer-coded categories (Marital status, Application mode, Course, Previous qualification,
  Nacionality, Mother's/Father's qualification, Mother's/Father's occupation) -> one-hot encoded
- 8 binary columns (0/1) -> kept as they are
- 7 numeric columns (Application order, Previous qualification (grade), Admission grade,
  Age at enrollment, Unemployment rate, Inflation rate, GDP) -> standardised

## 6. Preprocessing and train/test strategy
- Preprocessing is a `ColumnTransformer` inside a scikit-learn `Pipeline` with the model, so the
  scaler and one-hot encoder are **fitted on training data only** (also inside each CV fold).
  `OneHotEncoder(handle_unknown="ignore")` lets unseen categories pass without error.
- One stratified split: 80% train (3,539) / 20% test (885), `random_state=42`. Class proportions
  are the same in both parts.
- Model selection used **5-fold stratified cross-validation on the training set only**.
  The test set was evaluated once, after the models were fitted, and was never used for tuning.
- Class imbalance: both models use class weights (`balanced`) during training only.
  No resampling, no threshold changes, no hyperparameter tuning.
- No derived features were created.

## 7. Results (actual output of `python src/train_evaluate.py`, seed 42)

**Cross-validation on training data (model selection evidence)**

| Model | CV Accuracy | CV Macro F1 | CV Macro F1 std |
|---|---|---|---|
| Logistic Regression | 0.6030 | 0.5678 | 0.0152 |
| Random Forest | 0.6332 | 0.5720 | 0.0179 |

**Unseen test set (885 students)**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.5729 | 0.5632 | 0.5626 | 0.5497 | 0.5897 |
| Random Forest | 0.6102 | 0.5506 | 0.5521 | 0.5510 | 0.6091 |

**Per-class test results**

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Logistic Regression | Dropout | 0.686 | 0.577 | 0.627 | 284 |
| Logistic Regression | Enrolled | 0.301 | 0.522 | 0.382 | 159 |
| Logistic Regression | Graduate | 0.703 | 0.588 | 0.640 | 442 |
| Random Forest | Dropout | 0.603 | 0.641 | 0.621 | 284 |
| Random Forest | Enrolled | 0.340 | 0.321 | 0.330 | 159 |
| Random Forest | Graduate | 0.709 | 0.695 | 0.702 | 442 |

**Confusion matrices** (rows = actual, columns = predicted: Dropout / Enrolled / Graduate)

| Logistic Regression | Pred. Dropout | Pred. Enrolled | Pred. Graduate |
|---|---|---|---|
| Actual Dropout | 164 | 66 | 54 |
| Actual Enrolled | 20 | 83 | 56 |
| Actual Graduate | 55 | 127 | 260 |

| Random Forest | Pred. Dropout | Pred. Enrolled | Pred. Graduate |
|---|---|---|---|
| Actual Dropout | 182 | 41 | 61 |
| Actual Enrolled | 43 | 51 | 65 |
| Actual Graduate | 77 | 58 | 307 |

Figures: `outputs/confusion_matrix_*.png`. Full tables: `outputs/*.csv`.

## 8. Class imbalance and per-class behaviour
- **Enrolled** is the smallest class (17.9%) and the hardest: F1 is 0.38 (Logistic Regression) and
  0.33 (Random Forest). It sits "between" Dropout and Graduate, so enrollment-time information
  separates it poorly.
- **Graduate** is handled best (F1 0.64 / 0.70) and **Dropout** is moderate (F1 0.63 / 0.62).
- Class weights changed the trade-off: Logistic Regression finds more Enrolled students
  (recall 0.52) but with low precision (0.30) – it over-predicts Enrolled (276 predictions for
  159 real). Random Forest is more conservative on Enrolled (recall 0.32).
- Accuracy alone would hide this, which is why Macro F1 and per-class metrics are reported.

## 9. Final model choice
**Random Forest** was selected because it had the higher Macro F1 in cross-validation on the
training data (0.572 vs 0.568), and it also had higher accuracy (0.610 vs 0.573), weighted F1
(0.609 vs 0.590) and better Dropout recall and Graduate F1 on the test set.

Honest caveat: the two models are **almost tied on test Macro F1** (0.551 vs 0.550) and the CV
difference is smaller than the fold-to-fold standard deviation. Logistic Regression is clearly
better at catching Enrolled students (recall 0.52 vs 0.32). If finding Enrolled students matters
most, Logistic Regression is a reasonable alternative. The saved model is trained on the 80% training split only (it is not refitted on all data).

## 10. Feature importance (final model, permutation importance on the test set)
Importance = drop in Macro F1 when one original column is shuffled (10 repeats).

| Rank | Feature | Macro F1 drop |
|---|---|---|
| 1 | Tuition fees up to date | 0.0466 |
| 2 | Course | 0.0337 |
| 3 | Scholarship holder | 0.0243 |
| 4 | Mother's occupation | 0.0172 |
| 5 | Admission grade | 0.0139 |
| 6 | Inflation rate | 0.0101 |
| 7 | Mother's qualification | 0.0095 |
| 8 | Debtor | 0.0085 |
| 9 | Unemployment rate | 0.0081 |
| 10 | GDP | 0.0081 |

Consistent with the EDA: 86.6% of students whose tuition fees were *not* up to date dropped out
(vs 24.7% of those up to date), and 12.2% of scholarship holders dropped out (vs 38.7%).
**Feature importance shows what the model uses for prediction; it does not prove causation.**
Several lower-ranked importances (e.g. Unemployment rate, GDP) have standard deviations close to
their values, so their ranking is not reliable. See `outputs/feature_importance.csv`.

## 11. Sample predictions
`python src/predict.py` loads the saved pipeline and predicts 8 real students from the unseen
test set (shows actual label, predicted label and class probabilities). Output from my run
(6 of 8 correct – a tiny sample, not a performance estimate) is in `outputs/sample_predictions.csv`.
`predict_students(pipeline, dataframe)` in `src/predict.py` can be used with any DataFrame that
has the 24 enrollment-time columns.

## 12. Assumptions
- Unemployment rate, Inflation rate and GDP are treated as enrollment-time information (they are
  not on the prohibited list).
- "Tuition fees up to date" and "Debtor" are kept because they are not prohibited, but they may be
  recorded some time after enrollment. If they are not truly known at enrollment, the model's
  performance in real use could be lower. This is the main prediction-time assumption.
- Integer-coded columns are treated as categories, not as ordered numbers.
- "Enrolled" students have not finished yet; their final outcome is uncertain.

## 13. Limitations
- Moderate performance (Macro F1 ≈ 0.55); Enrolled is predicted poorly.
- Single 80/20 split: the test estimate has noise (885 students).
- Data come from one institution; results may not transfer elsewhere.
- Predictive association is **not** causal. For example, paying fees on time does not necessarily
  cause graduation, and giving scholarships would not automatically change outcomes.
- Code columns (Course, occupations) are not decoded here; no fairness analysis was done.

## 14. Completed / incomplete
- Completed: data inspection, EDA, leakage-safe pipeline, stratified split, two models, CV,
  full test evaluation (Accuracy, Precision, Recall, F1, Macro F1, confusion matrices), model
  selection, permutation feature importance, sample predictions, README/requirements/.gitignore.
- Not done (optional bonus): hyperparameter tuning, SHAP, unit tests, API/demo.

## 15. Verification record
- Ran `eda.py` -> `train_evaluate.py` -> `predict.py` in order with the supplied CSV
  (Python 3.12, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.8).
- Training was run twice and gave identical metrics (fixed seed 42).
- Confirmed no prohibited columns in `X` (assertion in `data_loader.py`) and that prediction
  uses the saved pipeline (same preprocessing as training).
- No credentials, `.env` files or API keys exist in the repository.
- Numbers in this README were copied from the script output. Small differences can appear with
  other library versions; always re-run and compare.
