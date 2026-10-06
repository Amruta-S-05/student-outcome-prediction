"""Project settings. Everything that is 'a decision' lives here so it is easy to explain."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Option_1_Student_Academic_Outcome.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_model.joblib"

# ---------- Reproducibility ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80% train / 20% final test
CV_FOLDS = 5              # cross-validation on the TRAINING set only (used for model selection)

# ---------- Target ----------
TARGET = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# ---------- Prediction-time rule from the PDF ----------
# These 12 semester-performance columns are NOT known at enrollment, so they are removed.
PROHIBITED_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# ---------- Feature groups (24 enrollment-time features = 36 - 12) ----------
# Integer codes that are really categories (the number has no order/size meaning) -> one-hot encode
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",              # spelled this way in the original UCI file
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Already 0/1 -> keep as they are
BINARY_COLUMNS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Real numbers -> standardise (mean 0, std 1)
NUMERIC_COLUMNS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
