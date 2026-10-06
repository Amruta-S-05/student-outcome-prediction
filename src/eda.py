"""Step 1: data inspection + simple EDA.
Run:  python src/eda.py
Everything printed is also saved to outputs/data_inspection.txt. Plots go to outputs/.
EDA plots use ONLY the permitted enrollment-time features.
"""
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from config import CLASS_ORDER, OUTPUT_DIR, PROHIBITED_COLUMNS, TARGET
from data_loader import get_features_and_target, load_data

lines = []
COLORS = ["#d9534f", "#f0ad4e", "#5cb85c"]  # Dropout, Enrolled, Graduate


def log(text=""):
    print(text)
    lines.append(str(text))


def inspect(df):
    log("=== 1. DATA INSPECTION ===")
    log(f"Rows: {df.shape[0]}")
    log(f"Columns: {df.shape[1]} (36 features + 1 target)")
    log("\nData types:")
    log(df.dtypes.astype(str).value_counts().to_string())
    log("\nTarget classes (count and %):")
    counts = df[TARGET].value_counts()
    for name in CLASS_ORDER:
        log(f"  {name:<9}: {counts[name]:>5}  ({counts[name] / len(df) * 100:.1f}%)")
    log(f"\nTotal missing values : {int(df.isnull().sum().sum())}")
    log(f"Duplicate rows       : {int(df.duplicated().sum())}")
    log(f"Prohibited semester columns present (will be dropped): "
        f"{sum(c in df.columns for c in PROHIBITED_COLUMNS)} of 12")


def data_quality_notes(X):
    log("\n=== 2. FEATURES AND DATA-QUALITY NOTES ===")
    log(f"Features kept for the enrollment-time model: {X.shape[1]}")
    log("\nNumeric summary:")
    num = ["Application order", "Previous qualification (grade)", "Admission grade",
           "Age at enrollment", "Unemployment rate", "Inflation rate", "GDP"]
    log(X[num].describe().T[["min", "mean", "max"]].round(2).to_string())
    log("\nNumber of distinct values in the integer-coded categorical columns:")
    cats = ["Marital status", "Application mode", "Course", "Previous qualification",
            "Nacionality", "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation"]
    log(X[cats].nunique().to_string())
    log("\nObservations:")
    log("- No missing values and no duplicate rows.")
    log("- Many columns are integer CODES for categories (e.g. Course, Application mode); "
        "the numbers have no size meaning, so they are one-hot encoded.")
    log("- 'Nacionality' keeps the original UCI spelling. Almost all students share one value.")
    log("- Age at enrollment is right-skewed (a few mature students, max 70).")
    log("- Classes are imbalanced: Enrolled is the smallest class.")


def plot_target(y):
    counts = y.value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=COLORS)
    ax.bar_label(bars)
    ax.set_title("Target distribution")
    ax.set_ylabel("Number of students")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_target_distribution.png", dpi=120)
    plt.close(fig)


def plot_numeric_vs_target(X, y):
    cols = ["Admission grade", "Previous qualification (grade)", "Age at enrollment"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, cols):
        data = [X.loc[y == c, col] for c in CLASS_ORDER]
        ax.boxplot(data, tick_labels=CLASS_ORDER)
        ax.set_title(f"{col} by outcome")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_numeric_vs_target.png", dpi=120)
    plt.close(fig)


def plot_categorical_vs_target(X, y):
    cols = ["Tuition fees up to date", "Debtor", "Scholarship holder",
            "Gender", "Displaced", "Daytime/evening attendance"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), cols):
        pct = pd.crosstab(X[col], y, normalize="index")[CLASS_ORDER] * 100
        pct.plot(kind="bar", stacked=True, ax=ax, color=COLORS, legend=False, rot=0)
        ax.set_title(col)
        ax.set_ylabel("% of students")
        ax.set_xlabel("value as coded in the dataset (0 / 1)")
    axes[0, 0].legend(CLASS_ORDER, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_categorical_vs_target.png", dpi=120)
    plt.close(fig)


def plot_course_dropout_rate(X, y):
    rate = (y == "Dropout").groupby(X["Course"]).mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rate.index.astype(str), rate.values, color="#d9534f")
    ax.set_title("Dropout rate (%) by Course code")
    ax.set_xlabel("Course code (as in dataset)")
    ax.set_ylabel("Dropout rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_dropout_rate_by_course.png", dpi=120)
    plt.close(fig)


def print_key_relationships(X, y):
    log("\n=== 3. KEY FEATURE-VS-TARGET NUMBERS (full dataset, for EDA only) ===")
    for col in ["Tuition fees up to date", "Debtor", "Scholarship holder"]:
        pct = (pd.crosstab(X[col], y, normalize="index")[CLASS_ORDER] * 100).round(1)
        log(f"\n% of each outcome by '{col}':")
        log(pct.to_string())
    log("\nMedian by outcome:")
    log(X.groupby(y)[["Admission grade", "Age at enrollment"]].median().loc[CLASS_ORDER].to_string())


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    inspect(df)
    X, y = get_features_and_target(df)
    data_quality_notes(X)
    print_key_relationships(X, y)
    plot_target(y)
    plot_numeric_vs_target(X, y)
    plot_categorical_vs_target(X, y)
    plot_course_dropout_rate(X, y)
    (OUTPUT_DIR / "data_inspection.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved plots and data_inspection.txt to the outputs/ folder.")


if __name__ == "__main__":
    main()
