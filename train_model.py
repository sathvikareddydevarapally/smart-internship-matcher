
import pickle
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DATASET_SIZE = 1200

FEATURES = [
    "python",
    "java",
    "sql",
    "ml",
    "web",
    "projects",
    "leetcode",
    "cgpa",
]

TRACKS = ["Data/ML", "Software", "Web Development"]


def create_synthetic_dataset(size=DATASET_SIZE):
    """Create reproducible, rule-labelled demonstration data."""

    rng = np.random.default_rng(RANDOM_STATE)

    data = pd.DataFrame({
        "python": rng.binomial(1, 0.55, size),
        "java": rng.binomial(1, 0.50, size),
        "sql": rng.binomial(1, 0.45, size),
        "ml": rng.binomial(1, 0.35, size),
        "web": rng.binomial(1, 0.40, size),
        "projects": rng.integers(0, 6, size),
        "leetcode": rng.integers(0, 251, size),
        "cgpa": np.round(rng.uniform(6.0, 10.0, size), 2),
    })

    # These illustrative rules generate labels.
    # They are NOT learned from real hiring outcomes.
    data_score = (
        2.5 * data["ml"]
        + 1.5 * data["python"]
        + 1.2 * data["sql"]
        + data["leetcode"] / 100
        + 0.25 * data["projects"]
    )

    software_score = (
        2.0 * data["java"]
        + 1.2 * data["python"]
        + 1.1 * data["sql"]
        + data["leetcode"] / 90
        + 0.35 * data["projects"]
        + 0.1 * data["cgpa"]
    )

    web_score = (
        2.0 * data["web"]
        + 0.5 * data["java"]
        + 0.8 * data["sql"]
        + 0.4 * data["projects"]
        + data["leetcode"] / 130
    )

    scores = np.column_stack([
        data_score,
        software_score,
        web_score,
    ])

    data["track"] = np.array(TRACKS)[np.argmax(scores, axis=1)]

    return data


def main():
    print("Creating synthetic demonstration dataset...")

    df = create_synthetic_dataset()

    print(f"\nDataset size: {len(df)}")
    print("\nClass distribution:")
    print(df["track"].value_counts())

    X = df[FEATURES]
    y = df["track"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Majority baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = {}

    print("\nFive-fold cross-validation on training data")
    print("-" * 55)

    for name, candidate in models.items():
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=5,
            scoring="f1_macro",
        )

        results[name] = scores.mean()

        print(
            f"{name}: macro-F1 = {scores.mean():.3f} "
            f"(std = {scores.std():.3f})"
        )

    # Select the better ML model using training-only cross-validation.
    ml_names = ["Logistic Regression", "Random Forest"]
    best_name = max(ml_names, key=lambda name: results[name])
    best_model = models[best_name]

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    print(f"\nSelected model: {best_name}")
    print(
        f"Held-out test accuracy: "
        f"{accuracy_score(y_test, predictions):.3f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=TRACKS,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=TRACKS,
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        matrix,
        index=TRACKS,
        columns=TRACKS,
    ))

    with open("internship_match_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    df.to_csv("synthetic_career_dataset.csv", index=False)

    print("\nSaved model: internship_match_model.pkl")
    print("Saved dataset: synthetic_career_dataset.csv")
    print("\nIMPORTANT:")
    print("All records and labels are synthetic.")
    print("These metrics measure agreement with generated rules,")
    print("not real internship suitability or hiring success.")


if __name__ == "__main__":
    main()
