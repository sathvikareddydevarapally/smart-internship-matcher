import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
FEATURES = [
    "python", "java", "sql", "ml",
    "web", "projects", "leetcode", "cgpa"
]
df = pd.read_csv("synthetic_career_dataset.csv")
X = df[FEATURES]
y = df["track"]
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
with open("internship_match_model.pkl", "rb") as file:
    model = pickle.load(file)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Held-out test accuracy: {accuracy:.3f}")
# Chart 1: Dataset class distribution
plt.figure(figsize=(8, 5))
df["track"].value_counts().plot(kind="bar")
plt.title("Synthetic Career-Track Distribution")
plt.xlabel("Career Track")
plt.ylabel("Number of Records")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
# Chart 2: Held-out test confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=["Data/ML", "Software", "Web Development"],
    cmap="Blues",
    values_format="d",
    xticks_rotation=20,
)
plt.title("Confusion Matrix — Synthetic Test Data")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved class_distribution.png")
print("Saved confusion_matrix.png")