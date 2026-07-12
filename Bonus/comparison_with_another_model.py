import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------------
print("Loading dataset...")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "feedback.csv"

df = pd.read_csv(DATA_PATH)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Number of samples:", len(df))
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# Step 2: Display Dataset
# --------------------------------------------------
print("\nFirst 5 samples:")
print(df.head())

# --------------------------------------------------
# Step 3: Prepare Input and Labels
# --------------------------------------------------
print("\nPreparing inputs...")

# Combine aspect and sentence
df["input"] = (
    "Aspect: "
    + df["aspect_term"].astype(str)
    + " Sentence: "
    + df["text"].astype(str)
)

# Convert labels
label_map = {-1: 0, 0: 1, 1: 2}

reverse_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

df["label"] = df["class"].map(label_map)

X = df["input"]
Y = df["label"]

print("Number of classes:", Y.nunique())

print("\nLabel distribution:")
print(Y.value_counts().sort_index())

# --------------------------------------------------
# Plot 1: Dataset Class Distribution
# --------------------------------------------------
label_names = ["Negative", "Neutral", "Positive"]
counts = Y.value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(label_names, counts)
plt.title("Dataset Class Distribution")
plt.xlabel("Sentiment Class")
plt.ylabel("Number of Samples")

for i, count in enumerate(counts):
    plt.text(i, count + 10, str(count), ha="center")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 4: Train/Test Split
# --------------------------------------------------
print("\nSplitting dataset...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("Training samples:", len(X_train_text))
print("Testing samples :", len(X_test_text))

# --------------------------------------------------
# Step 5: TF-IDF Features
# --------------------------------------------------
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2
)

# Fit ONLY on training data
X_train = vectorizer.fit_transform(X_train_text)

# Transform test data
X_test = vectorizer.transform(X_test_text)

print("Training feature matrix:", X_train.shape)
print("Testing feature matrix :", X_test.shape)

# --------------------------------------------------
# Step 6: Train Logistic Regression
# --------------------------------------------------
print("\nTraining Multinomial Naive Bayes...")

model = MultinomialNB()

model.fit(X_train, y_train)

print("Training complete!")

# --------------------------------------------------
# Step 7: Prediction
# --------------------------------------------------
print("\nMaking predictions...")

predictions = model.predict(X_test)

# --------------------------------------------------
# Step 8: Evaluation
# --------------------------------------------------
print("\n==============================")
print("BASELINE RESULTS")
print("==============================")

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)

# --------------------------------------------------
# Sentence Length Breakdown
# --------------------------------------------------
print("\nSentence Length Breakdown")

test_df = df.loc[y_test.index].copy()
test_df["prediction"] = predictions

test_df["length"] = test_df["text"].str.split().apply(len)

short = test_df[test_df["length"] < 10]
medium = test_df[(test_df["length"] >= 10) & (test_df["length"] <= 20)]
long = test_df[test_df["length"] > 20]

for name, subset in [
    ("Short", short),
    ("Medium", medium),
    ("Long", long)
]:
    acc = accuracy_score(subset["label"], subset["prediction"])
    print(f"{name}: {acc:.3f} ({len(subset)} samples)")

# --------------------------------------------------
# Plot 2: Evaluation Metrics
# --------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "Macro F1"]
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 4))
bars = plt.bar(metrics, scores)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Baseline Evaluation Metrics")

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.02,
        f"{score:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 3: Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 9: Custom Predictions
# --------------------------------------------------
print("\n==============================")
print("CUSTOM PREDICTIONS")
print("==============================")

examples = [
    ("The food was amazing but the service was terrible.", "food"),
    ("The food was amazing but the service was terrible.", "service"),
    ("The staff were polite and friendly.", "staff"),
    ("The menu was average.", "menu"),
]

for sentence, aspect in examples:

    model_input = f"Aspect: {aspect} Sentence: {sentence}"

    vector = vectorizer.transform([model_input])

    prediction = model.predict(vector)[0]

    print("\nSentence:")
    print(sentence)
    print("Aspect:", aspect)
    print("Predicted sentiment:", reverse_map[prediction])




print("\nMisclassified Examples")
print("=" * 50)

count = 0

for text, aspect, true_label, pred in zip(
    X_test_text,
    df.loc[y_test.index, "aspect_term"],
    y_test,
    predictions
):

    if true_label != pred:

        print("Sentence:")
        print(text)

        print("Aspect:", aspect)

        print("True:", reverse_map[true_label])

        print("Predicted:", reverse_map[pred])

        print("-" * 50)

        count += 1

    if count == 10:
        break

print("\nDone!")