import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(
    r"C:\Users\USER\Desktop\ABSA-Course-Feedback\Phase1\data\feedback.csv"
)

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

# IMPORTANT:
# Fit ONLY on the training data
X_train = vectorizer.fit_transform(X_train_text)

# Transform the test data using the fitted vectorizer
X_test = vectorizer.transform(X_test_text)

print("Training feature matrix:", X_train.shape)
print("Testing feature matrix :", X_test.shape)

# --------------------------------------------------
# Step 6: Train Logistic Regression
# --------------------------------------------------
print("\nTraining Logistic Regression baseline...")

model = LogisticRegression(
    solver="liblinear",
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

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

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

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

print("\nDone!")