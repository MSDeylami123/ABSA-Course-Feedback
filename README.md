# Aspect-Based Sentiment Analysis (ABSA) on Course Feedback

## Project Overview

This project implements a baseline machine learning model for Aspect-Based Sentiment Analysis (ABSA). The goal is to predict the sentiment (Positive, Neutral, or Negative) expressed toward a specific aspect mentioned in a sentence.

The baseline combines TF-IDF feature extraction with Logistic Regression and was developed as part of the Machine Learning course project.

---

## Dataset

The project uses the `feedback.csv` dataset containing aspect-level sentiment annotations.

Dataset statistics:

| Item | Value |
|------|------:|
| Total samples | 3,602 |
| Training samples | 2,881 |
| Testing samples | 721 |
| Number of classes | 3 |

Sentiment classes:

- Negative
- Neutral
- Positive

---

## Project Structure

```
project/
│
├── data/
│   └── feedback.csv
│
├── baseline.py
├── README.md
└── requirements.txt (optional)
```

---

## Requirements

Python 3.10 or later is recommended.

Required libraries:

- pandas
- matplotlib
- scikit-learn

Install them using:

```bash
pip install pandas matplotlib scikit-learn
```

---

## Running the Project

Run the baseline script:

```bash
python baseline.py
```

The script will automatically:

- Load the dataset
- Split the data into training and testing sets
- Create TF-IDF features
- Train the Logistic Regression classifier
- Evaluate the model
- Display the evaluation metrics
- Display the confusion matrix
- Run several example predictions
- Print representative misclassified examples

---

## Baseline Model

Feature extraction:

- TF-IDF Vectorizer
- Unigrams and Bigrams
- English stop-word removal

Classifier:

- Logistic Regression
- Balanced class weights
- liblinear solver

---

## Evaluation Metrics

The following evaluation metrics are reported:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification Report
- Confusion Matrix
- Sentence Length Breakdown

---

## Example Input

Sentence:

```
The food was amazing but the service was terrible.
```

Aspect:

```
food
```

Predicted sentiment:

```
Positive
```

---

Sentence:

```
The food was amazing but the service was terrible.
```

Aspect:

```
service
```

Predicted sentiment:

```
Positive
```

---

## Sample Output

```
Accuracy : 0.7143

Precision: 0.6432

Recall: 0.5964

Macro F1: 0.6131
```

---

## Known Limitations

The baseline has several limitations:

- Cannot fully understand semantic meaning
- Cannot effectively handle sarcasm or irony
- May struggle with multiple aspects in the same sentence
- Performance is affected by class imbalance
- Neutral sentiment is more difficult to classify than Positive sentiment

---

## Future Improvements

Possible future work includes:

- Replacing TF-IDF with transformer-based embeddings (e.g., BERT)
- Collecting additional balanced training data
- Evaluating stronger classification models
- Improving preprocessing and feature engineering

---

## Author

Machine Learning Course Project

Aspect-Based Sentiment Analysis (ABSA) Baseline