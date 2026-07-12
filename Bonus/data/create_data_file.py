from datasets import load_dataset

dataset = load_dataset("kshitij230/twitter_aspect_based_sentiment_analysis")

df = dataset["train"].to_pandas()

df.to_csv("feedback.csv", index=False)

print("Saved!")