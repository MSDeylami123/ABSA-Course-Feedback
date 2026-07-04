from datasets import load_dataset
import pandas as pd

dataset = load_dataset("introtogenairize/gen-ai-course-feedback")

print(dataset)
print()

print("Columns:")
print(dataset["train"].column_names)
print()

print("Number of rows:")
print(len(dataset["train"]))
print()

print("First example:")
print(dataset["train"][0])
print()

df = dataset["train"].to_pandas()
print(df.head())
print()

print(dataset["train"].features)