from datasets import load_dataset

dataset = load_dataset("introtogenairize/gen-ai-course-feedback")

df = dataset["train"].to_pandas()

df.to_csv("course_feedback.csv", index=False)

print("Saved!")