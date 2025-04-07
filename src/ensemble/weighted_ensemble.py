import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("val_probs.csv")

model_cols = ["prob_bow_dt", "prob_cat", "prob_rating"]

def get_accuracy(col):
    preds = df[col].apply(lambda p: "CG" if p > 0.5 else "OR")
    return (preds == df["label"]).mean()

def evaluate_data_dependent_weights(df):
    def get_w2(rating):
        if rating == 1 or rating == 2:
            return 0.9
        if rating == 3:
            return 0.8
        return 0

    def get_w1(category):
        if category == "Books_5":
            return 0.1
        if category == "Clothing_Shoes_and_Jewelry_5":
            return 0.2
        if category == "Kindle_Store_5":
            return 0.1
        if category == "Toys_and_Games_5":
            return 0.2
        return 0

    preds = []
    for _, row in df.iterrows():
        w2 = get_w2(row["rating"])
        w1 = get_w1(row["category"])
        w3 = 1 - w1 - w2

        final_prob = (
            w1 * row["prob_cat"] +
            w2 * row["prob_rating"] +
            w3 * row["prob_bow_dt"]
        )
        pred = "CG" if final_prob > 0.5 else "OR"
        preds.append(pred)

    df["predicted_label"] = preds

    # Print metrics
    print("=== Classification Report ===")
    print(classification_report(df["label"], df["predicted_label"], digits=4))
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(df["label"], df["predicted_label"]))

    accuracy = (df["label"] == df["predicted_label"]).mean()
    return accuracy

acc = evaluate_data_dependent_weights(df)
print(f"Accuracy: {acc:.4f}")
