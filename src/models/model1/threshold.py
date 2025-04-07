import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss

# Load dataset
train_df = pd.read_csv("category_review_relevance.csv")
test_df = pd.read_csv("val_category_relevance.csv")

train_df["label"] = train_df["label"].map({"OR": 1, "CG": 0})
test_df["label"] = test_df["label"].map({"OR": 1, "CG": 0})

# Store best thresholds and test metrics
best_thresholds = {}
metrics = {}

# Function to find the best threshold
def find_best_threshold(df_cat):
    best_acc = 0
    best_thresh = 0
    thresholds = np.linspace(df_cat["relevance"].min(), df_cat["relevance"].max(), 50)

    for thresh in thresholds:
        preds = (df_cat["relevance"] >= thresh).astype(int)
        acc = accuracy_score(df_cat["label"], preds)

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    return best_thresh

# Find thresholds using the training set
for category in train_df["category"].unique():
    train_cat = train_df[train_df["category"] == category]  # Filter train data for category

    # Find best threshold from train data
    best_thresh = find_best_threshold(train_cat)
    best_thresholds[category] = best_thresh

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

k = 10  # steepness of sigmoid

prob_results = []

for category in test_df["category"].unique():
    test_cat = test_df[test_df["category"] == category].copy()
    T = best_thresholds[category]

    # Convert relevance to probability using sigmoid centered at threshold
    test_cat["predicted_prob"] = sigmoid(k * (test_cat["relevance"] - T))

    # For evaluation: classify based on 0.5 threshold on probability
    test_cat["predicted_label"] = (test_cat["predicted_prob"] >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(test_cat["label"], test_cat["predicted_label"])
    precision = precision_score(test_cat["label"], test_cat["predicted_label"])
    recall = recall_score(test_cat["label"], test_cat["predicted_label"])
    f1 = f1_score(test_cat["label"], test_cat["predicted_label"])

    prob_results.append({
        "Category": category,
        "Threshold": T,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    })
    
metrics_df = pd.DataFrame(prob_results)
print(metrics_df)

metrics_df.to_csv("val_set_metrics.csv")
print("Saved test results to test_set_metrics.csv")

    
def return_prob(test_df, best_thresholds, k, output_file):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    probs = []
    preds = []
    
    for idx, row in test_df.iterrows():
        category = row["category"]
        T = best_thresholds[category]
        relevance = row["relevance"]
        p = sigmoid(k * (relevance - T))
        probs.append(p)
        preds.append(1 if p >= 0.5 else 0)

    test_df_with_probs = pd.DataFrame()
    test_df_with_probs["prob_cat"] = probs
    test_df_with_probs["label"] = test_df["label"]
    test_df_with_probs["predicted_label"] = preds

    # Compute and print accuracy
    acc = accuracy_score(test_df_with_probs["label"], test_df_with_probs["predicted_label"])
    print(f"Overall Accuracy on Test Set: {acc:.4f}")

    # Save CSV
    test_df_with_probs.drop(columns=["label", "predicted_label"], inplace=True)
    test_df_with_probs.to_csv(output_file, index=False)
    print(f"Saved predicted probabilities to {output_file}")

    return test_df_with_probs

    
return_prob(test_df, best_thresholds, k=10, output_file="val_results.csv")
