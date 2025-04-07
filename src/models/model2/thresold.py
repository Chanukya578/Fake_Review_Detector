import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model Definition
# ----------------------------
class BERTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)


# ----------------------------
# Dataset Class
# ----------------------------
class BERTDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(df['text_'].tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.ratings = torch.tensor(df['rating_scaled'].values, dtype=torch.float32)
        self.labels = df['label'].values  

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['rating'] = self.ratings[idx]
        item['label'] = self.labels[idx]
        return item


# ----------------------------
# Prediction Function
# ----------------------------
def get_preds(dataset):
    loader = DataLoader(dataset, batch_size=16)
    all_preds = []
    all_actuals = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            all_preds.extend(outputs.cpu().numpy())
            all_actuals.extend(batch['rating'].numpy())
            all_labels.extend(batch['label'])
    preds_real = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    preds_real = np.clip(preds_real, 0, 5)
    actuals_real = scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1)).flatten()
    return preds_real, actuals_real, np.array(all_labels)


# ----------------------------
# Find Best Threshold
# ----------------------------
def find_best_threshold_roc(preds, actuals, true_labels, rating_class):
    binary_labels = (true_labels == "OR").astype(int)
    diffs = np.abs(preds - rating_class)
    fpr, tpr, thresholds = roc_curve(binary_labels, -diffs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return -1 * thresholds[best_idx]


# ----------------------------
# Apply Threshold to Get Accuracy
# ----------------------------
def calc_accuracy(preds, actuals, true_labels, rating_class, threshold):
    mask = np.isclose(actuals, rating_class, atol=1e-2) 
    filtered_preds = preds[mask]
    filtered_labels = true_labels[mask]
    diffs = np.abs(filtered_preds - rating_class)
    pred_labels = np.where(diffs <= threshold, "OR", "CG")

    return accuracy_score(filtered_labels, pred_labels)


# ----------------------------
# Load Everything
# ----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTRegressor().to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/bert_rating_regressor.pt", map_location=device))
model.eval()

train_df = pd.read_csv("train_new.csv")
val_df = pd.read_csv("val_new.csv")
test_df = pd.read_csv("test_new.csv")

scaler = MinMaxScaler()
train_df['rating_scaled'] = scaler.fit_transform(train_df[['rating']])
val_df['rating_scaled'] = scaler.transform(val_df[['rating']])
test_df['rating_scaled'] = scaler.transform(test_df[['rating']])

train_dataset = BERTDataset(train_df)
val_dataset = BERTDataset(val_df)
test_dataset = BERTDataset(test_df)

train_preds, train_actuals, train_labels = get_preds(train_dataset)
val_preds, val_actuals, val_labels = get_preds(val_dataset)
test_preds, test_actuals, test_labels = get_preds(test_dataset)

# ----------------------------
# Evaluate Per Rating Class
# ----------------------------
results = []

for rating_class in [1.0, 2.0, 3.0, 4.0, 5.0]:
    threshold = find_best_threshold_roc(train_preds, train_actuals, train_labels, rating_class)

    train_acc = calc_accuracy(train_preds, train_actuals, train_labels, rating_class, threshold)
    val_acc = calc_accuracy(val_preds, val_actuals, val_labels, rating_class, threshold)
    test_acc = calc_accuracy(test_preds, test_actuals, test_labels, rating_class, threshold)

    results.append({
        "Rating": rating_class,
        "Best Threshold": round(threshold, 4),
        "Train Accuracy": f"{train_acc*100:.2f}%",
        "Val Accuracy": f"{val_acc*100:.2f}%",
        "Test Accuracy": f"{test_acc*100:.2f}%"
    })


results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("rating_threshold_accuracy.csv", index=False)
print("Saved to rating_threshold_accuracy.csv ✅")
