import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# Load & Preprocess Dataset
# ----------------------------
df = pd.read_csv('dataset.csv')
df = df[df['label'] == 'OR']
df = df.dropna(subset=['text_', 'rating'])

scaler = MinMaxScaler()
df['rating_scaled'] = scaler.fit_transform(df['rating'].values.reshape(-1, 1))

# ----------------------------
# Tokenization using BERT
# ----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERTDataset(Dataset):
    def __init__(self, texts, ratings):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['rating'] = self.ratings[idx]
        return item

# Split data
train_texts, test_texts, train_ratings, test_ratings = train_test_split(df['text_'].tolist(), df['rating_scaled'].tolist(), test_size=0.2, random_state=42)

train_dataset = BERTDataset(train_texts, train_ratings)
test_dataset = BERTDataset(test_texts, test_ratings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ----------------------------
# BERT Regressor Model
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
        return self.regressor(cls_output).squeeze()

model = BERTRegressor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# Training Setup
# ----------------------------
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3  

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ratings = batch['rating'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"\nEpoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}")

    # ----------------------------
    # Evaluate on Test Set
    # ----------------------------
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)

            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    rmse = root_mean_squared_error(actuals, preds)

    r2 = r2_score(actuals, preds)
    print(f"Epoch {epoch+1} - Test RMSE: {rmse:.4f}, R²: {r2:.4f}\n")

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), "bert_rating_regressor.pt")
print("Model saved as 'bert_rating_regressor.pt'")
