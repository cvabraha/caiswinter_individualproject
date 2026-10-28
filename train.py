import kagglehub
import os
import json
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertModel

# Download dataset
path = kagglehub.dataset_download("rmisra/news-headlines-dataset-for-sarcasm-detection")

# Build file path
json_file_path = os.path.join(path, 'Sarcasm_Headlines_Dataset.json')

data = []
with open(json_file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Dataframe setup
df = pd.DataFrame(data)
df = df[['headline', 'is_sarcastic']]

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "url", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", "", t)
    return t

df['clean_text'] = df['headline'].apply(clean_text)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['clean_text'].values,
    df['is_sarcastic'].values,
    test_size=0.2,
    random_state=42
)

# Dataloaders
train_ds = SarcasmDataset(train_texts, train_labels, tokenizer)
val_ds = SarcasmDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

class BertLSTMClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertLSTMClassifier, self).__init__()
        
        # BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # LSTM
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True, bidirectional=True)
        
        # Binary classifier
        self.classifier = nn.Linear(128 * 2, 2)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        
        # BERT --> LSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Classification
        logits = self.classifier(hidden_final)
        return logits

# Initialize Model and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertLSTMClassifier(freeze_bert=False) 
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

epochs = 3

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 10)

    # --- TRAINING ---
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        logits = model(input_ids, mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Average Training Loss: {total_loss / len(train_loader)}")

    # --- TRAINING EVALUATION ---
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            val_preds.extend(preds)
            val_true.extend(labels)

    acc = accuracy_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds)

    print(f"Validation Accuracy: {acc}")
    print(f"Validation F1 Score: {f1}")

# Save the model weights for the API
torch.save(model.state_dict(), "sarcasm_model.pth")
