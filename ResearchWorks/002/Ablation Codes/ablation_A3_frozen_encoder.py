# -*- coding: utf-8 -*-
"""
Ablation A3: Frozen Encoder (attention pooling retained)
Only the attention layer and regression head are trained.
Replace this result in Table IV row: "Frozen Encoder"
"""

from google.colab import drive
drive.mount('/content/drive')

import os, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, set_seed

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_DISABLED"] = "true"
MODEL_NAME = "roberta-base"
N_SPLITS = 5

df = pd.read_csv("/content/drive/MyDrive/Calorie-Prediction-from-Recipe/Dataset.csv")
df = df[['Recipe', 'Approximate Calorie per Serving']].dropna()
texts = df['Recipe'].astype(str).tolist()
labels = df['Approximate Calorie per Serving'].values


class RecipeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FrozenEncoderAttentionRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Only attention layer and regression head are trainable
        self.attention = nn.Linear(hidden_size, 1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs.last_hidden_state  # [B, T, H]

        # Token-level attention pooling
        scores = self.attention(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)

        preds = self.regressor(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.HuberLoss()(preds, labels)
        return {"loss": loss, "logits": preds}


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    return {"rmse": rmse, "mae": mae}


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(texts), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")

    X_train = [texts[i] for i in train_idx]
    X_test  = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train_log.reshape(-1, 1)).ravel()
    y_test_scaled  = scaler.transform(y_test_log.reshape(-1, 1)).ravel()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, padding=True, truncation=True, max_length=256)
    test_enc  = tokenizer(X_test,  padding=True, truncation=True, max_length=256)

    train_ds = RecipeDataset(train_enc, y_train_scaled)
    test_ds  = RecipeDataset(test_enc,  y_test_scaled)

    model = FrozenEncoderAttentionRegressor(MODEL_NAME).to(device)

    training_args = TrainingArguments(
        output_dir=f"./ablation_frozen_fold_{fold}",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=100,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    preds_scaled = trainer.predict(test_ds).predictions.squeeze()
    preds_log = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    preds = np.expm1(preds_log)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100

    all_results.append([rmse, mae, r2, mape])
    print(f"Fold {fold} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, MAPE: {mape:.2f}%")

results = np.array(all_results)

print("\n===== A3: Frozen Encoder — CV Results =====")
print(f"RMSE : {results[:,0].mean():.2f} ± {results[:,0].std():.2f}")
print(f"MAE  : {results[:,1].mean():.2f} ± {results[:,1].std():.2f}")
print(f"R²   : {results[:,2].mean():.3f}")
print(f"MAPE : {results[:,3].mean():.2f}%")
