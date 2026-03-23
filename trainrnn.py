print("Starting RNN/LSTM/GRU training on IMDB dataset...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext import data, datasets
import matplotlib.pyplot as plt
import random
import os
import numpy as np

# Device, seeds, output folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("../outputs/plots", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Fields
TEXT = data.Field(tokenize='basic_english', lower=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Vocab
TEXT.build_vocab(train_data, max_size=10000)
vocab_size = len(TEXT.vocab)

# Custom Dataset (small subset)
class IMDBDataset(Dataset):
    def __init__(self, data_list, max_len=200):
        self.data_list = data_list[:1000]  # small subset
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        example = self.data_list[idx]
        text = example.text[:self.max_len]
        token_ids = [TEXT.vocab.stoi[word] for word in text]
        label = float(example.label == 'pos')
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def collate_batch(batch):
    texts, labels = [], []
    for text, label in batch:
        texts.append(text)
        labels.append(label)
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded.to(device), labels.to(device)

# DataLoaders
batch_size = 32
train_loader = DataLoader(IMDBDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(IMDBDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

class RNNModel(nn.Module):
    def __init__(self, rnn_type='LSTM', vocab_size=10000, embed_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        else:
            raise ValueError("rnn_type must be RNN/LSTM/GRU")
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot loss
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
    plt.title(f'{type(model).__name__} Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"../outputs/plots/{type(model).__name__}_loss.png")
    plt.show()

for rnn_type in ['LSTM','GRU']:  # skip vanilla RNN for speed
    print(f"\nTraining {rnn_type} model...")
    model = RNNModel(rnn_type=rnn_type, vocab_size=vocab_size)
    train_model(model, train_loader, test_loader, epochs=5)