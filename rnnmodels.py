import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, rnn_type='LSTM', vocab_size=10000, embed_dim=100, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        else:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # take last timestep
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()