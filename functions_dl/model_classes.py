import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax


# ================================================== LSTM Classifier ==================================================

class LSTMClassifier(nn.Module):
    def __init__(self, dropout=0.2, hidden_sizes=(128, 64)):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[1], batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_sizes[1])
        self.attention = nn.Linear(hidden_sizes[1], 1)
        self.fc1 = nn.Linear(hidden_sizes[1], 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        attn_weights = softmax(self.attention(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        x = self.batch_norm(x)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
