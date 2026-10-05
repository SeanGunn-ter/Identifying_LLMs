import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextLSTM(nn.Module):
    """Embedding -> bidirectional LSTM -> final hidden states -> linear."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)  # forward + backward final states
        return self.fc(self.drop(h))
