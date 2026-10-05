import torch
from torch import nn


class TextCNN(nn.Module):
    """Embedding -> parallel 1D convs (3/4/5-word windows) -> max-pool -> linear."""

    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_dim=128,
        n_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths=None):
        e = self.emb(x).transpose(1, 2)  # (batch, emb, seq)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))
