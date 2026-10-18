import torch, math
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        B, T, C = Q.shape
        k = self.key(K)
        q = self.query(Q)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(V)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size, n_embd, dropout, block_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        out = torch.cat([h(Q, K, V) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)