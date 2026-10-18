import torch
from torch import nn
from torch.nn import functional as F
# from library import MultiHeadAttention, FeedForward

from tokenizers import Tokenizer
from datasets import load_dataset

torch.manual_seed(2026)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
block_size = 256 # max context length
n_embd = 384
lr = 3e-4
eval_iters = 200
eval_interval = 500
max_iters = 5000
dropout = 0.2
n_dlayers = 6
n_elayers = 6
n_heads = 6
vocab_size = 16384
n_blocks = 6

tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
dtr = "\n".join(ds["train"]["text"])
print(type(dtr))
dtr = torch.tensor(tokenizer.encode(dtr), dtype=torch.long)

def get_batch(split):
    # data = dtr if split == "train" else dval
    data = dtr
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)
    
    return x, y

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)
        
        self.mha1 = nn.MultiheadAttention(n_embd, n_heads, dropout)
        self.mha2 = nn.MultiheadAttention(n_embd, n_heads, dropout)
        
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx):
        x = self.embd(idx)
        
        x = self.mha1(x)
        x = self.ln1(x)
        
        x = self.mha2(x)
        x = self.ln2(x)
        
        x = self.ffwd(x)
        x = self.ln3(x)
        
        logits = self.lm_head(x)
        
        return logits
            
class GPT(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        
        self.blocks = nn.Sequential(*[Transformer() for _ in range(n_blocks)])
        
    def forward(self, idx, targets=None):
        logits = self.blocks(idx)
    
        if targets is None:
            loss = None
                    
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
gpt = GPT(n_blocks)

context = tokenizer.encode("Hello!")
    
"""class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)

        head_size = n_embd // n_head
        self.s_attn = MultiHeadAttention(n_head, head_size)
        self.f_fwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        enc_emb = self.pos_embd(self.enc_embd(idx))
        enc_output = self.encoders(enc_emb)

        x = self.pos_embd(self.dec_embd(idx))
        for decoder in self.decoders:
            x = decoder(x, enc_output)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx"""