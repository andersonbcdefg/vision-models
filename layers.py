import torch
from torch import nn

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, embed_dim, d_k, dropout=0.1):
    super().__init__()
    self.n_heads, self.d_k = n_heads, d_k
    self.W_qkv = nn.Linear(embed_dim, 3 * n_heads * d_k)
    self.attn_dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(n_heads * d_k, embed_dim)
    self.resid_dropout = nn.Dropout(dropout)

  def forward(self, X):
    Q, K, V = rearrange(self.W_qkv(X), "b l (h ddd) -> b h l ddd", h=self.n_heads).split(self.d_k, dim=-1)
    attn = torch.einsum("bhqd,bhkd->bhqk", Q, K) / self.d_k**0.5 
    attn_weights = self.attn_dropout(torch.softmax(attn, dim=-1))
    attn_out = torch.einsum("bhqk,bhkv->bhqv", attn_weights, V)
    proj = self.fc(rearrange(attn_out, "b h l d -> b l (h d)"))
    return self.resid_dropout(proj)

class FFN(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, num_hiddens=1, dropout=0.1, final_dropout=True):
    super().__init__()
    assert num_hiddens > 0, "Must have > 0 hidden layers."
    self.layers = nn.ModuleList([
        nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    ])
    
    for h in range(num_hiddens - 1):
        self.layers.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ))
    
    self.layers.append(nn.Linear(hidden_dim, out_dim))
    if final_dropout:
        self.layers.append(nn.Dropout(dropout))
    
  def forward(self, X):
    for layer in self.layers:
        X = layer(X)
    return X

class PreNormAndAdd(nn.Module):
  def __init__(self, embed_dim, sublayer):
    super().__init__()
    self.norm = nn.LayerNorm(embed_dim)
    self.sublayer = sublayer

  def forward(self, X):
    return X + self.sublayer(self.norm(X))

class TransformerBlock(nn.Module):
  def __init__(self, n_heads, embed_dim, d_k, ffn_hidden_dim, dropout):
    super().__init__()
    self.net = nn.Sequential(
        PreNormAndAdd(embed_dim, MultiHeadAttention(n_heads, embed_dim, d_k, dropout)),
        PreNormAndAdd(embed_dim, FFN(embed_dim, ffn_hidden_dim, embed_dim, num_hiddens=1, dropout=dropout))
    )
  
  def forward(self, X):
    return self.net(X)