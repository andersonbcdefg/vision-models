import torch
from torch import nn
from layers import TransformerBlock
from einops.layers.torch import Rearrange

class VisionTransformer(nn.Module):
  def __init__(self, img_size, patch_size, n_channels, n_classes,
               n_layers, n_heads, embed_dim, d_k, ffn_hidden_dim, dropout=0.1):
    super().__init__()
    assert img_size % patch_size == 0, "Image must divide evenly into patches."
    self.pos_embs = nn.Parameter(torch.randn(((img_size // patch_size)**2, embed_dim)))
    self.embed = nn.Sequential(
      Rearrange("b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)", h2=patch_size, w2=patch_size),
      nn.Linear(patch_size**2 * n_channels, embed_dim)
    )
    self.transformer = nn.Sequential(*[
        TransformerBlock(n_heads, embed_dim, d_k, ffn_hidden_dim, dropout) for _ in range(n_layers)
      ])
    # LayerNorm here?
    self.output_head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(embed_dim, n_classes)
    )

  def forward(self, X):
    seq = self.embed(X) + self.pos_embs.unsqueeze(0)
    for block in self.transformer:
      seq = block(seq)
    return self.output_head(torch.mean(seq, dim=1))