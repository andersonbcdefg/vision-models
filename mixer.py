import torch
from torch import nn
from layers import FFN, PreNormAndAdd
from einops.layers.torch import Rearrange

class MixerLayer(nn.Module):
  def __init__(self, embed_dim, seq_len, token_hidden, channel_hidden, dropout):
    super().__init__()
    self.token_mixer = PreNormAndAdd(embed_dim, nn.Sequential(
        Rearrange("b l c -> b c l"),
        FFN(seq_len, token_hidden, seq_len),
        Rearrange("b c l -> b l c")
      ))
    self.channel_mixer = PreNormAndAdd(embed_dim, FFN(embed_dim, channel_hidden, embed_dim))

  def forward(self, X):
    return self.channel_mixer(self.token_mixer(X))

class MLPMixer(nn.Module):
  def __init__(
      self, img_size, patch_size, n_channels, n_classes,
      embed_dim, token_hidden, channel_hidden, n_layers, dropout):
    super().__init__()
    assert img_size % patch_size == 0, "Image must be evenly divisible into patches"
    self.seq_len = int((img_size // patch_size) ** 2)
    self.embed = nn.Sequential(
      Rearrange("b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)", h2=patch_size, w2=patch_size),
      nn.Linear(patch_size**2 * n_channels, embed_dim)
    )
    self.mixer_layers = nn.ModuleList(
        [MixerLayer(embed_dim, self.seq_len, token_hidden, channel_hidden, dropout) for _ in range(n_layers)]
    )
    self.final_ln = nn.LayerNorm(embed_dim)
    self.output_head = nn.Linear(embed_dim, n_classes)

  def forward(self, X):
    seq = self.embed(X)
    for mixer_layer in self.mixer_layers:
      seq = mixer_layer(seq)
    pooled = torch.mean(self.final_ln(seq), dim=1)
    return self.output_head(pooled)