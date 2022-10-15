import torch
import copy
import numpy as np
from torch import nn
from layers import TransformerBlock
from einops.layers.torch import Rearrange
from einops import repeat

class MAEVisionTransformer(nn.Module):
  def __init__(self, img_size, patch_size, n_channels, mask_prob,
               enc_depth, enc_n_heads, enc_embed_dim, enc_d_k, enc_ffn_hidden_dim,
               dec_depth, dec_n_heads, dec_embed_dim, dec_d_k, dec_ffn_hidden_dim, 
               pretrain_out_hidden_dim, dropout=0.1):
    super().__init__()
    assert img_size % patch_size == 0, "Image must divide evenly into patches."
    self.seq_len = (img_size // patch_size) ** 2
    self.subsample_size = int(np.floor((1 - mask_prob) * self.seq_len))
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.enc_embed_dim = enc_embed_dim

    # Encoder
    self.enc_pos_embs = nn.Parameter(torch.randn((self.seq_len, enc_embed_dim)))
    self.embed = nn.Sequential(
      Rearrange("b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)", h2=patch_size, w2=patch_size),
      nn.Linear(patch_size**2 * n_channels, enc_embed_dim)
    )
    self.encoder = nn.Sequential(*[
        TransformerBlock(enc_n_heads, enc_embed_dim, enc_d_k, enc_ffn_hidden_dim, dropout) for _ in range(enc_depth)
      ])

    # Decoder
    self.mask_token = nn.Parameter(torch.randn((dec_embed_dim,)))
    self.dec_pos_embs = nn.Parameter(torch.randn((self.seq_len, dec_embed_dim)))
    self.enc_to_dec = nn.Linear(enc_embed_dim, dec_embed_dim)

    self.decoder = nn.Sequential(*[
        TransformerBlock(dec_n_heads, dec_embed_dim, d_k, dec_ffn_hidden_dim, dropout) for _ in range(dec_depth)
    ])

    # Pretraining
    self.pretrain_head = nn.Sequential(
        nn.Linear(dec_embed_dim, pretrain_out_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dec_ffn_hidden_dim, patch_size**2 * n_channels)
    )

  def forward(self, X, pretrain=True):    
    seq = self.embed(patches) + self.enc_pos_embs.unsqueeze(0)

    # If training, keep subsample of patches
    if pretrain:
      perm = torch.randperm(self.seq_len)
      seq = seq[:, perm[:self.subsample_size], :]
        
    # Apply encoder
    for block in self.encoder:
      seq = block(seq)

    if not pretrain:
      return seq

    # Add back masked patches, unshuffle, add positional embeddings
    masked_chunk = repeat(self.mask_token, "d -> b l d", b=X.shape[0], l=self.seq_len - self.subsample_size)
    seq = torch.cat([self.enc_to_dec(seq), masked_chunk], dim=1)
    seq = seq[:, torch.argsort(perm), :] + self.dec_pos_embs.unsqueeze(0)

    # Apply decoder
    for block in self.decoder:
      seq = block(seq)

    # Output
    return self.output_head(seq)

class MAEClassifier(nn.Module):
  def __init__(self, pretrained_model, out_hidden_dim, n_classes, dropout=0.1):
    super().__init__()
    self.encoder = copy.deepcopy(pretrained_encoder)
    self.classification_head = nn.Sequential(
      nn.Linear(self.encoder.enc_embed_dim, out_hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(out_hidden_dim, n_classes)
    )

  def forward(self, X):
    seq = self.encoder(X, pretrain=False)
    return self.classification_head(torch.mean(seq, dim=1))