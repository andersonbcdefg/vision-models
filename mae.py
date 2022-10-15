import torch
import numpy as np
from torch import nn
from layers import TransformerBlock
from einops.layers.torch import Rearrange
from einops import repeat

class PatchSample(nn.Module):
    def __init__(self, seq_len, mask_prob):
        super().__init__()
        self.n_unmasked = np.floor((1 - mask_prob) * seq_len)

    def forward(self, X, perm):    
        return X[:, perm[:self.n_unmasked], :]


class PatchUnsample(nn.Module):
    def __init__(self, seq_len, mask_prob):
        super().__init__()
        self.n_masked = seq_len - np.floor((1 - mask_prob) * seq_len)

    def forward(self, X, perm, mask_token):
        inv_perm = torch.argsort(perm)
        mask_chunk = repeat(mask_token, "d -> b l d", b=X.shape[0], l=self.n_masked)
        seq = torch.cat([seq, mask_chunk], dim=1)
        return seq[:, inv_perm, :]

class MAEVisionTransformer(nn.Module):
  def __init__(self, img_size, patch_size, n_channels, mask_prob,
               enc_depth, enc_n_heads, enc_embed_dim, enc_d_k, enc_ffn_hidden_dim,
               dec_depth, dec_n_heads, dec_embed_dim, dec_d_k, dec_ffn_hidden_dim, 
               pretrain_out_hidden_dim, finetune_out_hidden_dim, dropout=0.1):
    super().__init__()
    assert img_size % patch_size == 0, "Image must divide evenly into patches."
    self.subsample_size = int(np.floor((1 - mask_prob) * (img_size // self.patch_size)**2))
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.pos_embs = nn.Parameter(torch.randn(((img_size // patch_size)**2, embed_dim)))
    self.mask_token = nn.Parameter(torch.randn((embed_dim,)))
    self.embed = nn.Sequential(
      Rearrange("b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)", h2=patch_size, w2=patch_size),
      nn.Linear(patch_size**2 * n_channels, embed_dim)
    )
    self.encoder = nn.Sequential(*[
        TransformerBlock(enc_n_heads, enc_embed_dim, enc_d_k, enc_ffn_hidden_dim, dropout) for _ in range(enc_depth)
      ])
    self.decoder = nn.Sequential(*[
        TransformerBlock(dec_n_heads, dec_embed_dim, d_k, dec_ffn_hidden_dim, dropout) for _ in range(dec_depth)
    ])
    self.pretrain_head = nn.Sequential(
        nn.Linear(embed_dim, ffn_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(ffn_hidden_dim, patch_size**2 * n_channels)
    )
    self.finetune_head = nn.Sequential(

    )

  def forward(self, X, pretraining=True):    
    seq = self.embed(patches) + self.pos_embs.unsqueeze(0)
    b, l, d = seq.shape

    # If training, keep subsample of patches
    if training:
      with torch.no_grad():
        
    
    # Apply encoder
    for block in self.encoder:
      seq = block(seq)

    if not pretraining:
      return seq

    # Add back masked patches, positional embeddings, unshuffle
    

    # Apply decoder
    for block in self.decoder:
      seq = block(seq)

    # Output
    output =  self.output_head(seq) # batch, n_patches, patch_size
    return masked_idxs, output