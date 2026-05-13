"""
Gene expression encoder for spatial transcriptomics spots.

Two operating modes:
  - bulk  : the full (G,) vector is projected to embed_dim in one shot via an
             MLP, then processed by a small transformer. Fast, works for large G.
  - tokens: each gene becomes a separate token (B, G, 1) → (B, G+1, E). More
             expressive but memory-heavy for large gene panels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OmicsEncoder(nn.Module):
    """
    Maps a (B, G) log-normalised expression vector to (B, embed_dim).

    Args:
        num_genes:       Number of genes in the panel (G).
        embed_dim:       Output embedding dimension.
        hidden_dim:      Intermediate MLP width (bulk mode only).
        depth:           Number of transformer layers.
        num_heads:       Attention heads.
        dropout:         Dropout probability.
        use_gene_tokens: If True, treat every gene as its own token.
    """

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gene_tokens: bool = False,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.use_gene_tokens = use_gene_tokens

        if use_gene_tokens:
            self._build_token_mode(num_genes, embed_dim, depth, num_heads, dropout)
        else:
            self._build_bulk_mode(num_genes, embed_dim, hidden_dim, depth, num_heads, dropout)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_bulk_mode(self, G, D, H, depth, heads, dropout):
        """Full vector → single token → transformer."""
        self._mode = "bulk"
        self.input_proj = nn.Sequential(
            nn.Linear(G, H),
            nn.GELU(),
            nn.LayerNorm(H),
            nn.Linear(H, D),
            nn.LayerNorm(D),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=heads,
            dim_feedforward=D * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(D)

    def _build_token_mode(self, G, D, depth, heads, dropout):
        """Each gene value becomes its own token."""
        self._mode = "tokens"
        self.gene_proj = nn.Linear(1, D)
        self.gene_pos = nn.Embedding(G, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=heads,
            dim_feedforward=D * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(D)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, G) log-normalised gene expression values
        Returns:
            (B, embed_dim) omics embeddings
        """
        if self._mode == "bulk":
            # (B, G) → (B, D) → add seq dim → (B, 1, D)
            h = self.input_proj(x).unsqueeze(1)
            h = self.transformer(h)
            h = self.norm(h)
            return h.squeeze(1)                           # (B, D)

        # token mode
        B, G = x.shape
        gene_idx = torch.arange(G, device=x.device)
        # (B, G, 1) → (B, G, D) + positional
        tokens = self.gene_proj(x.unsqueeze(-1)) + self.gene_pos(gene_idx)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, G+1, D)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]                               # CLS token (B, D)
