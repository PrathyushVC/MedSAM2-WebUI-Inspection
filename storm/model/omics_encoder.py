from __future__ import annotations

import torch
import torch.nn as nn


class OmicsEncoder(nn.Module):
    """Transformer encoder for log-normalised gene expression profiles.

    Two operating modes are available:

    - **bulk**: the full ``(G,)`` expression vector is compressed to ``embed_dim``
      by an MLP before passing through a transformer. Efficient for large gene
      panels; recommended for G > 2000.
    - **tokens**: each gene is treated as its own transformer token. More
      expressive but memory scales with G; use for smaller curated panels.

    Args:
        num_genes: Number of genes in the expression panel (G).
        embed_dim: Output embedding dimension.
        hidden_dim: MLP bottleneck width used in bulk mode.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout probability applied throughout.
        use_gene_tokens: When ``True``, token mode is activated.
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        if use_gene_tokens:
            self._mode = "tokens"
            self.gene_proj = nn.Linear(1, embed_dim)
            self.gene_pos = nn.Embedding(num_genes, embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self._mode = "bulk"
            self.input_proj = nn.Sequential(
                nn.Linear(num_genes, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of expression profiles to fixed-size embeddings.

        Args:
            x: Float32 tensor of shape ``(B, G)`` containing log-normalised
               gene expression values.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        if self._mode == "bulk":
            h = self.input_proj(x).unsqueeze(1)
            h = self.transformer(h)
            return self.norm(h).squeeze(1)

        B, G = x.shape
        gene_idx = torch.arange(G, device=x.device)
        tokens = self.gene_proj(x.unsqueeze(-1)) + self.gene_pos(gene_idx)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.norm(tokens)[:, 0]
