import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """
    Transformer-based sequence classifier for variable-length inputs (e.g. early HAR).

    Maps an input sequence [B, L, input_dim] to class logits [B, num_classes] via
    input projection, sinusoidal positional encoding, stacked TransformerEncoder layers,
    optional pooling (mean/max/cls), and a small MLP head.
    """

    def __init__(self,
                 input_dim,
                 model_dim,
                 num_heads,
                 num_layers,
                 num_classes,
                 ff_dim=256,
                 dropout=0.1,
                 pooling='mean',
                 use_cls_token=False,
                 use_batchnorm=False,
                 classifier_hidden=128):
        """
        Args:
            input_dim: Feature dimension of each timestep.
            model_dim: Hidden size used in the Transformer (d_model).
            num_heads: Number of attention heads.
            num_layers: Number of TransformerEncoder layers.
            num_classes: Number of output classes.
            ff_dim: Feed-forward hidden dim in each encoder layer.
            dropout: Dropout rate applied in encoder and classifier.
            pooling: How to aggregate the sequence into a single vector.
                'mean' = mean over time; 'max' = max over time; 'cls' = use
                first position (requires use_cls_token=True).
            use_cls_token: If True, prepend a learnable [CLS] token and use
                its output when pooling='cls'.
            use_batchnorm: If True, apply BatchNorm1d after input projection.
            classifier_hidden: Hidden size of the 2-layer MLP classifier head.
        """
        super(TransformerClassifier, self).__init__()
        self.model_dim = model_dim
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.bn = nn.BatchNorm1d(model_dim) if use_batchnorm else None

        # Optional cls token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # x: [B, L, input_dim]
        x = self.input_proj(x)  # [B, L, model_dim]
        
        if self.bn:
            B, L, D = x.shape
            x = self.bn(x.transpose(1,2)).transpose(1,2)  # Apply BN over model_dim

        if self.use_cls_token:
            B = x.size(0)
            cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, model_dim]
            x = torch.cat([cls_token, x], dim=1)  # prepend cls_token

        x = self.pos_encoder(x)
        x = self.transformer(x)  # [B, L, model_dim]

        if self.pooling == 'mean':
            out = x.mean(dim=1)
        elif self.pooling == 'max':
            out, _ = x.max(dim=1)
        elif self.pooling == 'cls' and self.use_cls_token:
            out = x[:, 0, :]  # Use cls_token output
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling}")

        return self.classifier(out)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al.). Adds fixed, non-learned
    position information to sequences so the Transformer can use token order.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model: Model dimension (must match input feature size).
            dropout: Dropout applied after adding positional encoding.
            max_len: Maximum sequence length to precompute. Longer sequences
                are not supported.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal formula: PE(pos, 2i)=sin(pos/10000^(2i/d)), PE(pos, 2i+1)=cos(pos/10000^(2i/d))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Sequence tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Same shape as x, with positional encoding added and dropout applied.
        """
        # x: [B, L, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
