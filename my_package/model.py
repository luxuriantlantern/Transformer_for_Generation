import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, ff_dim: int = 256, num_heads: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.cond_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()  # 门控权重
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self_attn, _ = self.self_attn(x, x, x)
        x = x + self.dropout(self_attn)
        x = self.norm(x)

        cross_attn, _ = self.cross_attn(
            query=x,
            key=condition + x.mean(dim=1, keepdim=True),  # 强制对齐条件与内容均值
            value=condition
        )
        gate = self.cond_gate(torch.cat([x, condition], dim=-1))  # 动态门控
        cross_attn = gate * cross_attn  # 条件重要性加权
        x = x + self.dropout(cross_attn)
        x = self.norm(x)

        ffn = self.ffn(x)
        x = x + self.dropout(ffn)
        x = self.norm(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, ff_dim: int = 256, num_heads: int = 16, dropout: float = 0.1 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, condition)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int = 256, ff_dim: int = 256, num_heads: int = 16, dropout: float = 0.1, image_size : int = 28 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(num_heads)]
        )
        self.out = nn.Linear(embed_dim, image_size * image_size)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, condition)
        return x

class Transformer(nn.Module):
    def __init__(self, image_size: int = 28, num_classes : int = 10, embed_dim: int = 256, ff_dim: int = 256, num_heads: int = 16, dropout: float = 0.1 ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.input_proj = nn.Linear(image_size * image_size, embed_dim)
        self.concat_time = nn.Linear(embed_dim * 2, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.label_embed = nn.Sequential(
            nn.Embedding(num_classes, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoder = TransformerEncoder(embed_dim, ff_dim, num_heads, dropout)
        self.decoder = TransformerDecoder(embed_dim, ff_dim, num_heads, dropout)

        self.output_proj = nn.Linear(embed_dim, image_size * image_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.input_proj(x)

        t_embed = self.time_embed(t)

        label = label.squeeze(1)
        label_embed = self.label_embed(label)

        concat_time = torch.cat([x, t_embed], dim = 1)
        x = self.concat_time(concat_time)

        x = self.encoder(x, label_embed)
        x = self.decoder(x, label_embed)
        x = self.output_proj(x)
        x = x.view(B, C, H, W)

        return x

class Flow(nn.Module):
    def __init__(self, image_size: int = 28) -> None:
        super().__init__()
        self.net = Transformer(image_size = image_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.net(x, t, label)

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t_start = t_start.view(-1, 1).expand(x_t.shape[0], 1)
        t_end = t_end.view(-1, 1).expand(x_t.shape[0], 1)
        label = label.view(-1, 1).expand(x_t.shape[0], 1)

        delta_t = t_end[0][0] - t_start[0][0]
        x_t = x_t + delta_t * self(x=x_t, t=t_start, label=label)

        return x_t