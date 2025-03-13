import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
import numpy as np
from wandb.sdk.internal.profiler import torch_trace_handler

# ================== 配置参数 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 28
channels = 1
batch_size = 256
lr = 1e-4
epochs = 50
num_classes = 10

# ================== 数据加载 ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)  # [-1, 1] 归一化
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 64, ff_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
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

        cross_attn, _ = self.cross_attn(x, condition, condition)
        x = x + self.dropout(cross_attn)
        x = self.norm(x)

        ffn = self.ffn(x)
        x = x + self.dropout(ffn)
        x = self.norm(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, ff_dim: int = 256, num_heads: int = 8, dropout: float = 0.1 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, condition)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int = 64, ff_dim: int = 256, num_heads: int = 8, dropout: float = 0.1 ) -> None:
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
    def __init__(self, image_size: int = 28, embed_dim: int = 64, ff_dim: int = 256, num_heads: int = 8, dropout: float = 0.1 ) -> None:
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
            nn.Embedding(10, embed_dim),
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

        t_mid = t_start + (t_end - t_start) / 2

        x_mid = x_t + self(x=x_t, t=t_start, label=label) * (t_end[0][0] - t_start[0][0]) / 2

        x_t = x_t + (t_end[0][0] - t_start[0][0]) * self(x=x_mid, t=t_mid, label=label)

        return x_t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flow = Flow().to(device)
if os.path.exists("flow.pth"):
    flow.load_state_dict(torch.load("flow.pth"))
optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
loss_fn = nn.MSELoss()


@torch.no_grad()
def draw_flow(flow):
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow = flow.to(device)  # 将模型移动到 GPU

    # 初始化x为随机噪声 (10个样本，对应10个label)
    x = torch.randn(10, 1, image_size, image_size, device=device)  # 形状为 (10, 1, 28, 28)

    # 定义时间步 (从0到1，10等分)
    time_steps = torch.linspace(0, 1.0, 10, device=device)  # 形状为 (10,)

    # 定义label的值 (0到9)
    labels = torch.arange(10, dtype=torch.long, device=device).unsqueeze(1)  # 形状为 (10, 1)

    # 创建一个大的图像网格 (10行 x 10列)
    big_image = np.zeros((10 * image_size, 10 * image_size))

    for j in range(10):
        big_image[0: image_size, j * image_size:(j + 1) * image_size] = x[j, 0].cpu().numpy()

    # 遍历每个时间步
    for i in range(len(time_steps) - 1):
        t_start = time_steps[i]
        t_end = time_steps[i + 1]

        # 使用step方法更新x
        x = flow.step(x_t=x, t_start=t_start, t_end=t_end, label=labels)

        # 将当前时间步的图像放入大图像网格中
        for j in range(10):  # 遍历每个label
            image = x[j, 0].cpu().numpy()  # 获取第j个label的图像 (28, 28)
            big_image[(i + 1) * image_size:(i + 2) * image_size, j * image_size:(j + 1) * image_size] = image

    # 绘制大图像
    plt.figure(figsize=(10, 10))
    plt.imshow(big_image, cmap='gray')
    plt.axis('off')
    plt.show()


# draw_flow(flow)

for epoch in range(epochs):
    flow.train()
    for x_1, labels in train_loader:

        x_1 = x_1.to(device)
        labels = labels.to(device)
        B, C, H, W = x_1.shape

        x_0 = torch.randn_like(x_1, device = device)
        t = torch.rand(B, 1, 1, 1, device = device)
        x_t = (1 - t) * x_0 + t * x_1

        dx_t = x_1 - x_0

        t = t.view(B, 1)
        labels = labels.view(B, 1)

        optimizer.zero_grad()
        pred_dx_t = flow(x_t, t, labels)

        loss = loss_fn(pred_dx_t, dx_t)
        loss.backward()

        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
    if epoch % 10 == 0:
        draw_flow(flow)


torch.save(flow.state_dict(), 'flow.pth')