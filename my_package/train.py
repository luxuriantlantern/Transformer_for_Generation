import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from my_package.model import Flow
from my_package.utils import draw_flow
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 28
    channels = 1
    batch_size = 256
    lr = 1e-4
    epochs = 50
    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)  # [-1, 1] 归一化
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    flow = Flow().to(device)
    path_to_pth = "try_2out"
    path_to_pthout = path_to_pth + ".pth"
    path_to_pth = path_to_pth + ".pth"

    if os.path.exists(path_to_pth):
        flow.load_state_dict(torch.load(path_to_pth))
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter("runs/flow_experiment")

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

            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + epoch)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
        if epoch % 5 == 0:
            draw_flow(flow)

    writer.close()

    torch.save(flow.state_dict(), path_to_pthout)