import torch
import numpy as np
import matplotlib.pyplot as plt
from my_package.model import Flow

def draw_flow(flow : Flow, image_size : int = 28):
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
            image = x[j, 0].detach().cpu().numpy()  # 获取第j个label的图像 (28, 28)
            big_image[(i + 1) * image_size:(i + 2) * image_size, j * image_size:(j + 1) * image_size] = image

    # 绘制大图像
    plt.figure(figsize=(10, 10))
    plt.imshow(big_image, cmap='gray')
    plt.axis('off')
    plt.show()