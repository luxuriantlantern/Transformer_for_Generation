from my_package.model import Flow
from my_package.utils import draw_flow
import torch
import os

def test_flow(path_to_pth : str) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow = Flow().to(device)
    if os.path.exists(path_to_pth):
        flow.load_state_dict(torch.load(path_to_pth))
    draw_flow(flow)