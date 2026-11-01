"""Các hàm dùng chung cho train.py và evaluate.py.

Gồm: cố định hạt giống, đọc dữ liệu .npz, chuẩn hóa Z-score theo tập train,
các độ đo có mặt nạ (bỏ qua nhãn bằng 0) và dựng mô hình từ file cấu hình .json.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.stidgcn import STIDGCN
from utils.graph_utils import load_adj, row_normalize


# ---------------------------------------------------------------- môi trường
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    # Máy AMD Radeon không có CUDA -> tự chạy trên CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    """Chỉ đồng bộ khi chạy trên GPU NVIDIA; trên CPU không cần."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- dữ liệu
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"{split}.npz"))
    return d["x"].astype(np.float32), d["y"].astype(np.float32)


def get_dataloaders(data_dir, features=(0,), batch_size=64, seed=6666):
    """Đọc train/val/test.npz, chọn kênh đầu vào, chuẩn hóa kênh lưu lượng theo thống kê tập train.

    - x: (S, 12, N, 3) với kênh 0 = lưu lượng, 1 = thời điểm trong ngày, 2 = thứ trong tuần
    - y: (S, 12, N, 1) lưu lượng, KHÔNG chuẩn hóa (giá trị 0 = thiếu dữ liệu, dùng cho mặt nạ)
    """
    features = list(features)
    data = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    flow_train = data["train"][0][..., 0]
    scaler = StandardScaler(flow_train.mean(), flow_train.std())

    gen = torch.Generator()
    gen.manual_seed(seed)
    loaders = {}
    for split, (x, y) in data.items():
        x = x[..., features].copy()
        if 0 in features:                                   # chỉ chuẩn hóa kênh lưu lượng
            k = features.index(0)
            x[..., k] = scaler.transform(x[..., k])
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y[..., :1].copy()))
        is_train = split == "train"
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                                    generator=gen if is_train else None)
    return loaders, scaler, gen


# ---------------------------------------------------------------- độ đo có mặt nạ
def _mask(true):
    mask = (true != 0).float()
    return mask / mask.mean()                    # chia trung bình để giữ đúng thang sai số


def masked_mae(pred, true):
    loss = torch.abs(pred - true) * _mask(true)
    return torch.nan_to_num(loss).mean()


def masked_rmse(pred, true):
    loss = (pred - true) ** 2 * _mask(true)
    return torch.sqrt(torch.nan_to_num(loss).mean())


def masked_mape(pred, true):
    loss = torch.abs(pred - true) / true * _mask(true)   # vị trí true = 0 có mặt nạ 0 -> nan -> 0
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).mean() * 100


def all_metrics(pred, true):
    return (masked_mae(pred, true).item(),
            masked_rmse(pred, true).item(),
            masked_mape(pred, true).item())


# ---------------------------------------------------------------- mô hình
def build_model(cfg):
    """Dựng STIDGCN từ dict cấu hình (đọc từ file .json do train.py lưu)."""
    graph = cfg.get("graph", "adaptive")          # file .json cũ không có khóa này
    adj_norm = init_adj = None
    if graph == "dynamic":
        adj = load_adj(cfg["adj_path"])
        adj_norm = row_normalize(adj)
        init_adj = adj if cfg.get("svd_init", False) else None
    return STIDGCN(num_nodes=cfg["num_nodes"], in_dim=cfg["in_dim"], out_dim=1,
                   channels=cfg["channels"], seq_len=12, horizon=12,
                   dropout=cfg.get("dropout", 0.1), graph=graph,
                   adj_norm=adj_norm, init_adj=init_adj, tau=cfg.get("tau", 0.5))