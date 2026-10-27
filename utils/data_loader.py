import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.inspect_data import ensure_report


class StandardScaler:
    """Chuẩn hóa Z-score (chỉ áp dụng cho kênh lưu lượng)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_dataloader(data_dir, batch_size=64, features=(0,), check_data=True, seed=None):
    """Đọc train/val/test.npz đã cắt sẵn.

    features: các đặc trưng đầu vào. (0,) = chỉ lưu lượng;
              (0, 1, 2) = lưu lượng + thời điểm trong ngày + thứ trong tuần (giống code tác giả).
    check_data: kiểm tra dữ liệu qua báo cáo cache (chỉ chạy đầy đủ khi dữ liệu thay đổi).
    """
    if check_data:
        ensure_report(data_dir)  # in 2 dòng tóm tắt, dừng nếu dữ liệu có lỗi nghiêm trọng

    features = list(features)
    arrays = {}
    for s in ("train", "val", "test"):
        d = np.load(os.path.join(data_dir, f"{s}.npz"))
        arrays[f"x_{s}"] = d["x"][..., features].astype(np.float32)
        arrays[f"y_{s}"] = d["y"][..., 0:1].astype(np.float32)  # mục tiêu: chỉ lưu lượng

    # Chỉ chuẩn hóa kênh lưu lượng; time-of-day / day-of-week giữ nguyên
    scaler = StandardScaler(mean=arrays["x_train"][..., 0].mean(), std=arrays["x_train"][..., 0].std())
    for s in ("train", "val", "test"):
        arrays[f"x_{s}"][..., 0] = scaler.transform(arrays[f"x_{s}"][..., 0])

    g = torch.Generator().manual_seed(seed) if seed is not None else None

    def make(s, shuffle):
        ds = TensorDataset(torch.from_numpy(arrays[f"x_{s}"]), torch.from_numpy(arrays[f"y_{s}"]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)

    return make("train", True), make("val", False), make("test", False), scaler