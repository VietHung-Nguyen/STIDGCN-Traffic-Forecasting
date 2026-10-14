import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class StandardScaler:
    """Chuẩn hóa dữ liệu Z-score để mô hình hội tụ nhanh hơn."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_adjacency_matrix(adj_filepath):
    """Đọc ma trận kề tĩnh tĩnh đồ thị giao thông."""
    # Placeholder: Điều chỉnh cấu trúc đọc theo định dạng file .csv hoặc .npz của PEMS
    try:
        adj = np.load(adj_filepath)
        return torch.tensor(adj, dtype=torch.float32)
    except FileNotFoundError:
        print(f"Warning: Không tìm thấy file đồ thị tại {adj_filepath}.")
        return None

def get_dataloader(data_filepath, batch_size=64, train_ratio=0.6, val_ratio=0.2):
    """Load và chia tập train/val/test."""
    # Placeholder: Load data PEMS thực tế (shape: Sequence_length x Nodes x Features)
    # data = np.load(data_filepath)['data']
    data = np.random.rand(1000, 307, 3) # Dữ liệu giả lập 307 nút mạng (PEMS04)
    
    # Chia tập dữ liệu
    num_samples = data.shape[0]
    train_steps = int(num_samples * train_ratio)
    val_steps = int(num_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_steps]
    val_data = data[train_steps:val_steps]
    test_data = data[val_steps:]
    
    # Chuẩn hóa
    scaler = StandardScaler(mean=train_data[..., 0].mean(), std=train_data[..., 0].std())
    
    # TODO: Cắt window trượt (Sliding window) cho chuỗi thời gian (Vd: 12 steps -> 12 steps)
    
    return train_data, val_data, test_data, scaler
