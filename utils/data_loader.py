import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


class StandardScaler:
    """Chuẩn hóa dữ liệu Z-score để mô hình hội tụ nhanh hơn."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_dataset(data, seq_len, horizon):
    """
    Cắt chuỗi thời gian thành các cặp (Input, Target) theo cửa sổ trượt (sliding window).
    - seq_len: số lượng time steps dùng làm input (ví dụ: 12 bước = 1 tiếng).
    - horizon: số lượng time steps cần dự đoán (ví dụ: 12 bước = 1 tiếng).
    """
    x, y = [], []
    num_samples = data.shape[0]

    for i in range(num_samples - seq_len - horizon + 1):
        x.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + horizon])

    return np.array(x), np.array(y)


def get_dataloader(data_dir, batch_size=32):
    """Load dữ liệu đã được cắt sẵn từ thư mục chứa train/val/test.npz"""

    print(f"Đang tải dữ liệu từ thư mục: {data_dir}")

    # Đọc cả 3 file
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))

    # Các bộ dữ liệu chia sẵn thường đã cắt thành x và y
    x_train, y_train = train_data['x'], train_data['y']
    x_val, y_val = val_data['x'], val_data['y']
    x_test, y_test = test_data['x'], test_data['y']

    # ==========================================
    # 1. KIỂM TRA DỮ LIỆU KHUYẾT THIẾU (MISSING VALUES)
    # ==========================================
    print("\n" + "=" * 50)
    print("🔍 ĐANG KIỂM TRA DỮ LIỆU KHUYẾT THIẾU (MISSING VALUES)...")

    has_nan = np.isnan(x_train).any()
    total_nans = np.isnan(x_train).sum()
    print(f" - Dữ liệu có chứa giá trị NaN không? : {has_nan} (Tổng số lượng NaN: {total_nans})")

    total_zeros = (x_train == 0).sum()
    print(f" - Tổng số lượng giá trị bằng 0 (0.0)    : {total_zeros}")
    print("=" * 50 + "\n")

    # ==========================================
    # 2. IN RA 5 MẪU ĐẦU TIÊN CỦA DỮ LIỆU
    # ==========================================
    print("📋 IN RA 5 MẪU DỮ LIỆU ĐẦU TIÊN (x_train[:5]):")
    print(f" - Kích thước mảng x_train ban đầu: {x_train.shape}")
    print(x_train[:5])
    print("=" * 50 + "\n")

    # Chỉ lấy đặc trưng đầu tiên (Traffic Flow - Lưu lượng)
    # Shape ban đầu thường là (Batch, Seq_len, Nodes, Features)
    x_train = x_train[..., 0:1].astype(np.float32)
    y_train = y_train[..., 0:1].astype(np.float32)

    x_val = x_val[..., 0:1].astype(np.float32)
    y_val = y_val[..., 0:1].astype(np.float32)

    x_test = x_test[..., 0:1].astype(np.float32)
    y_test = y_test[..., 0:1].astype(np.float32)

    print(f"Kích thước tập Train sau khi lọc đặc trưng: X={x_train.shape}, Y={y_train.shape}")

    # Chuẩn hóa dữ liệu (Tính Mean/Std trên x_train)
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Đóng gói DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_scaled), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val_scaled), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_scaled), torch.tensor(y_test)),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler