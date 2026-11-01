"""Đọc và chuẩn hóa ma trận kề vật lý (adj_PEMS08.pkl)."""
import pickle

import numpy as np


def load_adj(path):
    """Đọc ma trận kề N x N từ file .pkl.

    Hỗ trợ hai định dạng thường gặp:
    - lưu trực tiếp một mảng N x N;
    - tuple (sensor_ids, sensor_id_to_ind, adj_mx) theo kiểu DCRNN / Graph WaveNet.
    """
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:          # file pickle tạo từ Python 2
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (tuple, list)):
        obj = obj[-1]
    adj = np.asarray(obj, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Ma trận kề phải vuông N x N, nhận được {adj.shape}")
    return adj


def row_normalize(adj, add_self_loop=True):
    """D^-1 (A + I): mỗi hàng tổng bằng 1, dùng cho lan truyền trên đồ thị."""
    a = adj.astype(np.float32).copy()
    if add_self_loop:
        a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return a / deg


def describe_adj(adj):
    """In vài thông số để kiểm tra đã đọc đúng file."""
    n = adj.shape[0]
    off = adj.copy()
    np.fill_diagonal(off, 0)
    nnz = int((off > 0).sum())
    sym = bool(np.allclose(adj, adj.T))
    print(f"[adj] N = {n} | phần tử khác 0 ngoài đường chéo = {nnz} | đối xứng = {sym} "
          f"| min = {adj.min():.4f} | max = {adj.max():.4f}")