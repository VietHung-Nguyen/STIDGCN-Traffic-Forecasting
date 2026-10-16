import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConvolution(nn.Module):
    """Module Tích chập Đồ thị Động (Dynamic Graph Convolution - DGCN)."""
    def __init__(self, in_channels, out_channels):
        super(DynamicGraphConvolution, self).__init__()
        # TODO: Cài đặt cơ chế sinh ma trận kề động theo thời gian thực
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, dynamic_adj):
        # x shape: (Batch, Nodes, Features)
        out = self.linear(x)
        out = torch.matmul(dynamic_adj, out)
        return F.relu(out)

class InteractiveLearningBlock(nn.Module):
    """Cơ chế học tương tác chia nhánh (Interactive Learning) phân rã chuỗi thời gian."""
    def __init__(self, channels):
        super(InteractiveLearningBlock, self).__init__()
        # TODO: Cài đặt nhánh chẵn/lẻ (even/odd sequences) và tương tác học đặc trưng
        self.conv1D = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1D(x)


class STIDGCN(nn.Module):
    """Mô hình cốt lõi STIDGCN."""

    def __init__(self, num_nodes, in_dim, out_dim, seq_len=12, horizon=12):
        super(STIDGCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon

        self.dgcn = DynamicGraphConvolution(in_dim, 64)
        self.interactive_learning = InteractiveLearningBlock(64)

        # Tầng dự báo đầu ra (chuyển từ seq_len sang horizon)
        self.regression_layer = nn.Linear(seq_len, horizon)

    def forward(self, x):
        # Input x shape: (Batch, Seq_len, Nodes, Features) -> (Batch, 12, 307, 1)

        # TẠM THỜI: Xây dựng một luồng forward đơn giản (Bypass) để code train.py chạy được không báo lỗi.
        # Chúng ta sẽ thay thế bằng logic đồ thị động (DGCN) và học tương tác (Interactive) ở các bước sau.

        # Chuyển đổi chiều: (Batch, Seq, Nodes, Feat) -> (Batch, Nodes, Feat, Seq)
        out = x.permute(0, 2, 3, 1)  # Shape: (Batch, 307, 1, 12)

        # Cho qua lớp Linear để dự đoán tương lai: 12 bước quá khứ -> 12 bước tương lai
        out = self.regression_layer(out)  # Shape: (Batch, 307, 1, 12)

        # Trả về đúng chiều ban đầu: (Batch, Horizon, Nodes, Features)
        out = out.permute(0, 3, 1, 2)  # Shape: (Batch, 12, 307, 1)

        return out
