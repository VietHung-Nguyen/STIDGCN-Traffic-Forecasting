"""
Bản STIDGCN ĐƠN GIẢN HÓA (tự cài đặt) — dùng cho phần thực nghiệm mở rộng.

Lưu ý: đây KHÔNG phải bản tái hiện đầy đủ của bài báo. So với code tác giả, bản này:
  - không có 4 lớp Conv1..Conv4 theo chiều thời gian,
  - dùng 4 module đồ thị riêng (bài báo dùng 1 DGCN dùng chung),
  - đồ thị là adaptive adjacency tĩnh (kiểu Graph WaveNet), không sinh từ dữ liệu đầu vào,
  - không có time embedding (giờ trong ngày, thứ trong tuần) nếu chỉ dùng features=(0,),
  - cấu trúc 2 khối nối tiếp thay vì cây 2 tầng (12 -> 6 -> 3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Splitting(nn.Module):
    """Chia chuỗi thời gian thành 2 nhánh Chẵn (Even) và Lẻ (Odd)."""

    def __init__(self):
        super(Splitting, self).__init__()

    def forward(self, x):
        # x shape: (Batch, Seq_len, Nodes, Channels)
        x_even = x[:, ::2, :, :]  # Các bước thời gian ở vị trí chẵn (0, 2, 4...)
        x_odd = x[:, 1::2, :, :]  # Các bước thời gian ở vị trí lẻ (1, 3, 5...)
        return x_even, x_odd


class DynamicGraphConvolution(nn.Module):
    """Tích chập đồ thị với ma trận kề thích nghi (học từ node embeddings)."""

    def __init__(self, channels, num_nodes, embed_dim=10, dropout=0.1):
        super(DynamicGraphConvolution, self).__init__()
        # Vector nhúng nút để tự học cấu trúc đồ thị
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.node_vec2 = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)

        self.proj = nn.Linear(channels * 2, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Nodes, Channels)
        # 1. Ma trận kề thích nghi kích thước (Nodes, Nodes), mỗi hàng tổng = 1
        adj = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=-1)

        # 2. Lan truyền thông tin giữa các nút: out[n] = sum_m adj[n, m] * x[m]
        # (ĐÃ SỬA: bản cũ dùng 'nn,bsnc->bsnc' -> einsum lấy ĐƯỜNG CHÉO của adj,
        #  nên mỗi nút chỉ nhân với chính nó, không có tích chập đồ thị)
        gcn_out = torch.einsum('nm,bsmc->bsnc', adj, x)

        # 3. Kết hợp đặc trưng gốc và đặc trưng đồ thị
        out = torch.cat([x, gcn_out], dim=-1)
        out = F.relu(self.proj(out))
        return self.dropout(out)


class InteractiveLearningBlock(nn.Module):
    """Khối học tương tác giữa nhánh chẵn/lẻ kết hợp tích chập đồ thị."""

    def __init__(self, channels, num_nodes, dropout=0.1, splitting=True):
        super(InteractiveLearningBlock, self).__init__()
        self.splitting = splitting
        self.split = Splitting()

        # 4 bộ biến đổi tương tác (bài báo gốc dùng 4 Conv riêng + 1 DGCN dùng chung)
        self.phi = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
        self.psi = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
        self.eta = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
        self.rho = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)

    def forward(self, x):
        if not self.splitting:
            return self.phi(x)

        # 1. Chia chuỗi (12 bước -> 2 nhánh 6 bước)
        x_even, x_odd = self.split(x)

        # 2. Giai đoạn 1: tỷ lệ hóa chéo (công thức kiểu SCINet: exp(tanh(.)))
        x_odd_s = x_odd * torch.exp(torch.tanh(self.phi(x_even)))
        x_even_s = x_even * torch.exp(torch.tanh(self.psi(x_odd)))

        # 3. Giai đoạn 2: cộng/trừ đặc trưng tương tác
        x_even_update = x_even_s + self.eta(x_odd_s)
        x_odd_update = x_odd_s - self.rho(x_even_s)

        # 4. Ghép xen kẽ 2 nhánh lại thành chuỗi 12 bước
        B, half_T, N, C = x_even_update.shape
        out = torch.zeros((B, half_T * 2, N, C), device=x.device, dtype=x.dtype)
        out[:, ::2, :, :] = x_even_update
        out[:, 1::2, :, :] = x_odd_update
        return out


class STIDGCN(nn.Module):
    """Kiến trúc tổng thể (bản đơn giản hóa)."""

    def __init__(self, num_nodes=170, in_dim=1, out_dim=1, channels=48, seq_len=12, horizon=12,
                 dropout=0.1, splitting=True):
        super(STIDGCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon

        # 1. Nhúng đầu vào: in_dim đặc trưng -> channels kênh ẩn
        self.start_conv = nn.Linear(in_dim, channels)

        # 2. Hai khối học tương tác nối tiếp, có kết nối tắt (residual)
        self.block1 = InteractiveLearningBlock(channels, num_nodes, dropout, splitting)
        self.block2 = InteractiveLearningBlock(channels, num_nodes, dropout, splitting)

        # 3. Đầu ra: chiếu chiều thời gian seq_len -> horizon, rồi kênh -> out_dim
        self.time_proj = nn.Linear(seq_len, horizon)
        self.end_conv = nn.Linear(channels, out_dim)

    def param_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x: (Batch, Seq_len=12, Nodes, in_dim)
        h = self.start_conv(x)  # -> (B, 12, N, C)

        h = h + self.block1(h)
        h = h + self.block2(h)

        # (B, 12, N, C) -> (B, N, C, 12) để chiếu theo thời gian
        h = h.permute(0, 2, 3, 1)
        h = F.relu(self.time_proj(h))  # -> (B, N, C, horizon)

        # -> (B, horizon, N, C)
        h = h.permute(0, 3, 1, 2)

        out = self.end_conv(h)  # -> (B, horizon, N, out_dim)
        return out