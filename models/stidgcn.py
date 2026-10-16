import torch
import torch.nn as nn
import torch.nn.functional as F


class Splitting(nn.Module):
    """Chia chuỗi thời gian thành 2 nhánh Chẵn (Even) và Lẻ (Odd)."""

    def __init__(self):
        super(Splitting, self).__init__()

    def forward(self, x):
        # x shape: (Batch, Seq_len, Nodes, Channels)
        x_even = x[:, ::2, :, :]  # Lấy các bước thời gian ở vị trí chẵn (0, 2, 4...)
        x_odd = x[:, 1::2, :, :]  # Lấy các bước thời gian ở vị trí lẻ (1, 3, 5...)
        return x_even, x_odd


class DynamicGraphConvolution(nn.Module):
    """Module Tích chập Đồ thị Động (Dynamic_GCN) học ma trận kề thích nghi."""

    def __init__(self, channels, num_nodes, embed_dim=10, dropout=0.1):
        super(DynamicGraphConvolution, self).__init__()
        # Khởi tạo vector nhúng nút (Node Embeddings) để tự học cấu trúc đồ thị động
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.node_vec2 = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)

        self.proj = nn.Linear(channels * 2, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Nodes, Channels)
        # 1. Sinh ma trận kề động (Dynamic Adjacency Matrix) kích thước (Nodes, Nodes)
        dynamic_adj = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=-1)

        # 2. Nhân tích chập đồ thị theo chiều không gian (Nodes)
        # Dùng einsum để nhân ma trận (Nodes, Nodes) với (Batch, Seq_len, Nodes, Channels)
        gcn_out = torch.einsum('nn,bsnc->bsnc', dynamic_adj, x)

        # 3. Kết hợp đặc trưng gốc và đặc trưng đồ thị
        out = torch.cat([x, gcn_out], dim=-1)
        out = F.relu(self.proj(out))
        return self.dropout(out)


class InteractiveLearningBlock(nn.Module):
    """Khối Học Tương Tác Không-Thời Gian (Interactive Learning + DGCN)."""

    def __init__(self, channels, num_nodes, dropout=0.1, splitting=True):
        super(InteractiveLearningBlock, self).__init__()
        self.splitting = splitting
        self.split = Splitting()

        # Các bộ lọc biến đổi tương tác giữa nhánh chẵn và nhánh lẻ
        self.phi = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
        self.psi = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
        self.eta = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
        self.rho = DynamicGraphConvolution(channels, num_nodes, dropout=dropout)

    def forward(self, x):
        if not self.splitting:
            return self.phi(x)

        # 1. Chia chuỗi thời gian (12 bước -> 2 nhánh 6 bước)
        x_even, x_odd = self.split(x)

        # 2. Tương tác Giai đoạn 1: Tỷ lệ hóa (Scaling) chéo giữa Chẵn và Lẻ
        x_odd_s = x_odd * torch.exp(torch.tanh(self.phi(x_even)))
        x_even_s = x_even * torch.exp(torch.tanh(self.psi(x_odd)))

        # 3. Tương tác Giai đoạn 2: Cộng/Trừ đặc trưng tương tác
        x_even_update = x_even_s + self.eta(x_odd_s)
        x_odd_update = x_odd_s - self.rho(x_even_s)

        # 4. Ghép nối (Interleave) 2 nhánh trở lại chuỗi thời gian 12 bước
        B, half_T, N, C = x_even_update.shape
        out = torch.zeros((B, half_T * 2, N, C), device=x.device)
        out[:, ::2, :, :] = x_even_update
        out[:, 1::2, :, :] = x_odd_update
        return out


class STIDGCN(nn.Module):
    """Kiến trúc tổng thể STIDGCN."""

    def __init__(self, num_nodes=170, in_dim=1, out_dim=1, channels=48, seq_len=12, horizon=12, dropout=0.1,
                 splitting=True):
        super(STIDGCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon

        # 1. Tầng nhúng đầu vào (Chuyển từ 1 feature lên số kênh ẩn channels=48)
        self.start_conv = nn.Linear(in_dim, channels)

        # 2. Khối Học tương tác & Đồ thị động (2 tầng nối tiếp kèm kết nối tắt Residual)
        self.block1 = InteractiveLearningBlock(channels, num_nodes, dropout, splitting)
        self.block2 = InteractiveLearningBlock(channels, num_nodes, dropout, splitting)

        # 3. Tầng dự báo đầu ra (Chuyển chiều thời gian từ seq_len sang horizon và về lại out_dim=1)
        self.time_proj = nn.Linear(seq_len, horizon)
        self.end_conv = nn.Linear(channels, out_dim)

    def forward(self, x):
        # Input x shape: (Batch, Seq_len=12, Nodes=170, Features=1)
        h = self.start_conv(x)  # -> (Batch, 12, 170, 48)

        # Qua các khối học tương tác (có Residual connection)
        h = h + self.block1(h)
        h = h + self.block2(h)

        # Chiếu theo chiều thời gian: (Batch, 12, 170, 48) -> (Batch, 170, 48, 12)
        h = h.permute(0, 2, 3, 1)
        h = F.relu(self.time_proj(h))  # -> (Batch, 170, 48, Horizon=12)

        # Đưa về lại chiều chuẩn: (Batch, Horizon=12, Nodes=170, Channels=48)
        h = h.permute(0, 3, 1, 2)

        # Dự báo đầu ra: -> (Batch, 12, 170, 1)
        out = self.end_conv(h)
        return out