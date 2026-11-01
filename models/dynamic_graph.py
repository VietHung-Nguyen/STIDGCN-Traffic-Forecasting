import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):

    def __init__(self, num_nodes, embed_dim=10, init_adj=None):
        super().__init__()
        if init_adj is not None:
            a = torch.as_tensor(init_adj, dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(a)
            sq = torch.sqrt(S[:embed_dim])
            e1 = U[:, :embed_dim] * sq              # (N, d)
            e2 = sq[:, None] * Vh[:embed_dim, :]    # (d, N)  ->  E1 E2 ~ xấp xỉ hạng d của A
        else:
            e1 = torch.randn(num_nodes, embed_dim)
            e2 = torch.randn(embed_dim, num_nodes)
        self.node_vec1 = nn.Parameter(e1)
        self.node_vec2 = nn.Parameter(e2)

    def forward(self):
        return F.softmax(F.relu(self.node_vec1 @ self.node_vec2), dim=-1)  # (N, N)


class GraphGenerator(nn.Module):
    """Sinh A_learn riêng cho từng mẫu từ đặc trưng ẩn H và ma trận kề vật lý A."""

    def __init__(self, channels, adj_norm, hidden_dim=32, tau=0.5):
        super().__init__()
        # Ma trận kề vật lý đã chuẩn hóa theo hàng: là hằng số, không phải tham số
        self.register_buffer("adj_norm", torch.as_tensor(adj_norm, dtype=torch.float32))
        self.gcn = nn.Linear(2 * channels, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim
        self.tau = tau

    def forward(self, h):                                              # h: (B, T, N, C)
        z = torch.einsum("nm,btmc->btnc", self.adj_norm, h)            # lan truyền một bước trên A
        z = F.relu(self.gcn(torch.cat([h, z], dim=-1)))                # GCN(H, A) với K = 1
        z = z.mean(dim=1)                                              # gộp trục thời gian -> (B, N, C)
        e = self.mlp(z)                                                # véc-tơ nút theo mẫu -> (B, N, d)
        scores = e @ e.transpose(1, 2) / math.sqrt(self.hidden_dim)    # (B, N, N)
        logits = torch.log(F.softmax(scores, dim=-1) + 1e-8)           # log A'
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        # Khi đánh giá: bỏ nhiễu Gumbel để kết quả tất định, giữ cùng nhiệt độ tau
        return F.softmax(logits / self.tau, dim=-1)


class DynamicGraphConvolutionV2(nn.Module):
    """Thay thế trực tiếp cho DynamicGraphConvolution: dùng A_dyn riêng cho từng mẫu."""

    def __init__(self, channels, num_nodes, adj_norm, embed_dim=10, dropout=0.1,
                 tau=0.5, init_adj=None, hidden_dim=32):
        super().__init__()
        self.adaptive = AdaptiveAdjacency(num_nodes, embed_dim, init_adj)
        self.generator = GraphGenerator(channels, adj_norm, hidden_dim, tau)
        self.alpha_logit = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0,5 lúc khởi tạo
        self.proj = nn.Linear(channels * 2, channels)
        self.dropout = nn.Dropout(dropout)
        self.last_adj = None                              # lưu A_dyn gần nhất để kiểm tra

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit).item()

    def forward(self, x):                                 # x: (B, T, N, C)
        a_adp = self.adaptive()                           # (N, N)
        a_learn = self.generator(x)                       # (B, N, N)
        alpha = torch.sigmoid(self.alpha_logit)
        a_dyn = alpha * a_adp.unsqueeze(0) + (1 - alpha) * a_learn   # (B, N, N), mỗi hàng tổng = 1
        self.last_adj = a_dyn.detach()
        z = torch.einsum("bnm,btmc->btnc", a_dyn, x)      # chú ý: 'bnm', ma trận riêng cho từng mẫu
        out = F.relu(self.proj(torch.cat([x, z], dim=-1)))
        return self.dropout(out)


def make_graph_module(channels, num_nodes, dropout, graph="adaptive",
                      adj_norm=None, init_adj=None, tau=0.5, hidden_dim=32):
    """Chọn loại mô-đun đồ thị. graph='adaptive' giữ đúng mô hình cũ để so sánh."""
    if graph == "adaptive":
        from models.stidgcn import DynamicGraphConvolution   # lớp cũ, chỉ có A_adp
        return DynamicGraphConvolution(channels, num_nodes, dropout=dropout)
    if graph == "dynamic":
        if adj_norm is None:
            raise ValueError("graph='dynamic' cần ma trận kề adj_norm (đọc từ adj_PEMS08.pkl)")
        return DynamicGraphConvolutionV2(channels, num_nodes, adj_norm, dropout=dropout,
                                         tau=tau, init_adj=init_adj, hidden_dim=hidden_dim)
    raise ValueError(f"graph không hợp lệ: {graph}")