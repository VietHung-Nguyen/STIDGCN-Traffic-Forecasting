"""Kiểm tra nhanh mô hình có đồ thị động trước khi huấn luyện thật."""
import argparse

import torch

from models.dynamic_graph import DynamicGraphConvolutionV2
from models.stidgcn import STIDGCN
from utils.graph_utils import describe_adj, load_adj, row_normalize

parser = argparse.ArgumentParser()
parser.add_argument("--adj_path", required=True)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--svd_init", action="store_true")
args = parser.parse_args()

torch.manual_seed(6666)
adj = load_adj(args.adj_path)
describe_adj(adj)                                    # PEMS08 kỳ vọng N = 170

model = STIDGCN(num_nodes=170, in_dim=1, out_dim=1, channels=args.channels,
                graph="dynamic", adj_norm=row_normalize(adj),
                init_adj=adj if args.svd_init else None)

# 1) Số tham số: so với công thức (hidden_dim = 32, 8 mô-đun đồ thị, 2 khối)
C = args.channels
expected = 32 * C * C + 275 * C + 36069
actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[params] thực tế = {actual:,} | công thức = {expected:,} | khớp = {actual == expected}")

# 2) Lan truyền thuận trên một lô ngẫu nhiên
x = torch.randn(4, 12, 170, 1)
model.train()
y = model(x)
print(f"[shape] vào {tuple(x.shape)} -> ra {tuple(y.shape)} (kỳ vọng (4, 12, 170, 1))")

# 3) Kiểm tra A_dyn của mô-đun đồ thị đầu tiên
g = next(m for m in model.modules() if isinstance(m, DynamicGraphConvolutionV2))
a = g.last_adj                                       # (B, N, N)
row_err = (a.sum(-1) - 1).abs().max().item()
diff = (a[0] - a[1]).abs().mean().item()
print(f"[A_dyn] kích thước {tuple(a.shape)} | sai lệch tổng hàng = {row_err:.2e} (≈ 0)")
print(f"[A_dyn] khác nhau giữa 2 mẫu = {diff:.2e} (> 0 nghĩa là đồ thị thực sự động)")
print(f"[alpha] = {g.alpha:.3f} (khởi tạo 0,5)")

# 4) Gradient chảy tới bộ sinh đồ thị và alpha
y.mean().backward()
print(f"[grad] alpha có gradient: {g.alpha_logit.grad is not None} | "
      f"bộ sinh đồ thị có gradient: {g.generator.gcn.weight.grad is not None}")