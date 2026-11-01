"""Đánh giá mô hình đã huấn luyện trên tập test PEMS08."""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from models.dynamic_graph import DynamicGraphConvolutionV2
from utils.experiment import all_metrics, build_model, get_dataloaders, get_device, set_seed, sync

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="file .json do train.py lưu")
parser.add_argument("--save_dir", default="results")
parser.add_argument("--plot_node", type=int, default=0)
parser.add_argument("--plot_start", type=int, default=11, help="mẫu bắt đầu vẽ (>= 11)")
parser.add_argument("--no_plot", action="store_true")
args = parser.parse_args()

with open(args.config, encoding="utf-8") as f:
    cfg = json.load(f)
name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
os.makedirs(args.save_dir, exist_ok=True)

set_seed(cfg.get("seed", 6666))
device = get_device()
loaders, scaler, _ = get_dataloaders(cfg["data_dir"], cfg.get("features", [0]),
                                     cfg.get("batch_size", 64), cfg.get("seed", 6666))

model = build_model(cfg).to(device)
model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
model.eval()                                   # đồ thị động: tự bỏ nhiễu Gumbel -> kết quả tất định
print(f"{name}: {model.param_num():,} tham số | thiết bị: {device}")

# ---------------------------------------------------------------- dự báo toàn bộ tập test
preds, trues = [], []
sync(device)
t0 = time.time()
with torch.no_grad():
    for x, y in loaders["test"]:              # tập test không xáo trộn -> giữ đúng thứ tự thời gian
        preds.append(scaler.inverse_transform(model(x.to(device))).cpu())
        trues.append(y)
sync(device)
infer_seconds = time.time() - t0
pred = torch.cat(preds)                       # (S, 12, N, 1)
true = torch.cat(trues)

# ---------------------------------------------------------------- sai số theo từng tầm
rows = []
print(f"\n{'Tầm':>4} {'Phút':>5} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8}")
for h in range(pred.shape[1]):
    mae, rmse, mape = all_metrics(pred[:, h], true[:, h])
    rows.append(dict(horizon=h + 1, minutes=5 * (h + 1), mae=mae, rmse=rmse, mape=mape))
    print(f"{h + 1:>4} {5 * (h + 1):>5} {mae:8.2f} {rmse:8.2f} {mape:8.2f}")

avg = {k: float(np.mean([r[k] for r in rows])) for k in ("mae", "rmse", "mape")}
print(f"\nTrung bình 12 tầm: MAE {avg['mae']:.2f} | RMSE {avg['rmse']:.2f} | MAPE {avg['mape']:.2f}%")
print(f"Thời gian suy luận toàn tập test: {infer_seconds:.2f} s ({device})")

alphas = [m.alpha for m in model.modules() if isinstance(m, DynamicGraphConvolutionV2)]
if alphas:
    print(f"alpha trung bình = {np.mean(alphas):.3f} (gần 1: tin A_adp; gần 0: tin A_learn)")

with open(os.path.join(args.save_dir, f"horizon_metrics_{name}.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary = dict(name=name, **avg, inference_seconds=infer_seconds, device=str(device),
               params=model.param_num(), alpha=float(np.mean(alphas)) if alphas else None)
with open(os.path.join(args.save_dir, f"test_{name}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- dự báo 24 giờ tại một nút
if not args.no_plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, s, L = args.plot_node, max(args.plot_start, 11), 288
    p, t = pred[..., n, 0].numpy(), true[..., n, 0].numpy()   # (S, 12)
    # Mẫu i, tầm h dự báo thời điểm i + h. Để 3 đường cùng chỉ một thời điểm:
    gt = t[s:s + L, 0]                 # thực tế tại thời điểm i + 1
    h1 = p[s:s + L, 0]                 # dự báo 5 phút cho thời điểm i + 1
    h12 = p[s - 11:s - 11 + L, 11]     # dự báo 60 phút cho CÙNG thời điểm (lấy từ mẫu sớm hơn 11 bước)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    steps = np.arange(L)
    ax.plot(steps, gt, label="Thực tế (Ground Truth)")
    ax.plot(steps, h1, "--", label="Dự báo trước 5 phút (horizon 1)")
    ax.plot(steps, h12, ":", label="Dự báo trước 60 phút (horizon 12)")
    ax.set_xlim(0, L)
    ax.set_xlabel("Bước thời gian (5 phút / bước)")
    ax.set_ylabel("Lưu lượng xe")
    ax.set_title(f"Dự báo lưu lượng 24 giờ – nút #{n} (Channels={cfg['channels']}, đồ thị {cfg.get('graph', 'adaptive')})")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(args.save_dir, f"forecast24h_{name}_node{n}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {out}")