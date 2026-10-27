"""
Đánh giá các checkpoint do train.py tạo ra.

Ví dụ:
    python evaluate.py                          # đánh giá channels 16 32 48 64
    python evaluate.py --channels 48 --plot_channels 48 --node 0

Kết quả:
    - In MAE / RMSE / MAPE cho đủ 12 horizon (tương ứng Fig. 4) và trung bình 12 horizon
      (tính giống code tác giả: trung bình của 12 giá trị theo từng horizon).
    - checkpoints/horizon_metrics_c{ch}.csv
    - checkpoints/horizon_curves.png        (sai số theo horizon, mọi cấu hình)
    - checkpoints/channels_analysis_fig6.png (ảnh hưởng của channels, tương ứng Fig. 6)
    - checkpoints/prediction_plot_c{ch}.png  (dự báo 24 giờ tại 1 nút)
"""
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae, masked_mape, masked_rmse

from config import DATA_DIR

# Kết quả STIDGCN trên PEMS08 trong Table 3 của bài báo (để đối chiếu)
PAPER_PEMS08 = {"mae": 14.03, "rmse": 23.35, "mape": 9.15}

_loader_cache = {}


def load_test_data(data_dir, features, batch_size=64):
    """Tải dữ liệu test một lần cho mỗi bộ features."""
    key = (data_dir, tuple(features))
    if key not in _loader_cache:
        _, _, test_loader, scaler = get_dataloader(data_dir=data_dir, batch_size=batch_size,
                                                   features=tuple(features))
        _loader_cache[key] = (test_loader, scaler)
    return _loader_cache[key]


def load_config(model_path, channels, args):
    """Đọc file .json lưu cùng checkpoint; nếu không có thì dùng tham số mặc định."""
    cfg_path = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  (Không thấy {cfg_path}, dùng cấu hình mặc định: features={args.features})")
    return {"channels": channels, "features": args.features, "in_dim": len(args.features),
            "num_nodes": None, "seq_len": 12, "horizon": 12, "dropout": 0.1}


def evaluate(model_path, channels, args, device, save_plot=False):
    cfg = load_config(model_path, channels, args)
    test_loader, scaler = load_test_data(args.data_dir, cfg["features"], args.batch_size)
    num_nodes = cfg["num_nodes"] or test_loader.dataset.tensors[0].shape[2]

    model = STIDGCN(num_nodes=num_nodes, in_dim=cfg["in_dim"], out_dim=1, channels=cfg["channels"],
                    seq_len=cfg["seq_len"], horizon=cfg["horizon"], dropout=cfg["dropout"],
                    splitting=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    total_params = model.param_num()

    preds_list, truths_list = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x.to(device))
            preds_list.append(scaler.inverse_transform(output.cpu()))
            truths_list.append(batch_y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    preds = torch.cat(preds_list, dim=0)    # (S, 12, N, 1)
    truths = torch.cat(truths_list, dim=0)  # (S, 12, N, 1)

    # Sai số theo từng horizon (giống cách tác giả in kết quả)
    horizon_rows = []
    for h in range(preds.shape[1]):
        p, t = preds[:, h], truths[:, h]
        horizon_rows.append({
            "horizon": h + 1,
            "minutes": (h + 1) * 5,
            "mae": masked_mae(p, t, null_val=0.0).item(),
            "rmse": masked_rmse(p, t, null_val=0.0).item(),
            "mape": masked_mape(p, t, null_val=0.0).item() * 100,
        })
    mae = float(np.mean([r["mae"] for r in horizon_rows]))
    rmse = float(np.mean([r["rmse"] for r in horizon_rows]))
    mape = float(np.mean([r["mape"] for r in horizon_rows]))

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ ({model_path} | Channels={cfg['channels']} | features={cfg['features']})")
    print(f" - Số tham số     : {total_params:,}")
    print(f" - Inference time : {inference_time:.2f} giây")
    print("-" * 70)
    print(f"{'Horizon':^9}|{'Phút':^7}|{'MAE':^10}|{'RMSE':^10}|{'MAPE (%)':^10}")
    for r in horizon_rows:
        mark = "  <-" if r["horizon"] in (3, 6, 12) else ""
        print(f"{r['horizon']:^9}|{r['minutes']:^7}|{r['mae']:^10.4f}|{r['rmse']:^10.4f}|"
              f"{r['mape']:^10.2f}{mark}")
    print("-" * 70)
    print(f" Trung bình 12 horizon: MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.2f}%")
    if "PEMS08" in args.data_dir.upper():
        print(f" Bài báo (Table 3)    : MAE {PAPER_PEMS08['mae']:.2f} | RMSE {PAPER_PEMS08['rmse']:.2f} "
              f"| MAPE {PAPER_PEMS08['mape']:.2f}%")
    print("=" * 70)

    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/horizon_metrics_c{cfg['channels']}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(horizon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(horizon_rows)

    if save_plot:
        plot_prediction(preds, truths, cfg["channels"], args.node)

    return {"channels": cfg["channels"], "params": total_params, "mae": mae, "rmse": rmse,
            "mape": mape, "time": inference_time, "horizons": horizon_rows}


def plot_prediction(preds, truths, channels, node_id, steps=288):
    """Dự báo 24 giờ (288 bước) tại 1 nút, ở horizon 1 và horizon 12."""
    steps = min(steps, preds.shape[0] - 11)
    t = np.arange(steps)
    y_true = truths[:steps, 0, node_id, 0].numpy()
    y_h1 = preds[:steps, 0, node_id, 0].numpy()
    # Mẫu i ở horizon 12 dự báo thời điểm i + 11 (so với horizon 1) -> dịch lại cho thẳng hàng
    y_h12 = preds[:steps, 11, node_id, 0].numpy()

    plt.figure(figsize=(12, 4.5))
    plt.plot(t, y_true, label="Thực tế (Ground Truth)", color="#1f77b4", linewidth=1.5)
    plt.plot(t, y_h1, label="Dự báo trước 5 phút (horizon 1)", color="#d62728", linestyle="--", linewidth=1.2)
    plt.plot(t + 11, y_h12, label="Dự báo trước 60 phút (horizon 12)", color="#2ca02c",
             linestyle=":", linewidth=1.2)
    plt.xlim(0, steps)
    plt.title(f"Dự báo lưu lượng 24 giờ – nút #{node_id} (Channels={channels})", fontsize=13)
    plt.xlabel("Bước thời gian (5 phút / bước)", fontsize=11)
    plt.ylabel("Lưu lượng xe", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"checkpoints/prediction_plot_c{channels}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ dự báo: {path}")


def plot_horizon_curves(results):
    """Sai số theo horizon cho mọi cấu hình (tương ứng Fig. 4)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, ax, title in zip(("mae", "rmse", "mape"), axes, ("MAE", "RMSE", "MAPE (%)")):
        for r in results:
            ax.plot([h["horizon"] for h in r["horizons"]], [h[key] for h in r["horizons"]],
                    marker="o", markersize=3, label=f"Channels={r['channels']}")
        ax.set_title(f"{title} theo horizon")
        ax.set_xlabel("Horizon (×5 phút)")
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("checkpoints/horizon_curves.png", dpi=300)
    plt.close()
    print("Đã lưu biểu đồ theo horizon: checkpoints/horizon_curves.png")


def plot_channels_comparison(results):
    """Ảnh hưởng của số kênh (tương ứng Fig. 6 trong bài báo)."""
    ch = [str(r["channels"]) for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    specs = [("mae", "MAE", "#1f77b4", "o"), ("rmse", "RMSE", "#ff7f0e", "s"),
             ("mape", "MAPE (%)", "#2ca02c", "^")]
    for ax, (key, label, color, marker) in zip(axes[:3], specs):
        ax.plot(ch, [r[key] for r in results], marker=marker, color=color, linewidth=2)
        ax.set_title(f"Ảnh hưởng của Channels đến {label}", fontsize=11)
        ax.set_xlabel("Số kênh (Channels)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[3].bar(ch, [r["params"] for r in results], color="#9467bd")
    axes[3].set_title("Số tham số theo Channels", fontsize=11)
    axes[3].set_xlabel("Số kênh (Channels)")
    axes[3].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("checkpoints/channels_analysis_fig6.png", dpi=300)
    plt.close()
    print("Đã lưu biểu đồ phân tích Channels: checkpoints/channels_analysis_fig6.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument("--features", type=int, nargs="+", default=[0],
                        help="chỉ dùng khi checkpoint không có file .json đi kèm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_channels", type=int, default=48, help="cấu hình dùng để vẽ biểu đồ dự báo")
    parser.add_argument("--node", type=int, default=0, help="nút dùng để vẽ biểu đồ dự báo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for ch in args.channels:
        path = f"checkpoints/best_stidgcn_c{ch}.pth"
        if not os.path.exists(path):
            print(f"Chưa tìm thấy {path}, hãy chạy train.py trước!")
            continue
        results.append(evaluate(path, ch, args, device, save_plot=(ch == args.plot_channels)))

    if not results:
        return

    # Bảng tổng hợp
    print("\n" + "#" * 80)
    print(" BẢNG TỔNG HỢP KHẢO SÁT CHANNELS (trung bình 12 horizon)")
    print("#" * 80)
    print(f"{'Channels':^10}|{'Số tham số':^14}|{'MAE':^10}|{'RMSE':^10}|{'MAPE (%)':^10}|"
          f"{'H12 MAE':^10}|{'Test (s)':^10}")
    print("-" * 80)
    for r in results:
        print(f"{r['channels']:^10}|{r['params']:^14,d}|{r['mae']:^10.4f}|{r['rmse']:^10.4f}|"
              f"{r['mape']:^10.2f}|{r['horizons'][-1]['mae']:^10.4f}|{r['time']:^10.2f}")
    if "PEMS08" in args.data_dir.upper():
        print(f"{'Bài báo':^10}|{'-':^14}|{PAPER_PEMS08['mae']:^10.2f}|{PAPER_PEMS08['rmse']:^10.2f}|"
              f"{PAPER_PEMS08['mape']:^10.2f}|{'-':^10}|{'-':^10}")
    print("#" * 80 + "\n")

    with open("checkpoints/summary_channels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channels", "params", "mae", "rmse", "mape", "test_time_sec"])
        for r in results:
            writer.writerow([r["channels"], r["params"], round(r["mae"], 4), round(r["rmse"], 4),
                             round(r["mape"], 2), round(r["time"], 2)])

    plot_horizon_curves(results)
    if len(results) > 1:
        plot_channels_comparison(results)


if __name__ == "__main__":
    main()