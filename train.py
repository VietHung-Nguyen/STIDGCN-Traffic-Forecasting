import argparse
import csv
import json
import os
import time

import torch

from models.dynamic_graph import DynamicGraphConvolutionV2
from utils.Experiment import (build_model, get_dataloaders, get_device,
                              masked_mae, set_seed)
from utils.graph_utils import describe_adj, load_adj

try:
    import config
except ImportError:
    config = None
DEFAULT_DATA_DIR = getattr(config, "DATA_DIR", "data/PEMS08")
DEFAULT_ADJ_PATH = getattr(config, "ADJ_PATH", os.path.join(DEFAULT_DATA_DIR, "adj_PEMS08.pkl"))

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="thư mục chứa train/val/test.npz")
parser.add_argument("--adj_path", default=DEFAULT_ADJ_PATH, help="file adj_PEMS08.pkl")
parser.add_argument("--save_dir", default="checkpoints")
parser.add_argument("--tag", default="", help="tiền tố tên lần chạy, mặc định = tên loại đồ thị")
parser.add_argument("--channels", type=int, nargs="+", default=[64])
parser.add_argument("--features", type=int, nargs="+", default=[0], help="0 = lưu lượng; 0 1 2 = thêm đặc trưng thời gian")
parser.add_argument("--num_nodes", type=int, default=170)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=0, help="0 = không dừng sớm")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--clip", type=float, default=5.0)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=6666)
parser.add_argument("--graph", choices=["adaptive", "dynamic"], default="adaptive")
parser.add_argument("--svd_init", action="store_true", help="khởi tạo A_adp bằng SVD của A")
parser.add_argument("--tau", type=float, default=0.5, help="nhiệt độ Gumbel-Softmax")
args = parser.parse_args()

device = get_device()
os.makedirs(args.save_dir, exist_ok=True)
print(f"Thiết bị: {device} | đồ thị: {args.graph} | đặc trưng: {args.features}")

loaders, scaler, gen = get_dataloaders(args.data_dir, args.features, args.batch_size, args.seed)
print(f"Chuẩn hóa theo tập train: mean = {scaler.mean:.2f}, std = {scaler.std:.2f}")
if args.graph == "dynamic":
    describe_adj(load_adj(args.adj_path))


def run_epoch(model, loader, optimizer=None):
    """Một lượt qua dữ liệu; có optimizer thì huấn luyện, không thì chỉ đánh giá. Trả về MAE trung bình."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = scaler.inverse_transform(model(x))          # về thang gốc trước khi so với nhãn
            loss = masked_mae(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def mean_alpha(model):
    a = [m.alpha for m in model.modules() if isinstance(m, DynamicGraphConvolutionV2)]
    return sum(a) / len(a) if a else None


for C in args.channels:
    set_seed(args.seed)
    gen.manual_seed(args.seed)                 # mọi cấu hình duyệt dữ liệu theo cùng thứ tự
    name = f"{args.tag or args.graph}_c{C}"
    cfg = dict(name=name, channels=C, in_dim=len(args.features), features=args.features,
               num_nodes=args.num_nodes, dropout=args.dropout, graph=args.graph,
               svd_init=args.svd_init, tau=args.tau, adj_path=args.adj_path,
               data_dir=args.data_dir, seed=args.seed, epochs=args.epochs,
               batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)

    model = build_model(cfg).to(device)
    n_params = model.param_num()
    print(f"\n===== {name}: {n_params:,} tham số =====")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_path = os.path.join(args.save_dir, f"{name}.pth")
    history, best_val, best_epoch, wait, epoch_times = [], float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mae = run_epoch(model, loaders["train"], optimizer)
        epoch_times.append(time.time() - t0)
        val_mae = run_epoch(model, loaders["val"])
        alpha = mean_alpha(model)

        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, wait = val_mae, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
        msg = (f"Chu kỳ {epoch:3d} | train MAE {train_mae:.3f} | val MAE {val_mae:.3f} "
               f"| {epoch_times[-1]:.1f} s" + (" | *" if improved else ""))
        if alpha is not None:
            msg += f" | alpha {alpha:.3f}"
        print(msg)
        history.append(dict(epoch=epoch, train_mae=train_mae, val_mae=val_mae,
                            seconds=epoch_times[-1], alpha=alpha))
        if args.patience and wait >= args.patience:
            print(f"Dừng sớm: val MAE không cải thiện sau {args.patience} chu kỳ")
            break

    with open(os.path.join(args.save_dir, f"{name}_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    cfg.update(params=n_params, best_epoch=best_epoch, best_val_mae=best_val,
               mean_epoch_seconds=sum(epoch_times) / len(epoch_times),
               final_alpha=mean_alpha(model), checkpoint=ckpt_path)
    with open(os.path.join(args.save_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Tốt nhất ở chu kỳ {best_epoch}: val MAE {best_val:.3f} -> đã lưu {ckpt_path}")