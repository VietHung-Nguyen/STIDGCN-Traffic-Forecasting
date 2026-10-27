"""
Huấn luyện bản STIDGCN đơn giản hóa, khảo sát số kênh (channels).

Ví dụ:
    python train.py                                   # channels 16 32 48 64, 20 epoch
    python train.py --channels 48 --epochs 100
    python train.py --features 0 1 2                  # thêm giờ trong ngày + thứ trong tuần

Mỗi cấu hình lưu vào checkpoints/:
    best_stidgcn_c{ch}.pth   trọng số tốt nhất trên tập validation
    best_stidgcn_c{ch}.json  cấu hình + kết quả (evaluate.py đọc file này)
    train_log_stidgcn_c{ch}.csv, loss_curve_stidgcn_c{ch}.png
"""
import argparse
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae

from config import DATA_DIR


def set_seed(seed):
    """Cố định seed để kết quả có thể lặp lại."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, scaler, epochs, lr, device,
                exp_name="stidgcn", weight_decay=1e-4, clip=5.0, patience=0):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/best_{exp_name}.pth"
    csv_path = f"checkpoints/train_log_{exp_name}.csv"
    plot_path = f"checkpoints/loss_curve_{exp_name}.png"

    best_val_loss, best_epoch, since_best = float("inf"), 0, 0
    history = []

    for epoch in range(1, epochs + 1):
        start_t = time.time()

        # --- Train ---
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            output = model(batch_x)
            predict = scaler.inverse_transform(output)

            loss = masked_mae(predict, batch_y, null_val=0.0)
            loss.backward()
            if clip:
                # Cắt gradient giống code tác giả (clip = 5) để tránh bùng nổ gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predict = scaler.inverse_transform(model(batch_x))
                loss = masked_mae(predict, batch_y, null_val=0.0)
                total_val_loss += loss.item() * batch_x.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        epoch_time = time.time() - start_t
        history.append([epoch, avg_train_loss, avg_val_loss, epoch_time])
        print(f"[{exp_name}] Epoch {epoch:03d}/{epochs} | Train MAE: {avg_train_loss:.4f} | "
              f"Val MAE: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss, best_epoch, since_best = avg_val_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Lưu checkpoint tốt nhất: {ckpt_path}")
        else:
            since_best += 1
            if patience and since_best >= patience:
                print(f"  -> Dừng sớm: {patience} epoch liên tiếp không cải thiện")
                break

    # 1. Lưu lịch sử huấn luyện ra CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_MAE", "Val_MAE", "Time_sec"])
        writer.writerows(history)

    # 2. Vẽ biểu đồ hội tụ
    ep = [r[0] for r in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(ep, [r[1] for r in history], label="Train MAE", marker="o", markersize=3)
    plt.plot(ep, [r[2] for r in history], label="Validation MAE", marker="s", markersize=3)
    plt.axvline(best_epoch, color="gray", linestyle=":", label=f"Best epoch = {best_epoch}")
    plt.title(f"Biểu đồ hội tụ hàm mất mát ({exp_name})", fontsize=12)
    plt.xlabel("Chu kỳ huấn luyện (Epoch)", fontsize=11)
    plt.ylabel("Sai số tuyệt đối trung bình (MAE)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_loss,
        "epochs_run": len(history),
        "avg_epoch_time_sec": float(np.mean([r[3] for r in history])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument("--features", type=int, nargs="+", default=[0],
                        help="0 = lưu lượng, 1 = giờ trong ngày, 2 = thứ trong tuần")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=5.0, help="0 = không cắt gradient")
    parser.add_argument("--patience", type=int, default=0, help="0 = không dừng sớm")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Khảo sát siêu tham số Channels {args.channels} trên: {device} ---")

    set_seed(args.seed)
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        data_dir=args.data_dir, batch_size=args.batch_size,
        features=tuple(args.features), seed=args.seed,
    )
    _, seq_len, num_nodes, in_dim = train_loader.dataset.tensors[0].shape
    horizon = train_loader.dataset.tensors[1].shape[1]

    for ch in args.channels:
        print("\n" + "#" * 60)
        print(f" BẮT ĐẦU HUẤN LUYỆN CẤU HÌNH: CHANNELS = {ch}")
        print("#" * 60)

        set_seed(args.seed)  # mỗi cấu hình khởi tạo trọng số từ cùng một seed
        model = STIDGCN(num_nodes=num_nodes, in_dim=in_dim, out_dim=1, channels=ch,
                        seq_len=seq_len, horizon=horizon, dropout=args.dropout,
                        splitting=True).to(device)
        n_params = model.param_num()
        print(f"Số tham số: {n_params:,}")

        exp_name = f"stidgcn_c{ch}"
        result = train_model(model, train_loader, val_loader, scaler, args.epochs, args.lr, device,
                             exp_name=exp_name, weight_decay=args.weight_decay,
                             clip=args.clip, patience=args.patience)

        # Lưu cấu hình đi kèm checkpoint để evaluate.py dựng lại đúng mô hình
        config = {
            "channels": ch, "features": args.features, "in_dim": int(in_dim),
            "num_nodes": int(num_nodes), "seq_len": int(seq_len), "horizon": int(horizon),
            "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "clip": args.clip, "seed": args.seed,
            "epochs": args.epochs, "params": n_params, "data_dir": args.data_dir, **result,
        }
        with open(f"checkpoints/best_{exp_name}.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Best epoch {result['best_epoch']} | Val MAE {result['best_val_mae']:.4f}")


if __name__ == "__main__":
    main()