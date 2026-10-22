import time
import csv
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae


def train_model(model, train_loader, val_loader, scaler, epochs, lr, device, exp_name='stidgcn'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    ckpt_path = f'checkpoints/best_{exp_name}.pth'
    csv_path = f'checkpoints/train_log_{exp_name}.csv'
    plot_path = f'checkpoints/loss_curve_{exp_name}.png'

    history = []

    for epoch in range(epochs):
        start_t = time.time()
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            output = model(batch_x)
            predict = scaler.inverse_transform(output)

            loss = masked_mae(predict, batch_y, null_val=0.0)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- Vòng lặp Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                predict = scaler.inverse_transform(output)

                loss = masked_mae(predict, batch_y, null_val=0.0)
                total_val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        epoch_time = time.time() - start_t

        history.append([epoch + 1, avg_train_loss, avg_val_loss, epoch_time])
        print(f"[{exp_name}] Epoch {epoch + 1:03d}/{epochs} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Đã lưu checkpoint mới tốt nhất vào: {ckpt_path}")

    # 1. Lưu lịch sử huấn luyện ra file CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_MAE', 'Val_MAE', 'Time_sec'])
        writer.writerows(history)

    # 2. Vẽ biểu đồ hội tụ Loss
    epochs_range = [row[0] for row in history]
    train_losses = [row[1] for row in history]
    val_losses = [row[2] for row in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs_range, train_losses, label='Train MAE', marker='o', markersize=4)
    plt.plot(epochs_range, val_losses, label='Validation MAE', marker='s', markersize=4)
    plt.title(f"Biểu đồ hội tụ hàm mất mát ({exp_name})", fontsize=12)
    plt.xlabel("Chu kỳ huấn luyện (Epoch)", fontsize=11)
    plt.ylabel("Sai số tuyệt đối trung bình (MAE)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy khảo sát siêu tham số Channels trên: {device} ---")

    data_dir = 'data/PEMS08'
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        data_dir=data_dir,
        batch_size=64
    )

    # Chạy 3 cấu hình: 2 lần < 48 (16, 32) và 1 lần > 48 (64)
    # Lưu ý: Mốc channels = 48 đã có sẵn trong file checkpoints/best_stidgcn.pth nên không cần chạy lại
    for ch in [16, 32, 48, 64]:
        print("\n" + "#" * 60)
        print(f" BẮT ĐẦU HUẤN LUYỆN CẤU HÌNH: CHANNELS = {ch}")
        print("#" * 60)

        model = STIDGCN(
            num_nodes=170,
            in_dim=1,
            out_dim=1,
            channels=ch,      # Thay đổi số kênh ẩn theo từng lần chạy
            seq_len=12,
            horizon=12,
            dropout=0.1,
            splitting=True
        ).to(device)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=scaler,
            epochs=20,
            lr=0.001,
            device=device,
            exp_name=f'stidgcn_c{ch}'
        )