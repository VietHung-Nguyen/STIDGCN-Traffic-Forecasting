import time
import csv
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae
from evaluate import evaluate_and_plot


def train_model(model, train_loader, val_loader, scaler, epochs, lr, device, save_prefix="stidgcn_no_split"):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    model_save_path = f'checkpoints/best_{save_prefix}.pth'
    csv_save_path = f'checkpoints/train_log_{save_prefix}.csv'

    train_losses, val_losses = [], []

    with open(csv_save_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_MAE', 'Val_MAE', 'Time_sec'])

        for epoch in range(epochs):
            start_epoch = time.time()
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
            epoch_time = time.time() - start_epoch

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            writer.writerow([epoch + 1, round(avg_train_loss, 4), round(avg_val_loss, 4), round(epoch_time, 2)])

            print(
                f"[{save_prefix}] Epoch {epoch + 1:03d}/{epochs} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> Đã lưu checkpoint mới tốt nhất vào: {model_save_path}")

    # Vẽ biểu đồ quá trình hội tụ Loss bằng tiếng Việt
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train MAE', marker='o', markersize=4)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation MAE', marker='s', markersize=4)
    plt.title(f"Biểu đồ Hội tụ Hàm mất mát qua 20 Epoch ({save_prefix})", fontsize=12)
    plt.xlabel("Chu kỳ huấn luyện (Epoch)", fontsize=11)
    plt.ylabel("Sai số tuyệt đối trung bình (MAE)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'checkpoints/loss_curve_{save_prefix}.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "#" * 60)
    print(" BẮT ĐẦU ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP TEST")
    print("#" * 60)

    # 1. Đánh giá mô hình STIDGCN đầy đủ (Splitting = True)
    evaluate_and_plot(
        model_path='checkpoints/best_stidgcn.pth',
        data_dir='data/PEMS08',
        device=device,
        splitting=True,
        save_plot_name='prediction_plot.png'
    )

    # 2. Đánh giá biến thể Ablation Study (Splitting = False)
    evaluate_and_plot(
        model_path='checkpoints/best_stidgcn_no_split.pth',
        data_dir='data/PEMS08',
        device=device,
        splitting=False,
        save_plot_name='prediction_plot_no_split.png'
    )