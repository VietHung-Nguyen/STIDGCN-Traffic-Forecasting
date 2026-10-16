import torch
import torch.optim as optim
import os
import torch.nn as nn
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader


def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss()  # Dùng MAE (L1 Loss) làm hàm mất mát

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        # --- Vòng lặp Train ---
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass: Đưa X qua mô hình
            output = model(batch_x)

            # Tính loss
            loss = criterion(output, batch_y)

            # Backward pass & cập nhật trọng số
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
                loss = criterion(output, batch_y)
                total_val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Lưu model tốt nhất ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_stidgcn.pth')
            print("  -> Đã lưu checkpoint mới tốt nhất!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    print("Đang tải và chuẩn bị dữ liệu...")
    seq_len = 12
    horizon = 12
    num_nodes = 170

    data_dir = 'data/PEMS08'

    train_loader, val_loader, test_loader, scaler = get_dataloader(
        data_dir=data_dir,
        batch_size=32
    )

    print("Khởi tạo mô hình STIDGCN...")
    model = STIDGCN(
        num_nodes=num_nodes,
        in_dim=1,
        out_dim=1,
        seq_len=seq_len,
        horizon=horizon
    ).to(device)

    print("Bắt đầu quá trình huấn luyện...")
    train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)