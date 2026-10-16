import torch
import torch.optim as optim
import os
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae


def train_model(model, train_loader, val_loader, scaler, epochs, lr, device):
    # Thiết lập tham số chuẩn theo bài báo: lr=0.001, weight_decay=0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # Dự báo và đưa ngược về giá trị lưu lượng xe thực tế
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

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_stidgcn.pth')
            print("  -> Đã lưu checkpoint mới tốt nhất!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    data_dir = 'data/PEMS08'
    train_loader, val_loader, test_loader, scaler = get_dataloader(
        data_dir=data_dir,
        batch_size=64  # Tham số chuẩn của bài báo
    )

    model = STIDGCN(
        num_nodes=170,
        in_dim=1,
        out_dim=1,
        channels=48,  # Tham số chuẩn của bài báo cho PEMS08
        seq_len=12,
        horizon=12,
        dropout=0.1,
        splitting=True
    ).to(device)

    train_model(model, train_loader, val_loader, scaler, epochs=20, lr=0.001, device=device)