import torch
import matplotlib.pyplot as plt
import numpy as np
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae, masked_rmse, masked_mape


def evaluate_and_plot(model_path, data_dir, device):
    # 1. Tải dữ liệu Test và bộ Scaler
    _, _, test_loader, scaler = get_dataloader(data_dir=data_dir, batch_size=64)

    # 2. Khởi tạo mô hình và nạp trọng số đã huấn luyện
    model = STIDGCN(num_nodes=170, in_dim=1, out_dim=1, channels=48, seq_len=12, horizon=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list, truths_list = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            # Đưa dự đoán về lại miền giá trị lưu lượng thực tế
            predict = scaler.inverse_transform(output.cpu())

            preds_list.append(predict)
            truths_list.append(batch_y)

    preds = torch.cat(preds_list, dim=0)
    truths = torch.cat(truths_list, dim=0)

    # 3. Tính toán các chỉ số đánh giá trên toàn bộ tập Test
    mae = masked_mae(preds, truths, null_val=0.0).item()
    rmse = masked_rmse(preds, truths, null_val=0.0).item()
    mape = masked_mape(preds, truths, null_val=0.0).item() * 100

    print("\n" + "=" * 45)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (PEMS08)")
    print(f" - MAE  : {mae:.4f}")
    print(f" - RMSE : {rmse:.4f}")
    print(f" - MAPE : {mape:.2f}%")
    print("=" * 45 + "\n")

    # 4. Vẽ biểu đồ so sánh tại Trạm cảm biến số 0 (Dự báo ở bước thời gian đầu tiên t+1)
    node_id = 0
    time_steps_to_plot = 288  # 288 bước * 5 phút = 1 ngày (24 giờ)

    y_true_plot = truths[:time_steps_to_plot, 0, node_id, 0].numpy()
    y_pred_plot = preds[:time_steps_to_plot, 0, node_id, 0].numpy()

    plt.figure(figsize=(12, 4.5))
    plt.plot(y_true_plot, label='Lưu lượng thực tế (Ground Truth)', color='#1f77b4', linewidth=1.5)
    plt.plot(y_pred_plot, label='Lưu lượng dự báo (STIDGCN)', color='#d62728', linestyle='--', linewidth=1.5)
    plt.title(f"So sánh Lưu lượng Giao thông trong 24 giờ (Nút cảm biến #{node_id} - PEMS08)", fontsize=13)
    plt.xlabel("Mốc thời gian (5 phút / bước)", fontsize=11)
    plt.ylabel("Lưu lượng xe (Traffic Flow)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('checkpoints/prediction_plot.png', dpi=300)
    print("Đã lưu biểu đồ vào: checkpoints/prediction_plot.png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_and_plot('checkpoints/best_stidgcn.pth', 'data/PEMS08', device)