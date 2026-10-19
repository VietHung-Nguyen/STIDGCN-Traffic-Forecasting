import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae, masked_rmse, masked_mape


def evaluate_and_plot(model_path, data_dir, device, splitting=True, save_plot_name='prediction_plot.png'):
    # 1. Tải dữ liệu Test và bộ Scaler
    _, _, test_loader, scaler = get_dataloader(data_dir=data_dir, batch_size=64)

    # 2. Khởi tạo mô hình và nạp trọng số đã huấn luyện
    model = STIDGCN(
        num_nodes=170, in_dim=1, out_dim=1, channels=48,
        seq_len=12, horizon=12, splitting=splitting
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list, truths_list = [], []

    # Bắt đầu đo thời gian dự báo (Inference Time - giống Table 5 trong bài báo)
    start_time = time.time()

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            predict = scaler.inverse_transform(output.cpu())

            preds_list.append(predict)
            truths_list.append(batch_y)

    inference_time = time.time() - start_time

    preds = torch.cat(preds_list, dim=0)
    truths = torch.cat(truths_list, dim=0)

    # 3. Tính toán các chỉ số đánh giá tổng thể
    mae = masked_mae(preds, truths, null_val=0.0).item()
    rmse = masked_rmse(preds, truths, null_val=0.0).item()
    mape = masked_mape(preds, truths, null_val=0.0).item() * 100

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ ({model_path} | Splitting={splitting})")
    print(f" - MAE            : {mae:.4f}")
    print(f" - RMSE           : {rmse:.4f}")
    print(f" - MAPE           : {mape:.2f}%")
    print(f" - Inference Time : {inference_time:.2f} giây")
    print("=" * 60)

    print("\nCHI TIẾT ĐÁNH GIÁ THEO TỪNG MỐC DỰ BÁO (Tương ứng Fig. 4):")
    for step, mins in [(2, 15), (5, 30), (11, 60)]:
        s_mae = masked_mae(preds[:, step, :, :], truths[:, step, :, :], null_val=0.0).item()
        s_rmse = masked_rmse(preds[:, step, :, :], truths[:, step, :, :], null_val=0.0).item()
        s_mape = masked_mape(preds[:, step, :, :], truths[:, step, :, :], null_val=0.0).item() * 100
        print(
            f" -> Dự báo sau {mins:2d} phút (Horizon {step + 1:2d}) | MAE: {s_mae:.4f} | RMSE: {s_rmse:.4f} | MAPE: {s_mape:.2f}%")
    print("=" * 60 + "\n")

    # 4. Vẽ biểu đồ so sánh tại Trạm cảm biến số 0
    node_id = 0
    time_steps_to_plot = 288

    y_true_plot = truths[:time_steps_to_plot, 0, node_id, 0].numpy()
    y_pred_plot = preds[:time_steps_to_plot, 0, node_id, 0].numpy()

    plt.figure(figsize=(12, 4.5))
    plt.plot(y_true_plot, label='Lưu lượng thực tế (Ground Truth)', color='#1f77b4', linewidth=1.5)
    plt.plot(y_pred_plot, label='Lưu lượng dự báo (STIDGCN)', color='#d62728', linestyle='--', linewidth=1.5)
    plt.title(f"So sánh Lưu lượng Giao thông trong 24 giờ (Nút #{node_id} - PEMS08)", fontsize=13)
    plt.xlabel("Mốc thời gian (5 phút / bước)", fontsize=11)
    plt.ylabel("Lưu lượng xe (Traffic Flow)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'checkpoints/{save_plot_name}', dpi=300)
    print(f"Đã lưu biểu đồ vào: checkpoints/{save_plot_name}")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Đánh giá mô hình chính bạn đã train xong
    evaluate_and_plot(
        model_path='checkpoints/best_stidgcn.pth',
        data_dir='data/PEMS08',
        device=device,
        splitting=True,
        save_plot_name='prediction_plot.png'
    )
