import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.stidgcn import STIDGCN
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae, masked_rmse, masked_mape


def evaluate_and_plot(model_path, data_dir, device, channels=48, splitting=True, save_plot_name=None):
    # 1. Tải dữ liệu Test và bộ Scaler
    _, _, test_loader, scaler = get_dataloader(data_dir=data_dir, batch_size=64)

    # 2. Khởi tạo mô hình theo đúng số channels của file trọng số
    model = STIDGCN(
        num_nodes=170, in_dim=1, out_dim=1, channels=channels,
        seq_len=12, horizon=12, splitting=splitting
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Đếm tổng số tham số của mô hình (Total Parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    preds_list, truths_list = [], []
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

    print("\n" + "=" * 65)
    print(f"KẾT QUẢ ĐÁNH GIÁ ({model_path} | Channels={channels} | Splitting={splitting})")
    print(f" - Tổng số tham số: {total_params:,} params")
    print(f" - MAE            : {mae:.4f}")
    print(f" - RMSE           : {rmse:.4f}")
    print(f" - MAPE           : {mape:.2f}%")
    print(f" - Inference Time : {inference_time:.2f} giây")
    print("=" * 65)

    print("CHI TIẾT ĐÁNH GIÁ THEO TỪNG MỐC DỰ BÁO (Tương ứng Fig. 4):")
    for step, mins in [(2, 15), (5, 30), (11, 60)]:
        s_mae = masked_mae(preds[:, step, :, :], truths[:, step, :, :], null_val=0.0).item()
        s_rmse = masked_rmse(preds[:, step, :, :], truths[:, step, :, :], null_val=0.0).item()
        s_mape = masked_mape(preds[:, step, :, :], truths[:, step, :, :], null_val=0.0).item() * 100
        print(f" -> Dự báo sau {mins:2d} phút (Horizon {step + 1:2d}) | MAE: {s_mae:.4f} | RMSE: {s_rmse:.4f} | MAPE: {s_mape:.2f}%")
    print("=" * 65 + "\n")

    # 4. Vẽ biểu đồ dự báo 24 giờ nếu có yêu cầu tên file
    if save_plot_name is not None:
        node_id = 0
        time_steps_to_plot = 288
        y_true_plot = truths[:time_steps_to_plot, 0, node_id, 0].numpy()
        y_pred_plot = preds[:time_steps_to_plot, 0, node_id, 0].numpy()

        plt.figure(figsize=(12, 4.5))
        plt.plot(y_true_plot, label='Lưu lượng thực tế (Ground Truth)', color='#1f77b4', linewidth=1.5)
        plt.plot(y_pred_plot, label=f'Dự báo STIDGCN (Channels={channels})', color='#d62728', linestyle='--', linewidth=1.5)
        plt.title(f"So sánh Lưu lượng Giao thông trong 24 giờ (Nút #{node_id} - PEMS08)", fontsize=13)
        plt.xlabel("Mốc thời gian (5 phút / bước)", fontsize=11)
        plt.ylabel("Lưu lượng xe (Traffic Flow)", fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'checkpoints/{save_plot_name}', dpi=300)
        plt.close()

    return {
        'channels': channels,
        'params': total_params,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'time': inference_time
    }


def plot_channels_comparison(results):
    """Vẽ biểu đồ phân tích độ nhạy siêu tham số Channels (Tương ứng Fig. 6 trong bài báo)."""
    ch_list = [str(r['channels']) for r in results]
    mae_list = [r['mae'] for r in results]
    rmse_list = [r['rmse'] for r in results]
    mape_list = [r['mape'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Biểu đồ 1: MAE
    axes[0].plot(ch_list, mae_list, marker='o', color='#1f77b4', linewidth=2)
    axes[0].set_title("Ảnh hưởng của Channels đến MAE", fontsize=11)
    axes[0].set_xlabel("Số kênh đặc trưng (Channels)")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.3)

    # Biểu đồ 2: RMSE
    axes[1].plot(ch_list, rmse_list, marker='s', color='#ff7f0e', linewidth=2)
    axes[1].set_title("Ảnh hưởng của Channels đến RMSE", fontsize=11)
    axes[1].set_xlabel("Số kênh đặc trưng (Channels)")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    # Biểu đồ 3: MAPE
    axes[2].plot(ch_list, mape_list, marker='^', color='#2ca02c', linewidth=2)
    axes[2].set_title("Ảnh hưởng của Channels đến MAPE (%)", fontsize=11)
    axes[2].set_xlabel("Số kênh đặc trưng (Channels)")
    axes[2].set_ylabel("MAPE (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'checkpoints/channels_analysis_fig6.png'
    plt.savefig(save_path, dpi=300)
    print(f"Đã lưu biểu đồ phân tích Channels (Fig. 6) vào: {save_path}")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/PEMS08'

    # Danh sách 4 cấu hình Channels cần đối sánh (16, 32, 48, 64)
    configs = [
        (16, 'checkpoints/best_stidgcn_c16.pth', None),
        (32, 'checkpoints/best_stidgcn_c32.pth', None),
        (48, 'checkpoints/best_stidgcn.pth', 'prediction_plot.png'),
        (64, 'checkpoints/best_stidgcn_c64.pth', None),
    ]

    channel_results = []
    for ch, path, plot_name in configs:
        if os.path.exists(path):
            res = evaluate_and_plot(
                model_path=path,
                data_dir=data_dir,
                device=device,
                channels=ch,
                splitting=True,
                save_plot_name=plot_name
            )
            channel_results.append(res)
        else:
            print(f"⚠️ Chưa tìm thấy file {path}, hãy chạy train.py trước!")

    # In bảng tổng hợp so sánh 4 cấu hình Channels ngay trên Terminal
    if len(channel_results) > 1:
        print("\n" + "#" * 75)
        print(" BẢNG TỔNG HỢP KHẢO SÁT SIÊU THAM SỐ CHANNELS (TƯƠNG ỨNG FIG. 6)")
        print("#" * 75)
        print(f"{'Channels':^10} | {'Số tham số':^14} | {'MAE':^10} | {'RMSE':^10} | {'MAPE (%)':^10} | {'Test Time (s)':^13}")
        print("-" * 75)
        for r in channel_results:
            print(f"{r['channels']:^10} | {r['params']:^14,d} | {r['mae']:^10.4f} | {r['rmse']:^10.4f} | {r['mape']:^10.2f} | {r['time']:^13.2f}")
        print("#" * 75 + "\n")

        plot_channels_comparison(channel_results)