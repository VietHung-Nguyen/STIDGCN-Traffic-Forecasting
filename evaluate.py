import torch
import matplotlib.pyplot as plt
import numpy as np


# from models.stidgcn import STIDGCN
# from utils.metrics import masked_mae, masked_rmse, masked_mape

def evaluate_and_plot(model_path, test_loader, scaler, device):
    print(f"Đang tải trọng số từ {model_path}...")
    # Khởi tạo mô hình và tải trọng số
    # model = STIDGCN(...).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    preds, truths = [], []
    # Vòng lặp test
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         out = model(x.to(device))
    #         preds.append(out.cpu())
    #         truths.append(y)

    # Giả lập dữ liệu để vẽ test (Xóa khi có code thật)
    y_true = np.sin(np.linspace(0, 10, 100))
    y_pred = y_true + np.random.normal(0, 0.1, 100)

    # Tính các chỉ số
    # mae = masked_mae(torch.tensor(y_pred), torch.tensor(y_true))
    # print(f"Test MAE: {mae:.4f}")

    # Trực quan hóa kết quả tại một nút giao thông cụ thể
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:100], label='Ground Truth', color='blue', linewidth=1.5)
    plt.plot(y_pred[:100], label='Prediction', color='red', linestyle='--', linewidth=1.5)
    plt.title("So sánh Lưu lượng Giao thông: Thực tế vs Dự báo", fontsize=14)
    plt.xlabel("Các mốc thời gian (Time Steps)", fontsize=12)
    plt.ylabel("Lưu lượng", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('checkpoints/prediction_plot.png')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_and_plot('checkpoints/best_stidgcn.pth', None, None, device)