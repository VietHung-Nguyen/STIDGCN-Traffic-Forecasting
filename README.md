# STIDGCN - Dự báo Lưu lượng Giao thông (Traffic Forecasting)
Báo cáo giữa kỳ: Nghiên cứu và triển khai bài báo *"Spatial–Temporal Dynamic Graph Convolutional Network With Interactive Learning for Traffic Forecasting"* (IEEE TITS, 2024) trên bộ dữ liệu PEMS08.

## 1. Cấu trúc thư mục
- `data/PEMS08/`: Chứa dữ liệu chuỗi thời gian giao thông (`train.npz`, `val.npz`, `test.npz`, `adj_PEMS08.pkl`).
- `models/stidgcn.py`: Định nghĩa kiến trúc STIDGCN, gồm khối chia tách chuỗi (`Splitting`), tích chập đồ thị động (`DynamicGraphConvolution`) và khối học tương tác (`InteractiveLearningBlock`).
- `utils/`: 
  - `data_loader.py`: Đọc dữ liệu, chuẩn hóa Z-score (`StandardScaler`) và khởi tạo `DataLoader`.
  - `metrics.py`: Các hàm đánh giá sai số có mặt nạ (`masked_mae`, `masked_rmse`, `masked_mape`).
- `checkpoints/`: Lưu trọng số mô hình (`.pth`), nhật ký huấn luyện (`.csv`) và biểu đồ trực quan hóa (`.png`).
- `train.py`: Kịch bản huấn luyện mô hình chính và thực nghiệm mở rộng (Ablation Study).
- `evaluate.py`: Đánh giá mô hình trên tập Test theo các mốc dự báo 15, 30, 60 phút và vẽ đồ thị so sánh.

## 2. Hướng dẫn cài đặt và chạy thực nghiệm
Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
