# Website Fingerprinting Project (Var-CNN & Deep Fingerprinting)

Dự án nghiên cứu và triển khai mô hình học sâu nhận diện lưu lượng truy cập web được mã hóa (Website Fingerprinting - WF) sử dụng kiến trúc mạng **Var-CNN** (kết hợp ResNet 1D dilated causal convolutions và Metadata MLP) cùng mô hình nền tảng **Deep Fingerprinting (DF)**.

Cấu trúc dự án được thiết kế chuẩn mực theo mô hình AI Project tiêu chuẩn.

---

## 📁 Cấu trúc thư mục dự án

```text
GR2/
├── configs/                     # Cấu hình siêu tham số, đường dẫn, pipeline (YAML/JSON)
│   ├── train_config.yaml        # Cấu hình huấn luyện, world size, batch size, epochs
│   └── model_config.yaml        # Cấu hình kiến trúc mô hình (Var-CNN, DF, layers)
│
├── data/                        # Quản lý dữ liệu
│   ├── raw/                     # Dữ liệu thô ban đầu (trích xuất pcap, csv, traces)
│   ├── processed/               # Dữ liệu đã qua làm sạch, chia train/val/test (.h5, .npz)
│   └── embeddings/              # Vector embeddings hoặc feature stores (nếu có)
│
├── notebooks/                   # Phục vụ EDA và thử nghiệm nhanh
│   ├── 01_data_exploration.ipynb   # Thống kê, phân bố hướng gói tin, thời gian
│   └── 02_model_prototyping.ipynb  # Thử nghiệm nhanh cấu trúc mạng Var-CNN và DF
│
├── src/                         # Mã nguồn chính của dự án (Module hóa hoàn chỉnh)
│   ├── __init__.py
│   ├── data/                    # Pipeline xử lý dữ liệu và Data loaders
│   │   ├── __init__.py
│   │   ├── dataset.py           # ThreadSafeIter, batch generator từ HDF5
│   │   └── preprocessing.py     # Xử lý raw traces, inter-arrival times, scaler, h5
│   ├── models/                  # Định nghĩa kiến trúc mạng
│   │   ├── __init__.py
│   │   ├── architectures.py     # Var-CNN multi-modal architecture & DF model
│   │   └── layers.py            # ResNet1D basic & dilated residual blocks
│   ├── training/                # Quá trình huấn luyện và tối ưu hóa
│   │   ├── __init__.py
│   │   ├── trainer.py           # Class Trainer quản lý training & validation loop
│   │   └── scheduler.py         # Callbacks: ReduceLROnPlateau, EarlyStopping, Checkpoint
│   ├── evaluation/              # Metrics và đánh giá mô hình
│   │   ├── __init__.py
│   │   ├── metrics.py           # TPR (2-class, multi-class), FPR, Accuracy
│   │   └── evaluate.py          # Tổng hợp predictions, ensemble và xuất kết quả
│   └── utils/                   # Tiện ích chung
│       ├── __init__.py
│       └── helpers.py           # Load config (YAML/JSON), set seed, logging, IO
│
├── app/                         # Tầng triển khai phục vụ mô hình (Inference & REST API)
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # REST API endpoints (/health, /predict, /predict/batch)
│   │   └── schemas.py           # Pydantic models xác thực input/output
│   └── main.py                  # Điểm khởi chạy FastAPI web service
│
├── artifacts/                   # Trọng số, checkpoints, kết quả đánh giá
│   ├── checkpoints/             # Lưu checkpoint trọng số tốt nhất (.h5)
│   ├── predictions/             # Softmax predictions của các sub-models (.npy)
│   └── README.md
│
├── tests/                       # Unit tests và kiểm thử tích hợp
│   ├── __init__.py
│   ├── test_data_pipeline.py    # Kiểm thử trích xuất đặc trưng & data loader
│   └── test_inference.py        # Kiểm thử khởi tạo mạng và dự đoán
│
├── scripts/                     # Scripts CLI tự động hóa
│   └── run_training.py          # Chạy toàn bộ pipeline từ CLI
│
├── .dockerignore
├── .gitignore
├── Dockerfile                   # Đóng gói môi trường container
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

Yêu cầu Python 3.8+ (khuyên dùng Python 3.9 hoặc 3.10).

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

---

### 2. Tiền xử lý dữ liệu (Data Preprocessing)

Chạy pipeline để chuyển đổi dữ liệu thô từ `data/raw/` sang tập dữ liệu đã chuẩn hóa lưu tại `data/processed/`:

```bash
python scripts/run_training.py --train-config configs/train_config.yaml --model-config configs/model_config.yaml
```

*Nếu file `.h5` chưa tồn tại, hệ thống sẽ tự động gọi module tiền xử lý dữ liệu.*

---

### 3. Huấn luyện mô hình (Model Training)

Huấn luyện mô hình Var-CNN:
```bash
python scripts/run_training.py --model-name var-cnn
```

Huấn luyện mô hình Deep Fingerprinting (DF):
```bash
python scripts/run_training.py --model-name df
```

---

### 4. Đánh giá mô hình (Model Evaluation)

Nếu đã có sẵn dự đoán trong `artifacts/predictions/` và muốn tính toán metrics đánh giá:
```bash
python scripts/run_training.py --eval-only
```
Kết quả đánh giá bao gồm:
- **Two-class TPR**: Tỉ lệ website được giám sát (monitored) được nhận diện chính xác.
- **Multi-class TPR / Accuracy**: Độ chính xác phân loại đúng từng website cụ thể.
- **FPR (False Positive Rate)**: Tỉ lệ website không giám sát (unmonitored) bị nhận diện nhầm là website giám sát.

Kết quả chi tiết được tự động lưu vào `artifacts/evaluation_results.json`.

---

### 5. Khởi chạy Web API phục vụ Inference

Khởi động FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI Interactive Docs**: Truy cập `http://localhost:8000/docs`
- **Health Check**: `GET http://localhost:8000/api/v1/health`
- **Predict Single Trace**: `POST http://localhost:8000/api/v1/predict`
- **Predict Batch Traces**: `POST http://localhost:8000/api/v1/predict/batch`

---

### 6. Chạy Unit Tests

Kiểm thử toàn bộ hệ thống pipeline và inference:
```bash
pytest tests/
# hoặc dùng unittest tiêu chuẩn
python -m unittest discover tests
```

---

### 7. Đóng gói Container với Docker

Build Docker image:
```bash
docker build -t website-fingerprinting:latest .
```

Khởi chạy container:
```bash
docker run -d -p 8000:8000 --name wf-service website-fingerprinting:latest
```
