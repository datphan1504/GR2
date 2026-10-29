# Artifacts Directory

Thư mục này được sử dụng để lưu trữ các tài nguyên được tạo ra trong quá trình huấn luyện và đánh giá:
- `checkpoints/`: Checkpoints trọng số tốt nhất trong từng epoch (`*.h5`).
- `best_model.pt` / `best_model.h5`: Trọng số tối ưu sau khi huấn luyện hoàn tất.
- `predictions/`: Dự đoán softmax (`.npy`) của các sub-models trên tập test.
- `evaluation_results.json`: Báo cáo chi tiết metrics đánh giá (Accuracy, TPR, FPR).
- Các bộ chuẩn hóa (`scaler.pkl`), tokenizer hoặc feature stores.
