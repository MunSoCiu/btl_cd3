# 📊 BOOK DATA ANALYSIS REPORT

## 1. Tổng quan

Dữ liệu gồm:

- Users
- Books
- Reading History

Pipeline xử lý:

- Load → Preprocess → Merge → Analysis → Recommendation → Time Series

## 2. Insight chính

### 📚 Thể loại

- Manga phổ biến nhất
- Người dùng thích nội dung giải trí + thực tiễn

### ⭐ Rating

- Trung bình ~4.0+
- Dữ liệu bias tích cực

### 👤 Người dùng

- Có sự phân hóa:
  - Heavy users
  - Casual users

### 💰 Giá vs Rating

- Không có tương quan rõ ràng

### 📈 Time Series

- Có xu hướng tăng theo thời gian
- Có thể dự báo được

## 3. Recommendation System

Áp dụng:

- Content-based filtering

Logic:

- Lấy thể loại yêu thích
- Gợi ý sách chưa đọc

(Ref: recommendation.py) :contentReference[oaicite:0]{index=0}

## 4. Kiến trúc hệ thống

- load_data → đọc dữ liệu :contentReference[oaicite:1]{index=1}
- preprocess → xử lý dữ liệu :contentReference[oaicite:2]{index=2}
- build_dataset → merge :contentReference[oaicite:3]{index=3}
- time_series → dự báo :contentReference[oaicite:4]{index=4}
- main pipeline → chạy toàn bộ :contentReference[oaicite:5]{index=5}

## 5. Hướng phát triển

- Collaborative Filtering
- Deep Learning Recommender
- Dashboard (Streamlit)
- Real-time recommendation

## 6. Kết luận

Hệ thống đã:

- Xây dựng pipeline hoàn chỉnh
- Phân tích hành vi đọc sách
- Triển khai hệ gợi ý cơ bản

👉 Sẵn sàng mở rộng thành hệ thống production
