# DPL302m Final Report & Group Presentation (SU25)

## 1. Tổng quan
Dự án này xây dựng mô hình nhận diện cảm xúc từ giọng nói (Speech Emotion Recognition) bằng deep learning, sử dụng TensorFlow/Keras và các thư viện xử lý âm thanh.

## 2. Cấu trúc thư mục
```text
DPL302m_Final_Report&Group_Presentation_SU25/
├── README.md
├── requirements.txt
├── src/
│   └── app.py
├── notebooks/
│   └── training_model.ipynb
├── models/
│   └── MardeusNet.keras
├── reports/
│   ├── final-report.pdf
│   └── group-presentation.pdf
└── dpl-env/
    ├── bin/
    ├── include/
    ├── lib/
    └── pyvenv.cfg
```

## 3. Mô tả thành phần
- `requirements.txt`: Danh sách thư viện Python cần cài đặt.
- `src/app.py`: File chạy ứng dụng hoặc suy luận mô hình.
- `notebooks/training_model.ipynb`: Notebook huấn luyện, đánh giá và trực quan hóa.
- `models/MardeusNet.keras`: Trọng số/mô hình đã lưu.
- `reports/`: Tài liệu báo cáo và slide trình bày.
- `dpl-env/`: Môi trường ảo Python cục bộ của dự án.

## 4. Thiết lập môi trường
```bash
python3 -m venv dpl-env
source dpl-env/bin/activate
pip install -r requirements.txt
```

## 5. Cách chạy
### Chạy ứng dụng
```bash
streamlit run src/app.py
```

### Mở notebook huấn luyện
```bash
jupyter notebook notebooks/training_model.ipynb
```

## 6. Ghi chú quản lý dự án
- Nên giữ môi trường ảo (`dpl-env`) ngoài version control.
- Nếu làm việc nhóm, ưu tiên cập nhật `requirements.txt` khi thêm thư viện.
- Không chỉnh sửa trực tiếp file mô hình `.keras` khi không cần thiết.
