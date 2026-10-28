# Phân cụm giao dịch IBFT và ứng dụng IBFT Explorer

Dự án phân nhóm giao dịch chuyển tiền IBFT theo số tiền, thời điểm và lịch sử giao dịch của người gửi với người nhận. Ba phương pháp được so sánh gồm K-means, Gaussian Mixture Model (GMM) và Autoencoder kết hợp K-means. Ứng dụng **IBFT Explorer** sử dụng mô hình K-means đã huấn luyện để xem đặc điểm các nhóm, tra cứu và xuất kết quả.

Dữ liệu thực nghiệm được mô phỏng lại các giá trị dựa trên cấu trúc trường và tên cột của dữ liệu giao dịch. Nhãn cụm biểu thị sự tương đồng giữa các giao dịch, chưa phải kết luận về gian lận hoặc mức rủi ro.

## Nội dung dự án

```text
app.py                              Ứng dụng Streamlit
src/ibft_utils.py                   Đọc, làm sạch, tạo đặc trưng và suy luận
.streamlit/config.toml             Cấu hình ứng dụng chạy cục bộ
notebooks/01_prepare_data.ipynb     Chuẩn bị dữ liệu và bộ tiền xử lý
models/kmeans.ipynb                 Thực nghiệm K-means
models/gmm.ipynb                    Thực nghiệm GMM
models/autoencoder_kmeans.ipynb     Thực nghiệm Autoencoder + K-means
notebooks/02_evaluate_visualize.ipynb
                                   So sánh và trực quan hóa kết quả
output/                            Bảng kết quả tổng hợp, cấu hình và biểu đồ
requirements.txt                   Các phiên bản thư viện sử dụng
```

## Phương pháp và kết quả hiện tại

- Từ 44.910 giao dịch đầu vào, quy trình giữ lại 44.832 giao dịch hợp lệ; 4.790 giao dịch thuộc bảy ngày đầu chỉ được dùng làm lịch sử nền.
- Bộ dữ liệu phân cụm gồm **40.042 giao dịch, 11 đặc trưng**, chia theo thời gian thành train (28.029), validation (6.006) và test (6.007).
- Đặc trưng lịch sử chỉ sử dụng các giao dịch xảy ra trước thời điểm đang xét. Các tham số điền thiếu và chuẩn hóa được học trên train.
- Khảo sát số cụm từ 3 đến 10 với năm seed. Số cụm và seed được chọn bằng Silhouette trên validation; test dùng để báo cáo kết quả cuối.
- Cả ba phương pháp chọn ba cụm trong phạm vi khảo sát. K-means có Silhouette validation trung bình cao nhất và mức biến động giữa các seed nhỏ nhất, nên được sử dụng trong ứng dụng với **K = 3, seed = 43**.

Các bảng so sánh nằm trong `output/comparison/`; kết quả từng phương pháp và biểu đồ tương ứng nằm trong các thư mục con của `output/`.

## Chuẩn bị môi trường

Môi trường sử dụng **Python 3.12**. Ví dụ trên Windows, chạy từ thư mục gốc dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Mở notebook bằng VS Code hoặc môi trường Jupyter và chọn kernel của `.venv`.

## Tạo dữ liệu và mô hình để chạy ứng dụng

Kho GitHub lưu mã nguồn, biểu đồ và các bảng kết quả tổng hợp. **Các tệp dữ liệu đầu vào trong `data/`, dữ liệu từng giao dịch (`.parquet`) và mô hình đã lưu (`.joblib`, `.pt`) được loại khỏi Git bằng `.gitignore`.** Vì vậy, sau khi clone cần có dữ liệu đầu vào phù hợp trên máy và chạy các notebook để tạo những tệp này.

1. Tạo thư mục `data/` tại thư mục gốc, đặt các tệp Excel đầu vào có sheet `IBFT` và cấu trúc theo mô tả trong notebook chuẩn bị dữ liệu.
2. Chạy toàn bộ `notebooks/01_prepare_data.ipynb` để tạo dữ liệu và bộ tiền xử lý trong `output/prepared/`.
3. Chạy ba notebook trong `models/` để huấn luyện và lưu kết quả từng phương pháp.
4. Chạy `notebooks/02_evaluate_visualize.ipynb` để tạo bảng so sánh và các biểu đồ tổng hợp.
5. Khởi chạy ứng dụng:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

Ứng dụng được cấu hình truy cập cục bộ tại `http://127.0.0.1:8501`.

## Sử dụng IBFT Explorer

- **Chọn dữ liệu:** sử dụng dữ liệu đã chuẩn bị hoặc tải lên Excel/CSV. Dữ liệu tải lên được xử lý trong bộ nhớ.
- **Xem tổng quan:** số giao dịch, số tiền trung vị, quy mô và đặc điểm các cụm, biểu đồ PCA.
- **Tra cứu:** lọc theo cụm, khoảng ngày, mã giao dịch hoặc tài khoản gửi/nhận.
- **Xuất CSV:** tải toàn bộ kết quả sau lọc; bảng trên giao diện hiển thị tối đa 500 dòng. Số tài khoản được che bớt theo mặc định trong cả bảng và tệp tải xuống.

CSV tải lên cần có các cột:

```text
referencenumber,transaction_datetime,accountsourcenumber,accountdestnumber,transactionamount,responsecode
```

Thời gian dùng định dạng `YYYY-MM-DD HH:MM:SS` theo giờ Việt Nam. Giữ mã giao dịch và tài khoản dưới dạng văn bản; số tiền dùng đơn vị VND và không có dấu phân cách hàng nghìn. Ứng dụng cung cấp tệp CSV mẫu trống cùng hướng dẫn về phần lịch sử cần có. Giới hạn mỗi tệp tải lên là 25 MB, tối đa 100.000 dòng trong một lần xử lý.

Khi chạy lại notebook, các kết quả hiển thị trong notebook có thể chứa dữ liệu từng giao dịch. Trước khi commit, xóa phần kết quả chạy trong notebook; giữ các bảng tổng hợp và hình ở `output/`.
