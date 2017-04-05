# Machine Learning - Text Classification w/ Recurrent Neural Network

Phan Ngọc Lân, Nguyễn Duy Mạnh

### Cài đặt
- Tải về vector từ (word embedding) Glove tại [đây](http://nlp.stanford.edu/data/glove.6B.zip) (~800MB). Giải nén glove.6B.100d.txt vào thư mục data/.
- Cài đặt dependency: Python >3.4, Tensorflow/Theano, glove-python, Keras, h5py, matplotlib, NLTK (Natural Language Toolkit), csv (Python).
- Dependency không bắt buộc: Kivy (demo đồ họa), feedparser (demo RSS).

### Triển khai
- Để tiến hành huấn luyện, sử dụng lệnh: `python3 train.py ([tên_mô_hình])`. Các tùy chọn của quá trình huấn luyện có thể được điều chỉnh trong settings.py. Các mô hình sẽ được lưu vào thư mục models/ với tên được chọn hoặc dưới định dạng `[dataset]_[năm-tháng-ngày]`.
- Để chạy đánh giá một mô hình đã huấn luyện (Precision, Recall, F1), sử dụng lệnh `python3 evaluate [đường_dẫn_đến_mô_hình] [tên_dataset]`.
- Để chạy demo đồ họa, sử dụng lệnh `python3 demo`. Để thực hiện dự đoán trên feed RSS, chạy `python3 demo_rss [đường_dẫn_đến_mô_hình]`. Kết quả dự đoán sẽ được lưu trong demo_rss/results.csv.

### Dataset
- Các dataset hiện có: ag_news, bbc, reuters. Tất cả các file được dùng nằm trong thư mục có định dạng data/tên_dataset_csv. Mỗi thư mục bao gồm: train.csv, test.csv (chứa dữ liệu tương ứng) và classes.txt chứa tên các chủ đề theo đúng thứ tự.
- Các file csv được định dạng theo quy ước: các cột cách nhau bởi dấu phẩy, nội dung được đặt trong dấu ngoặc kép. Mỗi file đầu vào có 3 cột: nhãn (số thứ tự của chủ đề bắt đầu từ 1), tiêu đề và nội dung.

![Model](GRU-Merge.png)
