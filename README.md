# Machine Learning - Text Classification w/ Recurrent Neural Network

Phan Ngọc Lân, Nguyễn Duy Mạnh

### Cài đặt
- Cài đặt dependency: yêu cầu Python>3.4 hoặc Python 2.7, pip.
Chạy lệnh `(sudo) pip install glove-python keras==1.2.2 h5py matplotlib nltk csv kivy feedparser` để tải về các gói cần thiết.
- Để bổ sung tính năng xử lý bằng card đồ họa (không bắt buộc, giúp tăng tốc quá trình học và dự đoán), cài đặt thư viện CUDA (dành cho GPU [NVIDIA](https://developer.nvidia.com/cuda-downloads)) và làm theo hướng dẫn tại [đây](http://deeplearning.net/software/theano/tutorial/using_gpu.html).

### Triển khai
- Để tiến hành huấn luyện, sử dụng lệnh: `python train.py ([tên_mô_hình])`. Các tùy chọn của quá trình huấn luyện có thể được điều chỉnh trong settings.py. Các mô hình sẽ được lưu vào thư mục models/ với tên được chọn hoặc dưới định dạng `[dataset]_[danh_sách_neuron_FC]`. Thư mục bao gồm cả các log huấn luyện và đồ thị kết quả.
- Để chạy đánh giá một mô hình đã huấn luyện (Precision, Recall, F1), sử dụng lệnh `python evaluate.py [đường_dẫn_đến_mô_hình] [tên_dataset]`.
- Để chạy demo đồ họa, sử dụng lệnh `python -m demo`. Chọn thư mục chứa mô hình muốn sử dụng trong ô chọn File bên phải giao diện (VD ag_news_[125]) và chọn load. Nhập tiêu đề và nội dung của văn bản cần phân loại vào các ô tương ứng phía bên trái và chọn Classify.
- Để thực hiện dự đoán trên feed RSS, chạy `python -m demo_rss [đường_dẫn_đến_mô_hình]`. Kết quả dự đoán sẽ được lưu trong demo_rss/results.csv.

### Dataset
- Các dataset hiện có: ag_news, bbc, reuters. Tất cả các file được dùng nằm trong thư mục có định dạng data/tên_dataset_csv. Mỗi thư mục bao gồm: train.csv, test.csv (chứa dữ liệu tương ứng) và classes.txt chứa tên các chủ đề theo đúng thứ tự.
- Các file csv được định dạng theo quy ước: các cột cách nhau bởi dấu phẩy, nội dung được đặt trong dấu ngoặc kép. Mỗi file đầu vào có 3 cột: nhãn (số thứ tự của chủ đề bắt đầu từ 1), tiêu đề và nội dung.

### Kiến trúc mạng neuron
![Model](docs/images/GRUModel.png)
