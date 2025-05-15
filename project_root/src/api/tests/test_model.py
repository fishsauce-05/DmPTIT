import os
import sys
import logging
import nltk
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần test
from train_model import preprocess_text, load_data, create_sample_data, train_model

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

# Tạo thư mục data/processed nếu chưa tồn tại
os.makedirs('data/processed', exist_ok=True)

# Tải các resource từ nltk nếu cần thiết
def setup_module():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Test hàm preprocess_text
def test_preprocess_text():
    # Test với văn bản tiếng Việt
    text = "Sản phẩm này rất tuyệt vời, tôi rất hài lòng!"
    processed = preprocess_text(text)
    assert isinstance(processed, str)
    assert "tuyệt vời" in processed
    assert "!" not in processed  # Dấu chấm than đã bị loại bỏ
    
    # Test với văn bản tiếng Anh
    text = "This product is amazing, I'm very satisfied with it!"
    processed = preprocess_text(text)
    assert isinstance(processed, str)
    assert "amazing" in processed
    assert "satisfied" in processed
    assert "!" not in processed  # Dấu chấm than đã bị loại bỏ
    
    # Test với các trường hợp đặc biệt
    assert preprocess_text("") == ""  # Chuỗi rỗng
    assert preprocess_text(123) == ""  # Không phải chuỗi
    assert preprocess_text(None) == ""  # None

# Test hàm load_data và create_sample_data
def test_data_loading():
    # Test tạo dữ liệu mẫu
    sample_data = create_sample_data()
    assert isinstance(sample_data, pd.DataFrame)
    assert 'review' in sample_data.columns
    assert 'sentiment' in sample_data.columns
    assert len(sample_data) > 0
    
    # Test load dữ liệu (mặc định sẽ tạo dữ liệu mẫu nếu không tìm thấy file)
    data = load_data('non_existent_file.csv')
    assert isinstance(data, pd.DataFrame)
    assert 'review' in data.columns
    assert 'sentiment' in data.columns
    assert len(data) > 0

# Test quá trình huấn luyện mô hình
def test_model_training():
    # Tạo dữ liệu mẫu
    sample_data = create_sample_data()
    
    # Huấn luyện mô hình với dữ liệu mẫu
    classifier = train_model(sample_data)
    
    # Kiểm tra classifier đã được tạo
    assert classifier is not None
    
    # Kiểm tra mô hình đã được lưu
    assert os.path.exists('models/sentiment_classifier.pkl')
    assert os.path.exists('models/vectorizer.pkl')
    assert os.path.exists('models/sentiment_model.pkl')
    
    # Test dự đoán với một số review
    predictions = classifier.predict(["Sản phẩm tuyệt vời"])
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]  # Đảm bảo nhãn dự đoán hợp lệ
    
    # Test dự đoán xác suất
    probas = classifier.predict_proba(["Sản phẩm tệ"])
    assert len(probas) == 1
    assert probas[0].shape[0] == 2  # Có 2 lớp (tích cực/tiêu cực)
    assert 0 <= probas[0][0] <= 1 and 0 <= probas[0][1] <= 1  # Xác suất từ 0-1

# Test với dữ liệu lớn hơn (cần tạo dữ liệu giả)
def test_with_larger_dataset():
    # Tạo tập dữ liệu lớn hơn
    n_samples = 100
    positive_samples = [f"Sản phẩm tuyệt vời {i}" for i in range(n_samples//2)]
    negative_samples = [f"Sản phẩm tệ {i}" for i in range(n_samples//2)]
    
    samples = positive_samples + negative_samples
    labels = [1] * (n_samples//2) + [0] * (n_samples//2)
    
    df = pd.DataFrame({'review': samples, 'sentiment': labels})
    
    # Tiền xử lý
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    # Phân chia train/test
    X = df['processed_review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tạo vectorizer và mô hình
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Huấn luyện mô hình
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Dự đoán và đánh giá
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Kiểm tra độ chính xác
    assert accuracy > 0.7  # Độ chính xác phải trên 70%

if __name__ == "__main__":
    # Chạy tất cả các test
    pytest.main(["-xvs", __file__])