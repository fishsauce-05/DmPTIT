import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import joblib
import os

# Tạo thư mục data/processed nếu chưa tồn tại
os.makedirs('data/processed', exist_ok=True)

# Tải dữ liệu
def load_data(file_path):
    """Đọc dữ liệu từ file CSV"""
    try:
        data = pd.read_csv(file_path)
        print(f"Đã tải dữ liệu từ {file_path}, kích thước: {data.shape}")
        return data
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {str(e)}")
        # Nếu không tìm thấy file, tạo dữ liệu mẫu
        print("Tạo dữ liệu mẫu cho việc huấn luyện...")
        return create_sample_data()

def create_sample_data():
    """Tạo dữ liệu mẫu để huấn luyện"""
    positive_reviews = [
        "Sản phẩm tuyệt vời, tôi rất hài lòng với chất lượng",
        "Giao hàng nhanh, đóng gói cẩn thận",
        "Giá cả hợp lý, chất lượng tốt",
        "Tôi sẽ mua lại sản phẩm này",
        "Dịch vụ khách hàng rất tốt",
        "Sản phẩm đúng như mô tả, rất đáng tiền",
        "Chất lượng vượt quá mong đợi của tôi",
        "Thiết kế đẹp, sử dụng dễ dàng",
        "Rất hài lòng với sản phẩm này",
        "Sản phẩm bền và đẹp"
    ]
    
    negative_reviews = [
        "Sản phẩm kém chất lượng",
        "Không đáng với số tiền bỏ ra",
        "Giao hàng chậm, đóng gói sơ sài",
        "Sản phẩm không giống như mô tả",
        "Dịch vụ khách hàng tệ",
        "Sản phẩm hỏng sau một tuần sử dụng",
        "Thất vọng với chất lượng sản phẩm",
        "Tôi sẽ không bao giờ mua lại",
        "Không đáng giá tiền",
        "Chất lượng kém, không đáng mua"
    ]
    
    # Tạo DataFrame
    reviews = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': labels
    })
    
    # Lưu dữ liệu mẫu
    df.to_csv('data/processed/sample_data.csv', index=False)
    print("Đã tạo và lưu dữ liệu mẫu")
    return df

# Tiền xử lý văn bản
def preprocess_text(text):
    """Tiền xử lý văn bản"""
    try:
        # Tải các tài nguyên cần thiết từ nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Định nghĩa stopwords cho tiếng Việt
        vietnamese_stopwords = ['và', 'của', 'cho', 'là', 'để', 'trong', 'với', 'được', 'có', 'không',
                               'những', 'các', 'từ', 'một', 'bạn', 'tôi', 'tại', 'về', 'đã', 'này', 'như']
        english_stopwords = stopwords.words('english')
        all_stopwords = vietnamese_stopwords + english_stopwords
        
        # Tạo lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        # Làm sạch văn bản
        if isinstance(text, str):
            # Chuyển về chữ thường
            text = text.lower()
            # Loại bỏ các ký tự đặc biệt
            text = re.sub(r'[^\w\s]', '', text)
            # Loại bỏ số
            text = re.sub(r'\d+', '', text)
            # Tách từ
            words = text.split()
            # Loại bỏ stopwords và lemmatize
            words = [lemmatizer.lemmatize(word) for word in words if word not in all_stopwords]
            # Ghép lại thành chuỗi
            text = ' '.join(words)
            return text
        return ""
    except Exception as e:
        print(f"Lỗi khi tiền xử lý văn bản: {str(e)}")
        return text if isinstance(text, str) else ""

# Phân tích thống kê dữ liệu
def analyze_data(df):
    """Phân tích và lưu thống kê dữ liệu"""
    try:
        # Thêm cột độ dài review
        df['length'] = df['review'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Phân tích phân phối độ dài
        length_stats = {
            'min_length': df['length'].min(),
            'max_length': df['length'].max(),
            'avg_length': df['length'].mean()
        }
        
        # Phân phối nhãn
        label_dist = df['sentiment'].value_counts().to_dict()
        
        # Lưu thống kê
        stats_df = pd.DataFrame({
            'length': df['length'],
            'sentiment': df['sentiment']
        })
        stats_df.to_csv('data/eda_statistics.csv', index=False)
        
        print("Phân tích dữ liệu:")
        print(f"Số lượng mẫu: {len(df)}")
        print(f"Phân phối nhãn: {label_dist}")
        print(f"Thống kê độ dài: {length_stats}")
        
        return stats_df
    except Exception as e:
        print(f"Lỗi khi phân tích dữ liệu: {str(e)}")
        return None

# Huấn luyện mô hình
def train_model(df):
    """Huấn luyện mô hình phân loại"""
    try:
        # Tiền xử lý dữ liệu
        df['processed_review'] = df['review'].apply(preprocess_text)
        
        # Chia tập dữ liệu
        X = df['processed_review']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tạo vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Huấn luyện mô hình
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Đánh giá mô hình
        y_pred = model.predict(X_test_vec)
        print("\nKết quả đánh giá mô hình:")
        print(classification_report(y_test, y_pred))
        
        # Lưu vectorizer và mô hình
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        joblib.dump(model, 'models/sentiment_model.pkl')
        
        # Tạo lớp dự đoán kết hợp cả vectorizer và model
        class SentimentClassifier:
            def __init__(self, vectorizer, model):
                self.vectorizer = vectorizer
                self.model = model
            
            def predict(self, texts):
                # Tiền xử lý
                processed_texts = [preprocess_text(text) for text in texts]
                # Vectorize
                vec_texts = self.vectorizer.transform(processed_texts)
                # Dự đoán
                return self.model.predict(vec_texts)
            
            def predict_proba(self, texts):
                # Tiền xử lý
                processed_texts = [preprocess_text(text) for text in texts]
                # Vectorize
                vec_texts = self.vectorizer.transform(processed_texts)
                # Dự đoán xác suất
                return self.model.predict_proba(vec_texts)
        
        # Tạo và lưu classifier
        classifier = SentimentClassifier(vectorizer, model)
        joblib.dump(classifier, 'models/sentiment_classifier.pkl')
        
        print("Đã lưu mô hình vào thư mục models/")
        return classifier
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        return None

if __name__ == "__main__":
    # Tạo cấu trúc thư mục
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Tải dữ liệu (thay đổi đường dẫn file nếu có)
    data = load_data('data/processed/reviews.csv')
    
    # Phân tích dữ liệu
    analyze_data(data)
    
    # Huấn luyện mô hình
    classifier = train_model(data)
    
    print("Quá trình huấn luyện hoàn tất!")