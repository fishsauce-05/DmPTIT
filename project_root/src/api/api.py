from flask import Flask, request, jsonify
import joblib
import traceback
import os
import pandas as pd
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Cấu hình logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('logs/api.log'), logging.StreamHandler()])

# Tải mô hình
@app.before_first_request
def load_model():
    global model, vectorizer, classifier
    try:
        # Tạo thư mục logs nếu chưa tồn tại
        os.makedirs('logs', exist_ok=True)
        
        # Tải mô hình
        model_path = 'models/sentiment_classifier.pkl'
        
        if os.path.exists(model_path):
            app.logger.info(f'Đang tải mô hình từ {model_path}')
            classifier = joblib.load(model_path)
            app.logger.info('Đã tải mô hình thành công!')
        else:
            app.logger.warning(f'Không tìm thấy mô hình tại {model_path}')
            app.logger.info('Sẽ tạo mô hình mới khi có yêu cầu đầu tiên')
            classifier = None
    except Exception as e:
        app.logger.error(f'Lỗi khi tải mô hình: {str(e)}')
        app.logger.error(traceback.format_exc())
        classifier = None

# Hàm tiền xử lý văn bản
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
        app.logger.error(f"Lỗi khi tiền xử lý văn bản: {str(e)}")
        return text if isinstance(text, str) else ""

# Lưu lịch sử dự đoán
def log_prediction(review_text, prediction, confidence):
    """Lưu lịch sử dự đoán vào file CSV"""
    try:
        os.makedirs('data/logs', exist_ok=True)
        log_file = 'data/logs/prediction_history.csv'
        
        # Tạo dữ liệu log
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'review': review_text,
            'prediction': prediction,
            'confidence': confidence
        }
        
        # Tạo DataFrame từ dữ liệu log
        log_df = pd.DataFrame([log_data])
        
        # Kiểm tra file tồn tại
        if os.path.exists(log_file):
            # Nếu tồn tại, append vào file
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            # Nếu không tồn tại, tạo file mới
            log_df.to_csv(log_file, index=False)
            
        app.logger.info(f"Đã lưu kết quả dự đoán vào {log_file}")
    except Exception as e:
        app.logger.error(f"Lỗi khi lưu lịch sử dự đoán: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """API dự đoán sentiment từ review"""
    try:
        # Lấy dữ liệu từ request
        data = request.get_json(force=True)
        
        # Kiểm tra dữ liệu đầu vào
        if 'review' not in data:
            return jsonify({'error': 'Vui lòng cung cấp trường "review" trong dữ liệu JSON'}), 400
        
        review_text = data.get('review', '')
        
        # Kiểm tra model đã được tải chưa
        global classifier
        if classifier is None:
            app.logger.warning('Mô hình chưa được tải. Sẽ tạo mô hình mới.')
            # Import các module cần thiết
            import sys
            sys.path.append('.')
            from train_model import train_model, load_data, analyze_data
            
            # Tải dữ liệu và train model
            data = load_data('data/processed/reviews.csv')
            analyze_data(data)
            classifier = train_model(data)
            
            if classifier is None:
                return jsonify({'error': 'Không thể tạo mô hình. Vui lòng kiểm tra logs.'}), 500
        
        # Dự đoán
        prediction = classifier.predict([review_text])
        probabilities = classifier.predict_proba([review_text])[0]
        
        # Định nghĩa nhãn
        sentiment_label = "Tích cực" if prediction[0] == 1 else "Tiêu cực"
        
        # Lấy độ tin cậy
        confidence = probabilities[1] if prediction[0] == 1 else probabilities[0]
        
        # Lưu lịch sử dự đoán
        log_prediction(review_text, sentiment_label, confidence)
        
        # Phân tích thêm
        review_length = len(review_text.split())
        additional_info = {
            'review_length': review_length,
            'processed_text': preprocess_text(review_text)
        }
        
        # Trả về kết quả
        return jsonify({
            'review': review_text,
            'label': sentiment_label,
            'confidence': float(confidence),
            'additional_info': additional_info
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """API lấy thống kê về dự đoán"""
    try:
        # Đường dẫn đến file log
        log_file = 'data/logs/prediction_history.csv'
        
        # Kiểm tra file tồn tại
        if not os.path.exists(log_file):
            return jsonify({'error': 'Chưa có dữ liệu dự đoán'}), 404
        
        # Đọc file log
        log_df = pd.read_csv(log_file)
        
        # Tính toán thống kê
        total_predictions = len(log_df)
        positive_count = len(log_df[log_df['prediction'] == 'Tích cực'])
        negative_count = len(log_df[log_df['prediction'] == 'Tiêu cực'])
        
        # Tính tỷ lệ
        positive_ratio = positive_count / total_predictions if total_predictions > 0 else 0
        negative_ratio = negative_count / total_predictions if total_predictions > 0 else 0
        
        # Trả về kết quả
        return jsonify({
            'total_predictions': total_predictions,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API kiểm tra trạng thái của server"""
    try:
        return jsonify({'status': 'ok', 'model_loaded': classifier is not None})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs('logs', exist_ok=True)
    
    # Khởi động server
    app.run(debug=True, host='0.0.0.0', port=5000)