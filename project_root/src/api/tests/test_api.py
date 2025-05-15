import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần test
from api import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Kiểm tra endpoint health hoạt động chính xác"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')
    
    @patch('api.classifier')
    def test_predict_endpoint_with_model(self, mock_classifier):
        """Kiểm tra endpoint predict khi model đã được tải"""
        # Mock dữ liệu trả về của classifier
        mock_classifier.predict.return_value = ["Tích cực"]
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]
        
        # Gửi request test
        response = self.app.post(
            '/predict',
            json={'review': 'Sản phẩm tuyệt vời'}
        )
        data = json.loads(response.data)
        
        # Kiểm tra kết quả
        self.assertEqual(response.status_code, 200)
        self.assertIn('label', data)
        self.assertEqual(data['label'], "Tích cực")
        self.assertIn('confidence', data)
        self.assertAlmostEqual(data['confidence'], 0.8)
    
    @patch('api.classifier', None)
    @patch('api.train_model')
    @patch('api.load_data')
    @patch('api.analyze_data')
    def test_predict_endpoint_without_model(self, mock_analyze, mock_load, mock_train):
        """Kiểm tra endpoint predict khi model chưa được tải"""
        # Mock các hàm được gọi
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["Tiêu cực"]
        mock_classifier.predict_proba.return_value = [[0.9, 0.1]]
        mock_train.return_value = mock_classifier
        
        # Gửi request test
        response = self.app.post(
            '/predict',
            json={'review': 'Sản phẩm tệ'}
        )
        
        # Kiểm tra các hàm đã được gọi
        mock_load.assert_called_once()
        mock_analyze.assert_called_once()
        mock_train.assert_called_once()
    
    def test_predict_endpoint_invalid_input(self):
        """Kiểm tra endpoint predict với dữ liệu không hợp lệ"""
        # Thiếu trường review
        response = self.app.post(
            '/predict',
            json={'text': 'Sản phẩm tuyệt vời'}  # Tên trường sai
        )
        
        self.assertEqual(response.status_code, 400)
        
        # JSON không hợp lệ
        response = self.app.post(
            '/predict',
            data='Invalid JSON',
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_stats_endpoint(self, mock_read_csv, mock_exists):
        """Kiểm tra endpoint stats"""
        # Mock dữ liệu
        mock_exists.return_value = True
        mock_df = MagicMock()
        mock_df.__len__.return_value = 100
        mock_df.loc.__getitem__.return_value.__len__.side_effect = [70, 30]  # Tích cực: 70, Tiêu cực: 30
        mock_read_csv.return_value = mock_df
        
        # Gửi request test
        response = self.app.get('/stats')
        data = json.loads(response.data)
        
        # Kiểm tra kết quả
        self.assertEqual(response.status_code, 200)
        self.assertIn('total_predictions', data)
        self.assertEqual(data['total_predictions'], 100)

if __name__ == '__main__':
    unittest.main()