import joblib
import os

class ComplaintPredictor:
    def __init__(self, model_dir='models/'):
        # Load model và vectorizer ngay khi khởi tạo
        model_path = os.path.join(model_dir, 'product_classifier.pkl')
        vec_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
        else:
            raise FileNotFoundError("Chưa tìm thấy file model. Hãy chạy train.py trước!")

    def predict(self, text):
        """Dự đoán cả Sản phẩm và Độ ưu tiên (Hybrid)"""
        if not text or len(text.strip()) < 10:
            return "N/A", 0, "Low", "Văn bản quá ngắn"

        # Dự đoán Sản phẩm bằng ML
        vec_text = self.vectorizer.transform([text])
        product = self.model.predict(vec_text)[0]
        confidence = self.model.predict_proba(vec_text).max() * 100

        # Dự đoán Priority bằng Hybrid Logic
        high_priority_keywords = [
            "foreclosure", "evict", "theft", "stolen", "fraud", 
            "harass", "threat", "police", "jail", "identity theft",
            "identity card", "fake", "unauthorized", "scam"
        ]
        is_urgent = any(kw in text.lower() for kw in high_priority_keywords)
        priority = "High" if is_urgent else "Low"
        
        reason = "Phát hiện từ khóa khẩn cấp" if is_urgent else "Xử lý thông thường"

        return product, confidence, priority, reason

# Debug
if __name__ == "__main__":
    predictor = ComplaintPredictor()
    sample = "My house is at risk of foreclosure because of a bank error!"
    res = predictor.predict(sample)
    print(f"Kết quả test: {res}")