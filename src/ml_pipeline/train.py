import pandas as pd
import sqlite3
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, db_path='../../data/database/sccra_db.sqlite', model_dir='models/'):
        self.db_path = db_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_data(self, sample_size=100000): # Tăng lên 100k để chính xác hơn
        print(f"--- Đang tải dữ liệu từ SQLite ({sample_size} dòng)... ---")
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT [Consumer complaint narrative], Product FROM complaints LIMIT {sample_size}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def train_product_classifier(self):
        start_time = time.time()
        df = self.load_data()
        
        # Tiền xử lý & Vector hóa
        print("--- Đang chuyển đổi văn bản (TF-IDF)... ---")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
        X_vec = vectorizer.fit_transform(df['Consumer complaint narrative'])
        y = df['Product']
        
        # Huấn luyện (Tối ưu tốc độ để giải quyết vấn đề > 3 phút)
        print("--- Đang huấn luyện Product Classifier (Optimized)... ---")
        # Giới hạn max_depth giúp model nhẹ và dự đoán cực nhanh (< 1s)
        model = RandomForestClassifier(n_estimators=50, max_depth=25, n_jobs=-1, random_state=42)
        model.fit(X_vec, y)
        
        # Lưu trữ
        print(f"--- Đang lưu model vào {self.model_dir} ---")
        joblib.dump(model, os.path.join(self.model_dir, 'product_classifier.pkl'), compress=3)
        joblib.dump(vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), compress=3)
        
        print(f"Hoàn tất trong {time.time() - start_time:.2f} giây!")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_product_classifier()
