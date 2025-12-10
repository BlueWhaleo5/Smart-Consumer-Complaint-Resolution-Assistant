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
        
        # 1. Tiền xử lý & Vector hóa
        print("--- Đang chuyển đổi văn bản (TF-IDF)... ---")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
        X_vec = vectorizer.fit_transform(df['Consumer complaint narrative'])
        y = df['Product']
        
        # 2. Huấn luyện (Tối ưu tốc độ để giải quyết vấn đề > 3 phút)
        print("--- Đang huấn luyện Product Classifier (Optimized)... ---")
        # Giới hạn max_depth giúp model nhẹ và dự đoán cực nhanh (< 1s)
        model = RandomForestClassifier(n_estimators=50, max_depth=25, n_jobs=-1, random_state=42)
        model.fit(X_vec, y)
        
        # 3. Lưu trữ
        print(f"--- Đang lưu model vào {self.model_dir} ---")
        joblib.dump(model, os.path.join(self.model_dir, 'product_classifier.pkl'), compress=3)
        joblib.dump(vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), compress=3)
        
        print(f"Hoàn tất trong {time.time() - start_time:.2f} giây!")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_product_classifier()

# from pyexpat import model
# import pandas as pd
# import sqlite3
# import joblib
# import os
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# class ModelTrainer:
#     def __init__(self, db_path='../../data/database/sccra_db.sqlite', model_dir='models/'):
#         self.db_path = db_path
#         self.model_dir = model_dir
#         os.makedirs(self.model_dir, exist_ok=True)
        
#     def load_data(self, sample_size=1300000):
#         print(f"--- Đang tải dữ liệu từ SQLite (Lấy mẫu {sample_size} dòng) ---")
#         conn = sqlite3.connect(self.db_path)
#         # Lấy mẫu ngẫu nhiên để đảm bảo tính đại diện
#         query = f"SELECT [Consumer complaint narrative], Product, priority FROM complaints LIMIT {sample_size}"
#         df = pd.read_sql(query, conn)
#         conn.close()
#         return df

#     def train(self):
#         start_time = time.time()
#         df = self.load_data()
        
#         # 2. Vector hóa văn bản (TF-IDF)
#         print("--- Đang Vector hóa (TF-IDF)... ---")
#         X = df['Consumer complaint narrative']
#         y_product = df['Product']
#         y_priority = df['priority'] # Lấy thêm nhãn priority từ DB
        
#         vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))  # Giảm từ 5000 xuống 3.000 để giảm thời gian huấn luyện
#         X_vec = vectorizer.fit_transform(X)
        
#         # 3. Chia tập dữ liệu (Train 80% - Test 20%)
#         # X_train, X_test, y_train, y_test = train_test_split(X_vec, y_product, test_size=0.2, random_state=42)
        
#         # 4. Huấn luyện Model phân loại Sản phẩm
#         print("--- Đang huấn luyện Model (Random Forest) ---")
#             # max_depth=20: Giới hạn chiều sâu để model không quá nặng
#             # n_estimators=50: Giảm số lượng cây (mặc định 100)
#             # n_jobs=-1: Sử dụng tất cả nhân CPU
#         model_prod = RandomForestClassifier(n_estimators=50, max_depth=25, class_weight='balanced', n_jobs=-1, random_state=42)
#         model_prod.fit(X_vec, y_product)
#         joblib.dump(model_prod, os.path.join(self.model_dir, 'product_classifier.pkl'), compress=3)    # Sử dụng compress=3 để nén file model nhỏ gọn hơn

#         # 5. Huấn luyện Model Dự đoán Priority
#             # Model này học cách các công ty phản hồi chậm trong quá khứ dựa trên văn bản
#         model_prio = RandomForestClassifier(n_estimators=50, max_depth=25, class_weight='balanced', n_jobs=-1, random_state=42)
#         model_prio.fit(X_vec, y_priority)
#         joblib.dump(model_prio, os.path.join(self.model_dir, 'priority_classifier.pkl'), compress=3)
        
#         # 6. Đánh giá
#         print(f"--- Huấn luyện xong trong {time.time() - start_time:.2f}s ---")
#         y_pred = model_prod.predict(X_vec)
#         print("\nKẾT QUẢ ĐÁNH GIÁ MODEL:")
#         print(classification_report(y_product, y_pred))
#         print("\nĐánh giá sơ bộ trên tập Test:")
#         print(f"Accuracy: {model_prod.score(X_vec, y_product):.2%}")
        
#         # 6. Lưu Vectorizer
#         joblib.dump(vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), compress=3)
#         print("--- Hoàn thành! ---")

# if __name__ == "__main__":
#     trainer = ModelTrainer()
#     trainer.train()
