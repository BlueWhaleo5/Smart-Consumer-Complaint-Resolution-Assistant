import pandas as pd
import sqlite3
import os
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer

class DataLoader:
    def __init__(self, db_path='../../data/database/sccra_db.sqlite'):
        self.db_path = db_path
        # Tự động tạo thư mục database nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()

    def run_pipeline(self, csv_path):
        print(f"--- Đang bắt đầu xử lý file: {csv_path} ---")
        
        # Đọc dữ liệu
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"Đã load {len(df)} dòng dữ liệu thô.")
        except Exception as e:
            print(f"Lỗi khi đọc file CSV: {e}")
            return

        # Làm sạch
        df = self.cleaner.clean(df)
        print(f"Đã lọc bỏ dòng trống. Còn lại {len(df)} dòng.")

        # Tạo Features
        df = self.engineer.transform(df)
        print("Đã tạo xong các cột tính năng bổ trợ.")

        # Lưu vào SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql('complaints', conn, if_exists='replace', index=False)
            conn.close()
            print(f"Thành công! Dữ liệu đã lưu tại: {self.db_path}")
        except Exception as e:
            print(f"Lỗi khi lưu Database: {e}")

if __name__ == "__main__":
    input_csv = "../../data/complaints.csv" 
    
    loader = DataLoader()
    loader.run_pipeline(input_csv)