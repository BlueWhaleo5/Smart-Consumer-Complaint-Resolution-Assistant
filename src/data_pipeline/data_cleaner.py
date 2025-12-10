import pandas as pd

class DataCleaner:
    def __init__(self):
        # Các cột chúng ta quan tâm theo thiết kế
        self.relevant_cols = [
            'Date received', 'Product', 'Consumer complaint narrative', 
            'Company', 'Timely response?'
        ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Chỉ giữ lại các cột cần thiết
        df = df[self.relevant_cols].copy()
        
        # 2. Loại bỏ các dòng thiếu nội dung khiếu nại (72.14% theo EDA của bạn)
        df = df.dropna(subset=['Consumer complaint narrative']).copy()
        
        # 3. Hợp nhất các nhãn sản phẩm tương đồng (Xử lý vấn đề bạn tìm thấy ở EDA)
        mapping = {
            'Credit reporting, credit repair services, or other personal consumer reports': 
            'Credit reporting or other personal consumer reports'
        }
        df['Product'] = df['Product'].replace(mapping)
        
        # 4. Làm sạch văn bản cơ bản (viết thường, bỏ khoảng trắng thừa)
        df['Consumer complaint narrative'] = df['Consumer complaint narrative'].astype(str).str.lower().str.strip()
        
        return df