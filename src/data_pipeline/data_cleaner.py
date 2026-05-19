import pandas as pd

class DataCleaner:
    def __init__(self):
        # Các cột chúng ta quan tâm theo thiết kế
        self.relevant_cols = [
            'Date received', 'Product', 'Consumer complaint narrative', 
            'Company', 'Timely response?'
        ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Giữ lại các cột cần thiết
        df = df[self.relevant_cols].copy()
        
        # Loại bỏ các dòng thiếu nội dung khiếu nại
        df = df.dropna(subset=['Consumer complaint narrative']).copy()
        
        # Hợp nhất các nhãn sản phẩm tương đồng
        mapping = {
            'Credit reporting, credit repair services, or other personal consumer reports': 
            'Credit reporting or other personal consumer reports'
        }
        df['Product'] = df['Product'].replace(mapping)
        
        # Làm sạch văn bản cơ bản (viết thường, bỏ khoảng trắng thừa)
        df['Consumer complaint narrative'] = df['Consumer complaint narrative'].astype(str).str.lower().str.strip()
        
        return df