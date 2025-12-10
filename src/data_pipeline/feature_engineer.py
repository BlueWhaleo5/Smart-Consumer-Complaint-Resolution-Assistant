import pandas as pd

class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Đảm bảo bản copy để tránh SettingWithCopyWarning
        df = df.copy()
        
        # 1. Tính số lượng từ (Phục vụ phân tích độ dài văn bản)
        df['word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(x.split()))
        
        # 2. Xử lý thời gian (Tách tháng/năm để vẽ biểu đồ xu hướng)
        df['Date received'] = pd.to_datetime(df['Date received'])
        df['year'] = df['Date received'].dt.year
        df['month_year'] = df['Date received'].dt.to_period('M').astype(str)
        
        # 3. Tạo nhãn Priority (Yêu cầu UC-ML-02)
        # Nếu Timely response? là "No" -> Ưu tiên cao (High)
        df['priority'] = df['Timely response?'].apply(lambda x: 'High' if x == 'No' else 'Low')
        
        return df