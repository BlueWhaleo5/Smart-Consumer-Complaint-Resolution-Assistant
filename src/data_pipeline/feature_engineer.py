import pandas as pd

class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Đảm bảo bản copy để tránh SettingWithCopyWarning
        df = df.copy()
        
        # Tính số lượng từ
        df['word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(x.split()))
        
        # Xử lý thời gian
        df['Date received'] = pd.to_datetime(df['Date received'])
        df['year'] = df['Date received'].dt.year
        df['month_year'] = df['Date received'].dt.to_period('M').astype(str)
        
        # Tạo nhãn Priority
        # Nếu Timely response? là "No" -> Ưu tiên cao (High)
        df['priority'] = df['Timely response?'].apply(lambda x: 'High' if x == 'No' else 'Low')
        
        return df