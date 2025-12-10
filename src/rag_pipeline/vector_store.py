import sqlite3
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, db_path=None, pdf_path=None, persist_directory=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.db_path = db_path or os.path.join(base_dir, 'data', 'database', 'sccra_db.sqlite')
        self.pdf_path = pdf_path or os.path.join(base_dir, 'SCCRA.pdf')
        self.persist_directory = persist_directory or os.path.join(base_dir, 'data', 'vector_db')
        
        # Sử dụng model embedding mã nguồn mở
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def load_pdf_data(self):
        """Đọc kiến thức từ file PDF dự án"""
        print(f"--- Đang đọc file PDF: {self.pdf_path} ---")
        if os.path.exists(self.pdf_path):
            loader = PyPDFLoader(self.pdf_path)
            return loader.load()
        else:
            print("!!! Cảnh báo: Không tìm thấy file PDF !!!")
            return []

    def get_db_summary(self):
        """Tạo bản tóm tắt thống kê từ DB để chatbot có thể trả lời các câu hỏi tổng quát"""
        print("--- Đang tạo bản tóm tắt thống kê từ Database ---")
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Tổng số khiếu nại
            total = pd.read_sql("SELECT COUNT(*) as count FROM complaints", conn).iloc[0]['count']
            
            # Top 3 sản phẩm bị khiếu nại nhiều nhất
            top_products = pd.read_sql(
                "SELECT Product, COUNT(*) as count FROM complaints GROUP BY Product ORDER BY count DESC LIMIT 3", 
                conn
            )
            products_str = ", ".join([f"{row['Product']} ({row['count']})" for _, row in top_products.iterrows()])
            
            # Thống kê độ ưu tiên
            priority_stats = pd.read_sql(
                "SELECT priority, COUNT(*) as count FROM complaints GROUP BY priority", 
                conn
            )
            priority_str = ", ".join([f"{row['priority']}: {row['count']}" for _, row in priority_stats.iterrows()])
            
            conn.close()

            summary_text = f"""
            SYSTEM DATA SUMMARY (Real-time Statistics):
            - Total Complaints in Database: {total}
            - Top 3 Products with Most Complaints: {products_str}
            - Priority Distribution: {priority_str}
            - Data source: SCCRA SQLite Database.
            - Current analysis shows that the product with the highest complaint volume (and likely the one the user is asking about in terms of priority/volume) is {top_products.iloc[0]['Product']}.
            """
            
            return [Document(page_content=summary_text, metadata={"source": "system_summary"})]
        except Exception as e:
            print(f"!!! Lỗi khi tạo tóm tắt DB: {e} !!!")
            return []

    def load_sqlite_data(self, limit=1000):
        """Đọc các khiếu nại từ SQLite để làm ví dụ đối chiếu"""
        print(f"--- Đang đọc {limit} khiếu nại từ SQLite ---")
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT [Consumer complaint narrative], Product FROM complaints WHERE [Consumer complaint narrative] IS NOT NULL LIMIT {limit}"
            df = pd.read_sql(query, conn)
            conn.close()

            documents = []
            for _, row in df.iterrows():
                # Biến mỗi dòng thành Document object của LangChain
                doc = Document(
                    page_content=row['Consumer complaint narrative'],
                    metadata={"source": "sqlite", "product": row['Product']}
                )
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"!!! Lỗi khi đọc SQLite: {e} !!!")
            return []

    def create_vector_store(self):
        # Thu thập dữ liệu
        pdf_docs = self.load_pdf_data()
        sqlite_docs = self.load_sqlite_data()
        summary_docs = self.get_db_summary()
        all_docs = pdf_docs + sqlite_docs + summary_docs

        # Chia nhỏ văn bản (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)
        print(f"--- Đã tạo ra {len(chunks)} đoạn văn bản nhỏ (chunks) ---")

        # Tạo Vector DB
        print(f"--- Đang tạo Vector DB tại {self.persist_directory} (Vui lòng đợi)... ---")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Vector Database đã được khởi tạo thành công.")
        return vector_db

if __name__ == "__main__":
    # manager = VectorStoreManager()
    manager = VectorStoreManager()
    manager.create_vector_store()