import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class ComplaintRetriever:
    def __init__(self, persist_directory=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.persist_directory = persist_directory or os.path.join(base_dir, 'data', 'vector_db')
        
        # Dùng cùng một Model Embedding với file vector_store.py
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Kết nối tới VD đã tồn tại
        if os.path.exists(self.persist_directory):
            try:
                self.vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("Đã kết nối thành công tới Vector Database.")
            except Exception as e:
                print(f"!!! Lỗi khi khởi tạo ChromaDB: {e} !!!")
                self.vector_db = None
        else:
            print(f"!!! Cảnh báo: Chưa tìm thấy Vector DB tại {self.persist_directory} !!!")
            self.vector_db = None

    def get_context(self, query, k=3):
        """
        Tìm kiếm các đoạn văn bản liên quan nhất
        k: Số lượng đoạn văn bản muốn lấy ra (mặc định là 3)
        """
        print(f"--- Đang tìm kiếm thông tin cho câu hỏi: '{query}' ---")
        if self.vector_db is None:
            return "!!! Vector Database chưa sẵn sàng hoặc bị lỗi. !!!", ["Error"]
        
        try:
            # Sử dụng Similarity Search để tìm các vector gần nhất
            docs = self.vector_db.similarity_search(query, k=k)
            
            # Gộp nội dung các đoạn văn bản lại thành một chuỗi context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Trích xuất nguồn (metadata)
            sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
            
            return context, sources
        except Exception as e:
            print(f"!!! Lỗi khi tìm kiếm dữ liệu: {e} !!!")
            return "!!! Đã xảy ra lỗi trong quá trình truy xuất thông tin. !!!", ["Error"]

# --- Debug ---
if __name__ == "__main__":
    try:
        retriever = ComplaintRetriever()
        
        # Thử đặt một câu hỏi liên quan đến PDF hoặc Khiếu nại
        test_query = "How to handle a mortgage complaint?"
        context, sources = retriever.get_context(test_query)
        
        print("\n=== CONTEXT TÌM THẤY ===")
        print(context[:500] + "...") # In 500 ký tự đầu
        print("\n=== NGUỒN TRÍCH DẪN ===")
        print(set(sources))
        
    except Exception as e:
        print(f"!!! Lỗi: {e} !!!")