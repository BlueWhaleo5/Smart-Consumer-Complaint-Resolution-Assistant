import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from .retriever import ComplaintRetriever

load_dotenv()
# Cấu hình API Key (Bạn nên để vào file .env)
#os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class QAChainManager:
    def __init__(self, use_local=False):
        # 1. Khởi tạo "Thủ thư"
        self.retriever_obj = ComplaintRetriever()
        self.retriever = self.retriever_obj.vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 4, 'lambda_mult': 0.5}
        )

        # 2. Lựa chọn khung xử lý
        if use_local:
            # --- Local (Ollama) ---
            # from langchain_community.llms import Ollama
            # self.llm = Ollama(model="llama3")
            pass 
        else:
            # --- API (Groq) ---
            model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
            self.llm = ChatGroq(
                model_name=model_name,
                temperature=0.3,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

        # 3. Định nghĩa Prompt Template
        template = """
        You are a professional Financial Complaint Assistant for the SCCRA system. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and professional.

        CONTEXT:
        {context}

        QUESTION: {question}

        HELPFUL ANSWER:"""
        
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def get_answer(self, query):
        """Hàm chính để lấy câu trả lời với xử lý lỗi API"""
        try:
            # Tạo chain xử lý: Tìm kiếm -> Đưa vào Prompt -> Hỏi LLM -> Trả lời
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
            )
            
            result = qa_chain.invoke({"query": query})
            
            # Trích xuất thông tin
            answer = result["result"]
            # Lấy nguồn trích dẫn từ metadata
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]))
            
            return answer, sources
        except Exception as e:
            error_str = str(e).lower()
            # Bắt đầu khi gặp lỗi API Gemini
            if "quota" in error_str or "api_key" in error_str or "429" in error_str:
                print(f"!!! Lỗi API Gemini ({error_str}). Đang tự động chuyển sang Ollama (Local LLM)...")
                return self._fallback_to_local(query, error_str)
                
            error_msg = f"!!! Lỗi khi gọi trợ lý AI: {str(e)} !!!"
            return error_msg, ["Error"]

    def _fallback_to_local(self, query, error_str=""):
        """Hàm dự phòng tự động chuyển sang dùng Ollama"""
        try:
            from langchain_community.llms import Ollama
            print("Đang khởi động Ollama (Llama3)...")
            local_llm = Ollama(model="llama3")
            
            qa_chain_local = RetrievalQA.from_chain_type(
                llm=local_llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
            )
            
            result = qa_chain_local.invoke({"query": query})
            
            answer = result["result"]
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]))
            
            return f"(Lưu ý: Đang trả lời bằng Local LLM - Llama3)\n\n" + answer, sources
            
        except Exception as local_e:
            return f"!!! Không thể tự động chuyển sang Local LLM. Vui lòng đảm bảo bạn đã cài đặt ứng dụng Ollama và chạy lệnh 'ollama pull llama3' trong terminal. Lỗi: {str(local_e)} !!!", ["Error"]

# --- Debug ---
if __name__ == "__main__":
    # Khởi tạo với API (Mặc định)
    qa_manager = QAChainManager(use_local=False)
    
    question = "How does SCCRA prioritize consumer complaints?"
    answer, sources = qa_manager.get_answer(question)
    
    print(f"\n AI ANSWER:\n{answer}")
    print(f"\n SOURCES: {sources}")

