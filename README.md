# Smart Consumer Complaint & Resolution Assistant (SCCRA)

**SCCRA** is an enterprise-grade AI ecosystem designed to streamline and intelligentize the end-to-end lifecycle of consumer complaint management. By bridging the gap between raw data and actionable insights, SCCRA empowers organizations to handle massive volumes of feedback with precision and speed.

## Core Pillars
1.  **Intelligent Data Triage**: Leverages Machine Learning (**Random Forest + TF-IDF**) to automatically categorize complaints and predict severity, reducing manual processing time by identifying high-priority cases in real-time.
2.  **Conversational Intelligence (RAG)**: Integrates a **Retrieval-Augmented Generation** assistant backed by **LangChain** and **Gemini/Llama-3**. It provides staff with instant, context-aware answers derived from internal policy documents and historical case resolutions.
3.  **Visual Analytics Hub**: A powerful **Streamlit-based EDA dashboard** that transforms complex datasets (1.3M+ records) into interactive visualizations, exposing seasonal trends, product pain points, and geographic sentiment.

## Why it Matters
In an era where customer satisfaction is a primary differentiator, SCCRA provides a scalable solution to handle the **7GB+** Consumer Complaint Database, ensuring no complaint goes unnoticed and every resolution is backed by historical intelligence.

## Key Features
- **Data EDA Dashboard**: Real-time visualization and analytics of complaint data
- **ML Classification Engine**: Predicts product categories and flags high-priority complaints in real-time
- **Smart RAG Assistant**: A conversational agent that can answer internal policy queries and search historical complaint cases

## Project Structure
```text
ComplaintAssistant/
│
├── app/                  # Streamlit Web App logic & UI
│   └── main.py           
├── data/                 # Data, Models, and Databases
│   ├── database/         # SQLite storage for complaints
│   └── vector_db/        # ChromaDB index for RAG (generated automatically)
├── src/                  
│   ├── data_pipeline/    # Data Loading, Cleaning, and Engineering
│   ├── ml_pipeline/      # Training script and model artifacts
│   └── rag_pipeline/     # QA Chain, Vector Store logic, and retrievers
├── SCCRA.pdf             # Project rules, timeline, and knowledge base
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── run.py                # Main script to run the project
```

## Setup & Execution

### 1. Requirements
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running The Application
> Ensure you are in the root directory `d:\Code\ComplaintAssistant`

**To run the Web App directly (Default):**
```bash
python run.py
# or
python run.py --app
```

**If this is your FIRST TIME, you must initialize the AI Vector DB:**
```bash
python run.py --rag
```

**If you want to use Local LLM (Ollama), setup the model first:**
```bash
python run.py --setup-local
```
*Wait for this script to chunk and embed `SCCRA.pdf` and historical DB cases.*

**To do both at once:**
```bash
python run.py --all
```

### 3. API Keys Configuration (For RAG Chatbot)
By default, SCCRA uses Groq `llama-3.3-70b-versatile` for the RAG engine.
You can define your API key in a `.env` file at the root level:
```env
GROQ_API_KEY="your-groq-key"
```
Or you can use localized models like Llama 3 via Ollama by changing `use_local=True` in `app/main.py`.

## Performance Evaluation (Đánh giá hiệu năng)

Dưới đây là tóm tắt các chỉ số vận hành thực tế của hệ thống SCCRA:

| Chỉ số | Thời gian / Đánh giá | Ghi chú |
| :--- | :--- | :--- |
| **Xử lý yêu cầu chung** | ~1 phút | Tổng thời gian trung bình cho các tác vụ đơn lẻ |
| **Phản hồi Trợ lý ảo (RAG)** | 2 - 3 phút | Phụ thuộc vào độ phức tạp của câu hỏi và lượng dữ liệu cần truy vấn |
| **Huấn luyện Model (Training)** | 45p - 1h+ | Bao gồm làm sạch dữ liệu, xử lý tính năng và train Random Forest |
| **Độ chính xác phân loại** | Rất tốt (Keyword-based) | Hoạt động cực kỳ hiệu quả khi dựa trên từ khóa mẫu; khả năng tự đánh giá (self-eval) đang được cải thiện |

> [!NOTE]
> Thời gian huấn luyện và xử lý dữ liệu bị ảnh hưởng lớn bởi kích thước file `complaints.csv` (thường > 7GB). Khuyến khích sử dụng các tập dữ liệu mẫu nhỏ hơn để test nhanh.

## Acknowledgments
- Phase 1 & 2: Data Pipeline Implemented
- Phase 3: ML Engine Implemented
- Phase 4: RAG Initialized
- Phase 5: Complete User Documentation

Video demo link: https://youtu.be/tIF_BW6scpQ
