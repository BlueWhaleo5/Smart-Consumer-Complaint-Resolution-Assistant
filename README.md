# Smart Consumer Complaint & Resolution Assistant (SCCRA)

**SCCRA** is an AI ecosystem designed to streamline and intelligentize the end-to-end lifecycle of consumer complaint management. SCCRA empowers organizations to handle massive volumes of feedback with precision and speed.

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

## Performance Evaluation

Below is a summary of the actual performance metrics of the SCCRA system:

| Metric | Time / Rating | Notes |
| :--- | :--- | :--- |
| **General Request Processing** | ~1 minute | Average total time for individual tasks |
| **Random Assistant Response (RAG)** | 2 - 3 minutes | Depends on the complexity of the query and the amount of data to be queried |
| **Model Training** | 45p - 1+ hour | Includes data cleaning, feature processing, and Random Forest training |
| **Classification Accuracy** | Very good (Keyword-based) | Extremely effective when based on sample keywords; self-eval capabilities are improving |

> [!NOTE]
> Training and data processing time is significantly affected by the size of the `complaints.csv` file (usually > 7GB). Using smaller sample datasets for faster testing is recommended.

## Acknowledgments
- Phase 1 & 2: Data Pipeline Implemented
- Phase 3: ML Engine Implemented
- Phase 4: RAG Initialized
- Phase 5: Complete User Documentation

Video demo link: https://youtu.be/tIF_BW6scpQ
