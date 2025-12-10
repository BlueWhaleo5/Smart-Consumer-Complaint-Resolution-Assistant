import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import sys

# Thêm đường dẫn thư mục gốc để có thể import từ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline.qa_chain import QAChainManager
from src.ml_pipeline.predict import ComplaintPredictor

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="SCCRA | Smart Consumer Complaint Assistant",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Background for Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }

    /* Glassmorphism Card Style */
    .stMetric, .stPlotlyChart, .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
    }

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(118, 75, 162, 0.1);
        border-bottom: 3px solid #764ba2 !important;
    }

    /* Chat Styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #764ba2;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

</style>
""", unsafe_allow_html=True)

# --- HÀM LOAD DỮ LIỆU EDA (TAB 1) ---
@st.cache_data(show_spinner="Đang nạp dữ liệu...")
def load_data_from_db():
    db_path = os.path.join(os.path.dirname(__file__), '../data/database/sccra_db.sqlite')
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM complaints", conn)
        conn.close()
        return df
    except Exception:
        return None

# --- HÀM KHỞI TẠO ---
@st.cache_resource
def get_predictor():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../src/ml_pipeline/models/')
        return ComplaintPredictor(model_dir=model_path)
    except Exception:
        return None

@st.cache_resource
def get_qa_manager():
    try:
        return QAChainManager(use_local=False)
    except Exception:
        return None

# --- UI HEADER ---
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-weight:800; font-size: 2.5rem;'>🛡️ Smart Consumer Complaint Assistant</h1>
        <p style='margin-top:10px; opacity: 0.9;'>Hệ thống hỗ trợ phân tích và giải quyết khiếu nại thông minh</p>
    </div>
""", unsafe_allow_html=True)

df = load_data_from_db()
predictor = get_predictor()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/carbon-copy/100/764BA2/security-shield.png", width=80)
    st.title("SCCRA Admin")
    st.markdown("---")
    if df is not None:
        all_products = df['Product'].unique().tolist()
        selected_products = st.multiselect("🔍 Lọc sản phẩm (Tab 1):", all_products, default=all_products[:2])
        filtered_df = df[df['Product'].isin(selected_products)] if selected_products else df
    else:
        st.warning("⚠️ Chưa có dữ liệu")
    
    st.markdown("---")
    if st.button("Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()

# --- TABS CHỨC NĂNG ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Phân loại AI", "💬 Trợ lý ảo"])

# === TAB 1: EDA DASHBOARD ===
with tab1:
    if df is not None:
        c11, c12, c13 = st.columns(3)
        with c11:
            st.metric("Tổng số khiếu nại", f"{len(filtered_df):,}", delta=None)
        with c12:
            st.metric("Tỉ lệ Ưu tiên cao", f"{(filtered_df['priority'] == 'High').mean():.1%}")
        with c13:
            st.metric("Độ dài văn bản trung bình", f"{filtered_df['word_count'].mean():.0f} từ")

        st.markdown("### 📈 Phân tích chi tiết")
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(filtered_df, names='Product', title="Phân bổ Sản phẩm", hole=0.5,
                             color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_hist = px.histogram(filtered_df, x='word_count', title="Phân bổ Độ dài nội dung", 
                                    color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.markdown("### 📅 Xu hướng khiếu nại theo thời gian")
        trend = filtered_df.groupby('month_year').size().reset_index(name='counts')
        fig_line = px.line(trend, x='month_year', y='counts', markers=True, 
                           color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("💡 Vui lòng chờ cập nhật dữ liệu từ hệ thống.")

# === TAB 2: ML PREDICTION ===
with tab2:
    st.markdown("### 🧠 Phân tích nội dung khiếu nại")
    st.write("Nhập nội dung khiếu nại (bằng tiếng Anh) để AI nhận diện loại hình sản phẩm và đánh giá mức độ khẩn cấp.")

    if predictor:
        user_input = st.text_area("Nội dung khiếu nại:", height=200, 
                                  placeholder="Ví dụ: I found unauthorized charges on my bank statement...")
        
        if st.button("Phân tích ngay", type="primary"):
            if user_input.strip():
                with st.spinner("Đang suy luận..."):
                    product, confidence, priority, reason = predictor.predict(user_input)
                
                st.markdown("---")
                st.success(f"**Kết quả dự đoán:** {product}")
                
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric("Độ tin cậy", f"{confidence:.1f}%")
                res_c2.metric("Mức độ ưu tiên", priority, delta="Cần xử lý gấp" if priority=="High" else None, delta_color="inverse")
                res_c3.metric("Lý do ưu tiên", reason)
                
                if priority == "High":
                    st.warning("**Lưu ý:** Khiếu nại này thuộc nhóm ưu tiên cao. Hãy ưu tiên xử lý sớm nhất có thể.")
            else:
                st.warning("Bạn chưa nhập nội dung khiếu nại.")
    else:
        st.error("Model AI hiện đang bảo trì.")

# === TAB 3: RAG CHAT ===
with tab3:
    st.markdown("### 💬 Trợ lý ảo")
    st.caption("Trợ lý có khả năng giải đáp mọi thắc mắc về quy trình xử lý khiếu nại của bạn.")

    qa_manager = get_qa_manager()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi tôi về quy trình xử lý..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if qa_manager:
            with st.chat_message("assistant"):
                with st.spinner("Đang tra cứu cơ sở kiến thức..."):
                    answer, sources = qa_manager.get_answer(prompt)
                    st.markdown(answer)
                    with st.expander("🔗 Xem nguồn trích dẫn"):
                        for s in sources:
                            st.write(f"{s}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("Trợ lý ảo hiện chưa sẵn sàng.")
