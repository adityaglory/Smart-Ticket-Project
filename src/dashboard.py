import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from sqlalchemy import create_engine

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Ticket AI Monitor",
    page_icon="🤖",
    layout="wide"
)

# --- INIT SESSION STATE (MEMORI) ---
# Ini agar data SOP tidak hilang saat refresh
if 'last_rag_result' not in st.session_state:
    st.session_state['last_rag_result'] = None

# --- KONEKSI DATABASE (READ ONLY) ---
DB_URL = "postgresql://admin:rahasia@localhost:5432/ticket_system"
engine = create_engine(DB_URL)

def load_data():
    try:
        query = "SELECT * FROM tickets ORDER BY created_at DESC LIMIT 50"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error connecting to DB: {e}")
        return pd.DataFrame()

# --- SIDEBAR: SIMULASI INPUT ---
st.sidebar.header("📝 Simulasi Tiket Masuk")
st.sidebar.write("Bertindaklah sebagai pelanggan:")

with st.sidebar.form("ticket_form"):
    customer_id = st.text_input("Customer ID", "CUST-001")
    text_input = st.text_area("Keluhan Pelanggan", "My card is not working!")
    submitted = st.form_submit_button("Kirim Tiket")

    if submitted:
        api_url = "http://localhost:8000/predict_ticket"
        payload = {"text": text_input, "customer_id": customer_id}
        
        try:
            with st.spinner('AI sedang berpikir...'):
                response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # SIMPAN KE MEMORI (SESSION STATE)
                st.session_state['last_rag_result'] = data['prediction']
                st.session_state['last_status'] = "success"
                
                # Paksa refresh agar tabel utama update
                st.rerun()
                
            else:
                st.session_state['last_status'] = "error"
                st.error("Gagal terhubung ke API")
        except Exception as e:
            st.error(f"API Error: {e}")

# --- TAMPILKAN HASIL RAG (DI LUAR FORM) ---
# Bagian ini akan tetap jalan meskipun halaman di-refresh
if st.session_state['last_rag_result']:
    pred = st.session_state['last_rag_result']
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Tiket Terkirim!")
    st.sidebar.markdown("### 🔍 Analisis AI")
    st.sidebar.write(f"**Topic:** {pred['topic']}")
    st.sidebar.write(f"**Urgency:** {pred['urgency']}")
    
    # Kotak Biru Saran SOP
    st.sidebar.info(f"💡 **Saran SOP (RAG):**\n\n{pred['suggested_sop']}")
    
    # Tombol untuk clear hasil (opsional)
    if st.sidebar.button("Reset Input"):
        st.session_state['last_rag_result'] = None
        st.rerun()

# --- HALAMAN UTAMA: MONITORING ---
st.title("🤖 Live Customer Support AI Dashboard")
st.markdown("---")

if st.button('🔄 Refresh Data'):
    st.rerun()

# 1. Load Data Real-time
df = load_data()

if not df.empty:
    # 2. Key Metrics
    col1, col2, col3 = st.columns(3)
    
    total_tickets = len(df)
    critical_count = len(df[df['urgency_level'] == 'CRITICAL'])
    
    avg_sentiment = df['sentiment_score'].mean() if not df['sentiment_score'].isna().all() else 0
    
    col1.metric("Total Tiket (Live)", f"{total_tickets}")
    col2.metric("Critical Issues", f"{critical_count}", delta_color="inverse")
    col3.metric("Rata-rata Sentimen", f"{avg_sentiment:.2f}")
    
    # 3. Visualisasi Grafik
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribusi Topik")
        fig_topic = px.bar(df['predicted_topic'].value_counts().reset_index(), 
                           x='predicted_topic', y='count',
                           title="Jumlah Tiket per Kategori")
        st.plotly_chart(fig_topic) 
        
    with c2:
        st.subheader("Tingkat Urgensi")
        color_map = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}
        fig_urgency = px.pie(df, names='urgency_level', 
                             title="Persentase Urgensi",
                             color='urgency_level',
                             color_discrete_map=color_map)
        st.plotly_chart(fig_urgency)

    # 4. Tabel Data Detail
    st.subheader("📋 Data Tiket Terbaru")
    
    def highlight_critical(s):
        if 'urgency_level' in s and s.urgency_level == 'CRITICAL':
            return ['background-color: #ffcccc'] * len(s)
        return [''] * len(s)

    st.dataframe(
        df[['created_at', 'customer_id', 'ticket_text', 'predicted_topic', 'urgency_level', 'sentiment_score']]
        .style.apply(highlight_critical, axis=1)
    )

else:
    st.info("Belum ada data tiket. Silakan input dari sidebar!")
