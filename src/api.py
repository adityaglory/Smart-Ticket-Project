import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from datetime import datetime
from transformers import pipeline

# --- IMPORT BARU: RAG ENGINE ---
# Kita panggil file yang baru saja kamu buat
from src.rag_engine import SimpleRAG

# 1. Inisialisasi Aplikasi
app = FastAPI(title="Smart Ticket AI System + RAG")

print("\n--- SYSTEM STARTUP ---")

# A. Load Model Klasifikasi Topik
print("1. Loading Topic Classifier...")
topic_model = joblib.load('models/ticket_classifier.pkl')

# B. Load Model Sentimen (Transformers)
print("2. Loading Sentiment Engine (DistilBERT)...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# C. Load RAG Engine (Knowledge Base)
print("3. Loading Knowledge Base (RAG)...")
rag_engine = SimpleRAG()

# Koneksi Database
DATABASE_URL = "postgresql://admin:rahasia@localhost:5432/ticket_system"
engine = create_engine(DATABASE_URL)

print("--- SYSTEM READY ---\n")

class TicketInput(BaseModel):
    text: str
    customer_id: str = "GUEST"

def determine_urgency(topic, sentiment_score):
    is_negative = sentiment_score['label'] == 'NEGATIVE'
    confidence = sentiment_score['score']

    if is_negative and confidence > 0.9:
        return "CRITICAL"
    elif topic == "Credit Card":
        return "HIGH"
    elif is_negative and confidence > 0.7:
        return "MEDIUM"
    else:
        return "LOW"

@app.post("/predict_ticket")
def predict_ticket(ticket: TicketInput):
    start_time = datetime.now()
    
    try:
        # 1. Prediksi Topik
        pred_topic = topic_model.predict([ticket.text])[0]
        topic_conf = float(topic_model.predict_proba([ticket.text]).max())
        
        # 2. Prediksi Sentimen (Deep Learning)
        sentiment_result = sentiment_pipeline(ticket.text)[0]
        
        # Format skor sentimen (-1 s/d 1) untuk visualisasi
        display_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
        
        # 3. Tentukan Urgensi
        urgency = determine_urgency(pred_topic, sentiment_result)
        
        # 4. RAG SEARCH (Pencarian Solusi Cerdas)
        # AI mencari SOP yang cocok dengan teks keluhan
        rag_result = rag_engine.search(ticket.text)
        
        # 5. Simpan ke Database
        # Kita simpan data tiketnya. (Saran SOP tidak disimpan ke DB, cuma dikirim ke Dashboard)
        with engine.connect() as conn:
            query_ticket = text("""
                INSERT INTO tickets 
                (ticket_text, customer_id, predicted_topic, confidence_score, sentiment_score, urgency_level)
                VALUES (:text, :cust_id, :topic, :conf, :sent, :urg)
                RETURNING id
            """)
            result = conn.execute(query_ticket, {
                "text": ticket.text,
                "cust_id": ticket.customer_id,
                "topic": pred_topic,
                "conf": topic_conf,
                "sent": display_score,
                "urg": urgency
            })
            ticket_id = result.fetchone()[0]
            
            # Log Time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            conn.execute(text("INSERT INTO model_logs (ticket_id, execution_time_ms, model_version) VALUES (:tid, :exec, 'v3-RAG')"),
                        {"tid": ticket_id, "exec": execution_time})
            conn.commit()

        # 6. Kembalikan Hasil Komplit (JSON)
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "prediction": {
                "topic": pred_topic,
                "urgency": urgency,
                "sentiment_label": sentiment_result['label'],
                "sentiment_score": display_score,
                # INI FITUR BARUNYA:
                "suggested_sop": rag_result['suggested_action']
            }
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
