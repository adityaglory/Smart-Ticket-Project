import sqlalchemy
from sqlalchemy import create_engine, text
import time

# Konfigurasi Koneksi (Sesuai docker-compose tadi)
DATABASE_URL = "postgresql://admin:rahasia@localhost:5432/ticket_system"

def init_database():
    engine = create_engine(DATABASE_URL)

    # Retry logic kalau database belum siap 100%
    max_retries = 5
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                print(f"Percobaan {i+1}: Berhasil terhubung ke Database!")

                # 1. Tabel Tiket
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id SERIAL PRIMARY KEY,
                    ticket_text TEXT NOT NULL,
                    customer_id VARCHAR(50),
                    predicted_topic VARCHAR(50),
                    confidence_score FLOAT,
                    sentiment_score FLOAT,
                    urgency_level VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'OPEN'
                );
                """))
                print("Tabel 'tickets' siap.")

                # 2. Tabel Logs (MLOps)
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_logs (
                    log_id SERIAL PRIMARY KEY,
                    ticket_id INTEGER REFERENCES tickets(id),
                    execution_time_ms FLOAT,
                    model_version VARCHAR(20),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """))
                print("Tabel 'model_logs' siap.")

                conn.commit()
                break

        except Exception as e:
            print(f"Belum connect, menunggu... ({e})")
            time.sleep(2)

if __name__ == "__main__":
    init_database()
