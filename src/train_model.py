import pandas as pd
import numpy as np
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Download resource NLTK (untuk sentimen nanti)
nltk.download('vader_lexicon', quiet=True)

def generate_dummy_data(n_samples=200):
    """
    Membuat data keluhan palsu tapi realistis untuk latihan.
    Di dunia nyata, ini diganti dengan pd.read_csv('data_asli.csv')
    """
    data = {
        'text': [],
        'category': []
    }
    
    # Pola keluhan untuk setiap kategori
    templates = {
        'Credit Card': [
            "My credit card was declined at the store.",
            "I was charged twice for the same transaction.",
            "Please cancel my credit card immediately, it was stolen.",
            "Why is my credit limit so low?",
            "Unexpected fee on my mastercard statement."
        ],
        'Mortgage': [
            "I need help with my mortgage payment plan.",
            "When is the next refinance rate available?",
            "My house loan application is stuck.",
            "Escrow account balance is incorrect.",
            "Can I defer my mortgage payment this month?"
        ],
        'Student Loan': [
            "I want to apply for student loan forgiveness.",
            "My interest rate on the student loan is too high.",
            "Navient has not updated my payment status.",
            "How do I consolidate my federal loans?",
            "Stop calling me about my student debt!"
        ]
    }
    
    # Generate random data
    for _ in range(n_samples):
        cat = np.random.choice(list(templates.keys()))
        # Ambil template acak dan tambah sedikit variasi
        base_text = np.random.choice(templates[cat])
        data['text'].append(base_text)
        data['category'].append(cat)
        
    return pd.DataFrame(data)

def train_and_save():
    print("1. Generating Data Latih...")
    df = generate_dummy_data(500)
    print(f"   -> Terbuat {len(df)} data sampel.")
    print(f"   -> Contoh: {df['text'].iloc[0]} ({df['category'].iloc[0]})")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42
    )

    # 2. Membangun Pipeline NLP
    # TfidfVectorizer: Mengubah teks jadi angka (frekuensi kata)
    # LogisticRegression: Algoritma klasifikasi yang cepat & akurat untuk teks
    print("\n2. Melatih Model NLP...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)
    
    # Evaluasi
    accuracy = pipeline.score(X_test, y_test)
    print(f"   -> Akurasi Model: {accuracy:.2f}")

    # 3. Simpan Model
    print("\n3. Menyimpan Model ke folder 'models/'...")
    joblib.dump(pipeline, 'models/ticket_classifier.pkl')
    print("   -> Sukses! File 'ticket_classifier.pkl' telah disimpan.")

if __name__ == "__main__":
    train_and_save()
