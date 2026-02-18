import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    def __init__(self):
        print("   -> Loading RAG Model (Sentence-Transformers)...")
        # Model ini mengubah kalimat menjadi deretan angka (vector)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # --- KNOWLEDGE BASE (SOP BANK - UPDATED) ---
        # Database SOP yang sudah dipertajam agar AI lebih pintar membedakan kasus
        self.knowledge_base = [
            {
                "id": 1,
                "topic": "Credit Card Declined (Technical/Limit)",
                "content": "SOP-001 (Technical Issue): Jika nasabah melapor 'card declined', 'transaction failed', atau 'payment rejected' saat bertransaksi, JANGAN blokir kartu. Langkah pertama: Cek 'Daily Transaction Limit' di sistem. Sarankan nasabah menaikkan limit via Aplikasi Mobile."
            },
            {
                "id": 2,
                "topic": "Student Loan Status",
                "content": "SOP-002 (Loan Inquiry): Untuk pertanyaan status aplikasi pinjaman pelajar (Student Loan), informasikan bahwa proses persetujuan memakan waktu 3-5 hari kerja. Pastikan dokumen (KTM, Transkrip) sudah lengkap."
            },
            {
                "id": 3,
                "topic": "CRITICAL: Stolen / Lost / Fraud",
                "content": "SOP-URGENT (Security Threat): Gunakan SOP ini HANYA jika nasabah menggunakan kata kunci: 'stolen', 'lost', 'hacked', 'thief', atau 'unauthorized transaction'. Langkah: 1. Segera BLOKIR KARTU permanen. 2. Minta nasabah isi Form Sanggahan (Form-112)."
            },
            {
                "id": 4,
                "topic": "Mortgage / KPR Rates",
                "content": "SOP-004 (Housing Loan): Suku bunga KPR saat ini adalah 5.5% fixed untuk 3 tahun pertama. Penawaran Refinancing hanya tersedia untuk nasabah dengan masa kredit berjalan > 2 tahun."
            },
            {
                "id": 5,
                "topic": "General Inquiry / Account Access",
                "content": "SOP-STD (General): Untuk keluhan umum (lupa password, ganti alamat, saldo tidak update), lakukan verifikasi identitas standar (Nama Ibu Kandung). Pandu nasabah menggunakan menu 'Settings' di aplikasi."
            }
        ]
        
        # --- MEMBANGUN INDEX VECTOR (FAISS) ---
        print("   -> Building Vector Index...")
        self.documents = [doc['content'] for doc in self.knowledge_base]
        self.doc_embeddings = self.model.encode(self.documents)
        
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.doc_embeddings))
        print(f"   -> RAG Siap! Memuat {len(self.documents)} SOP yang sudah dipertajam.")

    def search(self, query_text, top_k=1):
        """
        Fungsi untuk mencari SOP yang paling mirip dengan keluhan user
        """
        query_vector = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_vector), top_k)
        
        best_match_idx = indices[0][0]
        best_sop = self.knowledge_base[best_match_idx]
        
        return {
            "sop_topic": best_sop['topic'],
            "suggested_action": best_sop['content']
        }

if __name__ == "__main__":
    # Test kecil kalau file dijalankan langsung
    rag = SimpleRAG()
    print("\nTest 1 (Limit):", rag.search("My card declined at the shop")['sop_topic'])
    print("Test 2 (Fraud):", rag.search("Help! Someone stole my wallet and card")['sop_topic'])
