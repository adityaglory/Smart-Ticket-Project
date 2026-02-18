# Smart Ticket AI: Retrieval-Augmented Generation (RAG) Customer Support System

## Overview
This project implements an intelligent customer support automation system designed to classify tickets, detect urgency based on context, and recommend solutions using semantic search. The system addresses the limitations of traditional keyword-based support tools by utilizing Deep Learning (Transformers) and Vector Search technologies.

<img width="1919" height="920" alt="Screenshot 2026-02-18 150052" src="https://github.com/user-attachments/assets/f53cd52d-da03-4975-a513-48c5fdbca3a7" />

<img width="1917" height="921" alt="Screenshot 2026-02-18 150201" src="https://github.com/user-attachments/assets/cacd4b2f-0209-40f9-8647-5badf4bf4f1b" />
FIgure 1-2: AI model correctly identifies a technical issue based on input and give suggest as a feedback via RAG.

## Key Features

### 1. Context-Aware Sentiment Analysis
Unlike lexicon-based models (e.g., VADER) that rely on specific keywords, this system utilizes a fine-tuned DistilBERT transformer model. It is capable of detecting implicit urgency in customer complaints, such as financial distress signals in polite phrasing (e.g., "Transaction declined"), ensuring critical issues are prioritized correctly.

### 2. Retrieval-Augmented Generation (RAG)
The system employs a semantic search engine using FAISS (Facebook AI Similarity Search) and Sentence-Transformers. It retrieves the most relevant Standard Operating Procedure (SOP) from a vector database based on the semantic meaning of the ticket, distinguishing between technical failures and security threats.

### 3. Real-Time Monitoring Dashboard
A Streamlit-based interface provides real-time visualization of ticket influx, urgency distribution, and AI model performance metrics, enabling immediate decision-making for support agents.

## Technical Architecture
- **Language:** Python 3.10
- **API Framework:** FastAPI
- **Machine Learning:** Scikit-Learn, Hugging Face Transformers
- **Vector Database:** FAISS
- **Database:** PostgreSQL
- **Containerization:** Docker & Docker Compose

## Installation and Usage

1. Clone the repository to your local machine.
2. Start the infrastructure using Docker Compose:
   `docker-compose up -d`
3. Initialize the database schema:
   `python src/init_db.py`
4. Start the API server:
   `uvicorn src.api:app --reload`
5. Launch the dashboard:
   `streamlit run src/dashboard.py`

## Technical Report
For a detailed analysis of the model selection process and performance evaluation, please refer to the [ANALYSIS.md](./ANALYSIS.md) file included in this repository.
