# Technical Research Report: Automated Customer Support Triage System using Transformer-Based Sentiment Analysis and Retrieval-Augmented Generation (RAG)


---

## Abstract
In the modern banking sector, Customer Support (CS) teams face the challenge of analyzing high-volume unstructured text data in real-time. Traditional rule-based systems often fail to capture implicit urgency and struggle with context-dependent queries. This project proposes a hybrid AI architecture combining **DistilBERT** for context-aware sentiment classification and **Retrieval-Augmented Generation (RAG)** for semantic solution retrieval. Experimental results demonstrate that this architecture significantly outperforms lexicon-based approaches (VADER) in detecting critical financial distress signals and provides consistent, protocol-adherent responses to support agents.

---

## 1. Introduction

### 1.1. Background
Tier-1 support agents often spend 30-40% of their time classifying tickets and searching for relevant Standard Operating Procedures (SOPs). Manual triage leads to:
1.  **High Mean Time to Resolution (MTTR):** Delays in addressing critical issues.
2.  **Inconsistent Resolutions:** Agents may provide varying solutions to identical problems.
3.  **The "Politeness Paradox":** Urgent financial issues phrased politely (e.g., "My card transaction failed, please help") are often misclassified as low-priority by keyword-based systems.

### 1.2. Objectives
The primary objective is to develop an end-to-end pipeline that:
* Accurately detects urgency levels based on semantic context, not just keywords.
* Retrieves the exact SOP from a knowledge base using vector similarity search.
* Reduces the cognitive load on human agents via a real-time dashboard.

---

## 2. Theoretical Framework & Methodology

### 2.1. Sentiment Analysis: From Lexicon to Transformers
We compared two distinct approaches for the urgency detection module:

#### A. Baseline: VADER (Valence Aware Dictionary and sEntiment Reasoner)
VADER is a lexicon and rule-based sentiment analysis tool. It maps words to sentiment scores (e.g., "bad" = -1.0, "good" = +1.0).
* **Limitation:** It utilizes a "Bag-of-Words" approach, ignoring word order and context. In banking, the phrase "declined transaction" contains no inherently negative adjectives, leading VADER to score it as **Neutral (0.0)**.

#### B. Proposed: DistilBERT (Distilled Bidirectional Encoder Representations from Transformers)
We utilized `distilbert-base-uncased-finetuned-sst-2-english`.
* **Mechanism:** Unlike VADER, BERT models utilize **Self-Attention mechanisms**. This allows the model to weigh the importance of the word "declined" heavily when it appears in the context of "credit card" or "transaction."
* **Architecture:** DistilBERT is a smaller, faster, cheaper version of BERT, retaining 97% of BERT's performance while being 40% lighter, making it ideal for real-time inference on CPU-based containerized environments.

### 2.2. Retrieval-Augmented Generation (RAG) Pipeline
To solve the hallucination problem common in Generative AI, this system uses a Retrieval-based approach.

1.  **Vector Embeddings:**
    Incoming tickets are converted into dense vector representations using the `all-MiniLM-L6-v2` model. This model maps sentences to a 384-dimensional vector space.
    $$v_{query} = f_{embedding}(text_{input})$$

2.  **Indexing & Retrieval (FAISS):**
    We utilize **Facebook AI Similarity Search (FAISS)** for efficient similarity search. The system calculates the **L2 Distance (Euclidean)** or **Cosine Similarity** between the query vector ($v_{query}$) and the stored SOP vectors ($v_{sop}$).
    
    The system retrieves the SOP ($d$) that minimizes the distance:
    $$d = \arg\min_{i} \| v_{query} - v_{sop_i} \|^2$$

---

## 3. System Architecture

The system follows a Microservices architecture containerized via Docker:

1.  **Data Ingestion Layer:**
    * **FastAPI:** Asynchronous entry point handling JSON payloads.
    * **Pydantic:** Validates data schema (strict typing).

2.  **Inference Engine (The "Brain"):**
    * **Topic Classifier:** Scikit-Learn Pipeline (TF-IDF + Logistic Regression).
    * **Sentiment Engine:** Hugging Face Transformers pipeline (DistilBERT).
    * **RAG Engine:** Sentence-Transformers + FAISS Index (FlatL2).

3.  **Persistence Layer:**
    * **PostgreSQL:** Relational database storing transactional logs (`tickets` table) and MLOps metrics (`model_logs` table).

4.  **Presentation Layer:**
    * **Streamlit:** WebSocket-based dashboard for real-time visualization.

---

## 4. Experimental Results & Analysis

### 4.1. Sentiment Analysis Performance
We tested the system against edge cases specifically designed to trick traditional models.

| Input Query | VADER Score | DistilBERT Score | Analysis |
| :--- | :--- | :--- | :--- |
| *"I hate this service!"* | -0.6 (Negative) | -0.99 (Negative) | Both models perform well on explicit anger. |
| *"My card is declined"* | **0.0 (Neutral)** | **-0.98 (Negative)** | **DistilBERT accurately detects the technical failure as a negative event.** VADER fails. |
| *"I want to pay my loan"* | 0.1 (Positive) | -0.02 (Neutral) | DistilBERT correctly identifies this as a neutral intent. |

**Conclusion:** DistilBERT demonstrates superior capability in detecting implicit urgency, crucial for financial contexts.

### 4.2. Semantic Search (RAG) Accuracy
The RAG system was evaluated on its ability to disambiguate similar keywords with different intents.

* **Test Case 1: "Card Declined"**
    * *Keyword Match:* Could match "Lost Card" or "Credit Limit".
    * *Vector Match:* Closest to SOP-001 (Technical/Limit Issue).
    * *Result:* Success. System advises checking the daily limit rather than blocking the card.

* **Test Case 2: "Stolen Wallet"**
    * *Vector Match:* Closest to SOP-URGENT (Fraud Protocol).
    * *Result:* Success. System immediately suggests card blocking.

### 4.3. Latency Benchmarking
* **Average Inference Time:** ~150ms per ticket (running on CPU).
* **Bottleneck:** The Transformer model accounts for ~120ms of the latency.
* **Optimization:** The use of `DistilBERT` (66M parameters) instead of `BERT-Base` (110M parameters) reduced latency by approximately 40% without significant accuracy loss.

---

## 5. Limitations and Future Work

1.  **Cold Start Problem:** The RAG system relies entirely on the pre-defined Knowledge Base. If a user asks about a new policy not yet in the SOP database, the system may retrieve an irrelevant document (Hallucination via Retrieval).
    * *Future Mitigation:* Implement a "Confidence Threshold". If the nearest vector distance > $X$, return "No SOP Found".

2.  **Language Support:** Currently optimized for English.
    * *Future Work:* Implement Multilingual DistilBERT (mBERT) to support Bahasa Indonesia and regional languages.

3.  **Scalability:** The current FAISS implementation uses `IndexFlatL2` (Brute force). For a dataset > 1 million tickets, we would migrate to `IndexIVFFlat` (Inverted File Index) for approximate nearest neighbor search to maintain <100ms latency.

---

## 6. Conclusion
The "Smart Ticket AI" system successfully demonstrates that integrating Deep Learning (Transformers) with Semantic Search (RAG) provides a robust solution for automated customer support. The system resolves the "Politeness Paradox" in sentiment analysis and ensures strict adherence to banking protocols through vector-based retrieval, paving the way for fully autonomous Tier-1 support agents.
