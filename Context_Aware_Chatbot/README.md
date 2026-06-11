# 💬 Context-Aware Chatbot Using RAG

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> **"A conversational AI assistant featuring Retrieval-Augmented Generation (RAG) that preserves dialogue memory, retrieves domain-specific knowledge, and grounds its responses using a fine-tuned T5 language model."**

---

## 🎯 1. Objective of the task
The objective is to build a robust, context-aware conversational chatbot that:
- Retains memory of prior user turns for seamless multi-turn conversations.
- Retrieves highly relevant information from a vectorized custom knowledge base.
- Generates accurate, grounded answers to user questions while minimizing hallucinations.
- Generates deployable artifacts, including a Streamlit application script for immediate interactive web deployment.

---

## ⚙️ 2. Methodology / Approach
The implementation follows an end-to-end Retrieval-Augmented Generation (RAG) architecture:
- **Knowledge Base Construction**: A custom support-style corpus is created covering topics like billing, subscriptions, API limits, and app issues.
- **Text Chunking & Vectorization**: The knowledge base is segmented into smaller, overlapping chunks. These chunks are embedded into a vector space using a `TfidfVectorizer` to create a lightweight and fast retrieval matrix.
- **Retrieval Engine**: User queries are vectorized and compared against the document matrix using cosine similarity to fetch the top-k most relevant context chunks.
- **Conversation Memory**: The chatbot maintains a rolling window of recent dialogue turns, feeding both the conversation history and the retrieved context into the model's prompt.
- **Generation**: A Hugging Face sequence-to-sequence model (`google/flan-t5-small`) acts as the generator. It follows strict instruction-tuned rules to base its answers purely on the provided context and prior chat turns.
- **Deployment Generation**: The pipeline serializes all required artifacts (vectorizer, chunk data, matrix) and automatically writes a `streamlit_app.py` script to allow immediate local deployment.

---

## 📊 3. Key results or observations
- **High Retrieval Accuracy**: The system demonstrated exceptional top-1 retrieval accuracy on the evaluation dataset, consistently mapping user questions to the correct underlying topic.
- **Effective Context Awareness**: The integration of rolling conversation memory allowed the model to correctly resolve pronouns and follow-up questions in multi-turn dialogues.
- **Self-Contained Portability**: By utilizing a synthesized custom corpus and lightweight models (TF-IDF + T5-small), the entire workflow is highly portable, reproducible, and executable natively on cloud environments like Kaggle without requiring external database dependencies.
- **Instant Deployment**: The pipeline outputs a fully functional Streamlit application, bridging the gap between model prototyping and interactive user testing.
