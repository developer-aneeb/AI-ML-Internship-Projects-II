
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = Path(__file__).parent / "artifacts"
MODEL_NAME = "google/flan-t5-small"
TOP_K_CONTEXT = 3
MEMORY_TURNS = 4
MAX_NEW_TOKENS = 128
MIN_SIMILARITY = 0.15
ABSTAIN_MESSAGE = "I could not find that information in the knowledge base."

@st.cache_resource
def load_resources():
    chunks_df = pd.read_csv(BASE_DIR / "chunks.csv")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
    doc_matrix = joblib.load(BASE_DIR / "tfidf_matrix.joblib")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return chunks_df, vectorizer, doc_matrix, tokenizer, model, device

chunks_df, vectorizer, doc_matrix, tokenizer, model, device = load_resources()

if "history" not in st.session_state:
    st.session_state.history = []

def format_history(history, max_turns=MEMORY_TURNS):
    if not history:
        return "No prior context."
    recent = history[-2 * max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\\n".join(lines)

def retrieve_chunks(query, top_k=TOP_K_CONTEXT, min_similarity=MIN_SIMILARITY):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_score = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.0

    if top_score < min_similarity:
        return None

    results = chunks_df.iloc[top_idx].copy()
    results["score"] = sims[top_idx]
    return results.reset_index(drop=True)

def build_prompt(question, history_text, retrieved_df):
    context_blocks = []
    for _, row in retrieved_df.iterrows():
        context_blocks.append(
            f"[Source: {row['title']} | Topic: {row['topic']} | Chunk: {row['chunk_id']}]\\n{row['chunk_text']}"
        )
    context_text = "\\n\\n".join(context_blocks)

    return f"""
You are a retrieval-grounded support assistant.

STRICT RULES:
- Answer ONLY using the retrieved knowledge base context and the conversation history.
- Do NOT use outside knowledge.
- If the answer is not explicitly present, say: "I could not find that information in the knowledge base."
- Be concise, factual, and helpful.
- Do not invent policies, numbers, or details.

Conversation history:
{history_text}

Retrieved context:
{context_text}

User question:
{question}

Grounded answer:
""".strip()

@torch.no_grad()
def answer_question(question):
    history_text = format_history(st.session_state.history)
    retrieved = retrieve_chunks(question, top_k=TOP_K_CONTEXT)

    if retrieved is None:
        return ABSTAIN_MESSAGE, None

    prompt = build_prompt(question, history_text, retrieved)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if not answer:
        answer = ABSTAIN_MESSAGE

    sources = ", ".join(retrieved["title"].dropna().unique().tolist())
    answer = f"{answer}\\n\\nSources: {sources}"

    return answer, retrieved

st.title("Context-Aware RAG Chatbot")
st.caption("Ask questions about the knowledge base. The assistant retrieves relevant chunks and uses conversation history.")

if st.button("Clear chat"):
    st.session_state.history = []
    st.rerun()

user_input = st.chat_input("Type your question...")

if user_input:
    answer, retrieved = answer_question(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": answer})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.history:
    st.subheader("Most recent retrieval results")
    last_user = st.session_state.history[-2]["content"] if len(st.session_state.history) >= 2 else st.session_state.history[-1]["content"]
    retrieved_now = retrieve_chunks(last_user, top_k=TOP_K_CONTEXT)
    if retrieved_now is not None:
        st.dataframe(retrieved_now[["title", "topic", "score"]])
    else:
        st.info("No sufficiently relevant context was found for the latest question.")
