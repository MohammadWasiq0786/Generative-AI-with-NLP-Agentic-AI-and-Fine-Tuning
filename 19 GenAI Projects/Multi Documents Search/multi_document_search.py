import os
import io
import pdfplumber
import numpy as np
import faiss
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from uuid import uuid4

EURI_API_KEY="EURI_API_KEY"
EURI_CHAT_URL="https://api.euron.one/api/v1/euri/alpha/chat/completions"
EURI_EMBED_URL="https://api.euron.one/api/v1/euri/alpha/embeddings"




def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_euri_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    res = requests.post(EURI_EMBED_URL, headers=headers, json=payload)
    return np.array([d["embedding"] for d in res.json()["data"]])


def build_faiss_index(chunks):
    embeddings = get_euri_embeddings(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings


def search_chunks(query, chunks, index, embeddings, top_k=3):
    q_embed = get_euri_embeddings([query])[0]
    D, I = index.search(np.array([q_embed]), top_k)
    return [chunks[i] for i in I[0]], q_embed, [embeddings[i] for i in I[0]]

def get_answer_from_context(query, context, memory=[]):
    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are an intelligent document assistant. Based on the context, answer the query clearly.

Context:
{context}

Query: {query}
"""

    messages = [{"role": "system", "content": "Answer based only on the given context."}]
    messages += memory
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.3
    }

    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
    reply = res.json()['choices'][0]['message']['content']

    memory.append({"role": "user", "content": query})
    memory.append({"role": "assistant", "content": reply})
    return reply

def plot_vectors(vectors, query_vector):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(vectors + [query_vector]))
    doc_points = reduced[:-1]
    query_point = reduced[-1]

    fig, ax = plt.subplots()
    ax.scatter(doc_points[:, 0], doc_points[:, 1], label="Chunks", alpha=0.6)
    ax.scatter(query_point[0], query_point[1], color="red", label="Query", marker="x")
    ax.legend()
    st.pyplot(fig)


st.set_page_config(page_title="🧠 AI Document Search Pro", layout="wide")
st.title("📄 AI-Powered Multi-PDF Search Engine")

uploaded_files = st.file_uploader("Upload one or more PDF documents", type="pdf", accept_multiple_files=True)
query = st.text_input("🔍 Ask something about the documents:")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []
    
    
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        with open(f"temp_{uuid4()}.pdf", "wb") as f:
            f.write(file.read())
            text = extract_text_from_pdf(f.name)
            all_text += text + "\n"

    chunks = split_text(all_text)
    index, embeddings = build_faiss_index(chunks)

    st.success(f"{len(uploaded_files)} file(s) loaded and indexed successfully.")
    
if query:
        top_chunks, query_vector, retrieved_vectors = search_chunks(query, chunks, index, embeddings)
        context = "\n\n".join(top_chunks)
        answer = get_answer_from_context(query, context, st.session_state.chat_memory)

        st.markdown("### ✅ Answer:")
        st.write(answer)

        with st.expander("📜 Top Relevant Chunks"):
            for chunk in top_chunks:
                st.markdown(f"---\n{chunk}")

        with st.expander("📊 Vector Similarity Visualization"):
            plot_vectors(retrieved_vectors, query_vector)

        with st.expander("🧠 Conversation Memory"):
            st.json(st.session_state.chat_memory)
