
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# --- Load Vector Store ---
persist_directory = "rag_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# --- Load LLM ringan dari HuggingFace ---
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.3, "max_length": 512}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# --- UI Streamlit ---
st.set_page_config(page_title="Chatbot Budaya Papua", layout="wide")
st.title("💬 Chatbot Edukasi Cerita Rakyat Papua")
st.markdown("Tanya apa pun tentang cerita rakyat dari Papua yang kamu tahu!")

query = st.text_input("Masukkan pertanyaanmu di sini:")

if query:
    with st.spinner("Sedang mencari jawaban..."):
        result = qa_chain.run(query)
        st.success(result)
