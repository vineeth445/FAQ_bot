import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

questions = [
    "What is AI?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is NLP?",
    "What is Generative AI?"
]

answers = [
    "AI simulates human intelligence.",
    "ML learns patterns from data.",
    "Deep Learning uses neural networks.",
    "NLP helps machines understand language.",
    "Generative AI creates content."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(questions)

st.title("Semantic FAQ Bot")

query = st.text_input("Ask a question")

if query:
    query_embedding = model.encode([query])

    scores = cosine_similarity(
        faq_embeddings,
        query_embedding
    )

    best_match = np.argmax(scores)

    st.write("Matched Question:")
    st.write(questions[best_match])

    st.write("Answer:")
    st.success(answers[best_match])
