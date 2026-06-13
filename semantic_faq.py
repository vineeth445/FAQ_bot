from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Knowledge Base
questions = [
    "What is AI?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is NLP?",
    "What is Generative AI?"
]

# Answers
answers = [
    "AI simulates human intelligence.",
    "ML learns patterns from data.",
    "Deep Learning uses neural networks.",
    "NLP helps machines understand language.",
    "Generative AI creates content."
]

# Load Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for all FAQ questions
faq_embeddings = model.encode(questions)

# User query
query = "AI is the future"

# Convert query to embedding
query_embedding = model.encode([query])

# Calculate similarity scores
scores = cosine_similarity(
    faq_embeddings,
    query_embedding
)

# Find best matching question
best_match = np.argmax(scores)

# Display results
print("User Query:", query)
print("Best Matching Question:", questions[best_match])
print("Answer:", answers[best_match])
