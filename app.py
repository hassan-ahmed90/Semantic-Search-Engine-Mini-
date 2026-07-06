import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page config
st.set_page_config(page_title="Semantic Search App", page_icon="🔍")

st.title("🔍 Semantic Search")
st.write("Enter a query to find the top 5 most similar sentences.")

# Define the documents
documents = [
    # Customer Support (1–10)
    "How do I reset my password?",
    "Where can I update my email address?",
    "How do I cancel my subscription?",
    "What are your business hours?",
    "How can I change my billing information?",
    "How do I contact customer support?",
    "Where can I download my invoice?",
    "How do I track my order?",
    "What payment methods do you accept?",
    "How do I create a new account?",

    # Artificial Intelligence (11–20)
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing enables computers to understand text.",
    "Computer vision allows machines to analyze images.",
    "Generative AI can create text, images, and code.",
    "PyTorch is widely used for deep learning research.",
    "TensorFlow is a popular machine learning framework.",
    "Large language models are trained on massive text datasets.",
    "Data science combines statistics and programming.",
    "Feature engineering improves machine learning models.",

    # Programming (21–30)
    "Python is one of the most popular programming languages.",
    "Java is commonly used for enterprise applications.",
    "Git helps developers manage source code versions.",
    "GitHub is used to host software repositories.",
    "APIs allow different software systems to communicate.",
    "Object-oriented programming organizes code using classes.",
    "Debugging helps identify and fix software bugs.",
    "SQL is used to manage relational databases.",
    "Flutter enables cross-platform mobile app development.",
    "Docker simplifies application deployment using containers.",

    # Science & Education (31–40)
    "The Earth revolves around the Sun.",
    "Water boils at one hundred degrees Celsius.",
    "Gravity keeps planets in orbit around the Sun.",
    "Photosynthesis allows plants to produce food.",
    "Reading books improves vocabulary and knowledge.",
    "Mathematics is essential for engineering and science.",
    "Regular practice improves problem-solving skills.",
    "Online learning provides flexible education opportunities.",
    "The human brain contains billions of neurons.",
    "Electricity powers most modern electronic devices.",

    # Health, Travel & Lifestyle (41–50)
    "Regular exercise improves physical and mental health.",
    "A healthy diet includes fruits and vegetables.",
    "Drinking enough water keeps the body hydrated.",
    "Sleep is important for maintaining good health.",
    "Traveling helps people explore different cultures.",
    "Airplanes are one of the fastest modes of transportation.",
    "Hotels provide accommodation for travelers.",
    "Football is the world's most popular sport.",
    "Cricket is widely played in Pakistan and India.",
    "Cybersecurity protects systems from digital attacks."
]

# Cache the model and embeddings loading
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def get_document_embeddings(docs):
    model = load_model()
    return model.encode(docs)

# Load model and embeddings
with st.spinner("Loading model and calculating document embeddings..."):
    model = load_model()
    document_embeddings = get_document_embeddings(documents)

# Input from user
query = st.text_input("Enter your query:", placeholder="e.g., I want to learn artificial intelligence.")

# Search button
if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            # Encode query
            query_embedding = model.encode(query)
            
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(
                query_embedding.reshape(1, -1),
                document_embeddings
            ).flatten()
            
            # Get top 5 matches
            top_k = 5
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            
            st.subheader("Top 5 Similar Sentences:")
            for rank, index in enumerate(top_indices, start=1):
                score = similarity_scores[index]
                sentence = documents[index]
                
                # Display results nicely
                st.markdown(f"**{rank}.** {sentence}")
                st.caption(f"Similarity Score: `{score:.4f}`")
    else:
        st.warning("Please enter a query before searching.")
