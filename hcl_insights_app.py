import streamlit as st
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from google import genai

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("DB_NAME", "hcl")]
        collection = db[os.getenv("COLLECTION_NAME", "hcl-web")]
        return collection
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

# Initialize embedding model
@st.cache_resource
def init_embedding_model():
    return SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# Initialize Gemini client
@st.cache_resource
def init_gemini_client():
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        client = genai.Client(api_key=gemini_api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

def get_embeddings(text: str, model) -> List[float]:
    return model.encode(text).tolist()

def semantic_search(query: str, collection, embedding_model, top_k=5):
    """
    Performs semantic search using MongoDB's $vectorSearch operator
    
    Args:
        query (str): The search query
        collection: MongoDB collection
        embedding_model: SentenceTransformer model
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: List of matching documents with scores
    """
    # Generate embedding for query
    query_embedding = get_embeddings(query, embedding_model)

    # Define the search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "hcl_vector_index",  # Your vector index name
                "queryVector": query_embedding,
                "path": "embedding",
                "limit": top_k,
                "numCandidates": top_k * 10  # Optional: increases accuracy
            }
        },
        {
            "$project": {
                "filename": 1,
                "chunk_id": 1,
                "chunk_text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        st.error(f"Vector search failed: {str(e)}")
        return []

def generate_response(context: List[Dict], query: str, gemini_client) -> str:
    # Extract text from context
    context_text = "\n".join([r["chunk_text"] for r in context])
    
    # Create prompt for Gemini
    prompt = f"""Based on the following context, provide a clear and accurate answer to the question.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please state that clearly.

Answer:"""
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Failed to generate response"

def main():
    st.title("HCL Tech Semantic Search")
    st.write("Ask questions about HCL Tech Healthineers documents using semantic search")

    # Initialize components
    collection = init_mongodb()
    embedding_model = init_embedding_model()
    gemini_client = init_gemini_client()

    # Check initialization results
    if collection is None or gemini_client is None:
        st.error("Failed to initialize required components")
        return

    # Query input
    query = st.text_input("Enter your question:")

    if st.button("Search"):
        if query:
            with st.spinner("Searching documents..."):
                results = semantic_search(query, collection, embedding_model, top_k=5)

                if results:
                    with st.spinner("Generating answer..."):
                        response = generate_response(results, query, gemini_client)

                        # Display response
                        st.write("### Answer")
                        st.write(response)

                        # Display sources
                        st.write("### Sources")
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"Source {idx}"):
                                st.write(f"**Text:** {result['chunk_text']}")
                                st.write(f"**Relevance Score:** {result['score']:.2f}")
                else:
                    st.warning("No relevant documents found")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()
