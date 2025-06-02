import os
import faiss
import numpy as np
import pickle
import openai
import config
from datetime import datetime
from langsmith import traceable

class VectorDatabase:
    def __init__(self):
        self.dimension = 1536
        self.index = None
        self.documents = []
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
        elif not os.getenv("OPENAI_API_KEY"): 
             print("Warning: OPENAI_API_KEY not found in config.py or environment variables for direct OpenAI client usage.")
        self.load_or_create_index()

    @traceable(name="vectordb_load_or_create_index", run_type="tool")
    def load_or_create_index(self):
        if os.path.exists(f"{config.VECTOR_DB_PATH}/faiss_index.bin") and \
           os.path.exists(f"{config.VECTOR_DB_PATH}/documents.pkl"):
            try:
                self.index = faiss.read_index(f"{config.VECTOR_DB_PATH}/faiss_index.bin")
                with open(f"{config.VECTOR_DB_PATH}/documents.pkl", "rb") as f:
                    self.documents = pickle.load(f)
                if self.index.ntotal > 0 and self.index.d != self.dimension:
                    print(f"Warning: Loaded index dimension {self.index.d} differs from expected {self.dimension}. Re-initializing.")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.documents = []
            except Exception as e:
                print(f"Error loading index or documents: {e}. Starting fresh.")
                os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
        else:
            os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []

    @traceable(name="vectordb_get_embedding", run_type="embedding")
    def get_embedding(self, text: str):
        if not openai.api_key and config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
        
        if not openai.api_key and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not configured for direct openai client (embeddings). Set OPENAI_API_KEY environment variable or in config.py.")
        
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array([resp.data[0].embedding], dtype=np.float32)

    @traceable(name="vectordb_add_document", run_type="tool") 
    def add_document(self, document: str, metadata: dict = None):
        embedding = self.get_embedding(document)
        if self.index is None or self.index.d != embedding.shape[1]:
            print(f"Re-initializing index for dimension {embedding.shape[1]}")
            self.dimension = embedding.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embedding)
        self.documents.append({"text": document, "metadata": metadata or {}})
        self.save_index()
        return {"total_docs": len(self.documents)}

    @traceable(name="vectordb_search", run_type="retriever") 
    def search(self, query: str, k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return {"hits": []}
        q_emb = self.get_embedding(query)
        if q_emb.shape[1] != self.index.d:
            print(f"Query embedding dimension {q_emb.shape[1]} does not match index dimension {self.index.d}")
            return {"hits": []}
        distances, indices = self.index.search(q_emb, k)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                hits.append({
                    "distance": float(dist),
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                })
        return {"hits": hits}

    @traceable(name="vectordb_save_index", run_type="tool") 
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, f"{config.VECTOR_DB_PATH}/faiss_index.bin")
            with open(f"{config.VECTOR_DB_PATH}/documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)