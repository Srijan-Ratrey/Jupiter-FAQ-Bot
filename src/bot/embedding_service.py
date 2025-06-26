"""
Embedding Service for Jupiter FAQ Bot
This module handles vector embeddings for semantic search.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
import logging
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Sentence transformer model to use
            embedding_dim: Dimension of embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load the sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize FAISS index for fast similarity search
        self.index = None
        self.faq_data = None
        self.embeddings = None
        
    def create_embeddings(self, faq_df: pd.DataFrame, 
                         text_column: str = "question", 
                         combine_qa: bool = True) -> np.ndarray:
        """
        Create embeddings for FAQ data.
        
        Args:
            faq_df: DataFrame containing FAQ data
            text_column: Column to create embeddings for
            combine_qa: Whether to combine question and answer for embedding
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Creating embeddings for {len(faq_df)} FAQs...")
        
        # Prepare text for embedding
        if combine_qa:
            # Combine question and answer for richer embeddings
            texts = []
            for _, row in faq_df.iterrows():
                combined_text = f"Q: {row['question']} A: {row['answer']}"
                texts.append(combined_text)
        else:
            texts = faq_df[text_column].tolist()
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for fast similarity search.
        
        Args:
            embeddings: Array of embeddings to index
        """
        logger.info("Building FAISS index...")
        
        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # Add embeddings to index
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search_similar(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar FAQs using semantic similarity.
        
        Args:
            query: User query
            k: Number of similar FAQs to return
            
        Returns:
            List of (index, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results (convert distances to similarity scores)
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + dist)
            results.append((int(idx), float(similarity)))
        
        return results
    
    def get_relevant_faqs(self, query: str, k: int = 5, 
                         similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Get relevant FAQs for a query with metadata.
        
        Args:
            query: User query
            k: Number of FAQs to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant FAQ dictionaries
        """
        if self.faq_data is None:
            raise ValueError("FAQ data not loaded. Call process_faq_data() first.")
        
        # Search for similar FAQs
        similar_faqs = self.search_similar(query, k)
        
        # Filter by similarity threshold and prepare results
        relevant_faqs = []
        for idx, similarity in similar_faqs:
            if similarity >= similarity_threshold:
                faq_row = self.faq_data.iloc[idx]
                relevant_faq = {
                    'question': faq_row['question'],
                    'answer': faq_row['answer'],
                    'category': faq_row['category'],
                    'source': faq_row['source'],
                    'similarity_score': similarity,
                    'keywords': faq_row.get('keywords', [])
                }
                relevant_faqs.append(relevant_faq)
        
        logger.info(f"Found {len(relevant_faqs)} relevant FAQs for query: '{query[:50]}...'")
        return relevant_faqs
    
    def process_faq_data(self, faq_file: str = "data/processed/jupiter_faqs_processed.csv"):
        """
        Load and process FAQ data, create embeddings and index.
        
        Args:
            faq_file: Path to processed FAQ file
        """
        logger.info(f"Loading FAQ data from {faq_file}")
        
        # Load FAQ data
        self.faq_data = pd.read_csv(faq_file)
        
        # Create embeddings
        self.embeddings = self.create_embeddings(self.faq_data)
        
        # Build search index
        self.build_index(self.embeddings)
        
        logger.info("FAQ data processing completed successfully")
    
    def save_embeddings(self, save_dir: str = "data/embeddings"):
        """
        Save embeddings and index to disk.
        
        Args:
            save_dir: Directory to save embeddings
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings
        embedding_file = os.path.join(save_dir, "faq_embeddings.npy")
        np.save(embedding_file, self.embeddings)
        
        # Save FAISS index
        index_file = os.path.join(save_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_faqs': len(self.faq_data),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = os.path.join(save_dir, "embedding_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Embeddings saved to {save_dir}")
    
    def load_embeddings(self, save_dir: str = "data/embeddings"):
        """
        Load embeddings and index from disk.
        
        Args:
            save_dir: Directory containing saved embeddings
        """
        # Load embeddings
        embedding_file = os.path.join(save_dir, "faq_embeddings.npy")
        self.embeddings = np.load(embedding_file)
        
        # Load FAISS index
        index_file = os.path.join(save_dir, "faiss_index.bin")
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        metadata_file = os.path.join(save_dir, "embedding_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Embeddings loaded from {save_dir}")
        logger.info(f"Model: {metadata['model_name']}, FAQs: {metadata['num_faqs']}")
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about the embedding system."""
        if self.embeddings is None:
            return {"status": "No embeddings loaded"}
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_faqs": len(self.faq_data),
            "embedding_shape": self.embeddings.shape,
            "index_size": self.index.ntotal if self.index else 0
        }


# Test function
def test_embedding_service():
    """Test the embedding service with sample data."""
    embedding_service = EmbeddingService()
    
    # Process FAQ data
    embedding_service.process_faq_data()
    
    # Save embeddings
    embedding_service.save_embeddings()
    
    # Test search
    test_queries = [
        "How to open account?",
        "What is Jupiter Pro?",
        "Credit card application",
        "KYC verification process",
        "Money transfer options"
    ]
    
    print("\n=== EMBEDDING SERVICE TEST ===")
    for query in test_queries:
        relevant_faqs = embedding_service.get_relevant_faqs(query, k=3)
        print(f"\nQuery: '{query}'")
        for i, faq in enumerate(relevant_faqs, 1):
            print(f"  {i}. [{faq['similarity_score']:.3f}] {faq['question']}")
            print(f"     Category: {faq['category']}")
    
    # Print stats
    stats = embedding_service.get_embedding_stats()
    print(f"\n=== EMBEDDING STATS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_embedding_service() 