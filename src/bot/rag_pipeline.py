"""
RAG (Retrieval-Augmented Generation) Pipeline for Jupiter FAQ Bot
This module combines semantic search with LLM response generation.
"""

import os
import sys
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.embedding_service import EmbeddingService
from bot.llm_service import LLMService, setup_llm_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for FAQ bot.
    Combines semantic search with LLM response generation.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_provider: str = "openai",
                 openai_api_key: Optional[str] = None,
                 openrouter_api_key: Optional[str] = None,
                 similarity_threshold: float = 0.3,
                 max_retrieved_faqs: int = 5):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            llm_provider: Primary LLM provider
            openai_api_key: OpenAI API key
            similarity_threshold: Minimum similarity for FAQ retrieval
            max_retrieved_faqs: Maximum number of FAQs to retrieve
        """
        self.similarity_threshold = similarity_threshold
        self.max_retrieved_faqs = max_retrieved_faqs
        
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        
        # Initialize LLM service
        logger.info("Initializing LLM service...")
        self.llm_service = setup_llm_service({
            "primary_provider": llm_provider,
            "openai_api_key": openai_api_key,
            "openrouter_api_key": openrouter_api_key
        })
        
        # Conversation history
        self.conversation_history = []
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "successful_responses": 0,
            "avg_confidence": 0.0,
            "provider_usage": {}
        }
    
    def setup(self, faq_file: str = "data/processed/jupiter_faqs_processed.csv",
              force_rebuild: bool = False):
        """
        Setup the pipeline by loading FAQ data and embeddings.
        
        Args:
            faq_file: Path to processed FAQ file
            force_rebuild: Whether to rebuild embeddings even if they exist
        """
        logger.info("Setting up RAG pipeline...")
        
        embedding_dir = "data/embeddings"
        embeddings_exist = (
            os.path.exists(os.path.join(embedding_dir, "faq_embeddings.npy")) and
            os.path.exists(os.path.join(embedding_dir, "faiss_index.bin"))
        )
        
        if embeddings_exist and not force_rebuild:
            logger.info("Loading existing embeddings...")
            self.embedding_service.faq_data = pd.read_csv(faq_file)
            self.embedding_service.load_embeddings(embedding_dir)
        else:
            logger.info("Creating new embeddings...")
            self.embedding_service.process_faq_data(faq_file)
            self.embedding_service.save_embeddings(embedding_dir)
        
        logger.info("RAG pipeline setup complete!")
    
    def query(self, user_query: str, 
              style: str = "conversational",
              include_sources: bool = True,
              context_length: int = 3) -> Dict:
        """
        Process a user query and generate a response.
        
        Args:
            user_query: User's question
            style: Response style (conversational, formal, brief)
            include_sources: Whether to include source FAQs in response
            context_length: Number of relevant FAQs to use for context
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant FAQs using semantic search
            logger.info(f"Processing query: '{user_query[:50]}...'")
            
            relevant_faqs = self.embedding_service.get_relevant_faqs(
                query=user_query,
                k=self.max_retrieved_faqs,
                similarity_threshold=self.similarity_threshold
            )
            
            # Step 2: Generate response using LLM
            llm_response = self.llm_service.generate_faq_response(
                user_query=user_query,
                relevant_faqs=relevant_faqs[:context_length],
                style=style
            )
            
            # Step 3: Prepare final response
            response = {
                "query": user_query,
                "response": llm_response["response"],
                "confidence": llm_response["confidence"],
                "provider": llm_response["provider"],
                "style": style,
                "retrieved_faqs_count": len(relevant_faqs),
                "timestamp": datetime.now().isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Add source information if requested
            if include_sources and relevant_faqs:
                response["sources"] = [
                    {
                        "question": faq["question"],
                        "category": faq["category"],
                        "similarity": round(faq["similarity_score"], 3),
                        "source": faq["source"]
                    }
                    for faq in relevant_faqs[:3]  # Top 3 sources
                ]
            
            # Update metrics
            self.update_metrics(response)
            
            # Add to conversation history
            self.conversation_history.append({
                "query": user_query,
                "response": response["response"],
                "timestamp": response["timestamp"],
                "confidence": response["confidence"]
            })
            
            # Keep only last 10 conversations
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Query processed successfully (confidence: {response['confidence']:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Return error response
            error_response = {
                "query": user_query,
                "response": "I apologize, but I encountered an error while processing your query. Please try again or contact Jupiter support at +91 8655055086.",
                "confidence": 0.0,
                "provider": "Error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            return error_response
    
    def batch_query(self, queries: List[str], **kwargs) -> List[Dict]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            **kwargs: Additional arguments for query method
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries...")
        
        responses = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            response = self.query(query, **kwargs)
            responses.append(response)
        
        return responses
    
    def update_metrics(self, response: Dict):
        """Update performance metrics."""
        self.metrics["total_queries"] += 1
        
        if "error" not in response:
            self.metrics["successful_responses"] += 1
        
        # Update average confidence
        if response["confidence"] > 0:
            current_avg = self.metrics["avg_confidence"]
            n = self.metrics["successful_responses"]
            self.metrics["avg_confidence"] = (current_avg * (n - 1) + response["confidence"]) / n
        
        # Update provider usage
        provider = response.get("provider", "Unknown")
        self.metrics["provider_usage"][provider] = self.metrics["provider_usage"].get(provider, 0) + 1
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        success_rate = 0.0
        if self.metrics["total_queries"] > 0:
            success_rate = self.metrics["successful_responses"] / self.metrics["total_queries"]
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "conversation_history_length": len(self.conversation_history)
        }
    
    def get_conversation_history(self, last_n: int = 5) -> List[Dict]:
        """Get recent conversation history."""
        return self.conversation_history[-last_n:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_faq_categories(self) -> List[str]:
        """Get list of available FAQ categories."""
        if self.embedding_service.faq_data is not None:
            return sorted(self.embedding_service.faq_data['category'].unique().tolist())
        return []
    
    def search_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """
        Get FAQs by category.
        
        Args:
            category: FAQ category
            limit: Maximum number of FAQs to return
            
        Returns:
            List of FAQ dictionaries
        """
        if self.embedding_service.faq_data is None:
            return []
        
        category_faqs = self.embedding_service.faq_data[
            self.embedding_service.faq_data['category'] == category
        ].head(limit)
        
        return [
            {
                "question": row['question'],
                "answer": row['answer'],
                "category": row['category'],
                "source": row['source']
            }
            for _, row in category_faqs.iterrows()
        ]
    
    def export_conversation(self, filename: str = None) -> str:
        """
        Export conversation history to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_history_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_conversations": len(self.conversation_history),
            "metrics": self.get_metrics(),
            "conversations": self.conversation_history
        }
        
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Conversation history exported to {filepath}")
        return filepath

# Convenience function for quick setup
def create_rag_pipeline(openai_api_key: Optional[str] = None,
                       embedding_model: str = "all-MiniLM-L6-v2") -> RAGPipeline:
    """
    Create and setup a RAG pipeline with default configuration.
    
    Args:
        openai_api_key: OpenAI API key (optional)
        embedding_model: Embedding model to use
        
    Returns:
        Configured RAG pipeline
    """
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        llm_provider="openai" if openai_api_key else "fallback",
        openai_api_key=openai_api_key
    )
    
    pipeline.setup()
    return pipeline

# Test function
def test_rag_pipeline():
    """Test the RAG pipeline."""
    print("=== RAG PIPELINE TEST ===")
    
    # Create pipeline
    pipeline = create_rag_pipeline()
    
    # Test queries
    test_queries = [
        "How do I open a Jupiter account?",
        "What is Jupiter Pro?",
        "How to apply for credit card?",
        "KYC verification process",
        "Customer support contact"
    ]
    
    print(f"\nTesting {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        response = pipeline.query(query, style="conversational")
        
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Provider: {response['provider']}")
        print(f"Processing time: {response['processing_time']:.2f}s")
        print(f"Response: {response['response'][:200]}...")
        
        if "sources" in response:
            print(f"Sources: {len(response['sources'])} FAQs")
    
    # Print metrics
    metrics = pipeline.get_metrics()
    print(f"\n=== PIPELINE METRICS ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test category search
    categories = pipeline.get_faq_categories()
    print(f"\nAvailable categories: {categories}")
    
    if categories:
        category_faqs = pipeline.search_by_category(categories[0], limit=3)
        print(f"\nSample FAQs from '{categories[0]}' category:")
        for faq in category_faqs:
            print(f"- {faq['question']}")

if __name__ == "__main__":
    test_rag_pipeline() 