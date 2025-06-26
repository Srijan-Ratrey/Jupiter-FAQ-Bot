"""
Jupiter FAQ Bot - Main Interface
This module provides the main bot interface for the Jupiter FAQ system.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.rag_pipeline import RAGPipeline, create_rag_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JupiterFAQBot:
    """
    Jupiter FAQ Bot - Main interface for the conversational FAQ system.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 config_file: Optional[str] = None):
        """
        Initialize the Jupiter FAQ Bot.
        
        Args:
            openai_api_key: OpenAI API key for enhanced responses
            config_file: Path to configuration file
        """
        self.bot_name = "Jupiter Assistant"
        self.version = "1.0.0"
        self.startup_time = datetime.now()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize RAG pipeline
        logger.info("Initializing Jupiter FAQ Bot...")
        
        # Get API keys from environment or config
        openai_key = openai_api_key or self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        openrouter_key = self.config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
        
        # Create RAG pipeline with proper provider configuration
        self.rag_pipeline = RAGPipeline(
            embedding_model=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            llm_provider=self.config.get("primary_provider", "openrouter"),
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            similarity_threshold=self.config.get("similarity_threshold", 0.3),
            max_retrieved_faqs=self.config.get("max_sources", 5)
        )
        self.rag_pipeline.setup()
        
        # Bot personality and settings
        self.personality = {
            "greeting": "Hello! I'm your Jupiter banking assistant. I'm here to help answer your questions about Jupiter's services, features, and policies. What would you like to know?",
            "fallback": "I'm here to help with Jupiter banking questions. Could you please rephrase your question or ask about something specific like account opening, KYC, features, or support?",
            "goodbye": "Thank you for using Jupiter! If you need more help, feel free to ask anytime or contact our support team at +91 8655055086."
        }
        
        logger.info("Jupiter FAQ Bot initialized successfully!")
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file."""
        default_config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.3,
            "response_style": "conversational",
            "max_sources": 3,
            "include_sources": True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        return default_config
    
    def ask(self, question: str, style: Optional[str] = None, 
            include_sources: Optional[bool] = None) -> Dict:
        """
        Ask the bot a question and get a response.
        
        Args:
            question: User's question
            style: Response style (conversational, formal, brief)
            include_sources: Whether to include source FAQs
            
        Returns:
            Bot response dictionary
        """
        if not question or not question.strip():
            return {
                "response": "Please ask me a question about Jupiter banking services!",
                "confidence": 0.0,
                "query": question
            }
        
        # Use config defaults if not specified
        style = style or self.config.get("response_style", "conversational")
        include_sources = include_sources if include_sources is not None else self.config.get("include_sources", True)
        
        # Process query through RAG pipeline
        response = self.rag_pipeline.query(
            user_query=question.strip(),
            style=style,
            include_sources=include_sources
        )
        
        # Add bot personality touches
        if response["confidence"] < 0.2:
            response["response"] = f"{response['response']}\n\n{self.personality['fallback']}"
        
        return response
    
    def chat(self):
        """
        Start an interactive chat session.
        """
        print(f"\n🚀 {self.bot_name} v{self.version}")
        print("=" * 50)
        print(self.personality["greeting"])
        print("\nCommands:")
        print("• Type 'quit', 'exit', or 'bye' to end the chat")
        print("• Type 'help' for available commands")
        print("• Type 'stats' to see bot statistics")
        print("• Type 'categories' to see FAQ categories")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print(f"\n🤖 {self.bot_name}: {self.personality['goodbye']}")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                elif user_input.lower() == 'categories':
                    self._show_categories()
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.rag_pipeline.clear_history()
                    print("📝 Conversation history cleared!")
                    continue
                
                # Process regular question
                response = self.ask(user_input)
                
                print(f"\n🤖 {self.bot_name}: {response['response']}")
                
                # Show confidence and sources if available
                if response['confidence'] > 0:
                    print(f"\n📊 Confidence: {response['confidence']:.1%}")
                
                if response.get('sources'):
                    print(f"📚 Sources: {len(response['sources'])} relevant FAQs")
                    for i, source in enumerate(response['sources'][:2], 1):
                        print(f"   {i}. {source['question']} (similarity: {source['similarity']:.2f})")
                
                # Show query suggestions for better user experience
                if response['confidence'] > 0.3:  # Only show suggestions for decent matches
                    suggestions = self.get_query_suggestions(user_input, num_suggestions=3)
                    if suggestions:
                        print(f"\n💡 Related questions you might ask:")
                        for i, suggestion in enumerate(suggestions, 1):
                            print(f"   {i}. {suggestion}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 {self.bot_name}: {self.personality['goodbye']}")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🔧 Available Commands:
• help - Show this help message
• stats - Show bot performance statistics
• categories - List available FAQ categories
• history - Show recent conversation history
• clear - Clear conversation history
• quit/exit/bye - End the chat session

💡 Tips:
• Ask specific questions about Jupiter services
• Use natural language - no need for exact keywords
• Questions about account opening, KYC, features, etc. work well
• The bot learns from our comprehensive FAQ database
        """
        print(help_text)
    
    def _show_stats(self):
        """Show bot statistics."""
        metrics = self.rag_pipeline.get_metrics()
        uptime = datetime.now() - self.startup_time
        
        print(f"""
📈 Bot Statistics:
• Uptime: {str(uptime).split('.')[0]}
• Total queries: {metrics['total_queries']}
• Success rate: {metrics['success_rate']:.1%}
• Average confidence: {metrics['avg_confidence']:.1%}
• Conversation history: {metrics['conversation_history_length']} items

🔧 Provider usage:""")
        
        for provider, count in metrics['provider_usage'].items():
            print(f"• {provider}: {count} queries")
    
    def _show_categories(self):
        """Show available FAQ categories."""
        categories = self.rag_pipeline.get_faq_categories()
        
        print(f"\n📂 Available FAQ Categories ({len(categories)}):")
        for i, category in enumerate(categories, 1):
            print(f"   {i}. {category}")
        
        print("\n💡 You can ask questions about any of these topics!")
    
    def _show_history(self):
        """Show recent conversation history."""
        history = self.rag_pipeline.get_conversation_history(last_n=5)
        
        if not history:
            print("📝 No conversation history available.")
            return
        
        print(f"\n📝 Recent Conversations ({len(history)}):")
        for i, item in enumerate(history, 1):
            timestamp = item['timestamp'].split('T')[1].split('.')[0]  # Extract time
            print(f"\n{i}. [{timestamp}] Q: {item['query']}")
            print(f"   A: {item['response'][:100]}{'...' if len(item['response']) > 100 else ''}")
            print(f"   Confidence: {item['confidence']:.1%}")
    
    def batch_ask(self, questions: List[str], **kwargs) -> List[Dict]:
        """
        Ask multiple questions in batch.
        
        Args:
            questions: List of questions
            **kwargs: Additional arguments for ask method
            
        Returns:
            List of response dictionaries
        """
        return [self.ask(q, **kwargs) for q in questions]
    
    def get_faq_by_category(self, category: str, limit: int = 5) -> List[Dict]:
        """
        Get FAQs from a specific category.
        
        Args:
            category: FAQ category
            limit: Maximum number of FAQs
            
        Returns:
            List of FAQ dictionaries
        """
        return self.rag_pipeline.search_by_category(category, limit)
    
    def get_query_suggestions(self, user_query: str, num_suggestions: int = 3) -> List[str]:
        """
        Get related query suggestions based on semantic similarity.
        
        Args:
            user_query: User's current query
            num_suggestions: Number of suggestions to return
            
        Returns:
            List of suggested questions
        """
        # Get similar FAQs but with a lower threshold to find more diverse suggestions
        similar_faqs = self.rag_pipeline.embedding_service.get_relevant_faqs(
            query=user_query,
            k=num_suggestions * 2,  # Get more to filter from
            similarity_threshold=0.1  # Lower threshold for broader suggestions
        )
        
        suggestions = []
        seen_categories = set()
        
        for faq in similar_faqs:
            # Try to get diverse suggestions from different categories
            if len(suggestions) >= num_suggestions:
                break
                
            category = faq.get('category', '')
            question = faq.get('question', '')
            
            # Prefer questions from different categories for diversity
            if category not in seen_categories or len(suggestions) < num_suggestions // 2:
                suggestions.append(question)
                seen_categories.add(category)
        
                return suggestions[:num_suggestions]
    
    def get_statistics(self) -> Dict:
        """
        Get bot performance statistics.
        
        Returns:
            Dictionary with bot statistics
        """
        # Get conversation history stats
        conversation_history = getattr(self, 'conversation_history', [])
        total_queries = len(conversation_history)
        if total_queries > 0:
            confidences = [conv['confidence'] for conv in conversation_history if 'confidence' in conv]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        else:
            avg_confidence = 0
        
        # Get FAQ database stats
        faq_data = getattr(self.rag_pipeline.embedding_service, 'faq_data', [])
        categories = set()
        for faq in faq_data:
            if 'category' in faq:
                categories.add(faq['category'])
        
        return {
            "total_queries": total_queries,
            "average_confidence": avg_confidence,
            "total_faqs": len(faq_data),
            "categories_count": len(categories),
            "categories": sorted(list(categories)),
            "bot_name": self.bot_name,
            "embedding_model": self.rag_pipeline.embedding_service.model_name,
            "primary_llm_provider": self.rag_pipeline.llm_service.primary_provider
        }

    def export_conversation(self, filename: Optional[str] = None) -> str:
        """
        Export conversation history.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        return self.rag_pipeline.export_conversation(filename)
    
    def get_bot_info(self) -> Dict:
        """Get bot information and status."""
        metrics = self.rag_pipeline.get_metrics()
        
        return {
            "name": self.bot_name,
            "version": self.version,
            "startup_time": self.startup_time.isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "total_faqs": len(self.rag_pipeline.embedding_service.faq_data) if self.rag_pipeline.embedding_service.faq_data is not None else 0,
            "categories": len(self.rag_pipeline.get_faq_categories()),
            "metrics": metrics,
            "config": self.config
        }

# Factory function for easy bot creation
def create_jupiter_bot(openai_api_key: Optional[str] = None, 
                      config_file: Optional[str] = None) -> JupiterFAQBot:
    """
    Create a Jupiter FAQ Bot instance.
    
    Args:
        openai_api_key: OpenAI API key
        config_file: Configuration file path
        
    Returns:
        Jupiter FAQ Bot instance
    """
    return JupiterFAQBot(openai_api_key=openai_api_key, config_file=config_file)

# Demo function
def demo_bot():
    """Run a demo of the bot with sample questions."""
    print("🚀 Jupiter FAQ Bot Demo")
    print("=" * 30)
    
    # Create bot
    bot = create_jupiter_bot()
    
    # Demo questions
    demo_questions = [
        "How can I open a Jupiter account?",
        "What is Jupiter Pro and how do I get it?",
        "How do I complete my KYC verification?",
        "What are the customer support options?",
        "How can I apply for a credit card?"
    ]
    
    print(f"\nTesting {len(demo_questions)} sample questions...\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"🔹 Question {i}: {question}")
        response = bot.ask(question, style="conversational")
        print(f"🤖 Response: {response['response'][:150]}...")
        print(f"📊 Confidence: {response['confidence']:.1%}")
        print("-" * 50)
    
    # Show final stats
    info = bot.get_bot_info()
    print(f"\n📈 Demo completed!")
    print(f"Total queries processed: {info['metrics']['total_queries']}")
    print(f"Success rate: {info['metrics']['success_rate']:.1%}")

if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_bot()
        elif sys.argv[1] == "chat":
            bot = create_jupiter_bot()
            bot.chat()
    else:
        # Default: run demo
        demo_bot() 