"""
LLM Service for Jupiter FAQ Bot
This module handles integration with different LLM providers.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
import openai
from openai import OpenAI
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided")
    
    def generate_response(self, prompt: str, max_tokens: int = 500, 
                         temperature: float = 0.7, **kwargs) -> str:
        """Generate response using OpenAI GPT."""
        if not self.is_available():
            raise ValueError("OpenAI client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful Jupiter banking assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self.client is not None

class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider with Mistral models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"): #deepseek/deepseek-r1-0528-qwen3-8b:free
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None
            logger.warning("OpenRouter API key not provided")
    
    def generate_response(self, prompt: str, max_tokens: int = 500, 
                         temperature: float = 0.7, **kwargs) -> str:
        """Generate response using OpenRouter/Mistral."""
        if not self.is_available():
            raise ValueError("OpenRouter client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful Jupiter banking assistant. Provide clear, accurate, and friendly responses about Jupiter's banking services."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://jupiter.money",
                    "X-Title": "Jupiter FAQ Bot"
                }
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available."""
        return self.client is not None

class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider (e.g., Ollama, transformers)."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 500, 
                         temperature: float = 0.7, **kwargs) -> str:
        """Generate response using local LLM."""
        try:
            # Try Ollama API format
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                raise Exception(f"Local LLM API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Local LLM not available: {e}")
            return False

class FallbackProvider(BaseLLMProvider):
    """Fallback provider with template-based responses."""
    
    def __init__(self):
        self.templates = {
            "no_match": "I apologize, but I couldn't find a specific answer to your question in our FAQ database. Please contact Jupiter customer support at +91 8655055086 or email support@jupiter.money for personalized assistance.",
            "general_help": "For Jupiter banking assistance, you can:\n• Call customer care: +91 8655055086 (9 AM-7 PM weekdays)\n• Email: support@jupiter.money\n• Use the Jupiter app chat feature\n• Visit our community forums: community.jupiter.money"
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response using templates."""
        if context:
            return f"Based on our FAQ information:\n\n{context}\n\nFor more specific assistance, please contact Jupiter support."
        else:
            return self.templates["no_match"]
    
    def is_available(self) -> bool:
        """Fallback is always available."""
        return True

class LLMService:
    """Main LLM service that manages multiple providers."""
    
    def __init__(self, primary_provider: str = "openrouter", 
                 openai_api_key: Optional[str] = None,
                 openrouter_api_key: Optional[str] = None,
                 local_llm_url: str = "http://localhost:11434"):
        """
        Initialize LLM service with provider priority.
        
        Args:
            primary_provider: Primary LLM provider to use
            openai_api_key: OpenAI API key
            openrouter_api_key: OpenRouter API key
            local_llm_url: URL for local LLM service
        """
        self.providers = {}
        
        # Initialize providers
        self.providers["openai"] = OpenAIProvider(api_key=openai_api_key)
        self.providers["openrouter"] = OpenRouterProvider(api_key=openrouter_api_key)
        self.providers["local"] = LocalLLMProvider(base_url=local_llm_url)
        self.providers["fallback"] = FallbackProvider()
        
        self.primary_provider = primary_provider
        
        # Check provider availability
        self.check_providers()
    
    def check_providers(self):
        """Check availability of all providers."""
        logger.info("Checking LLM provider availability...")
        
        for name, provider in self.providers.items():
            available = provider.is_available()
            logger.info(f"{name.title()} provider: {'✓ Available' if available else '✗ Not available'}")
    
    def get_available_provider(self) -> BaseLLMProvider:
        """Get the first available provider."""
        # Try primary provider first
        if self.primary_provider in self.providers:
            provider = self.providers[self.primary_provider]
            try:
                if provider.is_available():
                    return provider
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_provider} failed availability check: {e}")
        
        # Try other providers in priority order (skip local if it's problematic)
        priority_order = ["openrouter", "openai", "fallback"]
        for name in priority_order:
            if name != self.primary_provider and name in self.providers:
                provider = self.providers[name]
                try:
                    if provider.is_available():
                        logger.info(f"Using {name} provider as fallback")
                        return provider
                except Exception as e:
                    logger.warning(f"Provider {name} failed availability check: {e}")
                    continue
        
        # Return fallback if nothing else works
        logger.info("All providers failed, using fallback provider")
        return self.providers["fallback"]
    
    def generate_faq_response(self, user_query: str, relevant_faqs: List[Dict], 
                             style: str = "conversational") -> Dict:
        """
        Generate a natural response based on relevant FAQs.
        
        Args:
            user_query: User's question
            relevant_faqs: List of relevant FAQ dictionaries
            style: Response style (conversational, formal, brief)
            
        Returns:
            Dictionary with response and metadata
        """
        provider = self.get_available_provider()
        
        if not relevant_faqs:
            # No relevant FAQs found
            response_text = provider.generate_response(
                f"User asked: '{user_query}'. No specific FAQ found.",
                context=None
            )
            return {
                "response": response_text,
                "provider": type(provider).__name__,
                "confidence": 0.0,
                "sources": [],
                "query": user_query
            }
        
        # Build context from relevant FAQs
        context_parts = []
        sources = []
        
        for i, faq in enumerate(relevant_faqs[:3], 1):  # Use top 3 FAQs
            context_parts.append(f"FAQ {i}:")
            context_parts.append(f"Q: {faq['question']}")
            context_parts.append(f"A: {faq['answer']}")
            context_parts.append(f"Category: {faq['category']}\n")
            
            sources.append({
                "question": faq['question'],
                "category": faq['category'],
                "similarity": faq['similarity_score']
            })
        
        context = "\n".join(context_parts)
        
        # Create prompt based on style
        if style == "conversational":
            prompt = f"""You are a friendly Jupiter banking assistant. A user asked: "{user_query}"

Here are the most relevant FAQs from our knowledge base:

{context}

Please provide a helpful, conversational response that:
1. Directly answers the user's question using the FAQ information
2. Is friendly and easy to understand
3. Mentions specific Jupiter features when relevant
4. Keeps the response concise but complete
5. If the FAQs don't fully answer the question, mention contacting support

Response:"""
        elif style == "brief":
            prompt = f"""User question: "{user_query}"

Relevant FAQ information:
{context}

Provide a brief, direct answer based on this information."""
        else:  # formal
            prompt = f"""Based on the following FAQ information, provide a professional response to: "{user_query}"

{context}

Please provide a professional, accurate response based on this information."""
        
        try:
            response_text = provider.generate_response(prompt)
            confidence = sum(faq['similarity_score'] for faq in relevant_faqs) / len(relevant_faqs)
            
            return {
                "response": response_text,
                "provider": type(provider).__name__,
                "confidence": confidence,
                "sources": sources,
                "query": user_query,
                "style": style
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to template response
            fallback_response = self.providers["fallback"].generate_response(
                user_query, context=relevant_faqs[0]['answer'] if relevant_faqs else None
            )
            
            return {
                "response": fallback_response,
                "provider": "FallbackProvider",
                "confidence": 0.5,
                "sources": sources,
                "query": user_query,
                "error": str(e)
            }

# Configuration and environment setup
def setup_llm_service(config: Optional[Dict] = None) -> LLMService:
    """
    Setup LLM service with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LLM service
    """
    if config is None:
        config = {}
    
    # Load from environment variables
    openai_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    openrouter_key = config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
    primary_provider = config.get("primary_provider", "openrouter")
    local_llm_url = config.get("local_llm_url", "http://localhost:11434")
    
    return LLMService(
        primary_provider=primary_provider,
        openai_api_key=openai_key,
        openrouter_api_key=openrouter_key,
        local_llm_url=local_llm_url
    )

# Test function
def test_llm_service():
    """Test the LLM service."""
    print("=== LLM SERVICE TEST ===")
    
    # Test with OpenRouter if available
    llm_service = setup_llm_service()
    
    # Mock relevant FAQs
    mock_faqs = [
        {
            "question": "How can I open a Savings account?",
            "answer": "To open a free Savings or Salary Bank Account on Jupiter - powered by Federal Bank - in 3 minutes, simply install the Jupiter App.",
            "category": "Account Opening",
            "similarity_score": 0.85
        }
    ]
    
    # Test response generation
    response = llm_service.generate_faq_response(
        user_query="How do I create a new account?",
        relevant_faqs=mock_faqs,
        style="conversational"
    )
    
    print(f"Query: {response['query']}")
    print(f"Provider: {response['provider']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Response: {response['response']}")

if __name__ == "__main__":
    test_llm_service() 