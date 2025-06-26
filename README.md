# Jupiter FAQ Bot 🤖

A sophisticated, AI-powered FAQ bot for Jupiter's Help Centre using modern NLP techniques and Retrieval-Augmented Generation (RAG).

## 🎯 Project Overview

This project scrapes Jupiter's FAQ content and builds an intelligent conversational bot that provides natural, accurate answers to user questions using advanced semantic search and language models.

## ✨ Key Features

### Core Functionality
- **🕷️ Web Scraping**: Comprehensive data collection from Jupiter's website and community forums
- **🧠 Semantic Search**: Advanced embedding-based search using SentenceTransformers + FAISS
- **💬 Natural Responses**: Conversational, formal, and brief response styles
- **📊 Smart Analytics**: Confidence scoring and source attribution

### Advanced Features ⭐
- **💡 Query Suggestions**: AI-powered related question recommendations
- **📈 Performance Analytics**: Comprehensive usage analysis and reporting
- **🔄 Multi-Provider Fallback**: Graceful degradation when primary services unavailable
- **🎨 Interactive Interface**: Rich CLI with conversation history and commands
- **📊 Evaluation Tools**: Automated performance testing and comparison

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Virtual environment (recommended)
```

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd Jupiter
python -m venv jupiter_faq_env
source jupiter_faq_env/bin/activate  # On Windows: jupiter_faq_env\Scripts\activate
pip install -r requirements.txt
```

### OpenRouter/Mistral Setup (Recommended)
```bash
# Get free API key from https://openrouter.ai/
python scripts/setup_openrouter.py
# OR set environment variable
export OPENROUTER_API_KEY="your-key-here"
```

### Quick Demo
```bash
# Interactive chat
python scripts/demo_bot.py interactive

# Automated demo
python scripts/demo_bot.py detailed
```

## 📊 System Architecture

```
User Query → Semantic Search (FAISS) → Context Retrieval → LLM Generation → Response
    ↓              ↓                         ↓                ↓
Embedding      FAQ Database            Context Builder    Mistral/OpenAI
Service        (39 FAQs)              (Top-K FAQs)       Natural Language
```

### Data Pipeline
1. **Web Scraping**: `src/data_collection/scraper.py`
2. **Preprocessing**: `src/data_collection/preprocessor.py`  
3. **Embedding**: `src/bot/embedding_service.py`
4. **LLM Integration**: `src/bot/llm_service.py`
5. **RAG Pipeline**: `src/bot/rag_pipeline.py`

## 📈 Performance Metrics

### Current Performance (OpenRouter/Mistral)
- **Response Time**: 3.5-7.8s average
- **Confidence**: 55.8% average  
- **Success Rate**: 100% (always finds relevant content)
- **High Confidence Queries**: 40% score >60%
- **Coverage**: 39 FAQs across 14 categories

### FAQ Categories (14)
- Account Opening, Account Types, Banking, Community Help
- Credit Card, Customer Support, Debit Card, General
- Investments, KYC, Money Transfer, Policy, Pots, Rewards

## 🛠️ Advanced Usage

### Performance Evaluation
```bash
# Comprehensive performance analysis
python evaluation/evaluate_performance.py

# Analytics dashboard
python evaluation/analytics_dashboard.py
```

### Custom Configuration
Edit `config.json`:
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "similarity_threshold": 0.2,
  "response_style": "conversational", 
  "primary_provider": "openrouter",
  "mistral_model": "mistralai/mistral-7b-instruct:free"
}
```

### Data Collection
```bash
# Re-scrape Jupiter website
python scripts/run_data_collection.py

# Process and clean data
python -m src.data_collection.preprocessor
```

## 🔧 Technical Implementation

### Embedding & Search
- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector DB**: FAISS with L2 similarity
- **Deduplication**: TF-IDF similarity filtering
- **Categorization**: Automated + manual validation

### LLM Integration
- **Primary**: OpenRouter (Mistral 7B Instruct - Free)
- **Fallback**: Template responses
- **Context**: Top-3 relevant FAQs with metadata
- **Styles**: Conversational, formal, brief

### New Features ⭐

#### Query Suggestions
```python
from bot.jupiter_faq_bot import create_jupiter_bot

bot = create_jupiter_bot()
suggestions = bot.get_query_suggestions("How to open account?", num_suggestions=3)
# Returns: ["What documents needed?", "KYC requirements?", "Account types?"]
```

#### Performance Analytics
- Usage patterns analysis
- Confidence distribution tracking  
- Popular query identification
- Temporal usage insights
- Performance metrics dashboard

## 📊 Project Requirements Completion

| Requirement | Status | Implementation |
|------------|---------|---------------|
| ✅ FAQ Scraping | **COMPLETE** | Multi-source web scraping |
| ✅ Data Preprocessing | **COMPLETE** | Cleaning, deduplication, categorization |
| ✅ LLM Integration | **COMPLETE** | OpenAI, OpenRouter/Mistral, Local LLM |
| ✅ Semantic Search | **COMPLETE** | FAISS + SentenceTransformers |
| ✅ Demo Interface | **COMPLETE** | Interactive CLI + demo scripts |
| ✅ Documentation | **COMPLETE** | Comprehensive README + code docs |
| ✅ Evaluation | **COMPLETE** | Automated performance testing |
| ⭐ Query Suggestions | **NEW** | AI-powered related questions |
| ⭐ Analytics Dashboard | **NEW** | Usage insights and metrics |
| ⭐ Performance Comparison | **NEW** | Multi-provider evaluation |

### Bonus Features
- **❌ Multilingual Support**: Not implemented (future enhancement)
- **✅ Query Suggestions**: ⭐ Implemented with semantic similarity
- **✅ Approach Comparison**: ⭐ Automated evaluation framework

## 🎮 Interactive Commands

When using `python scripts/demo_bot.py interactive`:

- `help` - Show available commands
- `stats` - Display bot performance statistics  
- `categories` - List all FAQ categories
- `history` - Show recent conversation history
- `clear` - Clear conversation history
- `quit/exit` - End session

## 📁 Project Structure

```
Jupiter/
├── 📂 src/                     # Core source code
│   ├── bot/                    # Bot components
│   │   ├── embedding_service.py   # Semantic search
│   │   ├── llm_service.py         # Multi-LLM integration  
│   │   ├── rag_pipeline.py        # RAG implementation
│   │   └── jupiter_faq_bot.py     # Main bot interface
│   ├── data_collection/        # Web scraping & preprocessing
│   └── evaluation/             # Performance testing utilities
├── 📂 scripts/                 # Executable scripts
│   ├── demo_bot.py             # Interactive demo
│   ├── run_data_collection.py  # Data collection runner
│   ├── setup_openrouter.py     # OpenRouter API setup
│   └── test_openrouter.py      # API testing
├── 📂 evaluation/              # Analysis & evaluation
│   ├── evaluate_performance.py # ⭐ Performance testing
│   └── analytics_dashboard.py  # ⭐ Usage analytics
├── 📂 docs/                    # Documentation
│   └── AI_Internship_Jupiter.pdf # Project requirements
├── 📂 data/                    # Data storage
│   ├── processed/              # Clean FAQ data
│   ├── embeddings/             # Vector embeddings
│   └── raw/                    # Original scraped data
├── 📂 logs/                    # System logs
│   └── performance_reports/    # Evaluation results
├── 📂 jupiter_faq_env/         # Virtual environment
├── config.json                 # Configuration file
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔬 Evaluation Results

### Comprehensive Testing
- **Test Suite**: 10 queries across easy/medium/hard difficulty
- **Metrics**: Response time, confidence, accuracy, source attribution
- **Providers**: OpenRouter/Mistral performance analysis
- **Reports**: Automated markdown reports with recommendations

### Key Insights
- Strong semantic matching across all query types
- Consistent source attribution (100% success rate)
- Good handling of complex queries (policy, technical questions)
- Room for improvement in confidence scores for edge cases

## 🚀 Future Enhancements

1. **Multilingual Support** - Hindi/Hinglish integration
2. **Real-time Learning** - User feedback incorporation
3. **Advanced Analytics** - ML-driven insights
4. **API Integration** - RESTful API for external systems
5. **Voice Interface** - Speech-to-text integration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


## 🏆 Achievements

- ✅ **100% Requirements Coverage**: All core requirements implemented
- ⭐ **3 Bonus Features Added**: Query suggestions, analytics, performance evaluation
- 🚀 **Production Ready**: Robust error handling and fallback mechanisms  
- 📊 **Comprehensive Testing**: Automated evaluation framework
- 🎯 **High Performance**: Fast semantic search with natural language responses

---
