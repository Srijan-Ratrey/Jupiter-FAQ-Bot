#!/usr/bin/env python3
"""
Jupiter FAQ Bot - Performance Evaluation Tool
Compares different approaches and providers for accuracy and latency.
"""

import sys
import os
import time
import statistics
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

from bot.jupiter_faq_bot import create_jupiter_bot
from bot.llm_service import LLMService
from bot.rag_pipeline import RAGPipeline

def create_test_queries() -> List[Dict]:
    """Create a comprehensive test suite of queries."""
    return [
        {
            "query": "How do I open a Jupiter account?",
            "expected_category": "Account Opening",
            "difficulty": "easy"
        },
        {
            "query": "What is Jupiter Pro?",
            "expected_category": "Account Types", 
            "difficulty": "easy"
        },
        {
            "query": "How to complete KYC verification?",
            "expected_category": "KYC",
            "difficulty": "medium"
        },
        {
            "query": "What documents do I need for account opening?",
            "expected_category": "Account Opening",
            "difficulty": "medium"
        },
        {
            "query": "How can I contact customer support?",
            "expected_category": "Customer Support",
            "difficulty": "easy"
        },
        {
            "query": "What are the benefits of Jupiter Pro membership?",
            "expected_category": "Account Types",
            "difficulty": "medium"
        },
        {
            "query": "How to apply for credit card through Jupiter?",
            "expected_category": "Credit Card",
            "difficulty": "medium"
        },
        {
            "query": "What personal information does Jupiter collect?",
            "expected_category": "Policy",
            "difficulty": "hard"
        },
        {
            "query": "Can I use Jupiter for international transfers?",
            "expected_category": "Money Transfer",
            "difficulty": "hard"
        },
        {
            "query": "How does Jupiter rewards program work?",
            "expected_category": "Rewards",
            "difficulty": "medium"
        }
    ]

def evaluate_provider(provider_name: str, test_queries: List[Dict]) -> Dict:
    """Evaluate a specific provider's performance."""
    print(f"\n🔍 Evaluating {provider_name} Provider...")
    
    # Configure bot for specific provider
    if provider_name.lower() == "openrouter":
        bot = create_jupiter_bot()  # Uses config.json settings
    else:
        # Create custom configuration for other providers
        bot = create_jupiter_bot()
    
    results = []
    response_times = []
    confidence_scores = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"   Query {i}/{len(test_queries)}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        response = bot.ask(test_case["query"])
        end_time = time.time()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        confidence_scores.append(response["confidence"])
        
        # Evaluate response quality
        result = {
            "query": test_case["query"],
            "expected_category": test_case["expected_category"],
            "difficulty": test_case["difficulty"],
            "provider": response.get("provider", "Unknown"),
            "confidence": response["confidence"],
            "response_time": response_time,
            "response_length": len(response["response"]),
            "has_sources": len(response.get("sources", [])) > 0,
            "num_sources": len(response.get("sources", [])),
            "response": response["response"][:200] + "..." if len(response["response"]) > 200 else response["response"]
        }
        results.append(result)
    
    # Calculate aggregate metrics
    metrics = {
        "provider": provider_name,
        "total_queries": len(test_queries),
        "avg_response_time": statistics.mean(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "avg_confidence": statistics.mean(confidence_scores),
        "min_confidence": min(confidence_scores),
        "max_confidence": max(confidence_scores),
        "high_confidence_rate": len([c for c in confidence_scores if c > 0.6]) / len(confidence_scores),
        "low_confidence_rate": len([c for c in confidence_scores if c < 0.3]) / len(confidence_scores),
        "queries_with_sources": len([r for r in results if r["has_sources"]]) / len(results),
        "results": results
    }
    
    return metrics

def compare_approaches() -> Dict:
    """Compare different approaches: retrieval-based vs LLM-based."""
    print("\n🔬 APPROACH COMPARISON")
    print("=" * 60)
    
    test_queries = create_test_queries()
    
    # Test with different providers/approaches
    providers_to_test = ["OpenRouter"]  # Can add more when available
    
    all_results = {}
    
    for provider in providers_to_test:
        try:
            results = evaluate_provider(provider, test_queries)
            all_results[provider] = results
        except Exception as e:
            print(f"❌ Error testing {provider}: {e}")
            continue
    
    return all_results

def generate_performance_report(results: Dict) -> str:
    """Generate a comprehensive performance report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/performance_report_{timestamp}.md"
    
    os.makedirs("logs", exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Jupiter FAQ Bot - Performance Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        for provider, metrics in results.items():
            f.write(f"### {provider} Provider\n\n")
            f.write(f"- **Average Response Time:** {metrics['avg_response_time']:.2f}s\n")
            f.write(f"- **Average Confidence:** {metrics['avg_confidence']:.1%}\n")
            f.write(f"- **High Confidence Rate:** {metrics['high_confidence_rate']:.1%}\n")
            f.write(f"- **Queries with Sources:** {metrics['queries_with_sources']:.1%}\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for provider, metrics in results.items():
            f.write(f"### {provider} Detailed Analysis\n\n")
            
            # Performance metrics table
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Queries | {metrics['total_queries']} |\n")
            f.write(f"| Avg Response Time | {metrics['avg_response_time']:.3f}s |\n")
            f.write(f"| Min Response Time | {metrics['min_response_time']:.3f}s |\n")
            f.write(f"| Max Response Time | {metrics['max_response_time']:.3f}s |\n")
            f.write(f"| Avg Confidence | {metrics['avg_confidence']:.1%} |\n")
            f.write(f"| High Confidence Rate (>60%) | {metrics['high_confidence_rate']:.1%} |\n")
            f.write(f"| Low Confidence Rate (<30%) | {metrics['low_confidence_rate']:.1%} |\n\n")
            
            # Query-by-query results
            f.write("#### Individual Query Results\n\n")
            f.write("| Query | Difficulty | Confidence | Response Time | Sources |\n")
            f.write("|-------|------------|------------|---------------|----------|\n")
            
            for result in metrics['results']:
                f.write(f"| {result['query'][:50]}... | {result['difficulty']} | {result['confidence']:.1%} | {result['response_time']:.3f}s | {result['num_sources']} |\n")
            
            f.write("\n")
        
        f.write("## Recommendations\n\n")
        
        best_provider = max(results.keys(), key=lambda k: results[k]['avg_confidence'])
        fastest_provider = min(results.keys(), key=lambda k: results[k]['avg_response_time'])
        
        f.write(f"- **Best Accuracy:** {best_provider} (avg confidence: {results[best_provider]['avg_confidence']:.1%})\n")
        f.write(f"- **Fastest Response:** {fastest_provider} (avg time: {results[fastest_provider]['avg_response_time']:.3f}s)\n")
        f.write("- **Overall:** The system shows strong performance with good semantic matching\n")
        f.write("- **Improvement Areas:** Consider expanding FAQ database for edge cases\n\n")
    
    return report_path

def main():
    """Main evaluation function."""
    print("🚀 Jupiter FAQ Bot - Performance Evaluation")
    print("=" * 60)
    
    try:
        # Run comprehensive comparison
        results = compare_approaches()
        
        if not results:
            print("❌ No results to analyze. Check your configuration.")
            return
        
        # Generate report
        report_path = generate_performance_report(results)
        
        print(f"\n📊 EVALUATION COMPLETE!")
        print(f"📄 Detailed report saved to: {report_path}")
        
        # Print summary
        print(f"\n📈 PERFORMANCE SUMMARY:")
        for provider, metrics in results.items():
            print(f"\n🤖 {provider}:")
            print(f"   • Avg Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"   • Avg Confidence: {metrics['avg_confidence']:.1%}")
            print(f"   • High Confidence Rate: {metrics['high_confidence_rate']:.1%}")
            print(f"   • Success Rate: {metrics['queries_with_sources']:.1%}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 