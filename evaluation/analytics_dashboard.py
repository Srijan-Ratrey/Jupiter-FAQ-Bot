#!/usr/bin/env python3
"""
Jupiter FAQ Bot - Analytics Dashboard
Provides insights into bot usage, popular queries, and performance trends.
"""

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, List
import glob

# Add src to path
sys.path.append('src')

def load_conversation_logs() -> List[Dict]:
    """Load all conversation logs from the logs directory."""
    log_files = glob.glob("logs/conversation_history_*.json")
    all_conversations = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                conversations = data.get('conversations', [])
                for conv in conversations:
                    conv['log_file'] = log_file
                    all_conversations.append(conv)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    return all_conversations

def analyze_query_patterns(conversations: List[Dict]) -> Dict:
    """Analyze patterns in user queries."""
    queries = [conv['query'].lower() for conv in conversations]
    
    # Common keywords
    all_words = []
    for query in queries:
        words = query.split()
        all_words.extend([word for word in words if len(word) > 3])
    
    common_keywords = Counter(all_words).most_common(10)
    
    # Query length distribution
    query_lengths = [len(query.split()) for query in queries]
    
    # Confidence distribution
    confidences = [conv['confidence'] for conv in conversations if 'confidence' in conv]
    
    return {
        "total_queries": len(queries),
        "unique_queries": len(set(queries)),
        "common_keywords": common_keywords,
        "avg_query_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "confidence_distribution": {
            "high (>60%)": len([c for c in confidences if c > 0.6]),
            "medium (30-60%)": len([c for c in confidences if 0.3 <= c <= 0.6]),
            "low (<30%)": len([c for c in confidences if c < 0.3])
        }
    }

def analyze_temporal_patterns(conversations: List[Dict]) -> Dict:
    """Analyze temporal usage patterns."""
    if not conversations:
        return {"error": "No conversations to analyze"}
    
    # Convert timestamps
    timestamps = []
    for conv in conversations:
        if 'timestamp' in conv:
            try:
                ts = datetime.fromisoformat(conv['timestamp'].replace('Z', '+00:00'))
                timestamps.append(ts)
            except:
                continue
    
    if not timestamps:
        return {"error": "No valid timestamps found"}
    
    # Hour of day analysis
    hours = [ts.hour for ts in timestamps]
    hour_distribution = Counter(hours)
    
    # Day of week analysis
    days = [ts.strftime('%A') for ts in timestamps]
    day_distribution = Counter(days)
    
    return {
        "total_sessions": len(timestamps),
        "date_range": {
            "start": min(timestamps).strftime('%Y-%m-%d'),
            "end": max(timestamps).strftime('%Y-%m-%d')
        },
        "peak_hour": max(hour_distribution, key=hour_distribution.get) if hour_distribution else "N/A",
        "peak_day": max(day_distribution, key=day_distribution.get) if day_distribution else "N/A",
        "hour_distribution": dict(hour_distribution),
        "day_distribution": dict(day_distribution)
    }

def generate_performance_insights(conversations: List[Dict]) -> Dict:
    """Generate performance and user satisfaction insights."""
    if not conversations:
        return {"error": "No conversations to analyze"}
    
    # Response quality indicators
    confidences = [conv['confidence'] for conv in conversations if 'confidence' in conv]
    response_lengths = [len(conv['response']) for conv in conversations if 'response' in conv]
    
    # Query complexity estimation (based on length and keywords)
    complex_keywords = ['how', 'why', 'what', 'when', 'where', 'difference', 'compare', 'benefits']
    complex_queries = 0
    
    for conv in conversations:
        query = conv['query'].lower()
        if any(keyword in query for keyword in complex_keywords) or len(query.split()) > 8:
            complex_queries += 1
    
    return {
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        "complex_query_rate": complex_queries / len(conversations) if conversations else 0,
        "high_confidence_rate": len([c for c in confidences if c > 0.6]) / len(confidences) if confidences else 0,
        "satisfaction_indicators": {
            "detailed_responses": len([r for r in response_lengths if r > 200]) / len(response_lengths) if response_lengths else 0,
            "confident_answers": len([c for c in confidences if c > 0.7]) / len(confidences) if confidences else 0
        }
    }

def create_visualizations(analytics_data: Dict):
    """Create visualization charts for the analytics data."""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confidence Distribution
    conf_dist = analytics_data['query_patterns']['confidence_distribution']
    axes[0, 0].pie(conf_dist.values(), labels=conf_dist.keys(), autopct='%1.1f%%')
    axes[0, 0].set_title('Confidence Score Distribution')
    
    # 2. Common Keywords
    keywords, counts = zip(*analytics_data['query_patterns']['common_keywords'][:8])
    axes[0, 1].barh(keywords, counts)
    axes[0, 1].set_title('Most Common Query Keywords')
    axes[0, 1].set_xlabel('Frequency')
    
    # 3. Hour Distribution (if available)
    if 'hour_distribution' in analytics_data['temporal_patterns']:
        hours = list(range(24))
        hour_counts = [analytics_data['temporal_patterns']['hour_distribution'].get(h, 0) for h in hours]
        axes[1, 0].bar(hours, hour_counts)
        axes[1, 0].set_title('Usage by Hour of Day')
        axes[1, 0].set_xlabel('Hour (24h format)')
        axes[1, 0].set_ylabel('Number of Queries')
    
    # 4. Performance Metrics
    perf_data = analytics_data['performance']
    metrics = ['Avg Confidence', 'High Conf Rate', 'Complex Query Rate', 'Satisfaction']
    values = [
        perf_data['avg_confidence'],
        perf_data['high_confidence_rate'],
        perf_data['complex_query_rate'],
        perf_data['satisfaction_indicators']['confident_answers']
    ]
    axes[1, 1].bar(metrics, values)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Score (0-1)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("logs", exist_ok=True)
    plot_path = f"logs/analytics_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    return plot_path

def generate_analytics_report(analytics_data: Dict) -> str:
    """Generate a comprehensive analytics report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/analytics_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Jupiter FAQ Bot - Analytics Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        qp = analytics_data['query_patterns']
        tp = analytics_data['temporal_patterns']
        perf = analytics_data['performance']
        
        f.write(f"- **Total Queries Processed:** {qp['total_queries']}\n")
        f.write(f"- **Unique Queries:** {qp['unique_queries']}\n")
        f.write(f"- **Average Confidence:** {qp['avg_confidence']:.1%}\n")
        f.write(f"- **High Confidence Rate:** {perf['high_confidence_rate']:.1%}\n")
        
        if 'date_range' in tp:
            f.write(f"- **Analysis Period:** {tp['date_range']['start']} to {tp['date_range']['end']}\n")
        
        f.write("\n")
        
        # Query Patterns
        f.write("## Query Analysis\n\n")
        f.write("### Most Common Keywords\n\n")
        f.write("| Keyword | Frequency |\n")
        f.write("|---------|----------|\n")
        for keyword, count in qp['common_keywords']:
            f.write(f"| {keyword} | {count} |\n")
        f.write("\n")
        
        f.write("### Confidence Distribution\n\n")
        for level, count in qp['confidence_distribution'].items():
            percentage = (count / qp['total_queries'] * 100) if qp['total_queries'] > 0 else 0
            f.write(f"- **{level}:** {count} queries ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Performance Insights
        f.write("## Performance Insights\n\n")
        f.write(f"- **Average Response Confidence:** {perf['avg_confidence']:.1%}\n")
        f.write(f"- **Complex Query Handling:** {perf['complex_query_rate']:.1%}\n")
        f.write(f"- **User Satisfaction Indicators:**\n")
        f.write(f"  - Detailed Responses: {perf['satisfaction_indicators']['detailed_responses']:.1%}\n")
        f.write(f"  - High Confidence Answers: {perf['satisfaction_indicators']['confident_answers']:.1%}\n")
        f.write("\n")
        
        # Usage Patterns
        if not tp.get('error'):
            f.write("## Usage Patterns\n\n")
            f.write(f"- **Peak Usage Hour:** {tp.get('peak_hour', 'N/A')}\n")
            f.write(f"- **Peak Usage Day:** {tp.get('peak_day', 'N/A')}\n")
            f.write(f"- **Total Sessions:** {tp.get('total_sessions', 0)}\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        if perf['avg_confidence'] < 0.5:
            f.write("- **Low Confidence Alert:** Consider expanding FAQ database or improving semantic matching\n")
        if perf['complex_query_rate'] > 0.6:
            f.write("- **Complex Queries:** Users are asking sophisticated questions - consider advanced NLP features\n")
        if qp['total_queries'] < 10:
            f.write("- **Low Usage:** Consider promoting the bot or improving discoverability\n")
        
        f.write("- **Data Quality:** Continue monitoring and expanding the FAQ database\n")
        f.write("- **User Experience:** Maintain fast response times and clear answers\n")
    
    return report_path

def main():
    """Main analytics function."""
    print("📊 Jupiter FAQ Bot - Analytics Dashboard")
    print("=" * 50)
    
    try:
        # Load conversation data
        conversations = load_conversation_logs()
        
        if not conversations:
            print("❌ No conversation logs found in logs/ directory")
            print("💡 Try running some bot interactions first!")
            return
        
        print(f"📈 Analyzing {len(conversations)} conversations...")
        
        # Perform analysis
        analytics_data = {
            'query_patterns': analyze_query_patterns(conversations),
            'temporal_patterns': analyze_temporal_patterns(conversations),
            'performance': generate_performance_insights(conversations)
        }
        
        # Generate visualizations
        try:
            plot_path = create_visualizations(analytics_data)
            print(f"📊 Visualizations saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️ Could not generate visualizations: {e}")
            plot_path = None
        
        # Generate report
        report_path = generate_analytics_report(analytics_data)
        
        print(f"\n📋 ANALYTICS SUMMARY:")
        print(f"📄 Detailed report: {report_path}")
        if plot_path:
            print(f"📊 Visualizations: {plot_path}")
        
        # Quick summary
        qp = analytics_data['query_patterns']
        perf = analytics_data['performance']
        
        print(f"\n🎯 KEY METRICS:")
        print(f"   • Total Queries: {qp['total_queries']}")
        print(f"   • Average Confidence: {qp['avg_confidence']:.1%}")
        print(f"   • High Confidence Rate: {perf['high_confidence_rate']:.1%}")
        print(f"   • Top Keywords: {', '.join([kw[0] for kw in qp['common_keywords'][:3]])}")
        
    except Exception as e:
        print(f"❌ Analytics failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 