#!/usr/bin/env python3
"""
Jupiter FAQ Bot - Data Collection Pipeline
This script orchestrates the complete data collection and preprocessing pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection.scraper import JupiterFAQScraper
from data_collection.preprocessor import FAQPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/embeddings',
        'models',
        'logs',
        'src/data_collection',
        'src/bot',
        'src/evaluation'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_data_collection():
    """Run the complete data collection pipeline"""
    try:
        logger.info("Starting Jupiter FAQ data collection pipeline...")
        
        # Create directory structure
        create_directory_structure()
        
        # Initialize scraper
        logger.info("Initializing scraper...")
        scraper = JupiterFAQScraper(delay=1.0)
        
        # Scrape all data
        logger.info("Scraping FAQ data from multiple sources...")
        faq_df = scraper.scrape_all()
        
        if len(faq_df) == 0:
            logger.error("No data was scraped. Exiting.")
            return False
        
        # Save raw data
        logger.info("Saving raw data...")
        scraper.save_data(faq_df, "data/raw/jupiter_faqs_raw.csv")
        
        # Initialize preprocessor
        logger.info("Initializing preprocessor...")
        preprocessor = FAQPreprocessor(similarity_threshold=0.85)
        
        # Process the data
        logger.info("Processing and cleaning FAQ data...")
        processed_df = preprocessor.process_dataframe(faq_df)
        
        # Save processed data
        logger.info("Saving processed data...")
        summary = preprocessor.save_processed_data(processed_df, "data/processed/jupiter_faqs_processed.csv")
        
        # Print final summary
        print("\n" + "="*60)
        print("🚀 DATA COLLECTION COMPLETED SUCCESSFULLY! 🚀")
        print("="*60)
        print(f"📊 Total FAQs collected: {summary['total_faqs']}")
        print(f"📁 Categories found: {len(summary['categories'])}")
        print(f"🔍 Sources scraped: {len(summary['sources'])}")
        print(f"📝 Average question length: {summary['avg_question_length']:.1f} characters")
        print(f"📖 Average answer length: {summary['avg_answer_length']:.1f} characters")
        
        print(f"\n📋 Category Distribution:")
        for category, count in summary['categories'].items():
            print(f"   • {category}: {count} FAQs")
        
        print(f"\n🌐 Data Sources:")
        for source, count in summary['sources'].items():
            print(f"   • {source}: {count} FAQs")
        
        print(f"\n📁 Files Created:")
        print(f"   • Raw data: data/raw/jupiter_faqs_raw.csv")
        print(f"   • Processed data: data/processed/jupiter_faqs_processed.csv")
        print(f"   • Summary: data/processed/jupiter_faqs_processed_summary.json")
        
        print(f"\n✅ Ready for next steps:")
        print(f"   1. Build embedding-based search system")
        print(f"   2. Integrate with LLM for natural responses")
        print(f"   3. Create user interface")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Data collection pipeline failed: {e}")
        return False
    
    finally:
        # Clean up
        if 'scraper' in locals():
            scraper.driver.quit()

def main():
    """Main function to run data collection"""
    success = run_data_collection()
    
    if success:
        logger.info("Data collection pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Data collection pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 