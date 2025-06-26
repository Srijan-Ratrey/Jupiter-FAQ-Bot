"""
Data Preprocessing for Jupiter FAQ Bot
This module cleans, normalizes, and structures the scraped FAQ data.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAQPreprocessor:
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the FAQ preprocessor.
        
        Args:
            similarity_threshold: Threshold for detecting duplicate questions
        """
        self.similarity_threshold = similarity_threshold
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features will be limited.")
            self.nlp = None
        
        # Category mapping for standardization
        self.category_mapping = {
            'general': 'General',
            'banking': 'Banking',
            'account opening': 'Account Opening',
            'account_opening': 'Account Opening',
            'debit card': 'Debit Card',
            'debit_card': 'Debit Card',
            'credit card': 'Credit Card',
            'credit_card': 'Credit Card',
            'money transfer': 'Money Transfer',
            'money_transfer': 'Money Transfer',
            'customer support': 'Customer Support',
            'customer_support': 'Customer Support',
            'kyc': 'KYC',
            'pots': 'Pots',
            'investments': 'Investments',
            'mutual funds': 'Mutual Funds',
            'mutual_funds': 'Mutual Funds',
            'rewards': 'Rewards',
            'policy': 'Policy',
            'community help': 'Community Help',
            'community_help': 'Community Help',
            'account types': 'Account Types',
            'account_types': 'Account Types'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for questions
        text = re.sub(r'[^\w\s\?\!\.\,\-\:\;]', '', text)
        
        return text.strip()
    
    def normalize_question(self, question: str) -> str:
        """Normalize questions for better matching"""
        question = self.clean_text(question)
        
        # Convert to lowercase
        question = question.lower()
        
        # Remove common question prefixes
        prefixes = ['how to', 'how can i', 'how do i', 'what is', 'what are', 'where is', 'where can i', 'when can i']
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
        
        # Remove question marks for comparison
        question = question.rstrip('?').strip()
        
        return question
    
    def categorize_faq(self, question: str, answer: str, existing_category: str = None) -> str:
        """Categorize FAQ based on content"""
        if existing_category and existing_category.lower() in self.category_mapping:
            return self.category_mapping[existing_category.lower()]
        
        text = f"{question} {answer}".lower()
        
        # Define keyword patterns for categories
        category_keywords = {
            'Banking': ['bank', 'savings', 'account', 'deposit', 'withdraw', 'federal bank', 'rbi'],
            'Debit Card': ['debit card', 'card', 'pin', 'atm', 'physical card', 'virtual card'],
            'Credit Card': ['credit card', 'edge', 'csb', 'federal', 'credit limit', 'credit score'],
            'KYC': ['kyc', 'verification', 'identity', 'document', 'vkyc', 'aadhaar', 'pan'],
            'Money Transfer': ['transfer', 'upi', 'neft', 'imps', 'rtgs', 'send money', 'payment'],
            'Pots': ['pot', 'pots', 'savings goal', 'goal based'],
            'Investments': ['invest', 'mutual fund', 'sip', 'portfolio', 'returns'],
            'Rewards': ['reward', 'cashback', 'points', 'redeem'],
            'Customer Support': ['support', 'help', 'customer care', 'contact', 'complaint', 'grievance'],
            'Account Opening': ['open account', 'create account', 'signup', 'registration', 'onboard'],
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'General' if no match
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'General'
    
    def detect_duplicates(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """Detect duplicate or very similar questions"""
        logger.info("Detecting duplicate questions...")
        
        questions = df['question_normalized'].tolist()
        
        # Use TF-IDF for similarity detection
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(questions)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        duplicates = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                similarity = similarity_matrix[i][j]
                if similarity >= self.similarity_threshold:
                    duplicates.append((i, j, similarity))
        
        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates
    
    def merge_duplicates(self, df: pd.DataFrame, duplicates: List[Tuple[int, int, float]]) -> pd.DataFrame:
        """Merge duplicate questions by combining their information"""
        if not duplicates:
            return df
        
        # Group duplicates
        duplicate_groups = defaultdict(set)
        for i, j, sim in duplicates:
            # Find existing group or create new one
            group_found = False
            for group_id, group in duplicate_groups.items():
                if i in group or j in group:
                    group.update([i, j])
                    group_found = True
                    break
            
            if not group_found:
                new_group_id = len(duplicate_groups)
                duplicate_groups[new_group_id] = {i, j}
        
        # Merge duplicates within each group
        rows_to_remove = set()
        
        for group_id, indices in duplicate_groups.items():
            indices = list(indices)
            if len(indices) <= 1:
                continue
            
            # Keep the first one and merge others into it
            main_idx = indices[0]
            
            # Combine answers
            answers = [df.loc[idx, 'answer'] for idx in indices if pd.notna(df.loc[idx, 'answer'])]
            combined_answer = ". ".join(set(answers))  # Remove duplicate sentences
            
            # Combine sources
            sources = [df.loc[idx, 'source'] for idx in indices if pd.notna(df.loc[idx, 'source'])]
            combined_sources = ", ".join(set(sources))
            
            # Update the main row
            df.loc[main_idx, 'answer'] = combined_answer
            df.loc[main_idx, 'source'] = combined_sources
            df.loc[main_idx, 'merged_from'] = len(indices) - 1
            
            # Mark others for removal
            rows_to_remove.update(indices[1:])
        
        # Remove duplicate rows
        df_cleaned = df.drop(index=list(rows_to_remove)).reset_index(drop=True)
        
        logger.info(f"Merged {len(rows_to_remove)} duplicate questions")
        return df_cleaned
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using spaCy"""
        if not self.nlp:
            # Fallback to simple keyword extraction
            words = word_tokenize(text.lower())
            keywords = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
            return keywords[:10]  # Return top 10 keywords
        
        doc = self.nlp(text)
        
        # Extract entities, noun phrases, and important words
        keywords = []
        
        # Named entities
        keywords.extend([ent.text.lower() for ent in doc.ents])
        
        # Noun phrases
        keywords.extend([chunk.text.lower() for chunk in doc.noun_chunks])
        
        # Important tokens (not stop words, not punctuation)
        keywords.extend([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']])
        
        # Remove duplicates and return
        return list(set(keywords))[:15]
    
    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful metadata to FAQ entries"""
        logger.info("Adding metadata...")
        
        # Add text length statistics
        df['question_length'] = df['question'].str.len()
        df['answer_length'] = df['answer'].str.len()
        df['question_word_count'] = df['question'].str.split().str.len()
        df['answer_word_count'] = df['answer'].str.split().str.len()
        
        # Extract keywords
        df['keywords'] = df.apply(lambda row: self.extract_keywords(f"{row['question']} {row['answer']}"), axis=1)
        
        # Add processing metadata
        df['processed_at'] = pd.Timestamp.now()
        df['version'] = '1.0'
        
        return df
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main processing pipeline for FAQ data"""
        logger.info("Starting FAQ data preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text fields
        processed_df['question'] = processed_df['question'].apply(self.clean_text)
        processed_df['answer'] = processed_df['answer'].apply(self.clean_text)
        
        # Remove empty entries
        processed_df = processed_df[
            (processed_df['question'].str.len() > 5) & 
            (processed_df['answer'].str.len() > 10)
        ].reset_index(drop=True)
        
        # Normalize questions for duplicate detection
        processed_df['question_normalized'] = processed_df['question'].apply(self.normalize_question)
        
        # Standardize categories
        processed_df['category'] = processed_df.apply(
            lambda row: self.categorize_faq(row['question'], row['answer'], row.get('category')), 
            axis=1
        )
        
        # Detect and merge duplicates
        duplicates = self.detect_duplicates(processed_df)
        processed_df = self.merge_duplicates(processed_df, duplicates)
        
        # Add metadata
        processed_df = self.add_metadata(processed_df)
        
        # Sort by category and question length
        processed_df = processed_df.sort_values(['category', 'question_length']).reset_index(drop=True)
        
        # Add final ID
        processed_df['processed_id'] = range(len(processed_df))
        
        logger.info(f"Preprocessing completed. Final dataset has {len(processed_df)} entries")
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str = "data/processed/jupiter_faqs_processed.csv"):
        """Save processed data"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
        
        # Save JSON for easy loading
        json_filepath = filepath.replace('.csv', '.json')
        
        # Convert to JSON-friendly format
        df_json = df.copy()
        df_json['keywords'] = df_json['keywords'].apply(lambda x: x if isinstance(x, list) else [])
        df_json.to_json(json_filepath, orient='records', indent=2)
        
        # Save category summary
        summary_filepath = filepath.replace('.csv', '_summary.json')
        summary = {
            'total_faqs': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'avg_question_length': df['question_length'].mean(),
            'avg_answer_length': df['answer_length'].mean(),
            'processing_date': pd.Timestamp.now().isoformat()
        }
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing summary saved to {summary_filepath}")
        
        return summary


if __name__ == "__main__":
    # Load raw data
    try:
        raw_df = pd.read_csv("data/raw/jupiter_faqs_raw.csv")
        logger.info(f"Loaded {len(raw_df)} raw FAQs")
        
        # Initialize preprocessor
        preprocessor = FAQPreprocessor()
        
        # Process the data
        processed_df = preprocessor.process_dataframe(raw_df)
        
        # Save processed data
        summary = preprocessor.save_processed_data(processed_df)
        
        # Print summary
        print(f"\n=== PREPROCESSING SUMMARY ===")
        print(f"Total FAQs processed: {summary['total_faqs']}")
        print(f"Categories: {list(summary['categories'].keys())}")
        print(f"Average question length: {summary['avg_question_length']:.1f} characters")
        print(f"Average answer length: {summary['avg_answer_length']:.1f} characters")
        print(f"\nCategory distribution:")
        for category, count in summary['categories'].items():
            print(f"  {category}: {count}")
        
    except FileNotFoundError:
        logger.error("Raw data file not found. Please run scraper.py first.")
    except Exception as e:
        logger.error(f"Processing failed: {e}") 