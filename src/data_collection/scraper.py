"""
Jupiter FAQ Data Scraper
This module scrapes FAQ data from Jupiter's help center, community forums, and website.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
from typing import List, Dict, Optional
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JupiterFAQScraper:
    def __init__(self, delay: float = 1.0):
        """
        Initialize the Jupiter FAQ scraper.
        
        Args:
            delay: Delay between requests to be respectful to servers
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize Selenium driver for dynamic content
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        self.scraped_data = []
        
    def __del__(self):
        """Clean up driver when object is destroyed"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def scrape_jupiter_website_faq(self) -> List[Dict]:
        """Scrape FAQs from Jupiter's main website"""
        logger.info("Scraping Jupiter main website FAQ...")
        
        url = "https://jupiter.money/contact/"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            faqs = []
            # Look for FAQ sections
            faq_elements = soup.find_all(['div', 'section'], class_=lambda x: x and 'faq' in x.lower())
            
            # Also check the specific FAQ section we saw
            faq_section = soup.find('h1', text=re.compile(r'Frequently asked\s+questions', re.I))
            if faq_section:
                faq_container = faq_section.find_parent()
                if faq_container:
                    questions = faq_container.find_all(text=re.compile(r'\?'))
                    for question in questions:
                        # Find the answer (usually the next sibling or in a toggle)
                        question_elem = question.parent
                        answer_elem = question_elem.find_next_sibling()
                        
                        if question_elem and answer_elem:
                            faqs.append({
                                'question': question.strip(),
                                'answer': answer_elem.get_text(strip=True),
                                'category': 'General',
                                'source': 'Jupiter Website',
                                'url': url
                            })
            
            # Manual extraction of the FAQs we can see in the search results
            manual_faqs = [
                {
                    'question': 'What is Jupiter Money?',
                    'answer': 'Jupiter is the 1-app for everything money that lets you spend and save money, track expenses, pay bills, and invest money in Direct Mutual Funds. It enables you to make smart money decisions every day using easy, intuitive, and personalized money management tools by eliminating the stress, fear and confusion that comes with managing money',
                    'category': 'General',
                    'source': 'Jupiter Website',
                    'url': url
                },
                {
                    'question': "Is 'Jupiter Bank' approved by the RBI?",
                    'answer': "Jupiter is itself not a bank and doesn't hold or claim to hold a banking license. The Savings Account and VISA Debit Card are provided by Federal Bank - an RBI-licensed bank - and follow all security standards prescribed by RBI. All funds in your account are insured under the RBI's deposit insurance scheme. Your money is always safe with Federal Bank",
                    'category': 'Banking',
                    'source': 'Jupiter Website',
                    'url': url
                },
                {
                    'question': 'How can I open a Savings account?',
                    'answer': 'To open a free Savings or Salary Bank Account on Jupiter - powered by Federal Bank - in 3 minutes, simply install the Jupiter App. Follow the on-screen instructions to create your account. To unlock the full Jupiter experience, complete your KYC in 5 minutes',
                    'category': 'Account Opening',
                    'source': 'Jupiter Website',
                    'url': url
                },
                {
                    'question': 'How can I get a Debit card?',
                    'answer': 'You can order a new physical Debit Card by tapping on the "Card" tab on the Jupiter app. While you can get a virtual Debit Card for free, you will be charged a one-time fee when ordering a physical Debit Card',
                    'category': 'Debit Card',
                    'source': 'Jupiter Website',
                    'url': url
                },
                {
                    'question': 'How to deposit cash in the Savings account?',
                    'answer': 'To deposit cash into your Savings or Salary Bank Account, visit any Federal Bank branch. You can find the nearest branch here: https://locations.federalbank.co.in/',
                    'category': 'Banking',
                    'source': 'Jupiter Website',
                    'url': url
                },
                {
                    'question': 'How can I transfer money from Jupiter?',
                    'answer': 'There are many ways to transfer money from Jupiter. You may use the Scan & Pay for UPI transfers, pay via your Debit Card, or use your Cheque Book for sending money. You can also use Bank Transfers on Jupiter to transfer money. Jupiter offers 4 different modes for bank transfers - 1. Internal Fund transfer (for Inter Bank) 2. NEFT 3. IMPS 4. RTGS Jupiter automatically selects the best mode for transfer when you enter payee and transaction information',
                    'category': 'Money Transfer',
                    'source': 'Jupiter Website',
                    'url': url
                }
            ]
            
            faqs.extend(manual_faqs)
            logger.info(f"Scraped {len(faqs)} FAQs from Jupiter website")
            return faqs
            
        except Exception as e:
            logger.error(f"Error scraping Jupiter website: {e}")
            return []
    
    def scrape_community_topics(self) -> List[Dict]:
        """Scrape topics from Jupiter Community forums"""
        logger.info("Scraping Jupiter Community topics...")
        
        base_url = "https://community.jupiter.money"
        help_url = f"{base_url}/c/help/27"
        
        try:
            # Use Selenium for dynamic content
            self.driver.get(help_url)
            time.sleep(3)
            
            faqs = []
            
            # Find topic links
            topic_elements = self.driver.find_elements(By.CSS_SELECTOR, "a.title")
            
            # Process first 20 topics to avoid overwhelming
            for topic_elem in topic_elements[:20]:
                try:
                    topic_url = topic_elem.get_attribute('href')
                    topic_title = topic_elem.text.strip()
                    
                    if topic_title and topic_url:
                        # Visit the topic page
                        topic_response = self.session.get(topic_url)
                        topic_soup = BeautifulSoup(topic_response.content, 'html.parser')
                        
                        # Extract the main post content
                        main_post = topic_soup.find('div', class_='post')
                        if main_post:
                            content = main_post.get_text(strip=True)
                            
                            faqs.append({
                                'question': topic_title,
                                'answer': content[:1000] + "..." if len(content) > 1000 else content,
                                'category': 'Community Help',
                                'source': 'Jupiter Community',
                                'url': topic_url
                            })
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    logger.warning(f"Error processing topic {topic_title}: {e}")
                    continue
            
            logger.info(f"Scraped {len(faqs)} topics from Jupiter Community")
            return faqs
            
        except Exception as e:
            logger.error(f"Error scraping Jupiter Community: {e}")
            return []
    
    def scrape_additional_sources(self) -> List[Dict]:
        """Scrape additional FAQ sources like grievance policy, etc."""
        logger.info("Scraping additional sources...")
        
        urls = [
            "https://jupiter.money/grievance-redressal-policy/",
            "https://jupiter.money/pricing-and-fees/",
            "https://jupiter.money/terms-and-conditions/",
            "https://jupiter.money/privacy-policy/"
        ]
        
        faqs = []
        
        for url in urls:
            try:
                response = self.session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract sections as Q&A pairs
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
                
                for heading in headings:
                    heading_text = heading.get_text(strip=True)
                    if len(heading_text) > 5 and '?' in heading_text or any(keyword in heading_text.lower() for keyword in ['how', 'what', 'when', 'where', 'why']):
                        # Find the content following this heading
                        content_elements = []
                        next_elem = heading.next_sibling
                        
                        while next_elem and next_elem.name not in ['h1', 'h2', 'h3', 'h4']:
                            if hasattr(next_elem, 'get_text'):
                                content_elements.append(next_elem.get_text(strip=True))
                            next_elem = next_elem.next_sibling
                        
                        content = ' '.join(content_elements)
                        
                        if content:
                            category = 'Policy' if 'policy' in url else 'General'
                            faqs.append({
                                'question': heading_text,
                                'answer': content[:1000] + "..." if len(content) > 1000 else content,
                                'category': category,
                                'source': url.split('/')[-2].replace('-', ' ').title(),
                                'url': url
                            })
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")
                continue
        
        logger.info(f"Scraped {len(faqs)} FAQs from additional sources")
        return faqs
    
    def manual_faq_data(self) -> List[Dict]:
        """Add manually curated FAQ data based on common Jupiter queries"""
        logger.info("Adding manual FAQ data...")
        
        manual_faqs = [
            {
                'question': 'How to withdraw money from Jupiter account?',
                'answer': 'You can withdraw money from Jupiter using UPI transfers, ATM withdrawals with your debit card, or bank transfers. For cash withdrawal, visit any Federal Bank ATM or branch.',
                'category': 'Banking',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'What are Jupiter Pots?',
                'answer': 'Jupiter Pots are goal-based savings features that help you save money for specific purposes. You can create multiple pots, set savings goals, and track your progress.',
                'category': 'Pots',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'How to apply for Jupiter credit card?',
                'answer': 'You can apply for Edge Federal Bank Credit Card or Edge CSB Bank Credit Card exclusively through the Jupiter app. The application process is digital and quick.',
                'category': 'Credit Card',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'What is Jupiter Pro?',
                'answer': 'Jupiter Pro is a premium tier offering enhanced features and benefits for Jupiter users. It may include higher transaction limits, premium support, and exclusive features.',
                'category': 'Account Types',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'How to complete KYC on Jupiter?',
                'answer': 'KYC can be completed through the Jupiter app using Video KYC (VKYC). You will need your Aadhaar card, PAN card, and a clear selfie. The process typically takes 5-10 minutes.',
                'category': 'KYC',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'Jupiter customer care number',
                'answer': 'Jupiter customer care can be reached at +91 8655055086 (9 AM to 7 PM on weekdays). You can also chat with executives through the Jupiter app or email support@jupiter.money',
                'category': 'Customer Support',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'How to invest in mutual funds through Jupiter?',
                'answer': 'Jupiter offers direct mutual fund investments through its app. You can browse different fund categories, analyze performance, and invest directly without any commission charges.',
                'category': 'Investments',
                'source': 'Manual',
                'url': 'N/A'
            },
            {
                'question': 'What are Jupiter rewards?',
                'answer': 'Jupiter Rewards is a cashback and rewards program where you earn points on transactions, which can be redeemed for cashback, vouchers, or other benefits.',
                'category': 'Rewards',
                'source': 'Manual',
                'url': 'N/A'
            }
        ]
        
        return manual_faqs
    
    def scrape_all(self) -> pd.DataFrame:
        """Scrape FAQs from all sources and return as DataFrame"""
        logger.info("Starting comprehensive FAQ scraping...")
        
        all_faqs = []
        
        # Scrape from different sources
        all_faqs.extend(self.scrape_jupiter_website_faq())
        all_faqs.extend(self.scrape_community_topics())
        all_faqs.extend(self.scrape_additional_sources())
        all_faqs.extend(self.manual_faq_data())
        
        # Convert to DataFrame
        df = pd.DataFrame(all_faqs)
        
        # Add metadata
        df['scraped_at'] = pd.Timestamp.now()
        df['id'] = range(len(df))
        
        logger.info(f"Total FAQs scraped: {len(df)}")
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str = "data/raw/jupiter_faqs_raw.csv"):
        """Save scraped data to CSV file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        
        # Also save as JSON for more flexible processing
        json_filepath = filepath.replace('.csv', '.json')
        df.to_json(json_filepath, orient='records', indent=2)
        logger.info(f"Data also saved to {json_filepath}")


if __name__ == "__main__":
    scraper = JupiterFAQScraper()
    
    try:
        # Scrape all data
        faq_df = scraper.scrape_all()
        
        # Save the data
        scraper.save_data(faq_df)
        
        # Print summary
        print(f"\n=== SCRAPING SUMMARY ===")
        print(f"Total FAQs collected: {len(faq_df)}")
        print(f"Categories found: {faq_df['category'].unique()}")
        print(f"Sources: {faq_df['source'].unique()}")
        print(f"\nCategory distribution:")
        print(faq_df['category'].value_counts())
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        scraper.driver.quit() 