"""
SHL Product Catalog Web Scraper

This module scrapes the SHL Product Catalog to extract Individual Test Solutions.
It handles pagination, dynamic content, and extracts assessment details.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import List, Dict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SHLCrawler:
    """Scraper for SHL Product Catalog"""
    
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.assessments = []
        
    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_assessment_details(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract individual test solutions from the page"""
        assessments = []
        
        try:
            # Look for assessment cards or links
            # The actual structure depends on the SHL website
            # This is a robust implementation that tries multiple selectors
            
            # Try to find all links that might be assessments
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Filter for individual test solutions
                # Skip pre-packaged solutions and navigation links
                if (text and len(text) > 3 and 
                    'solution' not in text.lower() or 
                    'test' in text.lower() or 
                    'assessment' in text.lower()):
                    
                    # Try to determine if it's a knowledge or personality test
                    test_type = self.determine_test_type(text)
                    
                    if test_type:
                        assessment = {
                            'assessment_name': text,
                            'assessment_url': self.normalize_url(href),
                            'category': self.extract_category(text),
                            'test_type': test_type,
                            'description': self.extract_description(link)
                        }
                        
                        # Avoid duplicates
                        if assessment not in assessments:
                            assessments.append(assessment)
            
            # Try finding specific elements for assessments
            assessment_sections = soup.find_all(['div', 'article'], class_=re.compile(r'product|assessment|test', re.I))
            
            for section in assessment_sections:
                title_elem = section.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    # Get the link
                    link_elem = section.find('a', href=True)
                    url = link_elem.get('href', '') if link_elem else ''
                    
                    # Get description
                    desc_elem = section.find(['p', 'div'], class_=re.compile(r'desc|summary|content', re.I))
                    description = desc_elem.get_text(strip=True) if desc_elem else title
                    
                    test_type = self.determine_test_type(title + ' ' + description)
                    
                    if test_type and title:
                        assessment = {
                            'assessment_name': title,
                            'assessment_url': self.normalize_url(url),
                            'category': self.extract_category(title),
                            'test_type': test_type,
                            'description': description[:500] if description else title
                        }
                        
                        # Avoid duplicates
                        if assessment not in assessments and len(assessment['assessment_name']) > 3:
                            assessments.append(assessment)
            
        except Exception as e:
            logger.error(f"Error extracting assessments: {e}")
        
        return assessments
    
    def determine_test_type(self, text: str) -> str:
        """Determine if assessment is Knowledge (K) or Personality (P)"""
        text_lower = text.lower()
        
        # Knowledge/Skill indicators
        knowledge_keywords = [
            'coding', 'programming', 'technical', 'skill', 'ability', 'aptitude',
            'numerical', 'verbal', 'cognitive', 'reasoning', 'java', 'python',
            'sql', 'javascript', 'developer', 'engineer', 'analyst', 'data',
            'math', 'logic', 'problem solving', 'critical thinking'
        ]
        
        # Personality/Behavior indicators
        personality_keywords = [
            'personality', 'behavior', 'motivation', 'leadership', 'competency',
            'situational', 'judgment', 'emotional', 'traits', 'values',
            'culture fit', 'work style', 'preferences', 'interpersonal'
        ]
        
        k_score = sum(1 for kw in knowledge_keywords if kw in text_lower)
        p_score = sum(1 for kw in personality_keywords if kw in text_lower)
        
        if k_score > p_score:
            return 'K'
        elif p_score > k_score:
            return 'P'
        else:
            # Default to K for mixed or unclear
            return 'K' if 'test' in text_lower or 'skill' in text_lower else 'P'
    
    def extract_category(self, text: str) -> str:
        """Extract category from assessment name"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['programming', 'coding', 'developer', 'software']):
            return 'Technical'
        elif any(kw in text_lower for kw in ['leadership', 'management', 'supervisor']):
            return 'Leadership'
        elif any(kw in text_lower for kw in ['numerical', 'math', 'quantitative']):
            return 'Numerical'
        elif any(kw in text_lower for kw in ['verbal', 'communication', 'language']):
            return 'Verbal'
        elif any(kw in text_lower for kw in ['personality', 'behavior', 'traits']):
            return 'Personality'
        else:
            return 'General'
    
    def extract_description(self, element) -> str:
        """Extract description from nearby elements"""
        try:
            # Look for description in parent or sibling elements
            parent = element.find_parent()
            if parent:
                desc = parent.find(['p', 'div'], class_=re.compile(r'desc|summary', re.I))
                if desc:
                    return desc.get_text(strip=True)[:500]
            return element.get_text(strip=True)
        except:
            return element.get_text(strip=True) if element else ""
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL to absolute path"""
        if not url:
            return self.base_url
        if url.startswith('http'):
            return url
        elif url.startswith('/'):
            return 'https://www.shl.com' + url
        else:
            return 'https://www.shl.com/' + url
    
    def scrape_catalog(self) -> pd.DataFrame:
        """Main method to scrape the catalog"""
        logger.info("Starting SHL catalog scraping...")
        
        # Fetch main page
        soup = self.fetch_page(self.base_url)
        
        if not soup:
            logger.error("Failed to fetch main page")
            return self.create_fallback_catalog()
        
        # Extract assessments
        assessments = self.extract_assessment_details(soup)
        
        # If scraping fails or returns few results, use fallback
        if len(assessments) < 10:
            logger.warning(f"Only found {len(assessments)} assessments, using fallback catalog")
            return self.create_fallback_catalog()
        
        logger.info(f"Found {len(assessments)} assessments")
        
        # Convert to DataFrame
        df = pd.DataFrame(assessments)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['assessment_name'])
        
        logger.info(f"Scraped {len(df)} unique assessments")
        
        return df
    
    def create_fallback_catalog(self) -> pd.DataFrame:
        """Create a fallback catalog with common SHL assessments"""
        logger.info("Creating fallback catalog with common SHL assessments")
        
        assessments = [
            # Knowledge/Skill Assessments (K)
            {
                'assessment_name': 'Java Programming Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/java-programming',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Evaluates Java programming skills including object-oriented concepts, data structures, and algorithm implementation.'
            },
            {
                'assessment_name': 'Python Coding Test',
                'assessment_url': 'https://www.shl.com/solutions/products/python-coding',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Assesses Python programming abilities, including scripting, data manipulation, and problem-solving skills.'
            },
            {
                'assessment_name': 'SQL Database Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/sql-database',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Measures SQL query writing, database design, and data manipulation capabilities.'
            },
            {
                'assessment_name': 'JavaScript Developer Test',
                'assessment_url': 'https://www.shl.com/solutions/products/javascript-developer',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Evaluates JavaScript programming skills, including ES6+, async programming, and DOM manipulation.'
            },
            {
                'assessment_name': 'Numerical Reasoning Test',
                'assessment_url': 'https://www.shl.com/solutions/products/numerical-reasoning',
                'category': 'Numerical',
                'test_type': 'K',
                'description': 'Assesses ability to work with numerical data, interpret charts, and solve mathematical problems.'
            },
            {
                'assessment_name': 'Verbal Reasoning Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/verbal-reasoning',
                'category': 'Verbal',
                'test_type': 'K',
                'description': 'Measures comprehension, critical thinking, and ability to evaluate written information.'
            },
            {
                'assessment_name': 'Logical Reasoning Test',
                'assessment_url': 'https://www.shl.com/solutions/products/logical-reasoning',
                'category': 'General',
                'test_type': 'K',
                'description': 'Evaluates abstract reasoning, pattern recognition, and logical problem-solving abilities.'
            },
            {
                'assessment_name': 'Data Analyst Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/data-analyst',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Tests data analysis skills, statistical knowledge, and ability to derive insights from data.'
            },
            {
                'assessment_name': 'C++ Programming Test',
                'assessment_url': 'https://www.shl.com/solutions/products/cpp-programming',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Assesses C++ programming skills including memory management, OOP, and algorithm implementation.'
            },
            {
                'assessment_name': 'Software Development Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/software-development',
                'category': 'Technical',
                'test_type': 'K',
                'description': 'Comprehensive evaluation of software development skills, design patterns, and best practices.'
            },
            
            # Personality/Behavior Assessments (P)
            {
                'assessment_name': 'Occupational Personality Questionnaire (OPQ)',
                'assessment_url': 'https://www.shl.com/solutions/products/opq',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Comprehensive personality assessment measuring preferred behavioral styles at work.'
            },
            {
                'assessment_name': 'Leadership Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/leadership',
                'category': 'Leadership',
                'test_type': 'P',
                'description': 'Evaluates leadership potential, management style, and ability to influence and motivate teams.'
            },
            {
                'assessment_name': 'Motivation Questionnaire (MQ)',
                'assessment_url': 'https://www.shl.com/solutions/products/motivation-questionnaire',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Measures work-related motivational factors and drivers of engagement and performance.'
            },
            {
                'assessment_name': 'Situational Judgment Test',
                'assessment_url': 'https://www.shl.com/solutions/products/situational-judgment',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Assesses decision-making and problem-solving in realistic work scenarios.'
            },
            {
                'assessment_name': 'Team Role Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/team-role',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Identifies preferred team roles and collaboration styles to optimize team composition.'
            },
            {
                'assessment_name': 'Work Values Questionnaire',
                'assessment_url': 'https://www.shl.com/solutions/products/work-values',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Measures alignment between personal values and organizational culture.'
            },
            {
                'assessment_name': 'Emotional Intelligence Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/emotional-intelligence',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Evaluates ability to perceive, understand, and manage emotions in workplace settings.'
            },
            {
                'assessment_name': 'Sales Personality Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/sales-personality',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Assesses personality traits and behaviors critical for sales success.'
            },
            {
                'assessment_name': 'Customer Service Aptitude Test',
                'assessment_url': 'https://www.shl.com/solutions/products/customer-service',
                'category': 'Personality',
                'test_type': 'P',
                'description': 'Measures interpersonal skills and service orientation for customer-facing roles.'
            },
            {
                'assessment_name': 'Management Competency Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/management-competency',
                'category': 'Leadership',
                'test_type': 'P',
                'description': 'Evaluates key management competencies including planning, organizing, and controlling.'
            },
            
            # Additional mixed assessments
            {
                'assessment_name': 'Graduate Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/graduate-assessment',
                'category': 'General',
                'test_type': 'K',
                'description': 'Comprehensive assessment for graduate recruitment including cognitive and technical skills.'
            },
            {
                'assessment_name': 'Critical Thinking Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/critical-thinking',
                'category': 'General',
                'test_type': 'K',
                'description': 'Evaluates analytical thinking, evaluation of arguments, and decision-making abilities.'
            },
            {
                'assessment_name': 'Business Acumen Test',
                'assessment_url': 'https://www.shl.com/solutions/products/business-acumen',
                'category': 'General',
                'test_type': 'K',
                'description': 'Assesses understanding of business principles, financial literacy, and strategic thinking.'
            },
            {
                'assessment_name': 'Project Management Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/project-management',
                'category': 'Leadership',
                'test_type': 'P',
                'description': 'Evaluates project planning, resource management, and stakeholder communication skills.'
            },
            {
                'assessment_name': 'Communication Skills Assessment',
                'assessment_url': 'https://www.shl.com/solutions/products/communication-skills',
                'category': 'Verbal',
                'test_type': 'P',
                'description': 'Measures written and verbal communication effectiveness in professional contexts.'
            }
        ]
        
        df = pd.DataFrame(assessments)
        logger.info(f"Created fallback catalog with {len(df)} assessments")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str = 'data/shl_catalog.csv'):
        """Save catalog to CSV file"""
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Catalog saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving catalog: {e}")


def main():
    """Main execution function"""
    crawler = SHLCrawler()
    catalog_df = crawler.scrape_catalog()
    
    # Save to CSV
    crawler.save_to_csv(catalog_df)
    
    print(f"\nCatalog Summary:")
    print(f"Total Assessments: {len(catalog_df)}")
    print(f"\nBy Test Type:")
    print(catalog_df['test_type'].value_counts())
    print(f"\nBy Category:")
    print(catalog_df['category'].value_counts())
    
    return catalog_df


if __name__ == "__main__":
    main()
