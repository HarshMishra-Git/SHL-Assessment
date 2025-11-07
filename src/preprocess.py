"""
Data Preprocessing Module

This module loads and preprocesses the Gen_AI Dataset.xlsx file,
cleaning queries and creating training mappings.
"""

import pandas as pd
import re
import logging
from typing import Dict, List, Tuple
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses training and test data from Gen_AI Dataset"""
    
    def __init__(self, excel_path: str = 'Data/Gen_AI Dataset.xlsx'):
        self.excel_path = excel_path
        self.train_df = None
        self.test_df = None
        self.train_mapping = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data from Excel file"""
        try:
            logger.info(f"Loading data from {self.excel_path}")
            
            # Read Excel file
            xls = pd.ExcelFile(self.excel_path)
            logger.info(f"Available sheets: {xls.sheet_names}")
            
            # Load Train-Set
            if 'Train-Set' in xls.sheet_names:
                self.train_df = pd.read_excel(self.excel_path, sheet_name='Train-Set')
                logger.info(f"Loaded Train-Set: {self.train_df.shape}")
            else:
                # Try alternative sheet names
                for sheet in xls.sheet_names:
                    if 'train' in sheet.lower():
                        self.train_df = pd.read_excel(self.excel_path, sheet_name=sheet)
                        logger.info(f"Loaded {sheet}: {self.train_df.shape}")
                        break
            
            # Load Test-Set
            if 'Test-Set' in xls.sheet_names:
                self.test_df = pd.read_excel(self.excel_path, sheet_name='Test-Set')
                logger.info(f"Loaded Test-Set: {self.test_df.shape}")
            else:
                # Try alternative sheet names
                for sheet in xls.sheet_names:
                    if 'test' in sheet.lower():
                        self.test_df = pd.read_excel(self.excel_path, sheet_name=sheet)
                        logger.info(f"Loaded {sheet}: {self.test_df.shape}")
                        break
            
            # If no sheets found, try to load all data from first sheet
            if self.train_df is None:
                logger.warning("No train sheet found, loading from first sheet")
                self.train_df = pd.read_excel(self.excel_path, sheet_name=0)
            
            return self.train_df, self.test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # Find URLs in text
        url_pattern = r'https?://[^\s,]+'
        urls = re.findall(url_pattern, text)
        
        return urls
    
    def parse_assessment_urls(self, url_column) -> List[str]:
        """Parse assessment URLs from various formats"""
        urls = []
        
        if pd.isna(url_column):
            return urls
        
        # If it's a string
        if isinstance(url_column, str):
            # Split by common separators
            parts = re.split(r'[,;\n\|]', url_column)
            for part in parts:
                part = part.strip()
                if 'http' in part or 'shl.com' in part:
                    urls.append(part)
                # Extract URLs from text
                extracted = self.extract_urls_from_text(part)
                urls.extend(extracted)
        
        # Remove duplicates and clean
        urls = list(set([url.strip() for url in urls if url]))
        
        return urls
    
    def create_train_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from query to assessment URLs"""
        if self.train_df is None:
            logger.error("Train data not loaded")
            return {}
        
        logger.info("Creating train mapping...")
        self.train_mapping = {}
        
        # Identify query and URL columns
        # Common column names to look for
        query_cols = ['query', 'job_description', 'jd', 'description', 'text', 'job query']
        url_cols = ['urls', 'assessment_urls', 'relevant_assessments', 'assessments', 'links']
        
        query_col = None
        url_col = None
        
        # Find query column
        for col in self.train_df.columns:
            col_lower = col.lower()
            if any(qc in col_lower for qc in query_cols):
                query_col = col
                logger.info(f"Found query column: {query_col}")
                break
        
        # Find URL column
        for col in self.train_df.columns:
            col_lower = col.lower()
            if any(uc in col_lower for uc in url_cols):
                url_col = col
                logger.info(f"Found URL column: {url_col}")
                break
        
        # If columns not found, use first two columns
        if query_col is None and len(self.train_df.columns) > 0:
            query_col = self.train_df.columns[0]
            logger.warning(f"Query column not identified, using: {query_col}")
        
        if url_col is None and len(self.train_df.columns) > 1:
            url_col = self.train_df.columns[1]
            logger.warning(f"URL column not identified, using: {url_col}")
        
        # Create mapping
        for idx, row in self.train_df.iterrows():
            query = self.clean_text(str(row[query_col]))
            
            if not query:
                continue
            
            # Parse URLs
            urls = self.parse_assessment_urls(row[url_col])
            
            # Store mapping
            if urls:
                self.train_mapping[query] = urls
        
        logger.info(f"Created {len(self.train_mapping)} query-URL mappings")
        return self.train_mapping
    
    def get_all_queries(self) -> Tuple[List[str], List[str]]:
        """Get all queries from train and test sets"""
        train_queries = []
        test_queries = []
        
        if self.train_df is not None:
            # Find query column
            query_col = None
            for col in self.train_df.columns:
                if any(qc in col.lower() for qc in ['query', 'job', 'description', 'text']):
                    query_col = col
                    break
            
            if query_col is None:
                query_col = self.train_df.columns[0]
            
            train_queries = [
                self.clean_text(str(q)) 
                for q in self.train_df[query_col] 
                if not pd.isna(q)
            ]
        
        if self.test_df is not None:
            # Find query column
            query_col = None
            for col in self.test_df.columns:
                if any(qc in col.lower() for qc in ['query', 'job', 'description', 'text']):
                    query_col = col
                    break
            
            if query_col is None:
                query_col = self.test_df.columns[0]
            
            test_queries = [
                self.clean_text(str(q)) 
                for q in self.test_df[query_col] 
                if not pd.isna(q)
            ]
        
        logger.info(f"Extracted {len(train_queries)} train queries and {len(test_queries)} test queries")
        return train_queries, test_queries
    
    def preprocess(self) -> Dict:
        """Main preprocessing pipeline"""
        # Load data
        self.load_data()
        
        # Create train mapping
        self.create_train_mapping()
        
        # Get all queries
        train_queries, test_queries = self.get_all_queries()
        
        # Summary
        logger.info("Preprocessing complete:")
        logger.info(f"  Train queries: {len(train_queries)}")
        logger.info(f"  Test queries: {len(test_queries)}")
        logger.info(f"  Train mappings: {len(self.train_mapping)}")
        
        return {
            'train_queries': train_queries,
            'test_queries': test_queries,
            'train_mapping': self.train_mapping,
            'train_df': self.train_df,
            'test_df': self.test_df
        }


def main():
    """Main execution function"""
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess()
    
    print("\n=== Preprocessing Summary ===")
    print(f"Train queries: {len(result['train_queries'])}")
    print(f"Test queries: {len(result['test_queries'])}")
    print(f"Train mappings: {len(result['train_mapping'])}")
    
    # Show sample
    if result['train_queries']:
        print(f"\nSample train query: {result['train_queries'][0][:100]}...")
    
    if result['train_mapping']:
        sample_key = list(result['train_mapping'].keys())[0]
        print(f"\nSample mapping:")
        print(f"  Query: {sample_key[:80]}...")
        print(f"  URLs: {result['train_mapping'][sample_key][:2]}")
    
    return result


if __name__ == "__main__":
    main()
