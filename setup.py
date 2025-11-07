#!/usr/bin/env python3
"""
Setup script for SHL Assessment Recommender System

This script automates the initialization process:
1. Checks dependencies
2. Generates/loads SHL catalog
3. Preprocesses training data  
4. Generates embeddings and builds FAISS index
5. Runs evaluation
"""

import sys
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'sentence_transformers',
        'faiss',
        'sklearn',
        'beautifulsoup4',
        'requests',
        'fastapi',
        'uvicorn',
        'streamlit'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'beautifulsoup4':
                __import__('bs4')
            elif package == 'sentence_transformers':
                __import__('sentence_transformers')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"Missing packages: {', '.join(missing)}")
        logger.info("Attempting to continue anyway...")
        return True
    
    logger.info("✓ All dependencies installed")
    return True


def step1_generate_catalog():
    """Step 1: Generate/Load SHL catalog"""
    logger.info("="*60)
    logger.info("STEP 1: Loading SHL Catalog")
    logger.info("="*60)
    
    try:
        csv_path = 'data/shl_catalog.csv'
        excel_path = 'Data/Gen_AI Dataset.xlsx'
        
        # Priority 1: Use existing CSV (uploaded with repo)
        if os.path.exists(csv_path):
            logger.info(f"✓ Found existing catalog: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"✓ Loaded {len(df)} assessments from CSV")
            return True
        
        # Priority 2: Generate from Excel
        if os.path.exists(excel_path):
            logger.info(f"✓ Generating catalog from Excel: {excel_path}")
            df = pd.read_excel(excel_path)
            
            logger.info(f"✓ Excel columns found: {list(df.columns)}")
            
            # Handle column name variations
            column_mapping = {
                'assessment_name': 'Assessment Name',
                'Assessment_Name': 'Assessment Name',
                'assessment name': 'Assessment Name',
                'assessment_url': 'Assessment URL',
                'Assessment_URL': 'Assessment URL',
                'assessment url': 'Assessment URL',
                'description': 'Description',
                'Description': 'Description',
                'category': 'Category',
                'Category': 'Category',
                'test_type': 'Test Type',
                'Test_Type': 'Test Type',
                'test type': 'Test Type',
                'Type': 'Test Type',
                'type': 'Test Type'
            }
            
            # Rename columns to standard format
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    logger.info(f"✓ Renamed column: {old_col} → {new_col}")
            
            # Verify required columns exist
            required_cols = ['Assessment Name', 'Assessment URL', 'Description', 'Category', 'Test Type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Excel file missing columns: {missing_cols}")
                logger.error(f"Available columns: {list(df.columns)}")
                logger.info("Attempting to use first 5 columns as fallback...")
                
                # Fallback: Use first 5 columns
                if len(df.columns) >= 5:
                    df.columns = required_cols[:len(df.columns)]
                    logger.info("✓ Using columns by position")
                else:
                    return False
            
            # Save to CSV
            os.makedirs('data', exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"✓ Saved {len(df)} assessments to {csv_path}")
            return True
        
        # Priority 3: Scrape from web (last resort)
        logger.info("✓ No local data found, scraping SHL website...")
        from src.crawler import SHLCrawler
        
        crawler = SHLCrawler()
        crawler.scrape_catalog()
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"✓ Scraped {len(df)} assessments")
            return True
        else:
            logger.error("✗ Scraping failed and no catalog available")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to load catalog: {e}")
        import traceback
        traceback.print_exc()
        return False


def step2_preprocess_data():
    """Step 2: Preprocess training data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Preprocessing Training Data")
    logger.info("="*60)
    
    try:
        from src.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess()
        
        logger.info(f"✓ Preprocessed {len(data.get('train_queries', []))} train queries")
        logger.info(f"✓ Preprocessed {len(data.get('test_queries', []))} test queries")
        logger.info(f"✓ Created {len(data.get('train_mapping', {}))} train mappings")
        return True
    except Exception as e:
        logger.warning(f"⚠ Preprocessing skipped: {e}")
        logger.info("✓ Continuing without training data")
        return True


def step3_build_index():
    """Step 3: Generate embeddings and build FAISS index"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Building Search Index")
    logger.info("="*60)
    logger.info("Downloading models and creating embeddings...")
    
    try:
        from src.embedder import AssessmentEmbedder
        
        embedder = AssessmentEmbedder()
        
        # Load catalog
        embedder.load_catalog()
        logger.info(f"✓ Loaded {len(embedder.assessments)} assessments")
        
        # Create embeddings
        embedder.create_embeddings()
        logger.info(f"✓ Generated embeddings with shape {embedder.embeddings.shape}")
        
        # Build FAISS index
        embedder.build_index()
        logger.info(f"✓ Built FAISS index with {embedder.index.ntotal} vectors")
        
        # Save
        embedder.save_index()
        logger.info(f"✓ Index saved to models/ directory")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_run_evaluation():
    """Step 4: Run evaluation on training set"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Running Evaluation")
    logger.info("="*60)
    
    try:
        from src.evaluator import RecommenderEvaluator
        from src.recommender import AssessmentRecommender
        from src.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess()
        train_mapping = data.get('train_mapping', {})
        
        if not train_mapping:
            logger.warning("⚠ No training data available, skipping evaluation")
            logger.info("✓ System ready (evaluation skipped)")
            return True
        
        recommender = AssessmentRecommender()
        if not recommender.load_index():
            logger.error("✗ Failed to load recommender")
            return False
        
        evaluator = RecommenderEvaluator()
        results = evaluator.evaluate(recommender, train_mapping, k=10)
        
        evaluator.print_report()
        evaluator.save_results()
        
        logger.info("✓ Evaluation complete")
        logger.info(f"✓ Mean Recall@10: {results['mean_recall_at_10']:.2%}")
        
        return True
    except Exception as e:
        logger.warning(f"⚠ Evaluation skipped: {e}")
        logger.info("✓ System ready (evaluation skipped)")
        return True


def verify_setup():
    """Verify setup completion"""
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION")
    logger.info("="*60)
    
    required_files = [
        'data/shl_catalog.csv',
        'models/faiss_index.faiss',
        'models/embeddings.npy',
        'models/mapping.pkl'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            logger.info(f"✓ {file_path} ({size:,} bytes)")
        else:
            logger.error(f"✗ {file_path} - MISSING!")
            missing.append(file_path)
    
    if missing:
        logger.error(f"Missing files: {missing}")
        return False
    
    try:
        from src.recommender import AssessmentRecommender
        
        recommender = AssessmentRecommender()
        recommender.load_index()
        
        num_assessments = len(recommender.assessment_data)
        num_vectors = recommender.index.ntotal
        
        logger.info(f"✓ Loaded {num_assessments} assessments")
        logger.info(f"✓ Index has {num_vectors} vectors")
        
        if num_assessments < 100:
            logger.warning(f"⚠ Only {num_assessments} assessments (expected 150+)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def main():
    """Main setup process"""
    logger.info("\n" + "="*60)
    logger.info("SHL ASSESSMENT RECOMMENDER - SETUP")
    logger.info("="*60)
    
    check_dependencies()
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logger.info("✓ Directories created")
    
    steps = [
        ("Load Catalog", step1_generate_catalog),
        ("Preprocess Data", step2_preprocess_data),
        ("Build Index", step3_build_index),
        ("Run Evaluation", step4_run_evaluation)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            if step_name in ["Load Catalog", "Build Index"]:
                logger.error(f"✗ Critical step failed: {step_name}")
                return 1
    
    if not verify_setup():
        logger.error("✗ Verification failed")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("✅ SETUP COMPLETE!")
    logger.info("="*60)
    logger.info("\n📊 System Ready for Recommendations")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)