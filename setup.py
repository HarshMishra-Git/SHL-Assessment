# #!/usr/bin/env python3
# """
# Setup script for SHL Assessment Recommender System

# This script automates the initialization process:
# 1. Generates SHL catalog
# 2. Preprocesses training data  
# 3. Generates embeddings and builds FAISS index
# 4. Runs evaluation
# """

# import sys
# import os
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# def check_dependencies():
#     """Check if all required packages are installed"""
#     required_packages = [
#         'pandas',
#         'numpy',
#         'torch',
#         'transformers',
#         'sentence_transformers',
#         'faiss',
#         'sklearn',
#         'beautifulsoup4',
#         'requests',
#         'fastapi',
#         'uvicorn',
#         'streamlit'
#     ]
    
#     missing = []
#     for package in required_packages:
#         try:
#             if package == 'sklearn':
#                 __import__('sklearn')
#             elif package == 'beautifulsoup4':
#                 __import__('bs4')
#             elif package == 'sentence_transformers':
#                 __import__('sentence_transformers')
#             else:
#                 __import__(package)
#         except ImportError:
#             missing.append(package)
    
#     if missing:
#         logger.error(f"Missing packages: {', '.join(missing)}")
#         logger.info("Please install requirements: pip install -r requirements.txt")
#         return False
    
#     logger.info("✓ All dependencies installed")
#     return True


# def step1_generate_catalog():
#     """Step 1: Generate SHL catalog"""
#     logger.info("="*60)
#     logger.info("STEP 1: Generating SHL Catalog")
#     logger.info("="*60)
    
#     try:
#         from src.crawler import SHLCrawler
        
#         crawler = SHLCrawler()
#         catalog_df = crawler.scrape_catalog()
#         crawler.save_to_csv(catalog_df)
        
#         logger.info(f"✓ Catalog generated with {len(catalog_df)} assessments")
#         return True
#     except Exception as e:
#         logger.error(f"✗ Failed to generate catalog: {e}")
#         return False


# def step2_preprocess_data():
#     """Step 2: Preprocess training data"""
#     logger.info("\n" + "="*60)
#     logger.info("STEP 2: Preprocessing Training Data")
#     logger.info("="*60)
    
#     try:
#         from src.preprocess import DataPreprocessor
        
#         preprocessor = DataPreprocessor()
#         data = preprocessor.preprocess()
        
#         logger.info(f"✓ Preprocessed {len(data['train_queries'])} train queries")
#         logger.info(f"✓ Preprocessed {len(data['test_queries'])} test queries")
#         logger.info(f"✓ Created {len(data['train_mapping'])} train mappings")
#         return True
#     except Exception as e:
#         logger.error(f"✗ Failed to preprocess data: {e}")
#         logger.warning("This is expected if Gen_AI Dataset.xlsx is not available")
#         return True  # Continue anyway


# def step3_build_index():
#     """Step 3: Generate embeddings and build FAISS index"""
#     logger.info("\n" + "="*60)
#     logger.info("STEP 3: Building Search Index")
#     logger.info("="*60)
#     logger.info("This may take a few minutes on first run (downloading models)...")
    
#     try:
#         from src.embedder import EmbeddingGenerator
        
#         embedder = EmbeddingGenerator()
#         index, embeddings, mapping = embedder.build_index()
        
#         logger.info(f"✓ Index built with {index.ntotal} vectors")
#         logger.info(f"✓ Embedding dimension: {embeddings.shape[1]}")
#         logger.info(f"✓ Files saved to models/ directory")
#         return True
#     except Exception as e:
#         logger.error(f"✗ Failed to build index: {e}")
#         return False


# def step4_run_evaluation():
#     """Step 4: Run evaluation on training set"""
#     logger.info("\n" + "="*60)
#     logger.info("STEP 4: Running Evaluation")
#     logger.info("="*60)
    
#     try:
#         from src.evaluator import RecommenderEvaluator
#         from src.recommender import AssessmentRecommender
#         from src.preprocess import DataPreprocessor
        
#         # Load data
#         preprocessor = DataPreprocessor()
#         data = preprocessor.preprocess()
#         train_mapping = data['train_mapping']
        
#         if not train_mapping:
#             logger.warning("No training data available, skipping evaluation")
#             return True
        
#         # Load recommender
#         recommender = AssessmentRecommender()
#         if not recommender.load_index():
#             logger.error("Failed to load recommender index")
#             return False
        
#         # Evaluate
#         evaluator = RecommenderEvaluator()
#         results = evaluator.evaluate(recommender, train_mapping, k=10)
        
#         # Print report
#         evaluator.print_report()
        
#         # Save results
#         evaluator.save_results()
        
#         logger.info("✓ Evaluation complete")
#         return True
#     except Exception as e:
#         logger.error(f"✗ Failed to run evaluation: {e}")
#         logger.warning("This is expected if training data is not available")
#         return True  # Continue anyway


# def main():
#     """Main setup process"""
#     logger.info("\n" + "="*60)
#     logger.info("SHL ASSESSMENT RECOMMENDER - SETUP")
#     logger.info("="*60)
    
#     # Check dependencies
#     if not check_dependencies():
#         logger.error("Setup aborted due to missing dependencies")
#         return 1
    
#     # Create directories
#     os.makedirs('data', exist_ok=True)
#     os.makedirs('models', exist_ok=True)
#     logger.info("✓ Directories created")
    
#     # Run setup steps
#     steps = [
#         ("Generate Catalog", step1_generate_catalog),
#         ("Preprocess Data", step2_preprocess_data),
#         ("Build Index", step3_build_index),
#         ("Run Evaluation", step4_run_evaluation)
#     ]
    
#     for step_name, step_func in steps:
#         if not step_func():
#             logger.error(f"Setup failed at step: {step_name}")
#             return 1
    
#     # Summary
#     logger.info("\n" + "="*60)
#     logger.info("SETUP COMPLETE!")
#     logger.info("="*60)
#     logger.info("\nNext steps:")
#     logger.info("  1. Start the API: python api/main.py")
#     logger.info("  2. Or start the UI: streamlit run app.py")
#     logger.info("\nFor more information, see README.md")
    
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*50)
    logger.info("SHL Assessment Recommender - Setup")
    logger.info("="*50)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Prepare catalog
    logger.info("\n📊 Step 1: Preparing assessment catalog...")
    
    csv_path = 'data/shl_catalog.csv'
    excel_path = 'Data/Gen_AI Dataset.xlsx'
    
    # Check if CSV already exists (uploaded with repo)
    if os.path.exists(csv_path):
        logger.info(f"✅ Found existing catalog: {csv_path}")
        df = pd.read_csv(csv_path)
    
    # Otherwise, create from Excel
    elif os.path.exists(excel_path):
        logger.info(f"📊 Creating catalog from Excel: {excel_path}")
        df = pd.read_excel(excel_path)
        df.to_csv(csv_path, index=False)
    
    # Last resort: scrape
    else:
        logger.info("🔍 No local data found, scraping SHL website...")
        from src.crawler import SHLCrawler
        crawler = SHLCrawler()
        crawler.scrape_catalog()
        df = pd.read_csv(csv_path)
    
    logger.info(f"✅ Catalog ready with {len(df)} assessments")
    
    # Step 2: Create embeddings and index
    logger.info("\n🔮 Step 2: Building search index...")
    
    from src.embedder import AssessmentEmbedder
    
    embedder = AssessmentEmbedder()
    embedder.load_catalog()
    
    logger.info(f"📚 Loaded {len(embedder.assessments)} assessments")
    
    embedder.create_embeddings()
    embedder.build_index()
    embedder.save_index()
    
    logger.info(f"✅ Index built successfully")
    
    # Step 3: Verify
    logger.info("\n✅ Step 3: Verifying setup...")
    
    from src.recommender import AssessmentRecommender
    
    recommender = AssessmentRecommender()
    recommender.load_index()
    
    logger.info(f"✅ Loaded {len(recommender.assessment_data)} assessments")
    logger.info(f"✅ Index has {recommender.index.ntotal} vectors")
    
    logger.info("\n" + "="*50)
    logger.info("✅ SETUP COMPLETE!")
    logger.info(f"✅ {len(recommender.assessment_data)} assessments ready")
    logger.info("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)