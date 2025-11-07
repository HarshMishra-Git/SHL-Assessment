#!/usr/bin/env python3
"""
Test script for SHL Assessment Recommender System

Tests basic functionality without requiring full model downloads.
"""

import sys
import os


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas
        import numpy
        import sklearn
        from bs4 import BeautifulSoup
        import requests
        print("✓ Data processing packages")
    except ImportError as e:
        print(f"✗ Data processing packages: {e}")
        return False
    
    try:
        from src import crawler, preprocess
        print("✓ Core modules (crawler, preprocess)")
    except ImportError as e:
        print(f"✗ Core modules: {e}")
        return False
    
    try:
        import fastapi
        import uvicorn
        import streamlit
        print("✓ API and UI packages")
    except ImportError as e:
        print(f"✗ API and UI packages: {e}")
        return False
    
    return True


def test_data_files():
    """Test that required data files exist"""
    print("\nTesting data files...")
    
    # Check training data
    if os.path.exists('Data/Gen_AI Dataset.xlsx'):
        print("✓ Training dataset found")
    else:
        print("✗ Training dataset not found (Data/Gen_AI Dataset.xlsx)")
    
    # Check catalog
    if os.path.exists('data/shl_catalog.csv'):
        print("✓ SHL catalog found")
        
        import pandas as pd
        df = pd.read_csv('data/shl_catalog.csv')
        print(f"  - {len(df)} assessments")
        print(f"  - K assessments: {len(df[df['test_type'] == 'K'])}")
        print(f"  - P assessments: {len(df[df['test_type'] == 'P'])}")
    else:
        print("⚠ SHL catalog not found (run: python src/crawler.py)")
    
    return True


def test_crawler():
    """Test the crawler module"""
    print("\nTesting crawler...")
    
    try:
        from src.crawler import SHLCrawler
        
        crawler = SHLCrawler()
        
        # Test text classification
        assert crawler.determine_test_type("Java programming test") == "K"
        assert crawler.determine_test_type("Personality assessment") == "P"
        print("✓ Test type classification works")
        
        # Test category extraction
        cat = crawler.extract_category("Leadership management")
        assert cat == "Leadership"
        print("✓ Category extraction works")
        
        return True
    except Exception as e:
        print(f"✗ Crawler test failed: {e}")
        return False


def test_preprocessor():
    """Test the preprocessor module"""
    print("\nTesting preprocessor...")
    
    try:
        from src.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Test text cleaning
        clean = preprocessor.clean_text("  Hello, WORLD!  ")
        assert clean == "hello, world!"
        print("✓ Text cleaning works")
        
        # Test URL extraction
        urls = preprocessor.extract_urls_from_text("Check https://example.com and http://test.com")
        assert len(urls) == 2
        print("✓ URL extraction works")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessor test failed: {e}")
        return False


def test_api_structure():
    """Test that API is properly structured"""
    print("\nTesting API structure...")
    
    try:
        from api.main import app
        
        # Check endpoints exist
        routes = [route.path for route in app.routes]
        
        assert "/health" in routes
        print("✓ /health endpoint exists")
        
        assert "/recommend" in routes
        print("✓ /recommend endpoint exists")
        
        return True
    except Exception as e:
        print(f"✗ API structure test failed: {e}")
        return False


def test_streamlit_app():
    """Test that Streamlit app can be imported"""
    print("\nTesting Streamlit app...")
    
    try:
        # Just check the file exists and is valid Python
        with open('app.py', 'r') as f:
            content = f.read()
        
        assert 'st.set_page_config' in content
        print("✓ Streamlit app file valid")
        
        assert 'SHL Assessment Recommender' in content
        print("✓ App title configured")
        
        return True
    except Exception as e:
        print(f"✗ Streamlit app test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("SHL ASSESSMENT RECOMMENDER - BASIC TESTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Crawler", test_crawler),
        ("Preprocessor", test_preprocessor),
        ("API Structure", test_api_structure),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All basic tests passed!")
        print("\nNote: Full system tests require:")
        print("  - Internet connection (for model downloads)")
        print("  - Running: python setup.py")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
