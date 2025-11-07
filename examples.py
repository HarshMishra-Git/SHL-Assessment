#!/usr/bin/env python3
"""
Example usage script for SHL Assessment Recommender System

This script demonstrates how to use the system programmatically.
"""

import sys
import os


def example_direct_usage():
    """Example: Using the recommender directly (without API)"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Direct Usage (Python)")
    print("="*60)
    
    from src.recommender import AssessmentRecommender
    from src.reranker import AssessmentReranker
    
    # Initialize recommender
    print("\nLoading recommender system...")
    recommender = AssessmentRecommender()
    
    # Load index
    if not recommender.load_index():
        print("Error: Please run 'python setup.py' first to build the index")
        return
    
    # Initialize reranker
    reranker = AssessmentReranker()
    
    # Example query
    query = "Looking for a Java developer who can lead a small team"
    print(f"\nQuery: {query}")
    
    # Get initial candidates
    print("\nGetting initial candidates...")
    candidates = recommender.recommend(query, k=15, method='faiss')
    
    # Rerank and balance
    print("Applying reranking and balancing...")
    results = reranker.rerank_and_balance(
        query=query,
        candidates=candidates,
        top_k=10,
        min_k=1,
        min_p=1
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Recommendations:")
    print('='*60)
    
    for assessment in results:
        print(f"\n{assessment['rank']}. {assessment['assessment_name']}")
        print(f"   Type: {assessment['test_type']}")
        print(f"   Category: {assessment['category']}")
        print(f"   Score: {assessment.get('score', 0):.4f}")
        print(f"   URL: {assessment['assessment_url']}")


def example_api_client():
    """Example: Using the API client"""
    print("\n" + "="*60)
    print("EXAMPLE 2: API Client Usage")
    print("="*60)
    
    import requests
    import json
    
    # API URL (assumes API is running)
    api_url = "http://localhost:8000"
    
    # Check health
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ API is running: {response.json()}")
        else:
            print(f"   ✗ API returned status {response.status_code}")
            print("   Please start the API: python api/main.py")
            return
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Cannot connect to API: {e}")
        print("   Please start the API: python api/main.py")
        return
    
    # Get recommendations
    print("\n2. Getting recommendations...")
    
    query = "Need a data analyst with SQL and Python skills"
    print(f"   Query: {query}")
    
    payload = {
        "query": query,
        "num_results": 5,
        "use_reranking": True,
        "min_k": 1,
        "min_p": 1
    }
    
    response = requests.post(
        f"{api_url}/recommend",
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n{'='*60}")
        print(f"Recommendations for: {result['query']}")
        print('='*60)
        
        for rec in result['recommendations']:
            print(f"\n{rec['rank']}. {rec['assessment_name']}")
            print(f"   Type: {rec['test_type']}")
            print(f"   Category: {rec['category']}")
            print(f"   Score: {rec['score']:.2%}")
    else:
        print(f"   ✗ Error: {response.status_code}")
        print(f"   {response.text}")


def example_batch_processing():
    """Example: Batch processing multiple queries"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    from src.recommender import AssessmentRecommender
    
    # Initialize recommender
    print("\nLoading recommender system...")
    recommender = AssessmentRecommender()
    
    if not recommender.load_index():
        print("Error: Please run 'python setup.py' first")
        return
    
    # Multiple queries
    queries = [
        "Java developer with team leadership",
        "Python data scientist",
        "Customer service representative",
        "Software engineer with problem-solving skills"
    ]
    
    print(f"\nProcessing {len(queries)} queries...")
    
    # Get recommendations for all queries
    all_recommendations = recommender.recommend_batch(queries, k=5)
    
    # Display results
    for query, recommendations in zip(queries, all_recommendations):
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('-'*60)
        
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            print(f"{i}. {rec['assessment_name']} ({rec['test_type']}) - {rec['score']:.4f}")


def example_custom_filtering():
    """Example: Custom filtering and post-processing"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Filtering")
    print("="*60)
    
    from src.recommender import AssessmentRecommender
    
    recommender = AssessmentRecommender()
    
    if not recommender.load_index():
        print("Error: Please run 'python setup.py' first")
        return
    
    query = "Software developer position"
    print(f"\nQuery: {query}")
    
    # Get recommendations
    recommendations = recommender.recommend(query, k=20)
    
    # Filter for only technical assessments
    technical = [r for r in recommendations if r['category'] == 'Technical']
    
    print(f"\nAll recommendations: {len(recommendations)}")
    print(f"Technical only: {len(technical)}")
    
    print("\nTechnical Assessments:")
    for i, rec in enumerate(technical[:5], 1):
        print(f"{i}. {rec['assessment_name']} - Score: {rec['score']:.4f}")
    
    # Filter for only K-type assessments
    k_type = [r for r in recommendations if r['test_type'] == 'K']
    
    print(f"\nKnowledge/Skill Assessments: {len(k_type)}")
    for i, rec in enumerate(k_type[:5], 1):
        print(f"{i}. {rec['assessment_name']} - {rec['category']}")


def example_evaluation():
    """Example: Running evaluation"""
    print("\n" + "="*60)
    print("EXAMPLE 5: System Evaluation")
    print("="*60)
    
    from src.evaluator import RecommenderEvaluator
    from src.recommender import AssessmentRecommender
    from src.preprocess import DataPreprocessor
    
    # Load data
    print("\nLoading training data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()
    train_mapping = data['train_mapping']
    
    if not train_mapping:
        print("No training data available")
        return
    
    print(f"Found {len(train_mapping)} training queries")
    
    # Load recommender
    print("\nLoading recommender...")
    recommender = AssessmentRecommender()
    if not recommender.load_index():
        print("Error: Please run 'python setup.py' first")
        return
    
    # Run evaluation
    print("\nRunning evaluation (this may take a moment)...")
    evaluator = RecommenderEvaluator()
    results = evaluator.evaluate(recommender, train_mapping, k=10)
    
    # Print report
    evaluator.print_report()


def main():
    """Main function - run all examples"""
    examples = [
        ("Direct Usage", example_direct_usage),
        ("API Client", example_api_client),
        ("Batch Processing", example_batch_processing),
        ("Custom Filtering", example_custom_filtering),
        ("Evaluation", example_evaluation)
    ]
    
    print("="*60)
    print("SHL ASSESSMENT RECOMMENDER - USAGE EXAMPLES")
    print("="*60)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nSelect an example (1-5) or 'all' to run all:")
    print("(Press Enter to run Example 1)")
    
    choice = input("> ").strip().lower()
    
    if not choice:
        choice = "1"
    
    if choice == "all":
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        try:
            examples[idx][1]()
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print("Invalid choice")
        return 1
    
    print("\n" + "="*60)
    print("For more information, see README.md")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
