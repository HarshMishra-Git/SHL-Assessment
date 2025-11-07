"""
Recommendation Engine Module

This module implements semantic search using FAISS and cosine similarity
to retrieve the most relevant assessments for a given query.
"""

import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AssessmentRecommender:
    """Recommender system using FAISS and embeddings"""
    
    def __init__(self):
        self.faiss_index = None
        self.embeddings = None
        self.assessment_mapping = {}
        self.embedder = None
        
    def load_index(self,
                  index_path: str = 'models/faiss_index.faiss',
                  embeddings_path: str = 'models/embeddings.npy',
                  mapping_path: str = 'models/mapping.pkl'):
        """Load FAISS index and related artifacts"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")
            
            # Load assessment mapping
            with open(mapping_path, 'rb') as f:
                self.assessment_mapping = pickle.load(f)
            logger.info(f"Loaded {len(self.assessment_mapping)} assessment mappings")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def load_embedder(self):
        """Load the embedding model for query encoding"""
        from src.embedder import EmbeddingGenerator
        
        if self.embedder is None:
            self.embedder = EmbeddingGenerator()
            self.embedder.load_model()
            logger.info("Embedding model loaded")
    
    def search_faiss(self, query_embedding: np.ndarray, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index for similar assessments"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded. Call load_index() first.")
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        return distances[0], indices[0]
    
    def search_cosine(self, query_embedding: np.ndarray, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Search using sklearn cosine similarity"""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_index() first.")
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_scores, top_k_indices
    
    def recommend(self, 
                 query: str, 
                 k: int = 15, 
                 method: str = 'faiss') -> List[Dict]:
        """
        Recommend assessments for a given query
        
        Args:
            query: Job description or query string
            k: Number of recommendations to return
            method: 'faiss' or 'cosine'
        
        Returns:
            List of recommended assessments with scores
        """
        # Load embedder if not loaded
        if self.embedder is None:
            self.load_embedder()
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search based on method
        if method == 'faiss':
            scores, indices = self.search_faiss(query_embedding, k)
        else:
            scores, indices = self.search_cosine(query_embedding, k)
        
        # Build results
        recommendations = []
        for idx, score in zip(indices, scores):
            if idx in self.assessment_mapping:
                assessment = self.assessment_mapping[idx].copy()
                assessment['score'] = float(score)
                assessment['index'] = int(idx)
                recommendations.append(assessment)
        
        logger.info(f"Found {len(recommendations)} recommendations for query")
        return recommendations
    
    def recommend_batch(self,
                       queries: List[str],
                       k: int = 15,
                       method: str = 'faiss') -> List[List[Dict]]:
        """
        Recommend assessments for multiple queries
        
        Args:
            queries: List of job descriptions or query strings
            k: Number of recommendations per query
            method: 'faiss' or 'cosine'
        
        Returns:
            List of recommendation lists
        """
        # Load embedder if not loaded
        if self.embedder is None:
            self.load_embedder()
        
        # Generate query embeddings
        query_embeddings = self.embedder.embed_queries(queries)
        
        # Get recommendations for each query
        all_recommendations = []
        
        for i, query_embedding in enumerate(query_embeddings):
            # Search
            if method == 'faiss':
                scores, indices = self.search_faiss(query_embedding, k)
            else:
                scores, indices = self.search_cosine(query_embedding, k)
            
            # Build results
            recommendations = []
            for idx, score in zip(indices, scores):
                if idx in self.assessment_mapping:
                    assessment = self.assessment_mapping[idx].copy()
                    assessment['score'] = float(score)
                    assessment['index'] = int(idx)
                    recommendations.append(assessment)
            
            all_recommendations.append(recommendations)
        
        logger.info(f"Generated recommendations for {len(queries)} queries")
        return all_recommendations
    
    def get_assessment_by_url(self, url: str) -> Dict:
        """Get assessment details by URL"""
        for idx, assessment in self.assessment_mapping.items():
            if assessment['assessment_url'] == url:
                return assessment
        return None
    
    def get_assessment_by_name(self, name: str) -> Dict:
        """Get assessment details by name"""
        name_lower = name.lower()
        for idx, assessment in self.assessment_mapping.items():
            if assessment['assessment_name'].lower() == name_lower:
                return assessment
        return None


def main():
    """Main execution function"""
    # Initialize recommender
    recommender = AssessmentRecommender()
    
    # Load index
    recommender.load_index()
    
    # Test queries
    test_queries = [
        "Looking for a Java developer with strong programming skills",
        "Need a team leader with excellent communication and management abilities",
        "Seeking a data analyst who can work with SQL and Python",
        "Want to assess personality traits for customer service role"
    ]
    
    print("\n=== Recommendation Test ===\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        # Get recommendations
        recommendations = recommender.recommend(query, k=5, method='faiss')
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   Category: {rec['category']}")
            print(f"   Type: {rec['test_type']}")
            print(f"   Score: {rec['score']:.4f}")
            print(f"   Description: {rec['description'][:100]}...")
    
    return recommender


if __name__ == "__main__":
    main()
