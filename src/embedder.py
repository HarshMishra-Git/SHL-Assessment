"""
Embedding Generation Module

This module generates embeddings for assessments and queries using
Hugging Face sentence transformers and creates a FAISS index for fast retrieval.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import os
from typing import List, Dict, Tuple
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings and creates FAISS index"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.faiss_index = None
        self.embeddings = None
        self.catalog_df = None
        self.assessment_mapping = {}
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_catalog(self, catalog_path: str = 'data/shl_catalog.csv') -> pd.DataFrame:
        """Load the SHL catalog"""
        try:
            self.catalog_df = pd.read_csv(catalog_path)
            logger.info(f"Loaded catalog with {len(self.catalog_df)} assessments")
            return self.catalog_df
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            raise
    
    def create_assessment_texts(self) -> List[str]:
        """Create text representations of assessments for embedding"""
        texts = []
        
        for idx, row in self.catalog_df.iterrows():
            # Combine relevant fields for embedding
            text_parts = []
            
            if pd.notna(row['assessment_name']):
                text_parts.append(str(row['assessment_name']))
            
            if pd.notna(row['category']):
                text_parts.append(f"Category: {row['category']}")
            
            if pd.notna(row['test_type']):
                type_full = 'Knowledge/Skill' if row['test_type'] == 'K' else 'Personality/Behavior'
                text_parts.append(f"Type: {type_full}")
            
            if pd.notna(row['description']):
                text_parts.append(str(row['description']))
            
            text = ' | '.join(text_parts)
            texts.append(text)
            
            # Create mapping from index to assessment details
            self.assessment_mapping[idx] = {
                'assessment_name': row['assessment_name'],
                'assessment_url': row['assessment_url'],
                'category': row['category'],
                'test_type': row['test_type'],
                'description': row['description']
            }
        
        logger.info(f"Created {len(texts)} assessment texts")
        return texts
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index for fast similarity search"""
        try:
            logger.info("Creating FAISS index...")
            
            # Dimensions of embeddings
            dimension = embeddings.shape[1]
            
            # Create index - using IndexFlatIP for inner product (cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS index created with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def save_artifacts(self, 
                      index_path: str = 'models/faiss_index.faiss',
                      embeddings_path: str = 'models/embeddings.npy',
                      mapping_path: str = 'models/mapping.pkl'):
        """Save FAISS index, embeddings, and mapping"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, index_path)
            logger.info(f"FAISS index saved to {index_path}")
            
            # Save embeddings
            np.save(embeddings_path, self.embeddings)
            logger.info(f"Embeddings saved to {embeddings_path}")
            
            # Save mapping
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.assessment_mapping, f)
            logger.info(f"Assessment mapping saved to {mapping_path}")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise
    
    def load_artifacts(self,
                      index_path: str = 'models/faiss_index.faiss',
                      embeddings_path: str = 'models/embeddings.npy',
                      mapping_path: str = 'models/mapping.pkl'):
        """Load FAISS index, embeddings, and mapping"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded from {index_path}")
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Embeddings loaded from {embeddings_path}")
            
            # Load mapping
            with open(mapping_path, 'rb') as f:
                self.assessment_mapping = pickle.load(f)
            logger.info(f"Assessment mapping loaded from {mapping_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            return False
    
    def build_index(self, catalog_path: str = 'data/shl_catalog.csv'):
        """Main method to build the complete index"""
        # Load catalog
        self.load_catalog(catalog_path)
        
        # Create assessment texts
        assessment_texts = self.create_assessment_texts()
        
        # Generate embeddings
        self.embeddings = self.generate_embeddings(assessment_texts)
        
        # Create FAISS index
        self.faiss_index = self.create_faiss_index(self.embeddings)
        
        # Save artifacts
        self.save_artifacts()
        
        logger.info("Index building complete!")
        
        return self.faiss_index, self.embeddings, self.assessment_mapping
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding[0]
    
    def embed_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple queries"""
        return self.generate_embeddings(queries, batch_size)


def main():
    """Main execution function"""
    # Initialize embedder
    embedder = EmbeddingGenerator()
    
    # Build index
    index, embeddings, mapping = embedder.build_index()
    
    print("\n=== Embedding Generation Summary ===")
    print(f"Total assessments indexed: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Assessment mapping entries: {len(mapping)}")
    
    # Test with a sample query
    test_query = "Looking for a Java developer with strong programming skills"
    query_embedding = embedder.embed_query(test_query)
    print(f"\nTest query embedding shape: {query_embedding.shape}")
    
    # Search test
    k = 5
    distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    
    print(f"\nTop {k} matches for test query:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        assessment = mapping[idx]
        print(f"\n{i+1}. {assessment['assessment_name']}")
        print(f"   Score: {dist:.4f}")
        print(f"   Type: {assessment['test_type']}")
    
    return embedder


if __name__ == "__main__":
    main()
