"""
Reranking Module

This module uses a cross-encoder model to rerank initial recommendations
and ensures balance between Knowledge (K) and Personality (P) assessments.
"""

import numpy as np
from typing import List, Dict
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AssessmentReranker:
    """Reranks recommendations using cross-encoder and ensures K/P balance"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Reranker using device: {self.device}")
        
    def load_model(self):
        """Load the cross-encoder model"""
        try:
            logger.info(f"Loading reranking model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Reranking model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def compute_cross_encoder_score(self, query: str, assessment_text: str) -> float:
        """Compute relevance score using cross-encoder"""
        if self.model is None:
            self.load_model()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                query,
                assessment_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get score
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[0][0].item()
            
            return score
            
        except Exception as e:
            logger.warning(f"Error computing cross-encoder score: {e}")
            return 0.0
    
    def create_assessment_text(self, assessment: Dict) -> str:
        """Create text representation of assessment for reranking"""
        parts = []
        
        if 'assessment_name' in assessment:
            parts.append(assessment['assessment_name'])
        
        if 'category' in assessment:
            parts.append(f"Category: {assessment['category']}")
        
        if 'test_type' in assessment:
            type_full = 'Knowledge/Skill Assessment' if assessment['test_type'] == 'K' else 'Personality/Behavior Assessment'
            parts.append(type_full)
        
        if 'description' in assessment:
            parts.append(assessment['description'])
        
        return ' | '.join(parts)
    
    def rerank(self, 
              query: str, 
              candidates: List[Dict], 
              top_k: int = 10,
              alpha: float = 0.5) -> List[Dict]:
        """
        Rerank candidates using cross-encoder scores
        
        Args:
            query: Original search query
            candidates: List of candidate assessments from initial retrieval
            top_k: Number of final results to return
            alpha: Weight for combining embedding score and cross-encoder score
                  (0.0 = only cross-encoder, 1.0 = only embedding)
        
        Returns:
            Reranked list of assessments
        """
        if not candidates:
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates...")
        
        # Compute cross-encoder scores
        for candidate in candidates:
            assessment_text = self.create_assessment_text(candidate)
            ce_score = self.compute_cross_encoder_score(query, assessment_text)
            
            # Store original embedding score
            embedding_score = candidate.get('score', 0.0)
            
            # Combine scores
            combined_score = alpha * embedding_score + (1 - alpha) * ce_score
            
            candidate['cross_encoder_score'] = ce_score
            candidate['embedding_score'] = embedding_score
            candidate['combined_score'] = combined_score
        
        # Sort by combined score
        reranked = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
        
        # Select top k
        reranked = reranked[:top_k]
        
        logger.info(f"Reranking complete, returning top {len(reranked)} results")
        return reranked
    
    def ensure_balance(self, 
                      assessments: List[Dict], 
                      min_k: int = 1, 
                      min_p: int = 1) -> List[Dict]:
        """
        Ensure balance between Knowledge (K) and Personality (P) assessments
        
        Args:
            assessments: List of assessments
            min_k: Minimum number of K assessments
            min_p: Minimum number of P assessments
        
        Returns:
            Balanced list of assessments
        """
        if not assessments:
            return []
        
        # Separate K and P assessments
        k_assessments = [a for a in assessments if a.get('test_type') == 'K']
        p_assessments = [a for a in assessments if a.get('test_type') == 'P']
        
        logger.info(f"Initial distribution - K: {len(k_assessments)}, P: {len(p_assessments)}")
        
        # Check if we need to adjust
        if len(k_assessments) < min_k or len(p_assessments) < min_p:
            logger.info("Adjusting to ensure minimum balance...")
            
            # Start with empty result
            result = []
            
            # Add minimum K assessments
            result.extend(k_assessments[:min_k])
            
            # Add minimum P assessments
            result.extend(p_assessments[:min_p])
            
            # Add remaining assessments by score
            remaining = [a for a in assessments if a not in result]
            remaining_sorted = sorted(remaining, key=lambda x: x.get('combined_score', x.get('score', 0)), reverse=True)
            
            # Fill up to desired total
            total_needed = len(assessments)
            result.extend(remaining_sorted[:total_needed - len(result)])
            
            # Sort final result by score
            result = sorted(result, key=lambda x: x.get('combined_score', x.get('score', 0)), reverse=True)
            
            logger.info(f"Balanced distribution - K: {len([a for a in result if a.get('test_type') == 'K'])}, "
                       f"P: {len([a for a in result if a.get('test_type') == 'P'])}")
            
            return result
        
        return assessments
    
    def rerank_and_balance(self,
                          query: str,
                          candidates: List[Dict],
                          top_k: int = 10,
                          min_k: int = 1,
                          min_p: int = 1,
                          alpha: float = 0.5) -> List[Dict]:
        """
        Rerank candidates and ensure K/P balance
        
        Args:
            query: Original search query
            candidates: List of candidate assessments
            top_k: Number of final results
            min_k: Minimum K assessments
            min_p: Minimum P assessments
            alpha: Weight for score combination
        
        Returns:
            Reranked and balanced list of assessments
        """
        # First rerank
        reranked = self.rerank(query, candidates, top_k=top_k * 2, alpha=alpha)  # Get more for balancing
        
        # Then ensure balance and trim to top_k
        balanced = self.ensure_balance(reranked, min_k=min_k, min_p=min_p)
        
        # Final trim to top_k
        final_results = balanced[:top_k]
        
        # Add rank
        for i, assessment in enumerate(final_results, 1):
            assessment['rank'] = i
        
        return final_results
    
    def normalize_scores(self, assessments: List[Dict]) -> List[Dict]:
        """Normalize scores to 0-1 range"""
        if not assessments:
            return assessments
        
        scores = [a.get('combined_score', a.get('score', 0)) for a in assessments]
        
        if not scores or max(scores) == min(scores):
            return assessments
        
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        for assessment in assessments:
            raw_score = assessment.get('combined_score', assessment.get('score', 0))
            normalized = (raw_score - min_score) / score_range
            assessment['score'] = normalized
        
        return assessments


def main():
    """Main execution function"""
    # Test the reranker
    reranker = AssessmentReranker()
    
    # Sample candidates
    candidates = [
        {
            'assessment_name': 'Java Programming Assessment',
            'category': 'Technical',
            'test_type': 'K',
            'description': 'Evaluates Java programming skills',
            'score': 0.85
        },
        {
            'assessment_name': 'Leadership Assessment',
            'category': 'Leadership',
            'test_type': 'P',
            'description': 'Evaluates leadership potential',
            'score': 0.75
        },
        {
            'assessment_name': 'Python Coding Test',
            'category': 'Technical',
            'test_type': 'K',
            'description': 'Assesses Python programming',
            'score': 0.80
        }
    ]
    
    query = "Looking for a Java developer with strong leadership skills"
    
    print("\n=== Reranking Test ===\n")
    print(f"Query: {query}\n")
    
    # Rerank and balance
    results = reranker.rerank_and_balance(query, candidates, top_k=5, min_k=1, min_p=1)
    
    print("Reranked Results:")
    for assessment in results:
        print(f"\n{assessment.get('rank', 0)}. {assessment['assessment_name']}")
        print(f"   Type: {assessment['test_type']}")
        print(f"   Embedding Score: {assessment.get('embedding_score', 0):.4f}")
        print(f"   Cross-Encoder Score: {assessment.get('cross_encoder_score', 0):.4f}")
        print(f"   Combined Score: {assessment.get('combined_score', 0):.4f}")
    
    return reranker


if __name__ == "__main__":
    main()
