"""
FastAPI routes embedded in Streamlit app
Access via: https://huggingface.co/spaces/Harsh-1132/SHL/api/recommend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
api_app = FastAPI(
    title="SHL Assessment Recommender API",
    description="AI-powered assessment recommendation system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10

class Assessment(BaseModel):
    assessment_name: str
    assessment_url: str
    description: str
    category: str
    test_type: str
    score: float

class RecommendResponse(BaseModel):
    query: str
    recommendations: List[dict]
    count: int

# Global recommender instances
recommender = None
reranker = None

def initialize_recommender():
    """Initialize recommender on first API call"""
    global recommender, reranker
    
    if recommender is None:
        logger.info("🚀 Initializing recommender for API...")
        
        from src.recommender import AssessmentRecommender
        from src.reranker import AssessmentReranker
        
        recommender = AssessmentRecommender()
        recommender.load_index()
        reranker = AssessmentReranker()
        
        logger.info("✅ Recommender initialized!")

@api_app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "SHL Assessment Recommender API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "recommend": "/api/recommend (POST)",
            "health": "/api/health (GET)",
            "catalog": "/api/catalog (GET)",
            "docs": "/api/docs",
            "ui": "/"
        }
    }

@api_app.get("/api/health")
async def health():
    """Health check endpoint"""
    initialize_recommender()
    
    return {
        "status": "healthy",
        "index_loaded": recommender is not None and recommender.index is not None,
        "catalog_size": len(recommender.assessment_data) if recommender and recommender.assessment_data else 0
    }

@api_app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get assessment recommendations
    
    **Request Body:**
    ```json
    {
        "query": "Java developer with leadership skills",
        "top_k": 10
    }
    ```
    """
    initialize_recommender()
    
    try:
        # Get recommendations
        candidates = recommender.recommend(request.query, k=20)
        
        # Rerank
        results = reranker.rerank_and_balance(
            query=request.query,
            candidates=candidates,
            top_k=request.top_k
        )
        
        return RecommendResponse(
            query=request.query,
            recommendations=results,
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/api/catalog")
async def get_catalog():
    """Get all assessments"""
    initialize_recommender()
    
    try:
        return {
            "assessments": recommender.assessment_data,
            "count": len(recommender.assessment_data),
            "types": {
                "K": sum(1 for a in recommender.assessment_data if a.get('test_type') == 'K'),
                "P": sum(1 for a in recommender.assessment_data if a.get('test_type') == 'P')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))