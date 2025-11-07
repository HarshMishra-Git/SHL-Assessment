"""
FastAPI Application for SHL Assessment Recommender

This module provides REST API endpoints for the recommendation system.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import AssessmentRecommender
from src.reranker import AssessmentReranker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
recommender = None
reranker = None


class RecommendRequest(BaseModel):
    """Request model for recommendation endpoint"""
    query: str = Field(..., description="Job description or query text", min_length=1)
    num_results: Optional[int] = Field(10, description="Number of recommendations to return", ge=1, le=20)
    use_reranking: Optional[bool] = Field(True, description="Whether to use reranking")
    min_k: Optional[int] = Field(1, description="Minimum knowledge assessments", ge=0)
    min_p: Optional[int] = Field(1, description="Minimum personality assessments", ge=0)


class AssessmentResponse(BaseModel):
    """Response model for a single assessment"""
    rank: int
    assessment_name: str
    url: str
    category: str
    test_type: str
    score: float
    description: str


class RecommendResponse(BaseModel):
    """Response model for recommendation endpoint"""
    query: str
    recommendations: List[AssessmentResponse]
    total_results: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global recommender, reranker
    
    try:
        logger.info("Loading recommender system...")
        
        # Load recommender
        recommender = AssessmentRecommender()
        success = recommender.load_index()
        
        if not success:
            logger.error("Failed to load recommender index")
            raise Exception("Failed to load recommender index")
        
        logger.info("Recommender loaded successfully")
        
        # Load reranker (lazy loading - will load on first use)
        reranker = AssessmentReranker()
        logger.info("Reranker initialized")
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the status of the API and current timestamp.
    """
    return {
        "status": "API is running",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Recommend SHL assessments based on query
    
    Args:
        request: RecommendRequest containing query and parameters
    
    Returns:
        RecommendResponse with list of recommended assessments
    """
    try:
        logger.info(f"Received recommendation request for query: {request.query[:50]}...")
        
        # Validate
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get initial recommendations
        initial_k = request.num_results * 2 if request.use_reranking else request.num_results
        candidates = recommender.recommend(
            query=request.query,
            k=initial_k,
            method='faiss'
        )
        
        if not candidates:
            logger.warning("No candidates found for query")
            return {
                "query": request.query,
                "recommendations": [],
                "total_results": 0
            }
        
        # Rerank if requested
        if request.use_reranking:
            logger.info("Applying reranking...")
            final_results = reranker.rerank_and_balance(
                query=request.query,
                candidates=candidates,
                top_k=request.num_results,
                min_k=request.min_k,
                min_p=request.min_p
            )
        else:
            # Just apply balancing
            final_results = reranker.ensure_balance(
                assessments=candidates[:request.num_results],
                min_k=request.min_k,
                min_p=request.min_p
            )
            # Add ranks
            for i, assessment in enumerate(final_results, 1):
                assessment['rank'] = i
        
        # Normalize scores
        final_results = reranker.normalize_scores(final_results)
        
        # Format response
        recommendations = []
        for assessment in final_results:
            recommendations.append({
                "rank": assessment.get('rank', 0),
                "assessment_name": assessment.get('assessment_name', ''),
                "url": assessment.get('assessment_url', ''),
                "category": assessment.get('category', ''),
                "test_type": assessment.get('test_type', ''),
                "score": round(assessment.get('score', 0.0), 4),
                "description": assessment.get('description', '')
            })
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        
        return {
            "query": request.query,
            "recommendations": recommendations,
            "total_results": len(recommendations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
