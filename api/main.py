# """
# FastAPI Application for SHL Assessment Recommender

# This module provides REST API endpoints for the recommendation system.
# """

# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from typing import List, Dict, Optional
# import logging
# from datetime import datetime
# import sys
# import os

# # Add parent directory to path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.recommender import AssessmentRecommender
# from src.reranker import AssessmentReranker

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="SHL Assessment Recommender API",
#     description="API for recommending SHL assessments based on job descriptions",
#     version="1.0.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, specify actual origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global instances
# recommender = None
# reranker = None


# class RecommendRequest(BaseModel):
#     """Request model for recommendation endpoint"""
#     query: str = Field(..., description="Job description or query text", min_length=1)
#     num_results: Optional[int] = Field(10, description="Number of recommendations to return", ge=1, le=20)
#     use_reranking: Optional[bool] = Field(True, description="Whether to use reranking")
#     min_k: Optional[int] = Field(1, description="Minimum knowledge assessments", ge=0)
#     min_p: Optional[int] = Field(1, description="Minimum personality assessments", ge=0)


# class AssessmentResponse(BaseModel):
#     """Response model for a single assessment"""
#     rank: int
#     assessment_name: str
#     url: str
#     category: str
#     test_type: str
#     score: float
#     description: str


# class RecommendResponse(BaseModel):
#     """Response model for recommendation endpoint"""
#     query: str
#     recommendations: List[AssessmentResponse]
#     total_results: int


# class HealthResponse(BaseModel):
#     """Response model for health check endpoint"""
#     status: str
#     timestamp: str


# @app.on_event("startup")
# async def startup_event():
#     """Load models on startup"""
#     global recommender, reranker
    
#     try:
#         logger.info("Loading recommender system...")
        
#         # Load recommender
#         recommender = AssessmentRecommender()
#         success = recommender.load_index()
        
#         if not success:
#             logger.error("Failed to load recommender index")
#             raise Exception("Failed to load recommender index")
        
#         logger.info("Recommender loaded successfully")
        
#         # Load reranker (lazy loading - will load on first use)
#         reranker = AssessmentReranker()
#         logger.info("Reranker initialized")
        
#         logger.info("API startup complete")
        
#     except Exception as e:
#         logger.error(f"Error during startup: {e}")
#         raise


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """
#     Health check endpoint
    
#     Returns the status of the API and current timestamp.
#     """
#     return {
#         "status": "API is running",
#         "timestamp": datetime.now().isoformat()
#     }


# @app.post("/recommend", response_model=RecommendResponse)
# async def recommend(request: RecommendRequest):
#     """
#     Recommend SHL assessments based on query
    
#     Args:
#         request: RecommendRequest containing query and parameters
    
#     Returns:
#         RecommendResponse with list of recommended assessments
#     """
#     try:
#         logger.info(f"Received recommendation request for query: {request.query[:50]}...")
        
#         # Validate
#         if not request.query or not request.query.strip():
#             raise HTTPException(status_code=400, detail="Query cannot be empty")
        
#         # Get initial recommendations
#         initial_k = request.num_results * 2 if request.use_reranking else request.num_results
#         candidates = recommender.recommend(
#             query=request.query,
#             k=initial_k,
#             method='faiss'
#         )
        
#         if not candidates:
#             logger.warning("No candidates found for query")
#             return {
#                 "query": request.query,
#                 "recommendations": [],
#                 "total_results": 0
#             }
        
#         # Rerank if requested
#         if request.use_reranking:
#             logger.info("Applying reranking...")
#             final_results = reranker.rerank_and_balance(
#                 query=request.query,
#                 candidates=candidates,
#                 top_k=request.num_results,
#                 min_k=request.min_k,
#                 min_p=request.min_p
#             )
#         else:
#             # Just apply balancing
#             final_results = reranker.ensure_balance(
#                 assessments=candidates[:request.num_results],
#                 min_k=request.min_k,
#                 min_p=request.min_p
#             )
#             # Add ranks
#             for i, assessment in enumerate(final_results, 1):
#                 assessment['rank'] = i
        
#         # Normalize scores
#         final_results = reranker.normalize_scores(final_results)
        
#         # Format response
#         recommendations = []
#         for assessment in final_results:
#             recommendations.append({
#                 "rank": assessment.get('rank', 0),
#                 "assessment_name": assessment.get('assessment_name', ''),
#                 "url": assessment.get('assessment_url', ''),
#                 "category": assessment.get('category', ''),
#                 "test_type": assessment.get('test_type', ''),
#                 "score": round(assessment.get('score', 0.0), 4),
#                 "description": assessment.get('description', '')
#             })
        
#         logger.info(f"Returning {len(recommendations)} recommendations")
        
#         return {
#             "query": request.query,
#             "recommendations": recommendations,
#             "total_results": len(recommendations)
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing recommendation: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     """Global exception handler"""
#     logger.error(f"Unhandled exception: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error"}
#     )


# if __name__ == "__main__":
#     import uvicorn
    
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         log_level="info"
#     )
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="AI-powered assessment recommendation system using semantic search and LLM reranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow all origins
app.add_middleware(
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
    recommendations: List[Assessment]
    count: int
    processing_time_ms: float

# Global variables for recommender
recommender = None
reranker = None

@app.on_event("startup")
async def startup_event():
    """Initialize recommender on startup"""
    global recommender, reranker
    
    logger.info("🚀 Starting SHL Assessment API...")
    
    try:
        # Check if models exist
        if not os.path.exists('models/faiss_index.faiss'):
            logger.info("🔧 First-time setup: Building index...")
            
            # Create directories
            os.makedirs('data', exist_ok=True)
            os.makedirs('models', exist_ok=True)
            os.makedirs('Data', exist_ok=True)
            
            # Run setup
            from src.crawler import SHLCrawler
            from src.embedder import AssessmentEmbedder
            
            logger.info("📊 Scraping SHL catalog...")
            crawler = SHLCrawler()
            crawler.scrape_catalog()
            
            logger.info("🔮 Building search index...")
            embedder = AssessmentEmbedder()
            embedder.load_catalog()
            embedder.create_embeddings()
            embedder.build_index()
            embedder.save_index()
            
            logger.info("✅ Setup complete!")
        
        # Load recommender
        from src.recommender import AssessmentRecommender
        from src.reranker import AssessmentReranker
        
        logger.info("📚 Loading recommender...")
        recommender = AssessmentRecommender()
        recommender.load_index()
        
        logger.info("🎯 Loading reranker...")
        reranker = AssessmentReranker()
        
        logger.info("✅ API ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "SHL Assessment Recommender API",
        "version": "1.0.0",
        "status": "running",
        "description": "AI-powered assessment recommendations using semantic search",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recommend": "/recommend (POST)",
            "catalog": "/catalog (GET)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if recommender and reranker else "initializing",
        "index_loaded": recommender is not None and recommender.index is not None,
        "catalog_size": len(recommender.assessment_data) if recommender and recommender.assessment_data else 0,
        "reranker_loaded": reranker is not None
    }

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get assessment recommendations for a job query
    
    - **query**: Job description or requirements
    - **top_k**: Number of recommendations to return (default: 10)
    """
    import time
    start_time = time.time()
    
    if not recommender or not reranker:
        raise HTTPException(status_code=503, detail="Service initializing, please try again in a moment")
    
    try:
        # Get initial recommendations
        logger.info(f"Processing query: {request.query[:50]}...")
        candidates = recommender.recommend(request.query, k=20)
        
        # Rerank and balance
        results = reranker.rerank_and_balance(
            query=request.query,
            candidates=candidates,
            top_k=request.top_k
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Returned {len(results)} recommendations in {processing_time:.0f}ms")
        
        return RecommendResponse(
            query=request.query,
            recommendations=results,
            count=len(results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/catalog")
async def get_catalog():
    """Get all available assessments"""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service initializing")
    
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

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    return {
        "total_assessments": len(recommender.assessment_data) if recommender.assessment_data else 0,
        "index_size": recommender.index.ntotal if recommender.index else 0,
        "embedding_dimension": 384,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)