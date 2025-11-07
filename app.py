"""
Streamlit Web Interface for SHL Assessment Recommender

This module provides a professional web interface for the recommendation system.
"""

import streamlit as st
# ========================================
# MOUNT FASTAPI FOR API ENDPOINTS
# ========================================
from streamlit.web import cli as stcli
import sys

# Check if we should serve API alongside Streamlit
if os.path.exists('api_routes.py'):
    try:
        from api_routes import api_app
        
        # This allows API access via /api/* routes
        # While Streamlit UI remains at /
        import streamlit.components.v1 as components
        
        # Log API availability
        print("✅ FastAPI mounted at /api/*")
        print("📚 API Docs: /api/docs")
        print("🔧 API Endpoints: /api/recommend, /api/health, /api/catalog")
        
    except Exception as e:
        print(f"⚠️ Could not mount API: {e}")

import pandas as pd
import requests
import json
import sys
import os
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender import AssessmentRecommender
from src.reranker import AssessmentReranker

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .assessment-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
        background-color: #f8f9fa;
    }
    .k-type {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .p-type {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .score-badge {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None


@st.cache_resource
def load_recommender():
    """Load and cache the recommender system"""
    try:
        recommender = AssessmentRecommender()
        success = recommender.load_index()
        if success:
            return recommender
        else:
            return None
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        return None


@st.cache_resource
def load_reranker():
    """Load and cache the reranker"""
    try:
        reranker = AssessmentReranker()
        return reranker
    except Exception as e:
        st.error(f"Error loading reranker: {e}")
        return None


def get_recommendations(query: str, num_results: int, use_reranking: bool, min_k: int, min_p: int):
    """Get recommendations from the system"""
    recommender = load_recommender()
    
    if recommender is None:
        st.error("Failed to load recommender system. Please check if models are available.")
        return []
    
    try:
        # Get initial candidates
        initial_k = num_results * 2 if use_reranking else num_results
        candidates = recommender.recommend(query, k=initial_k, method='faiss')
        
        if not candidates:
            return []
        
        # Apply reranking if requested
        if use_reranking:
            reranker = load_reranker()
            if reranker:
                final_results = reranker.rerank_and_balance(
                    query=query,
                    candidates=candidates,
                    top_k=num_results,
                    min_k=min_k,
                    min_p=min_p
                )
            else:
                final_results = candidates[:num_results]
        else:
            reranker = load_reranker()
            if reranker:
                final_results = reranker.ensure_balance(
                    assessments=candidates[:num_results],
                    min_k=min_k,
                    min_p=min_p
                )
            else:
                final_results = candidates[:num_results]
            
            # Add ranks
            for i, assessment in enumerate(final_results, 1):
                assessment['rank'] = i
        
        # Normalize scores
        if reranker:
            final_results = reranker.normalize_scores(final_results)
        
        return final_results
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []


def display_assessment(assessment: Dict, rank: int):
    """Display a single assessment card"""
    type_badge = f'<span class="k-type">Knowledge/Skill</span>' if assessment['test_type'] == 'K' else f'<span class="p-type">Personality/Behavior</span>'
    score_badge = f'<span class="score-badge">Score: {assessment.get("score", 0):.2%}</span>'
    
    st.markdown(f"""
    <div class="assessment-card">
        <h3>#{rank}. {assessment['assessment_name']}</h3>
        <p>{type_badge} &nbsp; <strong>Category:</strong> {assessment['category']} &nbsp; {score_badge}</p>
        <p><strong>Description:</strong> {assessment['description']}</p>
        <p><a href="{assessment['assessment_url']}" target="_blank">🔗 View Assessment</a></p>
    </div>
    """, unsafe_allow_html=True)


# Main UI
st.markdown('<h1 class="main-header">🎯 SHL Assessment Recommender System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered job assessment recommendations using semantic search</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    num_results = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=15,
        value=10,
        step=1
    )
    
    use_reranking = st.checkbox(
        "Use Advanced Reranking",
        value=True,
        help="Apply cross-encoder reranking for better accuracy"
    )
    
    st.subheader("Balance Settings")
    
    min_k = st.number_input(
        "Minimum Knowledge Assessments",
        min_value=0,
        max_value=5,
        value=1,
        help="Minimum number of knowledge/skill assessments"
    )
    
    min_p = st.number_input(
        "Minimum Personality Assessments",
        min_value=0,
        max_value=5,
        value=1,
        help="Minimum number of personality/behavior assessments"
    )
    
    st.markdown("---")
     # API Information
    st.markdown("### 🔧 API Access")
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #78D64B;
        font-size: 0.85rem;
    ">
        <p style="color: white; margin: 0;">
        <strong>API Endpoints:</strong><br>
        • <code>/api/recommend</code><br>
        • <code>/api/health</code><br>
        • <code>/api/catalog</code><br>
        <br>
        <strong>Docs:</strong> <a href="/api/docs" style="color: #78D64B;">/api/docs</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📖 About")
    st.markdown("""
    This system uses:
    - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
    - **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
    - **Search**: FAISS similarity search
    
    Recommends SHL Individual Test Solutions based on job descriptions.
    """)
    
    # Load evaluation results if available
    try:
        if os.path.exists('evaluation_results.json'):
            with open('evaluation_results.json', 'r') as f:
                eval_results = json.load(f)
            
            st.markdown("---")
            st.subheader("📊 Performance Metrics")
            st.metric("Mean Recall@10", f"{eval_results.get('mean_recall_at_10', 0):.2%}")
            st.metric("Mean Precision@10", f"{eval_results.get('mean_precision_at_10', 0):.2%}")
    except:
        pass


# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    query = st.text_area(
        "📝 Enter Job Description or Query",
        height=150,
        placeholder="e.g., Looking for a Java developer who can lead a small team and has strong communication skills...",
        help="Enter a job description, requirements, or natural language query"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Example queries dropdown
    example_queries = {
        "Java Developer + Leadership": "Looking for a Java developer who can lead a small team and mentor junior developers",
        "Data Analyst": "Need a data analyst with SQL and Python skills for business intelligence",
        "Customer Service Manager": "Seeking a customer service manager with excellent communication and problem-solving abilities",
        "Software Engineer": "Want to hire a software engineer with strong programming and analytical skills",
        "Sales Representative": "Looking for a sales representative with persuasive personality and negotiation skills"
    }
    
    selected_example = st.selectbox(
        "Or try an example:",
        [""] + list(example_queries.keys())
    )
    
    if selected_example:
        query = example_queries[selected_example]

# Get recommendations button
if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
    if not query or not query.strip():
        st.warning("⚠️ Please enter a query first!")
    else:
        with st.spinner("🔍 Searching for the best assessments..."):
            recommendations = get_recommendations(query, num_results, use_reranking, min_k, min_p)
            st.session_state.recommendations = recommendations

# Display results
if st.session_state.recommendations:
    recommendations = st.session_state.recommendations
    
    st.markdown("---")
    st.subheader(f"📋 Top {len(recommendations)} Recommended Assessments")
    
    # Summary statistics
    k_count = sum(1 for r in recommendations if r['test_type'] == 'K')
    p_count = sum(1 for r in recommendations if r['test_type'] == 'P')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    with col2:
        st.metric("Knowledge/Skill (K)", k_count)
    with col3:
        st.metric("Personality/Behavior (P)", p_count)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display each assessment
    for assessment in recommendations:
        display_assessment(assessment, assessment.get('rank', 0))
    
    # Download option
    st.markdown("---")
    
    # Prepare data for download
    download_data = []
    for assessment in recommendations:
        download_data.append({
            'Rank': assessment.get('rank', 0),
            'Assessment Name': assessment['assessment_name'],
            'Type': 'Knowledge/Skill' if assessment['test_type'] == 'K' else 'Personality/Behavior',
            'Category': assessment['category'],
            'Score': f"{assessment.get('score', 0):.2%}",
            'URL': assessment['assessment_url'],
            'Description': assessment['description']
        })
    
    df = pd.DataFrame(download_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="shl_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    # Show welcome message when no results
    st.info("👋 Welcome! Enter a job description above and click 'Get Recommendations' to find the best SHL assessments.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>SHL Assessment Recommender System | Powered by Generative AI</p>",
    unsafe_allow_html=True
)
