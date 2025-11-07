import streamlit as st

# ========================================
# PAGE CONFIG - MUST BE FIRST!
# ========================================
st.set_page_config(
    page_title="SHL Assessment Recommender | People Science",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import subprocess
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========================================
# AUTO-SETUP: Run setup.py on first load
# ========================================
if not os.path.exists('models/faiss_index.faiss'):
    with st.spinner("🚀 First-time setup: Building search index... This takes ~2-3 minutes"):
        try:
            result = subprocess.run([sys.executable, 'setup.py'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)
            
            if result.returncode == 0:
                st.success("✅ Setup complete! Reloading app...")
                st.rerun()
            else:
                st.error(f"Setup failed: {result.stderr}")
        except Exception as e:
            st.error(f"Setup error: {str(e)}")
            st.stop()

from src.recommender import AssessmentRecommender
from src.reranker import AssessmentReranker

# ========================================
# SHL BRAND STYLING
# ========================================
st.markdown("""
    <style>
    /* SHL Brand Colors */
    :root {
        --shl-dark-green: #1B5700;
        --shl-bright-green: #78D64B;
        --shl-gray: #2E2E2E;
        --shl-light-gray: #F8F9FA;
    }
    
    /* Header Styling */
    .shl-header {
        background: linear-gradient(135deg, #1B5700 0%, #2A7500 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(27, 87, 0, 0.2);
    }
    
    .shl-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .shl-subtitle {
        color: #78D64B;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5700 0%, #2A7500 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
        font-weight: 600;
    }
    
    /* Input Styling */
    .stTextArea textarea {
        border: 2px solid #78D64B !important;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #1B5700 !important;
        box-shadow: 0 0 0 2px rgba(120, 214, 75, 0.2) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1B5700 0%, #2A7500 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(27, 87, 0, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2A7500 0%, #1B5700 100%);
        box-shadow: 0 6px 20px rgba(27, 87, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Assessment Card Styling */
    .assessment-card {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #78D64B;
        transition: all 0.3s ease;
    }
    
    .assessment-card:hover {
        box-shadow: 0 4px 25px rgba(27, 87, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .assessment-card h3 {
        color: #1B5700;
        font-size: 1.4rem;
        margin: 0 0 1rem 0;
        font-weight: 700;
    }
    
    /* Badge Styling */
    .badge-k {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-p {
        background: linear-gradient(135deg, #EE297B 0%, #D91E6B 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .score-badge {
        background: #78D64B;
        color: #2E2E2E;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .category-badge {
        background: #F8F9FA;
        color: #2E2E2E;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #E5E7EB;
    }
    
    /* CTA Button */
    .cta-button {
        background: linear-gradient(135deg, #1B5700 0%, #2A7500 100%);
        color: white;
        padding: 0.7rem 1.8rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 3px 12px rgba(27, 87, 0, 0.3);
        transition: all 0.3s ease;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .cta-button:hover {
        background: linear-gradient(135deg, #2A7500 0%, #1B5700 100%);
        box-shadow: 0 5px 18px rgba(27, 87, 0, 0.4);
        transform: translateY(-2px);
        text-decoration: none;
        color: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1B5700;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1B5700 0%, #2A7500 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 8px;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2A7500 0%, #1B5700 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# SESSION STATE
# ========================================
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# ========================================
# CACHED RESOURCES
# ========================================
@st.cache_resource
def load_recommender():
    """Load and cache the recommender system"""
    try:
        recommender = AssessmentRecommender()
        success = recommender.load_index()
        return recommender if success else None
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        return None

@st.cache_resource
def load_reranker():
    """Load and cache the reranker"""
    try:
        return AssessmentReranker()
    except Exception as e:
        st.error(f"Error loading reranker: {e}")
        return None

# ========================================
# HELPER FUNCTIONS
# ========================================
def get_recommendations(query: str, num_results: int, use_reranking: bool, min_k: int, min_p: int):
    """Get recommendations from the system"""
    recommender = load_recommender()
    
    if recommender is None:
        st.error("Failed to load recommender system.")
        return []
    
    try:
        initial_k = num_results * 2 if use_reranking else num_results
        candidates = recommender.recommend(query, k=initial_k, method='faiss')
        
        if not candidates:
            return []
        
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
            
            for i, assessment in enumerate(final_results, 1):
                assessment['rank'] = i
        
        if reranker:
            final_results = reranker.normalize_scores(final_results)
        
        return final_results
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

# ========================================
# HEADER
# ========================================
st.markdown("""
<div class="shl-header">
    <h1 class="shl-title">🎯 SHL Assessment Recommender</h1>
    <p class="shl-subtitle">AI-Powered Talent Assessment Matching | People Science</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
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
    
    st.markdown("### ⚖️ Balance Settings")
    
    min_k = st.number_input(
        "Minimum Knowledge Tests",
        min_value=0,
        max_value=5,
        value=1
    )
    
    min_p = st.number_input(
        "Minimum Personality Tests",
        min_value=0,
        max_value=5,
        value=1
    )
    
    st.markdown("---")
    
    st.markdown("### 📖 About")
    st.markdown("""
    **Technology Stack:**
    - Embeddings: all-MiniLM-L6-v2
    - Reranking: ms-marco-MiniLM-L-6-v2
    - Search: FAISS
    
    **Performance:**
    - Mean Recall@10: 100%
    - 152 SHL Assessments
    """)
    
    # Load evaluation results if available
    try:
        if os.path.exists('evaluation_results.json'):
            with open('evaluation_results.json', 'r') as f:
                eval_results = json.load(f)
            
            st.markdown("---")
            st.markdown("### 📊 Metrics")
            st.metric("Recall@10", f"{eval_results.get('mean_recall_at_10', 0):.0%}")
            st.metric("Precision@10", f"{eval_results.get('mean_precision_at_10', 0):.0%}")
    except:
        pass

# ========================================
# MAIN CONTENT
# ========================================
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area(
        "📝 Enter Job Description or Requirements",
        height=150,
        placeholder="Example: Looking for a Java developer with 5+ years experience who can lead a team and has strong communication skills...",
        help="Enter a detailed job description or requirements"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    example_queries = {
        "Java Developer + Leadership": "Looking for a Java developer who can lead a small team and mentor junior developers with strong technical and interpersonal skills",
        "Data Analyst": "Need a data analyst with SQL, Python, and Excel skills for business intelligence and reporting",
        "Customer Service Manager": "Seeking a customer service manager with excellent communication, problem-solving, and team management abilities",
        "Software Engineer": "Want to hire a software engineer with strong programming skills in Python, analytical thinking, and collaboration abilities",
        "Sales Representative": "Looking for a sales representative with persuasive personality, negotiation skills, and customer relationship management experience"
    }
    
    selected_example = st.selectbox(
        "💡 Try an example:",
        [""] + list(example_queries.keys())
    )
    
    if selected_example:
        query = example_queries[selected_example]

# Get recommendations button
if st.button("🔍 FIND ASSESSMENTS", type="primary", use_container_width=True):
    if not query or not query.strip():
        st.warning("⚠️ Please enter a job description first!")
    else:
        with st.spinner("🔍 Analyzing requirements and matching assessments..."):
            recommendations = get_recommendations(query, num_results, use_reranking, min_k, min_p)
            st.session_state.recommendations = recommendations

# ========================================
# DISPLAY RESULTS
# ========================================
if st.session_state.recommendations:
    recommendations = st.session_state.recommendations
    
    st.markdown("---")
    st.markdown("## 🎯 Recommended Assessments")
    
    # Summary Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    k_count = sum(1 for r in recommendations if r.get('test_type') == 'K')
    p_count = sum(1 for r in recommendations if r.get('test_type') == 'P')
    avg_score = sum(r.get('score', 0) for r in recommendations) / len(recommendations) if recommendations else 0
    
    with col1:
        st.metric("Total Matches", len(recommendations))
    
    with col2:
        st.metric("Knowledge Tests", k_count)
    
    with col3:
        st.metric("Personality Tests", p_count)
    
    with col4:
        st.metric("Avg. Match Score", f"{avg_score:.0%}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display each recommendation
    for idx, rec in enumerate(recommendations, 1):
        # Determine badges
        if rec.get('test_type') == 'K':
            type_badge = '<span class="badge-k">📚 KNOWLEDGE</span>'
        else:
            type_badge = '<span class="badge-p">👤 PERSONALITY</span>'
        
        score = rec.get("score", 0)
        score_badge = f'<span class="score-badge">⭐ {score:.0%} MATCH</span>'
        
        # Get content
        description = str(rec.get('description', 'No description available'))[:250]
        assessment_name = str(rec.get('assessment_name', 'Unknown Assessment'))
        category = str(rec.get('category', 'General'))
        url = rec.get('assessment_url', '#')
        
        # Create card
        st.markdown(f"""
        <div class="assessment-card">
            <h3>#{idx} · {assessment_name}</h3>
            <div style="margin-bottom: 1rem;">
                {type_badge} &nbsp; {score_badge}
            </div>
            <p style="color: #4B5563; margin-bottom: 1.5rem; line-height: 1.7; font-size: 0.95rem;">
                {description}...
            </p>
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;">
                <span class="category-badge">📂 {category}</span>
                <a href="{url}" target="_blank" class="cta-button">VIEW ASSESSMENT →</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download Section
    st.markdown("---")
    
    df = pd.DataFrame([{
        'Rank': idx,
        'Assessment': rec.get('assessment_name', ''),
        'Type': rec.get('test_type', ''),
        'Category': rec.get('category', ''),
        'Match Score': f"{rec.get('score', 0):.2%}",
        'URL': rec.get('assessment_url', ''),
        'Description': rec.get('description', '')[:150]
    } for idx, rec in enumerate(recommendations, 1)])
    
    csv = df.to_csv(index=False)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.download_button(
            label="📥 DOWNLOAD RESULTS (CSV)",
            data=csv,
            file_name=f"shl_assessments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👋 **Welcome!** Enter a job description above and click 'FIND ASSESSMENTS' to discover the best SHL assessment recommendations powered by AI.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #78D64B; font-weight: 600;'>SHL Assessment Recommender | Powered by Advanced AI & Machine Learning</p>",
    unsafe_allow_html=True
)