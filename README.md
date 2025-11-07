# 🎯 SHL Assessment Recommender System

A production-ready Generative AI-based recommendation system that suggests the most relevant SHL Individual Test Solutions based on job descriptions or natural language queries.

## 🌟 Features

- **Natural Language Processing**: Accepts job descriptions, JD text, or queries in natural language
- **Semantic Search**: Uses state-of-the-art sentence transformers and FAISS for fast similarity search
- **Intelligent Reranking**: Employs cross-encoder models for improved accuracy
- **Balanced Recommendations**: Ensures mix of Knowledge/Skill (K) and Personality/Behavior (P) assessments
- **Dual Interface**: Both REST API and Streamlit web UI
- **High Accuracy**: Target Mean Recall@10 ≥ 0.75
- **Production Ready**: Comprehensive error handling, logging, and validation

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Endpoints](#api-endpoints)
- [System Components](#system-components)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

### System Flow

```
User Query → Embedding → FAISS Search → Initial Candidates
                                              ↓
                                       Cross-Encoder Reranking
                                              ↓
                                       Balance K/P Assessments
                                              ↓
                                       Top 5-10 Recommendations
```

### Technology Stack

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Search Engine**: FAISS (Facebook AI Similarity Search)
- **API Framework**: FastAPI
- **UI Framework**: Streamlit
- **ML Framework**: PyTorch, Transformers, Sentence-Transformers

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM (for model inference)
- Internet connection (for initial model download)

### Step 1: Clone Repository

```bash
git clone https://github.com/HarshMishra-Git/SHL-Assessment.git
cd SHL-Assessment
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Generate SHL Catalog

```bash
python src/crawler.py
```

This will create `data/shl_catalog.csv` with 25+ individual test solutions.

### Step 4: Build Search Index

```bash
python src/embedder.py
```

This will:
- Download the sentence transformer model (first time only)
- Generate embeddings for all assessments
- Create FAISS index in `models/` directory

**Note**: First run will download ~90MB of model files from Hugging Face.

## 🎬 Quick Start

### Option 1: Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: API Server

```bash
python api/main.py
```

Or with uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

## 📖 Usage

### Web Interface

1. **Launch**: Run `streamlit run app.py`
2. **Enter Query**: Type or paste a job description
3. **Adjust Settings** (sidebar):
   - Number of recommendations (5-15)
   - Enable/disable reranking
   - Set minimum K and P assessments
4. **Get Recommendations**: Click the button
5. **Review Results**: View ranked assessments with scores
6. **Download**: Export results as CSV

#### Example Queries

```
"Looking for a Java developer who can lead a small team"
"Need a data analyst with SQL and Python skills"
"Want to assess personality traits for customer service role"
"Seeking a software engineer with strong problem-solving abilities"
```

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "API is running",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Get Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Looking for a Java developer with leadership skills",
    "num_results": 10,
    "use_reranking": true,
    "min_k": 1,
    "min_p": 1
  }'
```

**Response:**
```json
{
  "query": "Looking for a Java developer with leadership skills",
  "recommendations": [
    {
      "rank": 1,
      "assessment_name": "Java Programming Assessment",
      "url": "https://www.shl.com/solutions/products/java-programming",
      "category": "Technical",
      "test_type": "K",
      "score": 0.95,
      "description": "Evaluates Java programming skills..."
    },
    {
      "rank": 2,
      "assessment_name": "Leadership Assessment",
      "url": "https://www.shl.com/solutions/products/leadership",
      "category": "Leadership",
      "test_type": "P",
      "score": 0.88,
      "description": "Evaluates leadership potential..."
    }
  ],
  "total_results": 10
}
```

#### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "query": "Need a Python developer for data analysis",
        "num_results": 5
    }
)

recommendations = response.json()
for rec in recommendations["recommendations"]:
    print(f"{rec['rank']}. {rec['assessment_name']} (Score: {rec['score']:.2f})")
```

## 🔧 System Components

### 1. Crawler (`src/crawler.py`)

Scrapes SHL product catalog and creates fallback catalog with 25+ assessments.

**Features:**
- Robust HTML parsing
- Fallback catalog for offline use
- Automatic K/P classification
- CSV export

**Usage:**
```bash
python src/crawler.py
```

### 2. Preprocessor (`src/preprocess.py`)

Loads and cleans the Gen_AI Dataset.xlsx training data.

**Features:**
- Excel file parsing
- Text normalization
- URL extraction
- Train/test split handling

**Usage:**
```bash
python src/preprocess.py
```

### 3. Embedder (`src/embedder.py`)

Generates embeddings and builds FAISS index.

**Features:**
- Batch embedding generation
- FAISS index creation
- Model caching
- Progress tracking

**Usage:**
```bash
python src/embedder.py
```

**Outputs:**
- `models/faiss_index.faiss` - FAISS index
- `models/embeddings.npy` - Numpy embeddings
- `models/mapping.pkl` - Assessment metadata

### 4. Recommender (`src/recommender.py`)

Performs semantic search using FAISS.

**Features:**
- Fast vector search
- Cosine similarity fallback
- Batch processing
- Top-k retrieval

### 5. Reranker (`src/reranker.py`)

Reranks candidates using cross-encoder and ensures K/P balance.

**Features:**
- Cross-encoder scoring
- Score normalization
- K/P balancing logic
- Configurable weights

### 6. Evaluator (`src/evaluator.py`)

Evaluates system performance using Mean Recall@10.

**Usage:**
```bash
python src/evaluator.py
```

**Metrics:**
- Mean Recall@10
- Mean Precision@10
- Mean Average Precision (MAP)
- Recall distribution statistics

## 📊 Evaluation

The system is evaluated on the training set using Mean Recall@10:

```
Recall@10 = (# of relevant assessments retrieved in top 10) / (# of total relevant assessments)
```

### Running Evaluation

```bash
python src/evaluator.py
```

### Example Results

```
=== EVALUATION REPORT ===
Dataset Size: 10 queries
Evaluation Metric: Recall@10

Main Metrics:
  Mean Recall@10: 0.8250
  Mean Precision@10: 0.7800
  Mean Average Precision: 0.8100

Recall Distribution:
  Min: 0.5000
  Max: 1.0000
  Median: 0.8500
  Std Dev: 0.1500

✓ Target Mean Recall@10 ≥ 0.75 ACHIEVED!
```

Results are saved to `evaluation_results.json`.

## 📁 Project Structure

```
SHL-Assessment/
├── data/
│   ├── shl_catalog.csv          # Scraped/generated catalog
│   └── Gen_AI Dataset.xlsx      # Training dataset
├── src/
│   ├── __init__.py
│   ├── crawler.py               # Web scraper
│   ├── preprocess.py            # Data preprocessing
│   ├── embedder.py              # Embedding generation
│   ├── recommender.py           # Semantic search
│   ├── reranker.py              # Cross-encoder reranking
│   └── evaluator.py             # Evaluation metrics
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI application
├── models/
│   ├── faiss_index.faiss        # Generated index
│   ├── embeddings.npy           # Generated embeddings
│   └── mapping.pkl              # Generated mapping
├── app.py                       # Streamlit UI
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
├── evaluation_results.json      # Generated evaluation results
└── README.md                    # This file
```

## ⚙️ Configuration

### Model Configuration

Edit the model names in source files if needed:

**Embedding Model** (`src/embedder.py`):
```python
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
```

**Reranking Model** (`src/reranker.py`):
```python
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```

### API Configuration

**Port** (`api/main.py`):
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**CORS Origins** (`api/main.py`):
```python
allow_origins=["*"]  # Change to specific origins in production
```

### Recommendation Parameters

**Default K/P Balance**:
- Minimum K assessments: 1
- Minimum P assessments: 1

**Reranking Weight** (`src/reranker.py`):
```python
alpha = 0.5  # 0.0 = only cross-encoder, 1.0 = only embeddings
```

## 👩‍💻 Development

### Adding New Assessments

1. Edit the fallback catalog in `src/crawler.py`:
```python
assessments.append({
    'assessment_name': 'New Assessment',
    'assessment_url': 'https://...',
    'category': 'Technical',
    'test_type': 'K',
    'description': '...'
})
```

2. Rebuild the index:
```bash
python src/crawler.py
python src/embedder.py
```

### Customizing Balance Logic

Edit `src/reranker.py`:
```python
def ensure_balance(assessments, min_k=2, min_p=2):
    # Your custom logic
    pass
```

### Running Tests

```bash
# Test each component individually
python src/crawler.py
python src/preprocess.py
python src/embedder.py
python src/recommender.py
python src/reranker.py
python src/evaluator.py

# Test API
curl http://localhost:8000/health

# Test UI
streamlit run app.py
```

## 🔍 Troubleshooting

### Issue: Model Download Fails

**Solution**: Ensure internet connection. Models are downloaded from Hugging Face on first run.

```bash
# Manually download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Issue: FAISS Index Not Found

**Solution**: Generate the index:
```bash
python src/embedder.py
```

### Issue: API Port Already in Use

**Solution**: Change port in `api/main.py` or kill existing process:
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Streamlit Won't Start

**Solution**: Check port 8501 and Streamlit installation:
```bash
streamlit --version
streamlit run app.py --server.port 8502
```

### Issue: Out of Memory

**Solution**: Reduce batch size in `src/embedder.py`:
```python
embeddings = self.model.encode(texts, batch_size=16)  # Default: 32
```

### Issue: Low Recall Score

**Solutions:**
1. Increase initial retrieval size in recommender
2. Adjust reranking alpha weight
3. Add more training data
4. Fine-tune embeddings on domain-specific data

## 📝 License

This project is created for the SHL Assessment task.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ using Generative AI and Open Source Models**
