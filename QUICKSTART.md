# Quick Reference - SHL Assessment Recommender

## Installation (One-Time Setup)
```bash
pip install -r requirements.txt
python setup.py
```

## Start Services

### Web Interface
```bash
streamlit run app.py
# Open: http://localhost:8501
```

### API Server
```bash
python api/main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer with leadership skills"}'
```

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={"query": "Python data analyst", "num_results": 5}
)

for rec in response.json()["recommendations"]:
    print(f"{rec['rank']}. {rec['assessment_name']} - {rec['score']:.2%}")
```

## Direct Usage (No API)
```python
from src.recommender import AssessmentRecommender
from src.reranker import AssessmentReranker

# Initialize
recommender = AssessmentRecommender()
recommender.load_index()
reranker = AssessmentReranker()

# Get recommendations
query = "Software engineer"
candidates = recommender.recommend(query, k=15)
results = reranker.rerank_and_balance(query, candidates, top_k=10)

# Display
for assessment in results:
    print(f"{assessment['rank']}. {assessment['assessment_name']}")
```

## Common Commands

### Run Tests
```bash
python test_basic.py
```

### Run Examples
```bash
python examples.py
```

### Run Evaluation
```bash
python src/evaluator.py
```

### Regenerate Catalog
```bash
python src/crawler.py
```

### Rebuild Index
```bash
python src/embedder.py
```

## Project Structure
```
src/           - Core modules
api/           - FastAPI application
data/          - Catalog and datasets
models/        - Generated models (after setup)
app.py         - Streamlit UI
setup.py       - Automated setup
test_basic.py  - Test suite
examples.py    - Usage examples
```

## Configuration

### Number of Results
- Web UI: Use slider (5-15)
- API: Set `num_results` parameter (1-20)

### K/P Balance
- Web UI: Adjust "Minimum K/P Assessments"
- API: Set `min_k` and `min_p` parameters

### Reranking
- Web UI: Toggle "Use Advanced Reranking"
- API: Set `use_reranking` to true/false

## Files Generated on First Run
```
models/faiss_index.faiss  - Search index (~10KB)
models/embeddings.npy     - Embeddings (~40KB)
models/mapping.pkl        - Metadata (~5KB)
evaluation_results.json   - Results (~1KB)
```

## Troubleshooting

### Models not found
```bash
python setup.py  # Re-run setup
```

### Port in use
```bash
# Change port in code or kill process
lsof -ti:8000 | xargs kill -9
```

### Import errors
```bash
pip install -r requirements.txt
```

### Out of memory
```bash
# Reduce batch size in src/embedder.py
batch_size = 16  # Default: 32
```

## Key Features

✅ Natural language queries
✅ Semantic search with FAISS
✅ Cross-encoder reranking
✅ K/P assessment balancing
✅ REST API + Web UI
✅ Batch processing
✅ Evaluation metrics
✅ Production-ready

## Documentation

- README.md - Full documentation
- DEPLOYMENT.md - Deployment guide
- SUMMARY.md - Project summary
- This file - Quick reference

## Support

Questions? Check:
1. README.md troubleshooting section
2. DEPLOYMENT.md for production setup
3. examples.py for code samples
4. GitHub issues for help
