# Deployment Guide

## Quick Start Deployment

### Prerequisites
- Python 3.8+ installed
- pip package manager
- Internet connection (for initial model download)
- 2GB+ RAM

### Step-by-Step Deployment

#### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/HarshMishra-Git/SHL-Assessment.git
cd SHL-Assessment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Initialize System

```bash
# Run automated setup (generates catalog, builds index)
python setup.py
```

This will:
- Generate SHL catalog (25+ assessments)
- Preprocess training data (if available)
- Download models from Hugging Face (~150MB total)
- Build FAISS search index
- Run evaluation on training set

**Note**: First run takes 5-10 minutes due to model downloads.

#### 3. Start Services

**Option A: API Server**
```bash
# Start FastAPI server
python api/main.py

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access API at: http://localhost:8000
API Docs at: http://localhost:8000/docs

**Option B: Web Interface**
```bash
# Start Streamlit UI
streamlit run app.py
```

Access UI at: http://localhost:8501

**Option C: Both (separate terminals)**
```bash
# Terminal 1 - API
python api/main.py

# Terminal 2 - UI
streamlit run app.py
```

## Production Deployment

### Using Gunicorn (API)

```bash
# Install gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Start API
pm2 start "uvicorn api.main:app --host 0.0.0.0 --port 8000" --name shl-api

# Start UI
pm2 start "streamlit run app.py --server.port 8501" --name shl-ui

# View logs
pm2 logs

# Stop services
pm2 stop all
```

### Using Systemd (Linux)

Create `/etc/systemd/system/shl-api.service`:
```ini
[Unit]
Description=SHL Assessment API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/SHL-Assessment
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl start shl-api
sudo systemctl enable shl-api
sudo systemctl status shl-api
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/shl
server {
    listen 80;
    server_name your-domain.com;

    # API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # UI
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/shl /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Cloud Deployment

### AWS EC2

1. Launch EC2 instance (t2.medium or larger)
2. Install Python 3.8+
3. Clone repository
4. Follow deployment steps above
5. Configure security groups (ports 8000, 8501)

### Google Cloud Run

Create `Dockerfile` and deploy:
```bash
gcloud run deploy shl-api --source .
```

### Heroku

Create `Procfile`:
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Deploy:
```bash
heroku create shl-recommender
git push heroku main
```

## Environment Variables

Create `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Performance
BATCH_SIZE=32
MAX_WORKERS=4

# Paths
DATA_DIR=data
MODELS_DIR=models
```

Load in code:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Expected response:
# {"status":"API is running","timestamp":"..."}
```

### Logging

Logs are written to stdout. Capture with:
```bash
# API logs
python api/main.py > logs/api.log 2>&1

# UI logs
streamlit run app.py > logs/ui.log 2>&1
```

### Performance Monitoring

Add monitoring endpoints in `api/main.py`:
```python
@app.get("/metrics")
async def metrics():
    return {
        "total_requests": request_counter,
        "avg_response_time": avg_response_time,
        "uptime": uptime
    }
```

## Scaling

### Horizontal Scaling

Deploy multiple API instances behind load balancer:
```bash
# Instance 1
uvicorn api.main:app --port 8000

# Instance 2
uvicorn api.main:app --port 8001

# Instance 3
uvicorn api.main:app --port 8002
```

Use nginx load balancing:
```nginx
upstream shl_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    location /api/ {
        proxy_pass http://shl_api/;
    }
}
```

### Caching

Add Redis caching for frequent queries:
```python
import redis
cache = redis.Redis(host='localhost', port=6379)

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    cache_key = f"query:{hash(request.query)}"
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Generate recommendations
    result = ...
    cache.setex(cache_key, 3600, json.dumps(result))
    return result
```

## Security

### API Authentication

Add API key authentication:
```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/recommend", dependencies=[Depends(verify_api_key)])
async def recommend(request: RecommendRequest):
    ...
```

### HTTPS

Use certbot for Let's Encrypt SSL:
```bash
sudo certbot --nginx -d your-domain.com
```

### Rate Limiting

Add rate limiting:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/recommend")
@limiter.limit("10/minute")
async def recommend(request: Request, ...):
    ...
```

## Troubleshooting

### Models Not Loading
```bash
# Download models manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Port Already in Use
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

### Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=16

# Or use swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Backup and Recovery

### Backup Important Files
```bash
# Backup models and data
tar -czf backup.tar.gz models/ data/ evaluation_results.json

# Restore
tar -xzf backup.tar.gz
```

### Automated Backups
```bash
# Add to crontab
0 2 * * * tar -czf ~/backups/shl-$(date +\%Y\%m\%d).tar.gz /path/to/SHL-Assessment/models /path/to/SHL-Assessment/data
```

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review troubleshooting section
3. Open GitHub issue
4. Contact support team
