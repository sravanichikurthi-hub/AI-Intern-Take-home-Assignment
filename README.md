# SHL Assessment Recommender API

Conversational FastAPI agent that recommends SHL Individual Test Solutions.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. (Optional) Re-scrape the catalog — only needed if SHL updates their site
python scraper.py   # writes catalog.json

# 4. Run the API
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` when ready |
| POST | `/chat` | Conversational assessment recommender |

## Request / Response Schema

### POST /chat

**Request**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a senior Python data engineer"},
    {"role": "assistant", "content": "What level of stakeholder interaction does this role require?"},
    {"role": "user", "content": "High — they present findings to leadership"}
  ]
}
```

**Response**
```json
{
  "reply": "Based on your requirements, here are my top recommendations...",
  "recommendations": [
    {
      "name": "Python (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

## Deployment (Render free tier)

1. Push this repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `ANTHROPIC_API_KEY = sk-ant-...`

## Architecture

```
POST /chat
    │
    ▼
ChatRequest validation (Pydantic)
    │
    ▼
SHLAgent.chat(messages)
    ├── _extract_query_from_history()     ← last 3 user turns joined
    ├── _detect_test_type_hints()         ← keyword-based type pre-filter
    └── CatalogRetriever.search(query)    ← FAISS cosine similarity
            │
            ▼
        Top-20 catalog items injected into system prompt
            │
            ▼
        Anthropic claude-sonnet-4 API call
            │
            ▼
        _parse_response() + URL validation
            │
            ▼
ChatResponse (Pydantic serialisation)
```

## Test Cases

```bash
# Health check
curl http://localhost:8000/health

# Vague query — agent should clarify
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need an assessment"}]}'

# Role-specific query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who collaborates with stakeholders"}]}'
```
