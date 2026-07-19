from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import yaml

from .database.vector_store import VectorStoreManager

# Load config
import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Dynamically resolve config path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try multiple candidate locations so running from project root or inside backend works
candidate_paths = [
    os.path.join(BASE_DIR, "config", "config.yaml"),
    os.path.join(BASE_DIR, "backend", "config", "config.yaml"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")
]

config = None
for p in candidate_paths:
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info("Loaded config from %s", p)
            break
        except Exception as e:
            logger.warning("Failed to load config at %s: %s", p, e)

if config is None:
    # Fallback to minimal defaults so the API can start for development
    logger.warning(
        "No config file found at expected locations. Falling back to safe defaults: %s",
        candidate_paths,
    )
    config = {
        'api': {
            'host': '127.0.0.1',
            'port': 8000,
            'cors_origins': ['*']
        },
        'vector_store': {},
    }

app = FastAPI(title="FBSL-KAGS API with Finnish Floor Plans", version="1.0.0")

# Initialize services
vector_store = VectorStoreManager()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the simple frontend from the same FastAPI app (if present)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    # Serve directory and allow index.html to be returned for directory requests
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    @app.get("/frontend", include_in_schema=False)
    @app.get("/frontend/", include_in_schema=False)
    async def _frontend_root():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        return RedirectResponse("/frontend/")

    @app.get("/frontend-debug", include_in_schema=False)
    async def _frontend_debug():
        """Developer debug endpoint: returns index.html or error details."""
        try:
            index_path = os.path.join(FRONTEND_DIR, "index.html")
            if not os.path.exists(index_path):
                return {"error": "index.html not found", "path": index_path}
            return FileResponse(index_path, media_type="text/html")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return {
                "error": "exception serving frontend",
                "detail": str(e),
                "traceback": tb
            }

class RoomSearchRequest(BaseModel):
    room_type: str
    embedding_type: str = "composite"
    top_k: int = 10

class SimilaritySearchRequest(BaseModel):
    query_text: str
    embedding_type: str = "composite"
    top_k: int = 5
    filter_room_type: Optional[str] = None

@app.get("/")
async def root():
    finnish_status = "loaded" if vector_store.finnish_embeddings else "not loaded"
    
    stats = {}
    if vector_store.finnish_embeddings:
        stats = vector_store.finnish_embeddings.get_plan_statistics()
    
    return {
        "message": "FBSL-KAGS API with Finnish Floor Plan Embeddings",
        "status": "success",
        "finnish_embeddings": finnish_status,
        "stats": stats
    }

@app.post("/finnish/search/rooms")
async def search_finnish_rooms(request: RoomSearchRequest):
    """Search Finnish floor plans by room type"""
    results = vector_store.search_finnish_rooms(
        request.room_type,
        embedding_type=request.embedding_type,
        top_k=request.top_k
    )
    
    return {
        "query": request.room_type,
        "results_count": len(results),
        "results": results
    }

@app.post("/finnish/search/similar")
async def search_similar_spaces(request: SimilaritySearchRequest):
    """Search for similar spaces using text query"""
    # Encode query to embedding
    query_embedding = vector_store.embedding_model.encode(request.query_text)
    
    # Search
    results = vector_store.search_similar_finnish_spaces(
        query_embedding,
        embedding_type=request.embedding_type,
        top_k=request.top_k,
        filter_room_type=request.filter_room_type
    )
    
    return {
        "query": request.query_text,
        "filter": request.filter_room_type,
        "results_count": len(results),
        "results": results
    }

@app.get("/finnish/plan/{plan_id}/adjacencies")
async def get_plan_adjacencies(plan_id: str):
    """Get room adjacencies for a specific plan"""
    adjacencies = vector_store.get_finnish_room_adjacencies(plan_id)
    
    if adjacencies is None:
        raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
    
    return {
        "plan_id": plan_id,
        "adjacencies": adjacencies
    }

@app.get("/finnish/stats")
async def get_finnish_stats():
    """Get statistics about Finnish embeddings"""
    if vector_store.finnish_embeddings is None:
        raise HTTPException(status_code=404, detail="Finnish embeddings not loaded")
    
    return vector_store.finnish_embeddings.get_plan_statistics()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "connected",
        "finnish_embeddings": "loaded" if vector_store.finnish_embeddings else "not loaded"
    }


class PipelineRequest(BaseModel):
    project_name: Optional[str] = None
    requirements: Optional[str] = None
    context: Optional[dict] = None
    max_alternatives: Optional[int] = 5
    use_got: Optional[bool] = True
    enable_convergence_loop: Optional[bool] = True

    # GoT tuning parameters
    got_delta: Optional[float] = None
    got_patience: Optional[int] = None
    got_max_nodes: Optional[int] = None
    got_selection_metric: Optional[str] = None


@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """Run the full pipeline (synchronously) and return results.

    Accepts GoT tuning parameters: `got_delta`, `got_patience`, `got_max_nodes`, `got_selection_metric`.
    """
    from backend.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()

    # Build request dict expected by orchestrator.process_design_request
    req = {
        'project_name': request.project_name,
        'requirements': request.requirements,
        'context': request.context or {},
        'max_alternatives': request.max_alternatives,
        'use_got': request.use_got,
        'enable_convergence_loop': request.enable_convergence_loop,
        'got_delta': request.got_delta,
        'got_patience': request.got_patience,
        'got_max_nodes': request.got_max_nodes,
        'got_selection_metric': request.got_selection_metric
    }

    result = await orchestrator.process_design_request(req)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])