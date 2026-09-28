from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.creative import router as creative_router

app = FastAPI(
    title="Ascent AI API",
    description="Backend API for Ascent AI Creative Studio",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    creative_router,
    prefix="/api"
)


@app.get("/")
def home():
    return {
        "message": "Ascent AI backend is running!"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Ascent AI"
    }