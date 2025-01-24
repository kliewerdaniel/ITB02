from fastapi import FastAPI
from backend.api.routers.story import router as story_router

app = FastAPI()
app.include_router(story_router, prefix="/story")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}