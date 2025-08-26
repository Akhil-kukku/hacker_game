from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from game_logic import GameEngine
import uvicorn

app = FastAPI(title="Hacker Game API", version="1.0.0")
engine = GameEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Hacker Game API is running"}

@app.get("/levels")
def get_levels():
    return engine.get_level_list()

@app.post("/start")
async def start_level(request: Request):
    data = await request.json()
    level_index = data.get("level", 0)
    return engine.start_level(level_index)

@app.post("/action")
async def take_action(request: Request):
    data = await request.json()
    return engine.process_action(data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
