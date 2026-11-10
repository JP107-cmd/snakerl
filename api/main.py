from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from model.agent import Agent
import os

frontend_urls = os.getenv("FRONTEND_URLS")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=frontend_urls,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/cnn")
async def run_cnn():
    agent = Agent(hyperparameter_set="snakev3-2", use_cnn=True)
    return StreamingResponse(agent.display(), media_type="application/x-ndjson")

@app.get("/mlp")
async def run_cnn():
    agent = Agent(hyperparameter_set="snakev2-1", use_cnn=False)
    return StreamingResponse(agent.display(), media_type="application/x-ndjson")