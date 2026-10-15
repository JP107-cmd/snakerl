from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from model.agent import Agent

app = FastAPI()

@app.get("/cnn")
async def run_cnn():
    agent = Agent(hyperparameter_set="snakev3-0", use_cnn=True)
    return StreamingResponse(agent.display(), media_type="text/event-stream")