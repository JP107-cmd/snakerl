import uvicorn
import os

api_host = os.getenv("API_HOST")
api_port = os.getenv("API_PORT")
api_reload = os.getenv("API_RELOAD")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=api_host,
        port=api_port,
        reload=api_reload,
    )
