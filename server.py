from fastapi import FastAPI, HTTPException
from experiment import experiment
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=4)

perf_folder = os.path.join('.', 'perf_data')

if not os.path.exists(os.path.join(perf_folder, "binary")):
    os.makedirs(os.path.join(perf_folder, "binary"), exist_ok=True)

if not os.path.exists(os.path.join(perf_folder, "logs")):
    os.makedirs(os.path.join(perf_folder, "logs"), exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Hello World"}

class ExperimentRequest(BaseModel):
    scheduler: str
    cpu: int
    cpu_method: str
    io: int
    mem_load: int
    duration: float
    interval: float

@app.post("/experiment")
async def run_experiment(req: ExperimentRequest):
    """
    Run an experiment and return the results.
    Executes in a thread so the FastAPI server stays fast.
    """
    try:
        loop = asyncio.get_event_loop()

        # Run experiment in background thread
        result = await loop.run_in_executor(
            executor,
            lambda: experiment(
                req.scheduler,
                req.cpu,
                req.cpu_method,
                req.io,
                req.mem_load,
                req.duration,
                req.interval
            )
        )

        return {"status": "ok", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))