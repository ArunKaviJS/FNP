import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4

# Import unified pipeline from complete.py
from prod_mongo import run_pipeline, fetch_job_status, update_job_status

app = FastAPI(title="Invoice Processing API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ======= Request Models =======
class FileJsonPayload(BaseModel):
    clusterId: str
    userId: str
    fileName: str


# ======= Routes =======

# Health Check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Invoice OCR API running"}


# Job Status
@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = fetch_job_status(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})
    return job


# Upload + Process OCR → LLM → Store in Mongo
@app.post("/upload/")
def upload_json(payload: FileJsonPayload):
    filename = payload.fileName

    # Check local folder first, then uploads folder
    local_path = os.path.join(os.getcwd(), filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(local_path):
        file_path = local_path
    elif os.path.exists(upload_path):
        file_path = upload_path
    else:
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found in current directory or uploads folder."
        )

    job_id = str(uuid4())
    update_job_status(job_id, "processing", "OCR processing started")

    try:
        # Prepare payload for run_pipeline
        pipeline_payload = {
            "clusterId": payload.clusterId,
            "userId": payload.userId,
            "fileName": filename
        }

        # Run complete pipeline (OCR + extraction + insert + enrichment)
        run_pipeline(file_path, pipeline_payload)

        # Update job status as completed
        update_job_status(
            job_id,
            "completed",
            "Invoice processing completed successfully",
            {"fileName": filename, "clusterId": payload.clusterId, "userId": payload.userId}
        )

        return fetch_job_status(job_id)

    except Exception as e:
        update_job_status(job_id, "failed", str(e))
        return fetch_job_status(job_id)


# Run locally with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
