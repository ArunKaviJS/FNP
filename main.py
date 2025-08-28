# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4

# Import functions from extract.py
from complete import run_local_ocr, update_job_status, fetch_job_status, itemdescription_function , process_invoice

app = FastAPI(title="Invoice Processing API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ======= Request Models =======
class FileJsonPayload(BaseModel):
    fileId: str


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
    filename = payload.fileId

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
    # ---- OCR Extract ----
        text, pages = run_local_ocr(file_path)

        # ---- Extract + Store Structured JSON in Mongo ----
        structured_response = itemdescription_function(text)

        # ---- Process Invoice (e.g., store in Mongo, enrich fields) ----
        sent = process_invoice(text)

        # ---- Save job status ----
        update_job_status(
            job_id,
            "completed",
            "OCR + Item description completed (stored in MongoDB)",
            {
                "raw_text": text,
                "pages": pages,
                "structured": structured_response,
                "invoice_status": sent,   # include invoice process result
            }
        )

        # ---- Return final job status ----
        return fetch_job_status(job_id)

    except Exception as e:
        update_job_status(job_id, "failed", str(e))
        return fetch_job_status(job_id)


# Run locally with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
