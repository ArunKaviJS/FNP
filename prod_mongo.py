import os
import time
import traceback
from datetime import datetime, timezone
from typing import Tuple, List
import numpy as np
import json
from uuid import uuid4

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
import httpx
from openai import AzureOpenAI, RateLimitError
from bson import ObjectId

# ========= Load ENV =========
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

AZURE_EMBED_API_KEY = os.getenv("AZURE_EMBED_API_KEY")
AZURE_EMBED_ENDPOINT = os.getenv("AZURE_EMBED_ENDPOINT")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_EMBED_API_VERSION")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT")

# ========= MongoDB Connection =========
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
col_fnp = client["ak_fnp_embeds"]["fnp_embeddings"]
# invoice collection
PROD_MONGO_URI = os.getenv("PROD_MONGO_URI")
PRODclient = MongoClient(PROD_MONGO_URI)
db_invoice = PRODclient["yc-dev"]["tb_file_details"]

# ========= Jobs tracking =========
jobs = {}
invoice_number_var = None   # Global var to store invoice number (optional)

MIN_SIMILARITY = 0.75


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_job_status(job_id, status, message=None, data=None):
    jobs[job_id] = {
        "jobId": job_id,
        "status": status,
        "message": message,
        "data": data,
        "updatedAt": utc_now_iso()
    }


def fetch_job_status(job_id):
    return jobs.get(job_id)


# ========= OCR / PDF Extraction =========
def run_local_ocr(file_path: str) -> Tuple[str, int]:
    """Convert PDF/image → OCR text. Returns: Tuple[text:str, pages:int]"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return "", 0

        text_chunks = []
        pages_count = 0

        if file_path.lower().endswith(".pdf"):
            try:
                pages = convert_from_path(file_path, dpi=300)
                pages_count = len(pages)
                for page in pages:
                    text_chunks.append(pytesseract.image_to_string(page))
            except Exception:
                traceback.print_exc()
                return "", 0
        else:
            try:
                img = Image.open(file_path)
                text_chunks.append(pytesseract.image_to_string(img))
                pages_count = 1
            except Exception:
                traceback.print_exc()
                return "", 0

        return "\n".join(text_chunks).strip(), pages_count
    except Exception:
        traceback.print_exc()
        return "", 0


# ========= Azure LLM Agent =========
class AzureLLMAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            http_client=httpx.Client(),
        )
        self.model = AZURE_OPENAI_DEPLOYMENT
        self.RateLimitError = RateLimitError

    # ---- Embedding helper ----
    def embed_text(self, text: str) -> list:
        try:
            client = AzureOpenAI(
                api_key=AZURE_EMBED_API_KEY,
                azure_endpoint=AZURE_EMBED_ENDPOINT,
                api_version=AZURE_EMBED_API_VERSION,
                http_client=httpx.Client(),
            )
            resp = client.embeddings.create(
                model=AZURE_EMBED_DEPLOYMENT,
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return []

    # ---- LLM completion (generic) ----
    def complete(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert floral invoice processing system that extracts structured "
                            "invoice details (vendor, invoice number, date, items). "
                            "Always return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except self.RateLimitError:
            print("⚠️ Rate limit hit. Retrying after 5 seconds…")
            time.sleep(5)
            return self.complete(prompt)
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "{}"

    def build_prompt(self, extracted_text: str):
        schema = {
            "vendorName": "Vendor Name",
            "invoiceNo": "Invoice Number",
            "invoiceDate": "YYYY-MM-DD",
            "items": [
                {
                    "productCode": None,
                    "skuProductName": "product description from the invoice",
                    "quantity": 0,
                    "unitPrice": 0.0,
                    "totalAmount": 0.0,
                    "currency": "INR",
                    "matchConfidence": 0.0,
                }
            ]
        }

        return (
            f"You are an expert floral invoice processing system.\n"
            f"Return ONLY valid JSON (no explanations, no markdown).\n\n"
            f"Important rules:\n"
            f"- Always use `null` for missing or unknown values (e.g. productCode).\n"
            f"- If the invoice currency is detected as 'AED', replace it with 'INR'.\n"
            f"- All currency values in the final JSON must be 'INR'.\n"
            f"- Follow this schema strictly:\n{json.dumps(schema, indent=2)}\n\n"
            f"Extract JSON from this invoice text:\n{extracted_text}"
        )

    def extract_invoice_and_items(self, ocr_text: str) -> dict:
        """Use LLM to extract invoice number and item descriptions. JSON-only."""
        prompt = f"""
You are given raw OCR text extracted from an invoice.

Task:
1. Extract the Invoice Number (example: 71281).
2. Extract all the product/item description names (strings).

Return strictly in JSON format:
{{
  "invoiceNumber": "<string or null>",
  "itemDescriptions": ["<item1>", "<item2>", "..."]
}}

OCR text:
---
{ocr_text}
---
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
            )
            raw_text = resp.choices[0].message.content.strip()
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            json_text = raw_text[start:end]
            data = json.loads(json_text)
        except Exception as e:
            print(f"⚠️ LLM parsing error: {e}")
            data = {"invoiceNumber": None, "itemDescriptions": []}
        return data

    def choose_best_code(self, sku_name: str, candidates: list) -> str:
        """Ask LLM to pick the most relevant productCode from vector candidates."""
        prompt = f"""
You are a floral product matcher.
Original itemDescription: "{sku_name}"

Candidates:
{json.dumps(candidates, indent=2)}

Pick ONLY the best matching candidate's productCode.
Return strictly in JSON:
{{"productCode": "xxxx"}}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a JSON-only responder."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            json_text = text[start:end]
            parsed = json.loads(json_text)
            return parsed.get("productCode")
        except Exception as e:
            print(f"⚠️ LLM selection error: {e}")
            return None


# ========= Similarity & Retrieval =========
def cosine_similarity(vec1, vec2) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def get_top_chunks(helper: AzureLLMAgent, search_text: str, top_k: int = 3) -> List[dict]:
    query_embedding = helper.embed_text(search_text)
    if not query_embedding:
        return []

    docs = list(col_fnp.find({}, {"embeddings": 1, "text": 1, "rmdtc_code": 1, "_id": 0}))
    similarities = []

    for doc in docs:
        text = doc.get("text", "")
        rmdtc_code = doc.get("rmdtc_code")
        for emb in doc.get("embeddings", []):
            score = cosine_similarity(query_embedding, emb)
            if score >= MIN_SIMILARITY:
                code = rmdtc_code
                if not code and "code:" in text.lower():
                    try:
                        after = text.lower().split("code:")[1].strip()
                        code = after.split()[0]
                    except Exception:
                        pass
                similarities.append({
                    "productCode": code,
                    "itemDescription": text,
                    "score": score
                })

    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]


# ========= Phase 2: Product code enrichment =========
def process_invoice(ocr_text: str, payload: dict):
    """
    Takes raw OCR text, extracts invoice number + item descriptions,
    and updates MongoDB updatedExtractedValues with productCode.
    """
    helper = AzureLLMAgent()
    parsed = helper.extract_invoice_and_items(ocr_text)
    invoice_no = parsed.get("invoiceNumber")
    item_descs = parsed.get("itemDescriptions", []) or []

    print(f"\n📄 Invoice No: {invoice_no}")
    final_mappings = []

    for item in item_descs:
        print(f"\n🔎 Processing item: {item}")

        top_chunks = get_top_chunks(helper, item, top_k=3)
        if not top_chunks:
            print(f"⚠️ No matches found for {item}")
            continue

        best_code = helper.choose_best_code(item, top_chunks)
        if best_code:
            final_mappings.append({"itemDescription": item, "productCode": best_code})
            print(f"✅ {item} → {best_code}")

            # ----- MongoDB update in updatedExtractedValues -----
            result = db_invoice.update_one(
                {"clusterId": payload["clusterId"], "userId": payload["userId"], "fileName": payload["fileName"]},
                {"$set": {"updatedExtractedValues.items.$[elem].productCode": best_code}},
                array_filters=[{"elem.skuProductName": item}]
            )
            if result.modified_count > 0:
                print(f"📝 Updated MongoDB for {item} with productCode {best_code}")
            else:
                print(f"⚠️ No MongoDB row updated for {item} (check exact string match)")
        else:
            print(f"⚠️ Could not assign productCode for {item}")

    print("\n=== Final Results ===")
    for mapping in final_mappings:
        print(mapping)


# ========= Phase 1: Structured Extraction =========def itemdescription_function(extracted_text: str):
def itemdescription_function(extracted_text: str):
    """
    Builds a structured invoice JSON using AzureLLMAgent.
    Returns dict (not inserted).
    """
    global invoice_number_var
    agent = AzureLLMAgent()

    # ---- Call LLM with structured schema prompt ----
    prompt = agent.build_prompt(extracted_text)
    structured_json_text = agent.complete(prompt)

    # ---- Canonical fallback extraction ----
    canon = agent.extract_invoice_and_items(extracted_text)
    canon_invoice_no = canon.get("invoiceNumber")
    canon_items = canon.get("itemDescriptions", []) or []

    try:
        s = structured_json_text.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:].strip()

        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed structure is not a JSON object.")

        # Ensure invoice number is set
        if canon_invoice_no and not parsed.get("invoiceNo"):
            parsed["invoiceNo"] = canon_invoice_no

        # If LLM returned no items, fall back to canon_items
        if not parsed.get("items"):
            rebuilt_items = []
            for name in canon_items:
                rebuilt_items.append({
                    "productCode": None,
                    "skuProductName": name,
                    "quantity": None,       # leave None instead of 0
                    "unitPrice": None,
                    "totalAmount": None,
                    "currency": "INR",
                    "matchConfidence": 0.0,
                })
            parsed["items"] = rebuilt_items
        else:
            # Normalize currency to INR in all items
            for item in parsed["items"]:
                if not item.get("currency"):
                    item["currency"] = "INR"
                elif item["currency"].upper() == "AED":
                    item["currency"] = "INR"
                else:
                    item["currency"] = "INR"

        # Store invoice number globally
        invoice_number_var = parsed.get("invoiceNo")

        return parsed

    except Exception as e:
        print(f"⚠️ Could not parse invoice JSON: {e}")
        return {
            "rawStructured": structured_json_text,
            "invoiceNo": canon_invoice_no,
            "itemDescriptions": canon_items,
        }



# ========= End-to-End Runner =========

def run_pipeline(file_path: str, payload: dict):
    """
    Full pipeline:
    1) OCR the file
    2) Update existing MongoDB document with extractedValues & updatedExtractedValues
    3) Enrich productCode for each item
    """
    job_id = str(uuid4())
    update_job_status(job_id, "started", f"Processing file: {file_path}")

    try:
        # ---- OCR ----
        text, pages = run_local_ocr(file_path)
        if not text:
            update_job_status(job_id, "failed", "OCR failed or empty text")
            print("❌ OCR returned no text.")
            return

        # ---- Phase 1: Structured JSON ----
        structured_response = itemdescription_function(text)

        # ---- Build wrapper invoice doc ----
        invoice_doc = {
            "pagesCount": pages,
            "processingStatus": "Completed",
            "processingMessage": "Processing completed successfully",
            "extractedText": text,
            "extractedValues": structured_response,
            "updatedExtractedValues": structured_response,
            "updatedAt": datetime.now(timezone.utc),
        }

        # ---- Ensure clusterId & userId are ObjectId ----
        try:
            cluster_oid = ObjectId(payload["clusterId"])
            user_oid = ObjectId(payload["userId"])
        except Exception as e:
            print(f"❌ Invalid ObjectId format in payload: {e}")
            update_job_status(job_id, "failed", "Invalid ObjectId in payload")
            return

        # ---- Update existing Mongo row (no insert) ----
        result = db_invoice.update_one(
            {
                "clusterId": cluster_oid,
                "userId": user_oid,
                "fileName": payload["fileName"]
            },
            {"$set": invoice_doc},
            upsert=False  # ⚡ Do NOT insert if missing
        )

        if result.matched_count > 0:
            print(f"📝 Updated existing MongoDB row for {payload['fileName']}")
        else:
            print(f"⚠️ No matching document found for {payload['fileName']} (nothing updated)")

        # ---- Phase 2: Enrich product codes ----
        process_invoice(text, {
            "clusterId": str(cluster_oid),  # pass back as str for consistency
            "userId": str(user_oid),
            "fileName": payload["fileName"]
        })

        update_job_status(
            job_id,
            "completed",
            "OCR + Structured update + Product code enrichment done",
            {"pages": pages}
        )
        print(f"✅ Job {job_id} completed.")

    except Exception as e:
        traceback.print_exc()
        update_job_status(job_id, "failed", f"Unhandled error: {e}")
