import os
import time
import traceback
from datetime import datetime, timezone
from typing import Tuple , List
import numpy as np
from uuid import uuid4
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
import httpx
from openai import AzureOpenAI, RateLimitError

# === Local Import for Product Code Filler ===
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

# NOTE: keep invoice_collection under ak_fnp_embeddings as you said earlier
col_fnp = client["ak_fnp_embeds"]["fnp_embeddings"]  # embeddings collection

db_invoice = client["ak_fnp_embeddings"]["invoice_collection"]

# ========= Jobs tracking =========
jobs = {}
invoice_number_var = None   # Global var to store invoice number

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

    # ---- Search (optional future use) ----
    def search_similar(self, query_text: str, top_k: int = 3):
        query_emb = self.embed_text(query_text)
        if not query_emb:
            return []
        results = db_invoice.aggregate([
            {
                "$vectorSearch": {
                    "queryVector": query_emb,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": top_k,
                    "index": "vector_index"
                }
            }
        ])
        return list(results)

    # ---- LLM completion ----
    def complete(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert floral invoice processing system that extracts structured "
                            "invoice details (vendor, invoice number, date, items)."
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

    # ---- Prompt builder ----
    def build_prompt(self, extracted_text: str):
        schema = {
            "vendorName": "Vendor Name",
            "invoiceNo": "Invoice Number",
            "invoiceDate": "YYYY-MM-DD",
            "items": [
                {
                    "productCode": None,   # <-- force null
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
    
    # ---- Prompt builder (invoice no + items only) ----
    def extract_invoice_and_items(self, ocr_text: str) -> dict:
        """Use LLM to extract invoice number and item descriptions"""
        prompt = f"""
You are given raw OCR text extracted from an invoice.

Task:
1. Extract the Invoice Number (example: 71281).
2. Extract all the product/item description names.

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
        """Ask LLM to pick the most relevant productCode"""
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

    
def cosine_similarity(vec1, vec2) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


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
                if not code and "code:" in text:
                    code = text.split("code:")[1].strip().split()[0]
                similarities.append({
                    "productCode": code,
                    "itemDescription": text,
                    "score": score
                })

    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]




def process_invoice(text):
    helper = AzureLLMAgent()
    
    parsed = helper.extract_invoice_and_items(text)
    invoice_no = parsed.get("invoiceNumber")
    item_descs = parsed.get("itemDescriptions", [])

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

            # ----- MongoDB update -----
            result = db_invoice.update_one(
                {"invoiceNo": invoice_no, "items.skuProductName": item},
                {"$set": {"items.$.productCode": best_code}}
            )
            if result.modified_count > 0:
                print(f"📝 Updated MongoDB for {item} with productCode {best_code}")
            else:
                print(f"⚠️ No MongoDB row updated for {item}")

        else:
            print(f"⚠️ Could not assign productCode for {item}")

    print("\n=== Final Results ===")
    for mapping in final_mappings:
        print(mapping)

# ========= Structured Extraction =========
def itemdescription_function(extracted_text: str):
    global invoice_number_var
    agent = AzureLLMAgent()

    prompt = agent.build_prompt(extracted_text)
    structured_json = agent.complete(prompt)

    try:
        # cleanup markdown fences if LLM wraps output
        structured_json = structured_json.strip()
        if structured_json.startswith("```"):
            structured_json = structured_json.strip("`")
            if structured_json.lower().startswith("json"):
                structured_json = structured_json[4:].strip()

        parsed = json.loads(structured_json)

        # normalize productCode → None
        if "items" in parsed and isinstance(parsed["items"], list):
            for item in parsed["items"]:
                if not item.get("productCode") or str(item["productCode"]).lower() in ["null", "n/a", ""]:
                    item["productCode"] = None

        # set invoice number variable
        invoice_number_var = parsed.get("invoiceNo")

        # 👉 insert into MongoDB
        result = db_invoice.insert_one(parsed)

        # ✅ fill missing product codes (post-processing step)
        

        # return parsed + Mongo _id
        parsed["_id"] = str(result.inserted_id)
        return parsed

    except Exception as e:
        print(f"⚠️ Could not parse invoice JSON: {e}")
        return {"rawStructured": structured_json}
    

# Example run
if __name__ == "__main__":
    process_invoice("uploads/Invoice.pdf")
