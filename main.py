from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from openai import OpenAI
import re
from io import StringIO
import csv
from datetime import datetime
from dotenv import load_dotenv
import os

# === SETTINGS ===
load_dotenv()  # Load environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Tesseract path setup
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # Linux (server)
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ✅ Poppler path (Windows only, Linux uses system installation)
POPLER_PATH = r"C:\poppler\Library\bin" if os.name == "nt" else None

# === FASTAPI APP ===
app = FastAPI(title="Invoice Extractor Website")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# === HELPER FUNCTIONS ===
def detect_supplier(ocr_text: str) -> str:
    match = re.search(r'\b(UAB|MB)\s+"?([A-Za-z0-9\s]+)"?', ocr_text, re.IGNORECASE)
    if match:
        supplier = match.group(2).strip()
        if supplier.lower() != "maisto sprendimai":
            return supplier
    return "Unknown Supplier"


def extract_table_lines(ocr_text: str) -> str:
    return "\n".join([
        line.strip()
        for line in ocr_text.split("\n")
        if any(char.isdigit() for char in line) and len(line.split()) > 2
    ])


def process_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    # ✅ Convert PDF to images (Poppler only needed for Windows)
    images = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPLER_PATH)
    ocr_text = ""

    for page in images:
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        ocr_text += f"\n{pytesseract.image_to_string(img, config='--psm 6')}\n"

    supplier = detect_supplier(ocr_text)
    cleaned_ocr_text = extract_table_lines(ocr_text)

    # ✅ Send to OpenAI for table extraction
    prompt = f"""
    Extract ONLY the product table from this invoice.
    Output clean CSV in this format:
    Pavadinimas,Kiekis,Kaina be PVM,Suma EUR

    OCR text:
    {cleaned_ocr_text}

    Return only CSV, no explanation.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise OCR table reconstruction assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    table_csv = response.choices[0].message.content.strip()

    # ✅ Parse CSV into DataFrame
    rows = []
    reader = csv.reader(StringIO(table_csv))
    for row in reader:
        if len(row) == 1 and "," in row[0]:
            row = [col.strip() for col in row[0].split(",")]
        rows.append(row)

    if len(rows) > 1:
        fixed_rows = []
        for r in rows:
            if len(r) > 4:
                pavadinimas = " ".join(r[:-3]).strip()
                fixed_rows.append([pavadinimas, r[-3], r[-2], r[-1]])
            elif len(r) < 4:
                r += [""] * (4 - len(r))
                fixed_rows.append(r)
            else:
                fixed_rows.append(r)

        df = pd.DataFrame(fixed_rows[1:], columns=["Pavadinimas", "Kiekis", "Kaina be PVM", "Suma EUR"])
    else:
        df = pd.DataFrame(columns=["Pavadinimas", "Kiekis", "Kaina be PVM", "Suma EUR"])

    df["Supplier"] = supplier
    return df


# === ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        df = process_pdf(pdf_bytes)

        if df.empty:
            return JSONResponse({"message": "No table extracted."}, status_code=200)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        df.to_excel(output_file, index=False, engine="openpyxl")

        return FileResponse(output_file, filename=os.path.basename(output_file))

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
