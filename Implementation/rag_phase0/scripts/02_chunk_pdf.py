import pdfplumber
import re
import pickle
import os

PDF_PATH = "data/pdfs/Basics_of_math.pdf"
CHUNKS_PATH = "data/chunks.pkl"

def detect_chunk_type(text):
    if re.search(r"example", text, re.I):
        return "example"
    if re.search(r"exercise", text, re.I):
        return "exercise"
    if re.search(r"solution", text, re.I):
        return "solution"
    return "explanation"

chunks = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        sections = text.split("\n\n")
        for sec in sections:
            if len(sec.split()) < 50:
                continue

            chunks.append({
                "text": sec.strip(),
                "chunk_type": detect_chunk_type(sec),
                "difficulty": "beginner"
            })

print(f"Extracted {len(chunks)} chunks")

# Save chunks to disk
os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"Saved chunks to {CHUNKS_PATH}")
