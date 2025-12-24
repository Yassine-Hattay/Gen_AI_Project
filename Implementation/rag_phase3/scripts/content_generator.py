# scripts/content_generator.py
import faiss
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
import os


import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import DB_CONFIG

# FAISS + embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss", "index.faiss")
index = faiss.read_index(FAISS_PATH)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Qwen2.5-3B-Instruct integration
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

def generate_content(next_concept, profile):
    """LLM + RAG content generation using Qwen2.5"""
    prefs = profile["preferences"]

    query = (
        f"{prefs['difficulty']} level, {prefs['modality']} explanation of "
        f"{next_concept} for a {profile['background']} student, "
        f"goal: {profile['goal']}, verbosity: {prefs['verbosity']}"
    )

    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb, dtype='float32'), k=3)

    retrieved_texts = []
    for idx in I[0]:
        cur.execute("SELECT text FROM chunks WHERE faiss_index=%s", (int(idx),))
        row = cur.fetchone()
        if row and row[0].strip():
            retrieved_texts.append(row[0])

    if not retrieved_texts:
        retrieved_texts = ["This is a default explanation placeholder."]
    
    # ===============================
    # ✅ FIX 1: CLEAN RETRIEVED TEXTS
    # ===============================
    cleaned_texts = []
    for txt in retrieved_texts:
        lines = txt.splitlines()
        lines = [
            l for l in lines
            if l.strip().lower() not in ("foreword", "introduction")
        ]
        cleaned_texts.append("\n".join(lines).strip())

    retrieved_texts = [t for t in cleaned_texts if t]
    # Build Qwen chat-style messages
    prefs = profile["preferences"]

    messages = [
        {
    "role": "system",
    "content": (
        "You are an educational tutor.\n"
        "- Do NOT repeat headings or words.\n"
        "- Do NOT write 'Foreword' or generic headers.\n"
        "- Produce a single, coherent explanation.\n"
        "- Avoid repetition.\n"
        "- Start directly with content."
    )
}
,
        {"role": "user", "content": (
            f"Create a {prefs['difficulty']} level, {prefs['modality']} explanation "
            f"with {prefs['verbosity']} verbosity "
            f"about '{next_concept}' for a {profile['background']} student "
            f"(goal: {profile['goal']}).\n\n"
            f"Use the following sources:\n" + "\n".join(retrieved_texts)
        )}
    ]


    # Apply Qwen chat template
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model_llm.device)

    # Generate response
    generated_ids = model_llm.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=False,   # IMPORTANT
        temperature=0.0,   # IMPORTANT
        pad_token_id=tokenizer.eos_token_id
        )   


    # Remove input tokens from output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    final_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return final_text
