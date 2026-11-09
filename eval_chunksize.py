import os
import time

import faiss
import numpy as np
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rag_pipeline import extract_text, chunk_text, search_chunks

PDF_PATH = r"C:\Users\HP\OneDrive\Documents\Explainable_Deep_Learning_Credit_Risk final .pdf"
EMBED_MODEL = "gemini-embedding-001"
SIZES = [250, 150]

TESTS = [
    ("What AUC did the deep learning model reach on the credit default task?", "0.856"),
    ("What AUC did the deep learning model reach on the loan approval task?", "0.995"),
    ("At the default threshold, what percentage of creditworthy borrowers under 35 were wrongly flagged as high risk?", "44.9%"),
    ("What was the overall accuracy before and after group-specific thresholds?", "77.5%"),
    ("What were the underlying default rates for the three age bands?", "11.8%"),
    ("How many age bands was the model audited across?", "three age bands"),
    ("What AUC did gradient boosting achieve on the default task?", "0.868"),
    ("Which group was wrongly flagged most often, and how large was the gap to the oldest group?", "nearly four times"),
    ("Did making the model fairer cost any overall accuracy?", "83.8%"),
    ("Why did the author say some difference in treatment across age groups was expected?", "genuinely differ"),
    ("Which researcher argued against relying on explanations produced after the model is trained?", "Rudin"),
    ("How stable were the results when the data was split several different ways?", "0.837"),
    ("By how much did the oldest group's error rate rise after the fix?", "15.0%"),
]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed(texts, task):
    vectors = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i + 10]
        for attempt in range(4):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task),
                )
                vectors.extend(e.values for e in result.embeddings); time.sleep(20)
                break
            except Exception as e:
                busy = any(code in str(e) for code in ("503", "429", "UNAVAILABLE"))
                if busy and attempt < 3:
                    time.sleep(30)
                else:
                    raise
    array = np.array(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


text = extract_text(PDF_PATH)
chunks_by_size = {s: chunk_text(text, chunk_size=s) for s in SIZES}

# keep only questions whose phrase sits inside one chunk at every size
common = [(q, p) for q, p in TESTS
          if all(any(p.lower() in c.lower() for c in chunks_by_size[s]) for s in SIZES)]
dropped = [p for q, p in TESTS if (q, p) not in common]
print(f"Questions used at every chunk size: {len(common)} of {len(TESTS)}")
if dropped:
    print("Dropped because the phrase was split across chunks:", dropped)

query_vectors = embed([q for q, _ in common], "RETRIEVAL_QUERY")

for size in SIZES:
    chunks = chunks_by_size[size]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(chunks)
    vectors = embed(chunks, "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    scores = {"keyword": 0, "tf-idf": 0, "embeddings": 0}
    emb_misses = []
    for i, (q, p) in enumerate(common):
        kw = search_chunks(q, chunks, k=3)
        sims = linear_kernel(tfidf.transform([q]), matrix).ravel()
        tf = [chunks[j] for j in sims.argsort()[::-1][:3]]
        _, ids = index.search(query_vectors[i:i + 1], 3)
        em = [chunks[j] for j in ids[0]]
        for name, top3 in (("keyword", kw), ("tf-idf", tf), ("embeddings", em)):
            hit = any(p.lower() in c.lower() for c in top3)
            scores[name] += hit
            if name == "embeddings" and not hit:
                emb_misses.append(q[:70])
    print(f"\nChunk size {size} words ({len(chunks)} chunks): "
          + ", ".join(f"{m} {s}/{len(common)}" for m, s in scores.items()))
    for q in emb_misses:
        print("   embeddings missed:", q)
