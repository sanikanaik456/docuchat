import time

import PyPDF2
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MODELS = ["gemini-3.5-flash", "gemini-3.7-flash"]
MAX_TRIES = 3
RETRY_WAIT_SECONDS = 15


def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def search_chunks(query, chunks, k=3):
    """Original keyword search: counts words shared with the question."""
    query_words = set(query.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]]


def search_chunks_tfidf(query, chunks, k=3):
    """Rank chunks with TF-IDF, which ignores common words and weights rare ones.
    Falls back to the keyword search if TF-IDF cannot be built."""
    if not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(chunks)
        scores = linear_kernel(vectorizer.transform([query]), matrix).ravel()
    except ValueError:
        return search_chunks(query, chunks, k)
    best = scores.argsort()[::-1][:k]
    return [chunks[i] for i in best]


def _is_busy_error(message):
    return any(code in message for code in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def generate_answer(query, relevant_chunks, api_key):
    client = genai.Client(api_key=api_key)
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Based on the following context only, answer the question.
If the answer is not in the context, say "I cannot find this in the document."

Context:
{context}

Question: {query}

Answer:"""

    for model in MODELS:
        for attempt in range(MAX_TRIES):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                if not _is_busy_error(str(e)):
                    raise
                if attempt < MAX_TRIES - 1:
                    time.sleep(RETRY_WAIT_SECONDS)
    raise RuntimeError(
        "Gemini is busy, or the free-tier limit (5 requests per minute) was reached. "
        "Please wait a minute and try again."
    )