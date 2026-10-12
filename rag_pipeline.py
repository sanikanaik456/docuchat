import PyPDF2
from google import genai

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def search_chunks(query, chunks, k=3):
    query_words = set(query.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]]

def generate_answer(query, relevant_chunks, api_key):
    client = genai.Client(api_key=api_key)
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Based on the following context only, answer the question.
If the answer is not in the context, say "I cannot find this in the document."

Context:
{context}

Question: {query}

Answer:"""
    
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt
    )
    return response.text