\# DocuChat — RAG-Powered PDF Question Answering



Upload any PDF and ask questions about it using Google Gemini AI.



\## Features

\- Upload any PDF document

\- Ask natural language questions

\- Answers grounded in real document content (no hallucination!)

\- Built with Python, Streamlit and Google Gemini API



\## Tech Stack

\- Python

\- Streamlit

\- Google Gemini API (google-genai)

\- PyPDF2



\## How to Run

pip install streamlit google-genai PyPDF2

streamlit run app.py



\## How it Works

1\. PDF uploaded and text extracted

2\. Text split into chunks

3\. Relevant chunks found using keyword search

4\. Gemini AI generates grounded answer


## Retrieval evaluation
I tested how well retrieval finds the passage that answers a question, using 13 questions I wrote about my own 48-page MSc dissertation (7 worded like the text, 6 paraphrased). A question counts as a hit if the top 3 chunks contain the answer phrase.

| Chunk size | Keyword (current app) | TF-IDF | Embeddings (Gemini + FAISS) |
|---|---|---|---|
| 500 words | 11/13 | 12/13 | 12/13 |
| 250 words | 11/13 | 12/13 | 13/13 |
| 150 words | 10/13 | 11/13 | 10/13 |

Notes: only 13 questions that I wrote myself, so a difference of one question is not meaningful. This measures retrieval only, not the quality of Gemini's final answer. The app itself still uses keyword matching with 500-word chunks. The embedding and FAISS code is only in eval_chunksize.py. Embeddings hit the free-tier rate limit (429), so the script pauses between batches. To run it, set GEMINI_API_KEY and edit PDF_PATH in the script.
